"""
author: Seongho Son (seong.son.22@ucl.ac.uk)

repeat training of multiple agents connected through network.
* partner selection mechanism added.
parallelized repetitions.
"""

import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd
import ray
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, os.path.realpath("."))
from env.agentnet import AgentNetwork  # noqa: E402
from env.matgame import MatrixGameEnv  # noqa: E402
from env.statemanager import StateManager  # noqa: E402
from models.dan import DANAgent  # noqa: E402
from models.ray.dqn import RayDQNAgent  # noqa: E402
from models.ray.sac import RaySACAgent  # noqa: E402
from models.ray.onlyc import RayOnlyCooperateAgent  # noqa: E402
from models.ray.onlyd import RayOnlyDefectAgent  # noqa: E402
from models.ray.rtft import RayReverseTitForTatAgent  # noqa: E402
from models.ray.tft import RayTitForTatAgent  # noqa: E402
from models.tabularq import TabularQLearningAgent  # noqa: E402
from utils.configs import prepare_config  # noqa: E402
from utils.results import collect_qvalues_agents  # noqa: E402
from utils.results import create_results_dir  # noqa: E402
from utils.results import process_summary  # noqa: E402
from utils.results import provide_network_analysis  # noqa: E402
from utils.results import summarise_qvalues_agents  # noqa: E402
from utils.results import summarise_selectors  # noqa: E402
from utils.results import collect_summary, summarise_agents  # noqa: E402
from utils.time_profiler import TimeProfiler  # noqa: E402
from utils.visualize import plot_dynamics  # noqa: E402
from utils.visualize import plot_network_agents  # noqa: E402
from utils.visualize import plot_network_selection_actioninfo  # noqa: E402
from utils.visualize import plot_network_selection_lasthistory  # noqa: E402
from utils.visualize import plot_qvalues  # noqa: E402

RAY_NUM_GPUS = 0.
RAY_NUM_CPUS = 0.01


def parse_args():
    parser = argparse.ArgumentParser(description="arguments for training code")
    # determine which configuration to use
    parser.add_argument("config", type=str)
    parser.add_argument("--set_id", type=str, default="default")
    parser.add_argument("--run_id", type=str, default="run0")
    return parser.parse_args()


def get_type_agent(k):
    if k == "tabularq":
        return TabularQLearningAgent
    elif k == "tft":
        return RayTitForTatAgent
    elif k == "dqn":
        return RayDQNAgent
    elif k == "sac":
        return RaySACAgent
    elif k == "dan":
        return DANAgent
    elif k == "rtft":
        return RayReverseTitForTatAgent
    elif k == "onlyc":
        return RayOnlyCooperateAgent
    elif k == "onlyd":
        return RayOnlyDefectAgent

def find_ps_name(config):
    for k in config:
        if "_ps" in k:
            return k
    return None

def set_population(
    config,
    idx_rep,
    rng,
    game,
    stm,
):
    """
    set population for agents and other modules.
    """
    population = list()
    selectors = list()
    if torch.cuda.is_available():
        ray_actual_gpus = RAY_NUM_GPUS
    else:
        ray_actual_gpus = 0

    num_agents = 0
    for key in config.population.keys():
        num_agents += config.population[key]

    # model initialisation
    idx_population = 0
    for key in config.population.keys():
        for i in range(config.population[key]):
            seed_agent = (
                idx_rep + int(rng.integers(0, num_agents)) + config.iteration_modifier
            )
            population.append(
                get_type_agent(key)
                .options(num_gpus=ray_actual_gpus, num_cpus=RAY_NUM_CPUS)
                .remote(
                    idx_population,
                    game,
                    config,
                    stm,
                    name=key,
                    seed_modifier=seed_agent,
                )
            )

            # partner selector
            if config.select_partner:
                seed_selector = (
                    idx_rep
                    + int(rng.integers(0, num_agents))
                    + config.iteration_modifier
                )
                name_selector = find_ps_name(config)
                selectors.append(
                    get_type_agent(name_selector.split("_")[0])
                    .options(num_gpus=ray_actual_gpus, num_cpus=RAY_NUM_CPUS)
                    .remote(
                        idx_population,
                        game,
                        config,
                        stm,
                        name_selector,
                        seed_modifier=seed_selector,
                    )
                )
                selectors[-1].add_name.remote(population[-1].getattr.remote("name"))
            idx_population += 1

    if idx_rep == 0:
        current_device = ray.get(population[0].getattr.remote("device"))
        print(f"[train] using device ${current_device}$ for torch")

    # history initialisation: random action history
    for idx_agent in range(len(population)):
        ids_except_me = list(range(0, num_agents))
        ids_except_me.remove(idx_agent)
        ids_except_me = np.array(ids_except_me)
        samples_op = rng.integers(
            0, num_agents - 1, size=(config.state.len_history,)
        ).tolist()
        ids_op = ids_except_me[samples_op]

        agent = population[idx_agent]
        for j in range(config.state.len_history):

            agent.store.remote("ids_op", ids_op[j])

            actions_init = rng.integers(0, 2, size=(2,)).tolist()
            agent.store.remote("actions", actions_init)
            reward_init = config.matgame.scores[0][actions_init[0]]
            agent.store.remote("rewards", reward_init)

            if config.select_partner:
                selectors[idx_agent].store.remote(
                    "actions", 0
                )  # TODO: could change afterwards
                selectors[idx_agent].store.remote("rewards", reward_init)

        state_init = stm.encode(
            ray.get(agent.getattr.remote("history")), id_me=idx_agent
        )
        agent.store.remote("states", state_init)
        if config.select_partner:
            selectors[idx_agent].store.remote("states", state_init)

    return population, selectors


def reset_nums_actionpairs(config, num_agents):
    """
    reset the matrix for recording action pairs agents experience.
    """
    res = np.zeros((num_agents, num_agents, config.matgame.num_actions**2))
    return res


def reset_nums_selected_lasthistory(config, num_agents):
    """
    reset the matrix for recording action pairs agents experience.
    """
    if config.state.history_type == "single":
        res = np.zeros((num_agents, num_agents, config.matgame.num_actions))
    elif config.state.history_type == "pair":
        res = np.zeros((num_agents, num_agents, config.matgame.num_actions * 2))
    return res


def convert_lasthistory_to_index(config, last_history):
    """
    convert last action history to a single index.
    """
    if config.state.history_type == "single":
        res = last_history.argmax()
    elif config.state.history_type == "pair":
        numact = config.matgame.num_actions
        res = (
            2 * last_history[:numact].argmax()
            + last_history[numact : 2 * numact].argmax()
        )
    return res


def reset_returns_agents(num_agents):
    return [list() for i in range(num_agents)]


def get_idxs_add_agent(config, anet):
    """
    get steps at which an agent is added to the network,
    when network is set to grow
    """
    if config.agentnet.growing:
        target_maxnet = int(config.num_steps * config.agentnet.growing_schedule)
        remaining_agents = anet.network.shape[0] - anet.current_agents
        idxs_add_agent = np.arange(1, remaining_agents + 1) * (
            target_maxnet // remaining_agents
        )
        return idxs_add_agent
    return None


def get_records(
    idx_rep,
    num_steps,
    config,
    anet,
    stm,
    returns_agents,
    population,
    selectors,
    num_agents,
    nums_actionpairs,
    nums_selected_lasthistory,
):
    """
    get records during the training.
    """
    names_population = [ray.get(agent.getattr.remote("name")) for agent in population]
    names_selectors = [
        ray.get(selector.getattr.remote("name")) for selector in selectors
    ]

    records = dict()
    sum_reward = [np.sum(returns_agents[i]) for i in range(anet.current_agents)]

    records["returns_agents"] = reset_returns_agents(num_agents)
    records["summary_agents"] = summarise_agents(
        names_population,
        idx_rep,
        num_steps,
        anet.current_agents,
        sum_reward,
        nums_actionpairs,
    )
    if config.select_partner:
        records["summary_selectors"] = summarise_selectors(
            names_selectors,
            idx_rep,
            num_steps,
            anet.current_agents,
            sum_reward,
            nums_selected_lasthistory,
        )

    if (num_steps + 1) < config.num_steps:
        records["nums_actionpairs"] = reset_nums_actionpairs(config, num_agents)
        if config.select_partner or config.select_partner2:
            records["nums_selected_lasthistory"] = reset_nums_selected_lasthistory(
                config, num_agents
            )

    if config.profile_qvalues:
        records["summary_qv"] = summarise_qvalues_agents(
            population,
            stm,
            idx_rep,
            num_steps,
            anet.current_agents,
        )
    return records


def append_nparray(name, path, data_tosave):
    """
    if given np.array was never saved, create one.
    if it existed, append given data to it.
    """

    if os.path.exists(path):
        # load saved networks
        for i in range(100):
            try:
                data_saved = np.load(path, allow_pickle=True)
                data_tosave = np.append(data_saved, data_tosave, axis=0)
                break
            except Exception as e:
                time.sleep(3)
                print(e)
                print(f"[append_nparray-{name}-{i}] waiting for the other process")

    # save data
    with open(path, "wb") as f:
        np.save(f, data_tosave)


def save_run(
    idx_rep,
    config,
    path,
    summaries_agents,
    summaries_selectors,
    anet,
    net_interactions,
):
    """
    save per-run result.
    """

    path_network = path + "/networks/nets.npy"
    path_netparam = path + "/networks/netparam.npy"

    # run summary - agents, selectors
    collect_summary(summaries_agents, path, config, "agent", "summary_agent_raw")
    if config.select_partner:
        collect_summary(
            summaries_selectors, path, config, "selector", "summary_selector_raw"
        )

    # idx_rep
    append_nparray("iteration", path + "/iterations.npy", np.array([idx_rep]))

    # agent network
    append_nparray("agent network", path_network, np.expand_dims(anet.network, axis=0))
    append_nparray("network parameters", path_netparam, np.array([anet.gen_param]))

    # save actionpair information
    append_nparray(
        "action pairs",
        path + "/actionpairs.npy",
        np.array(net_interactions["actionpairs"]),
    )
    if config.select_partner:
        # save partner selection information
        append_nparray(
            "selections",
            path + "/selections.npy",
            np.array(net_interactions["selections"]),
        )
    print(f"    [save_run()] results for run {idx_rep} saved")


def plot_iteration(
    idx_rep,
    config,
    anet,
    population,
    selectors,
    nums_actionpairs,
    nums_selections,
    qvalues_agents,
):
    """
    plot information accumulated during the training iteration.
    """
    # save visualized network structure,
    # with strategies of agents and partner selectors depicted as node colours
    path_nimage = results_dir + "/networks/imgs/"
    names_agents = [
        ray.get(population[i].getattr.remote("name")) for i in range(len(population))
    ]
    plot_network_agents(
        idx_rep,
        config.num_steps,
        names_agents,
        anet.network,
        nums_actionpairs,
        path_nimage + f"/agents_run{idx_rep}_step{config.num_steps}.png",
    )
    if config.select_partner:
        names_selectors = [
            ray.get(selectors[i].getattr.remote("name")) for i in range(len(selectors))
        ]
    elif config.select_partner2:
        names_selectors = [
            ray.get(population[i].getattr.remote("name")) for i in range(len(selectors))
        ]

    if config.select_partner or config.select_partner2:
        plot_network_selection_actioninfo(
            idx_rep,
            config.num_steps,
            names_selectors,
            anet.network,
            nums_actionpairs,
            nums_selections,
            path_nimage
            + f"/selection_actioninfo_run{idx_rep}_step{config.num_steps}.png",
        )
        plot_network_selection_lasthistory(
            idx_rep,
            config.num_steps,
            names_selectors,
            anet.network,
            nums_actionpairs,
            nums_selections,
            path_nimage
            + f"/selection_lasthistory_run{idx_rep}_step{config.num_steps}.png",
        )

    # save q-value plots
    if config.profile_qvalues:
        qvalues_agents = pd.concat(qvalues_agents)
        cols_qv = list(qvalues_agents.columns)[
            len(
                [
                    "iteration",
                    "step",
                    "name",
                    "idx_agent",
                ]
            ) :
        ]
        idx_agent_qv = 0
        path_save = (
            results_dir + f"/subfigures/iter{idx_rep}/qvalues_{idx_agent_qv}.png"
        )
        plot_qvalues(
            config,
            qvalues_agents,
            cols_qv,
            iteration=idx_rep,
            idx_agent=idx_agent_qv,
            path_save=path_save,
        )


def process_train(config, idx_rep, results_dir):
    # initialise time profiler
    tp = TimeProfiler(config.profile_time)

    # prepare setting
    game = MatrixGameEnv(config.matgame)

    summaries_agents = list()
    summaries_selectors = list()
    qvalues_agents = list()

    # prepare random number generator: seed modified by idx_rep
    rng = np.random.default_rng(
        seed=config.random_seeds.experiment + idx_rep + config.iteration_modifier
    )

    # create statemanager
    stm = StateManager(config, seed_modifier=idx_rep + config.iteration_modifier)

    tp.start("set_population")
    # initialise agents with assigned properties
    population, selectors = set_population(
        config,
        idx_rep,
        rng,
        game,
        stm,
    )

    num_agents = len(population)
    tp.end("set_population")

    net_interactions = {"actionpairs": list(), "selections": list()}
    nums_actionpairs = reset_nums_actionpairs(config, num_agents)
    nums_selected_lasthistory = reset_nums_selected_lasthistory(config, num_agents)
    returns_agents = reset_returns_agents(num_agents)

    # define network of agents
    anet = AgentNetwork(
        config, num_agents, seed_modifier=idx_rep + config.iteration_modifier
    )

    # initialise the environment
    _ = game.reset()
    states_for_op = np.zeros((len(population), stm.dim_state), dtype=np.float32)
    states_selected = np.zeros((states_for_op.shape[0], 2, states_for_op.shape[1]))

    # learning
    for i in tqdm(range(config.num_steps)):
        train_this_step = (i + 1) % config.update_frequency == 0

        tp.start("state_encoding")
        # encode states of every agent, at once
        states_remote = [
            population[idx_agent].produce_state.remote()
            for idx_agent in range(anet.current_agents)
        ]
        states_for_op = np.array(ray.get(states_remote))
        tp.end("state_encoding")

        tp.start("loop_train_selector")
        if config.select_partner and train_this_step:
            # training for partner selection model
            for idx_agent in range(anet.current_agents):
                cand_agents = anet.connected_agents(idx_agent)
                selectors[idx_agent].step_train.remote(
                    states_for_op[cand_agents],
                )
        tp.end("loop_train_selector")

        tp.start("partner selection")
        tp.start("partner selection-select_partner")
        agent_pairs = np.zeros((anet.current_agents, 2), dtype=np.int32) - 1
        agent_pairs[:, 0] = np.arange(anet.current_agents)
        temp_ops = list()
        for idx_agent in range(anet.current_agents):
            # select partner of agent IDX_AGENT from the network
            cand_agents = anet.connected_agents(idx_agent)
            if len(cand_agents) == 0:
                raise ValueError
            if config.select_partner:
                temp_ops.append(
                    selectors[idx_agent].select_partner.remote(
                        cand_agents, states_for_op
                    )
                )
            elif config.select_partner2:
                temp_ops.append(
                    population[idx_agent].select_partner2.remote(
                        cand_agents, states_for_op
                    )
                )
            else:
                agent_pairs[idx_agent, 1] = cand_agents[
                    rng.integers(0, len(cand_agents))
                ]
        tp.end("partner selection-select_partner")

        tp.start("partner selection-select_partner_get")
        if config.select_partner or config.select_partner2:
            agent_pairs[:, 1] = ray.get(temp_ops)
        tp.end("partner selection-select_partner_get")

        tp.start("partner selection-record")
        # constitute pairs of agents and states based on selection
        for idx_agent in range(anet.current_agents):
            states_selected[idx_agent] = states_for_op[
                [agent_pairs[idx_agent, 1], agent_pairs[idx_agent, 0]]
            ]
            # save the last action history of the selected opponent
            if config.select_partner or config.select_partner2:
                idx_last_history = convert_lasthistory_to_index(
                    config, stm.get_last_history(states_selected[idx_agent, 0])
                )
                nums_selected_lasthistory[
                    agent_pairs[idx_agent, 0],
                    agent_pairs[idx_agent, 1],
                    idx_last_history,
                ] += 1
        tp.end("partner selection-record")
        tp.end("partner selection")

        tp.start("loop_train_agent")
        # update agent models, after all agents has played a step
        if train_this_step:
            for idx_agent in range(anet.current_agents):
                population[idx_agent].step_train.remote(states_selected[idx_agent, 0])
        tp.end("loop_train_agent")

        # act
        tp.start("loop_act")
        tp.start("loop_act-remote")
        actions = np.zeros((anet.current_agents, 2), dtype=np.int32)
        rewards = np.zeros((anet.current_agents, 2))
        temp_actions = []
        is_me = [True, False]
        acting_agents = np.arange(anet.current_agents)
        if config.selected_history:
            rng.shuffle(acting_agents)
        for j in range(2):
            temp_actions.append(
                [
                    population[agent_pairs[idx_agent, j]].act.remote(
                        states_selected[idx_agent, j], is_me[j]
                    )
                    for idx_agent in acting_agents
                ]
            )
        tp.end("loop_act-remote")

        tp.start("loop_act-get")
        for j in range(2):
            actions[acting_agents, j] = ray.get(temp_actions[j])
        tp.end("loop_act-get")

        tp.start("loop_act-step_record")
        for idx_agent in acting_agents:
            # receive reward from the environment
            rewards_pair, new_states = game.step(actions[idx_agent])
            rewards[idx_agent] = rewards_pair

            # store state and action in the history
            d_store = dict()
            d_store["actions"] = actions[idx_agent]
            d_store["ids_op"] = agent_pairs[idx_agent, 1]
            population[idx_agent].store_multiple.remote(d_store)

            # record action pair information
            # store rewards in the training buffer score buffer
            for j in range(2):
                idx_op = 1 - j
                nums_actionpairs[
                    agent_pairs[idx_agent, j],
                    agent_pairs[idx_agent, idx_op],
                    2 * actions[idx_agent, j] + actions[idx_agent, idx_op],
                ] += 1
                population[agent_pairs[idx_agent, j]].store_buffer.remote(
                    "rewards", rewards[idx_agent, j]
                )
                returns_agents[agent_pairs[idx_agent, j]].append(rewards[idx_agent, j])

            if config.select_partner:
                selectors[idx_agent].store_buffer.remote(
                    "rewards", rewards[idx_agent, 0]
                )
        tp.end("loop_act-step_record")
        tp.end("loop_act")

        # rewire
        tp.start("loop_rewire")
        rewire_ray = list()
        rewire_results = list()
        if (config.freq_rewire > 0) and ((i + 1) % config.freq_rewire == 0):
            for idx_agent in range(anet.current_agents):
                rewire_ray.append(
                    population[idx_agent].rewire.remote(
                        anet.connected_agents(idx_agent), states_for_op
                    )
                )

            rewire_results = ray.get(rewire_ray)
            rng.shuffle(rewire_results)
            for res_rw in rewire_results:
                if res_rw["attach"] is not None:
                    anet.set_edge(res_rw["idx_me"], res_rw["attach"], 1)
                if (
                    (res_rw["detach"] is not None)
                    and (len(anet.connected_agents(res_rw["idx_me"])) > 1)
                    and (len(anet.connected_agents(res_rw["detach"])) > 1)
                ):
                    anet.set_edge(res_rw["idx_me"], res_rw["detach"], 0)

        tp.end("loop_rewire")

        tp.start("record")
        if (i + 1) % config.record_step == 0 or (i + 1) == config.num_steps:
            # save pairwise interaction history
            if (i + 1) % config.agentnet.netsave_step == 0 or (
                i + 1
            ) == config.num_steps:
                net_interactions["actionpairs"].append(nums_actionpairs)
                if config.select_partner or config.select_partner2:
                    net_interactions["selections"].append(nums_selected_lasthistory)

            # process and accumulate summaries
            records = get_records(
                idx_rep,
                i,
                config,
                anet,
                stm,
                returns_agents,
                population,
                selectors,
                num_agents,
                nums_actionpairs,
                nums_selected_lasthistory,
            )
            returns_agents = records["returns_agents"]

            summaries_agents += records["summary_agents"]
            if config.select_partner:
                summaries_selectors += records["summary_selectors"]

            # training not finished
            if (i + 1) < config.num_steps:
                nums_actionpairs = records["nums_actionpairs"]
                if config.select_partner or config.select_partner2:
                    nums_selected_lasthistory = records["nums_selected_lasthistory"]

            if config.profile_qvalues:
                summary_qv = records["summary_qv"]
                qvalues_agents.append(summary_qv)
        tp.end("record")

        # modify network structure when needed
        anet.modify_structure(i)

    """
    Save results: that can generate all the others
        - raw interaction data
        - generated network
        - action pair information
    """
    save_run(
        idx_rep,
        config,
        results_dir,
        summaries_agents,
        summaries_selectors,
        anet,
        net_interactions,
    )

    # plot accumulated information
    plot_iteration(
        idx_rep,
        config,
        anet,
        population,
        selectors,
        nums_actionpairs,
        nums_selected_lasthistory,
        qvalues_agents,
    )

    # save models
    tp.start("saving_models")
    if config.save_models:
        path_models = results_dir + f"/models/iter{idx_rep}/"
        for i in range(num_agents):
            name_model = ray.get(population[i].get_name.remote())
            # population[i].save.remote(path_models + f"model{i}_{name_model}.pth")
            torch.save(
                ray.get(population[i].state_dict.remote()),
                path_models + f"model{i}_{name_model}.pth",
            )

            if config.select_partner:
                # selectors[i].save.remote(path_models +
                # f"selector{i}_{name_model}.pth")
                torch.save(
                    ray.get(selectors[i].state_dict.remote()),
                    path_models + f"selector{i}_{name_model}.pth",
                )
    tp.end("saving_models")

    # print out idx_rep for progress check
    freq_check = max((config.repeat_experiment // 20), config.num_processes)
    if (idx_rep + 1) % freq_check == 0:
        print(
            f"{(idx_rep+1):>8d}/{config.repeat_experiment}"
            + f"({(idx_rep+1)/config.repeat_experiment * 100:.2f}%) completed"
        )
    tp.summarise(results_dir + "/time_profile_summary.txt", idx_rep)
    # print(f"[END] iteration {idx_rep}")

    # accumulate results
    return [
        qvalues_agents,
    ]


def postprocess_results(
    config,
    results_dir,
    result_qvalues,
):
    """
    use accumulated results to generate processed summary and plots.
    """

    path_network = results_dir + "/networks"

    process_summary(
        results_dir,
        config,
        "agent",
        "summary_agent_raw",
        "summary_agent_processed",
    )

    provide_network_analysis(
        results_dir + "/summary_agent_processed_peragent.csv",
        path_network + "/nets.npy",
    )

    if config.select_partner:
        process_summary(
            results_dir,
            config,
            "selector",
            "summary_selector_raw",
            "summary_selector_processed",
        )

    if config.profile_qvalues:
        collect_qvalues_agents(
            result_qvalues, results_dir, config, "summary_qvalues_agents"
        )

    # visualize results
    plot_dynamics(
        results_dir + "/summary_agent_raw.gz", results_dir + "/dynamics_agent"
    )
    if config.select_partner:
        plot_dynamics(
            results_dir + "/summary_selector_raw.gz",
            results_dir + "/dynamics_selector",
        )


def setup_slackmessenger(config):
    """
    set up SlackMessenger when needed.
    """
    slm = None
    if "slack_cfg" in config:
        # Imported lazily: slack-sdk is only needed if run progress is being
        # posted to Slack, which is off unless `slack_cfg` is set.
        from utils.slack_messenger import SlackMessenger

        slm = SlackMessenger(config.slack_cfg)

    return slm


if __name__ == "__main__":
    # load config
    args = parse_args()
    d_set = "" if args.set_id == "default" else args.set_id + "/"
    config = OmegaConf.load(f"configs/{d_set}{args.config}.yaml")
    config = prepare_config(d_set, config)
    title_run = f"{args.set_id}-{args.config}: {args.run_id}"

    # set SlackMessenger if configuration is specified
    slm = setup_slackmessenger(config)

    # prepare for saving results
    results_dir = create_results_dir(
        args.set_id, args.config, args.run_id, config.repeat_experiment
    )
    result_agents = list()
    result_selectors = list()
    result_networks = list()
    result_actionpairs = list()
    result_lasthistory = list()
    result_qvalues = list()

    # backup configuration
    config_save = "_".join(args.config.split("/"))
    with open(results_dir + f"/{config_save}.yaml", "w") as fp:
        OmegaConf.save(config, fp)

    # check elapsed time for the experiment
    time_start = time.time()

    print(f"\n  [{title_run}] started")
    print(f"    start: parallelized with {config.num_processes} processes")
    with mp.Pool(processes=config.num_processes) as pool:
        results = pool.starmap(
            process_train,
            [[config, i, results_dir] for i in range(config.repeat_experiment)],
        )
    print(f"    experiment with {config.repeat_experiment} repetitions ended")

    # save results
    time_elapsed = time.time() - time_start

    # summarise results
    for result in results:
        result_qvalues += result[0]

    time_postprocess = time.time()
    postprocess_results(config, results_dir, result_qvalues)
    time_postprocess = time.time() - time_postprocess

    # record statistics
    s_summary = ""
    s_summary += f"[{title_run}]\n"
    s_time = f"{time_elapsed:.2f} seconds elapsed for the experiment\n"
    s_time_post = f"{time_postprocess:.2f} seconds elapsed for postprocessing results\n"
    s_summary += s_time + s_time_post

    print("    " + s_time)
    print("    " + s_time_post)
    with open(results_dir + "/experiment_summary.txt", "w") as fp:
        fp.write(s_summary)
    if slm is not None:
        slm.post_message(s_summary)
    print(f"  [{title_run}] finished")
