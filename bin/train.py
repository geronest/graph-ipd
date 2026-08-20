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

# import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, os.path.realpath("."))
from env.agentnet import AgentNetwork  # noqa: E402
from env.matgame import MatrixGameEnv  # noqa: E402
from env.statemanager import StateManager  # noqa: E402
from models.dan import DANAgent  # noqa: E402
from models.dqn import DQNAgent  # noqa: E402
from models.onlyc import OnlyCooperateAgent  # noqa: E402
from models.onlyd import OnlyDefectAgent  # noqa: E402
from models.rtft import ReverseTitForTatAgent  # noqa: E402
from models.tabularq import TabularQLearningAgent  # noqa: E402
from models.tft import TitForTatAgent  # noqa: E402
from utils.configs import init_config  # noqa: E402
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
from utils.visualize import plot_qvalues  # noqa: E402


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
        return TitForTatAgent
    elif k == "dqn":
        return DQNAgent
    elif k == "dan":
        return DANAgent
    elif k == "rtft":
        return ReverseTitForTatAgent
    elif k == "onlyc":
        return OnlyCooperateAgent
    elif k == "onlyd":
        return OnlyDefectAgent


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

    num_agents = 0
    for key in config.population.keys():
        for i in range(config.population[key]):
            num_agents += 1

    # model initialisation
    for key in config.population.keys():
        for i in range(config.population[key]):
            seed_agent = idx_rep + int(rng.integers(0, num_agents))
            population.append(
                get_type_agent(key)(
                    game,
                    config,
                    stm,
                    name=key,
                    seed_modifier=seed_agent,
                )
            )

            # partner selector
            if config.select_partner:
                seed_selector = idx_rep + int(rng.integers(0, num_agents))
                selectors.append(
                    get_type_agent("dqn")(
                        game,
                        config,
                        stm,
                        "dqn_ps",
                        seed_modifier=seed_selector,
                    )
                )
                selectors[-1].add_name(population[-1].name)

    if idx_rep == 0:
        current_device = population[0].device
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

            agent.store("ids_op", ids_op[j])

            actions_init = rng.integers(0, 2, size=(2,)).tolist()
            agent.store("actions", actions_init)
            reward_init = config.matgame.scores[0][actions_init[0]]
            agent.store("rewards", reward_init)

            if config.select_partner:
                selectors[idx_agent].store(
                    "actions", 0
                )  # TODO: could change afterwards
                selectors[idx_agent].store("rewards", reward_init)

        state_init = stm.encode(agent.history, id_me=idx_agent)
        agent.store("states", state_init)
        if config.select_partner:
            selectors[idx_agent].store("states", state_init)

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


def store_train(models, idx_model, cache_states, cache_actions, cache_rewards):
    """
    store experience, proceed training
    """
    model = models[idx_model]
    history = model.history

    old_state = history["states"][-1]
    old_reward = history["rewards"][-1]
    model.store("states", cache_states[idx_model])
    model.store("rewards", cache_rewards[idx_model])

    if "_ps" in model.name:
        # partner selection model
        old_action = history["actions"][-1]
        model.store("actions", 0)
    else:
        # agent model
        old_action = history["actions"][-1][0]
        model.store("actions", cache_actions[idx_model])

    model.step_train(old_state, old_action, old_reward, history["states"][-1])


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
    names_population = [agent.name for agent in population]
    names_selectors = [selector.name for selector in selectors]

    records = dict()
    avg_reward = [np.mean(returns_agents[i]) for i in range(anet.current_agents)]

    records["returns_agents"] = reset_returns_agents(num_agents)
    records["summary_agents"] = summarise_agents(
        names_population,
        idx_rep,
        num_steps,
        anet.current_agents,
        avg_reward,
        nums_actionpairs,
    )
    if config.select_partner:
        records["summary_selectors"] = summarise_selectors(
            names_selectors,
            idx_rep,
            num_steps,
            anet.current_agents,
            avg_reward,
            nums_selected_lasthistory,
        )

    if (num_steps + 1) < config.num_steps:
        records["nums_actionpairs"] = reset_nums_actionpairs(config, num_agents)
        if config.select_partner:
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


def plot_iteration(
    idx_rep,
    config,
    anet,
    population,
    selectors,
    nums_actionpairs,
    qvalues_agents,
):
    """
    plot information accumulated during the training iteration.
    """
    # save visualized network structure,
    # with strategies of agents and partner selectors depicted as node colours
    path_nimage = results_dir + "/networks/imgs/"
    plot_network_agents(
        anet.network,
        nums_actionpairs,
        path_nimage + f"/agents_{idx_rep}.png",
    )
    if config.select_partner:
        # NOTE: plot_network_selectors() was split into
        # plot_network_selection_actioninfo() and
        # plot_network_selection_lasthistory() (commit 256b37d), which take a
        # different argument list. Only bin/train_ray.py was migrated. Partner
        # selection experiments in the paper were run with bin/train_ray.py;
        # use that script rather than this one.
        raise NotImplementedError(
            "bin/train.py does not support partner selection "
            "(select_partner: True). Use bin/train_ray.py instead, which is "
            "the entrypoint used for the partner selection experiments."
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
    # print(f"[START] iteration {idx_rep}")

    # initialise time profiler
    tp = TimeProfiler(config.profile_time)

    # prepare setting
    game = MatrixGameEnv(config.matgame)

    summaries_agents = list()
    summaries_selectors = list()
    qvalues_agents = list()

    # prepare random number generator: seed modified by idx_rep
    rng = np.random.default_rng(seed=config.random_seeds.experiment + idx_rep)

    # create statemanager
    stm = StateManager(config, seed_modifier=idx_rep)

    tp.start("set_population")
    # initialise agents with assigned properties
    population, selectors = set_population(
        config,
        idx_rep,
        rng,
        game,
        stm,
    )
    tp.end("set_population")

    num_agents = len(population)
    nums_actionpairs = reset_nums_actionpairs(config, num_agents)
    nums_selected_lasthistory = reset_nums_selected_lasthistory(config, num_agents)
    returns_agents = reset_returns_agents(num_agents)

    # define network of agents
    anet = AgentNetwork(config, num_agents, seed_modifier=idx_rep)
    idxs_add_agent = get_idxs_add_agent(config, anet)

    # initialise the environment
    _ = game.reset()
    states_for_op = np.zeros((len(population), stm.dim_state), dtype=np.float32)

    # learning
    for i in tqdm(range(config.num_steps)):
        # add agents in the network in a linear pace so that
        # the network becomes full when 90% of experiment is done
        if config.agentnet.growing and i in idxs_add_agent:
            anet.add_agent()

        tp.start("state_encoding")
        # encode states of every agent, at once
        for idx_agent in range(anet.current_agents):
            states_for_op[idx_agent] = stm.encode(
                population[idx_agent].history, id_me=idx_agent
            )
        tp.end("state_encoding")

        cache_states = list()
        cache_actions = list()
        cache_rewards = list()

        tp.start("loop_act")
        for idx_agent in range(anet.current_agents):
            # select partner of agent IDX_AGENT from the network
            cand_agents = anet.connected_agents(idx_agent)
            if len(cand_agents) == 0:
                continue

            tp.start("partner_selection")
            if config.select_partner:
                qvalues_ps = selectors[idx_agent](states_for_op[cand_agents])
                if rng.random() > config.dqn_ps.epsilon:
                    # collect agents with maximum Q-value, randomly choose among them
                    qvalues_ps_np = qvalues_ps.detach().cpu().reshape(-1).numpy()

                    agents_maxq = cand_agents[qvalues_ps_np == qvalues_ps_np.max()]

                    idx_sample = int(rng.random() * len(agents_maxq))
                    idx_agents = [idx_agent, agents_maxq[idx_sample]]
                else:
                    idx_sample = int(rng.random() * len(cand_agents))
                    idx_agents = [idx_agent, cand_agents[idx_sample]]
                selectors[idx_agent].cached_Q = qvalues_ps[idx_sample]
            else:
                # with probability of epsilon, select randomly
                idx_sample = int(rng.random() * len(cand_agents))
                idx_agents = [idx_agent, cand_agents[idx_sample]]

            # constitute pairs of agents and states based on selection
            agents = [population[idx_agents[0]], population[idx_agents[1]]]
            states_pair = states_for_op[[idx_agents[1], idx_agents[0]]]

            # save the last action history of the selected opponent
            if config.select_partner:
                idx_last_history = stm.get_last_history(states_pair[0]).argmax()
                nums_selected_lasthistory[
                    idx_agents[0], idx_agents[1], idx_last_history
                ] += 1
            tp.end("partner_selection")

            # act
            tp.start("action")
            actions = list()
            prepare_train = [True, False]
            for j in range(2):
                actions.append(agents[j].act(states_pair[j], prepare_train[j]))
            nums_actionpairs[
                idx_agents[0], idx_agents[1], 2 * actions[0] + actions[1]
            ] += 1
            tp.end("action")

            # receive reward from the environment
            rewards, new_states = game.step(tuple(actions))

            cache_states.append(states_pair[0])
            cache_actions.append(actions)
            cache_rewards.append(rewards[0])
        tp.end("loop_act")

        tp.start("loop_train")
        # update agent models, after all agents has played a step
        for idx_agent in range(anet.current_agents):
            # store history & train for the first agent only (for equal sampling)
            returns_agents[idx_agent].append(cache_rewards[idx_agent])
            store_train(
                population, idx_agent, cache_states, cache_actions, cache_rewards
            )

            if config.select_partner:
                # training for partner selection model
                store_train(
                    selectors, idx_agent, cache_states, cache_actions, cache_rewards
                )
        tp.end("loop_train")

        tp.start("record")
        if (i + 1) % config.record_step == 0 or (i + 1) == config.num_steps:
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
                if config.select_partner:
                    nums_selected_lasthistory = records["nums_selected_lasthistory"]

            if config.profile_qvalues:
                summary_qv = records["summary_qv"]
                qvalues_agents.append(summary_qv)
        tp.end("record")

    # save models
    tp.start("saving_models")
    if config.save_models:
        path_models = results_dir + f"/models/iter{idx_rep}/"
        for i in range(num_agents):
            name_model = population[i].name
            population[i].save(path_models + f"model{i}_{name_model}.pth")

            if config.select_partner:
                selectors[i].save(path_models + f"selector{i}_{name_model}.pth")
    tp.end("saving_models")

    # plot accumulated information
    plot_iteration(
        idx_rep, config, anet, population, selectors, nums_actionpairs, qvalues_agents
    )

    # print out idx_rep for progress check
    freq_check = max((config.repeat_experiment // 20), config.num_processes)
    if (idx_rep + 1) % freq_check == 0:
        print(
            f"{(idx_rep+1):>8d}/{config.repeat_experiment}"
            + f"({(idx_rep+1)/config.repeat_experiment * 100:.2f}%) completed"
        )
    tp.summarise()
    # print(f"[END] iteration {idx_rep}")

    # accumulate results
    return [
        summaries_agents,
        summaries_selectors,
        anet.network,
        nums_actionpairs,
        nums_selected_lasthistory,
        qvalues_agents,
    ]


if __name__ == "__main__":
    # load config
    args = parse_args()
    d_set = "" if args.set_id == "default" else args.set_id + "/"
    config = OmegaConf.load(f"configs/{d_set}{args.config}.yaml")
    config = init_config(config)
    title_run = f"{args.set_id}-{args.config}: {args.run_id}"

    # prepare for saving results
    results_dir = create_results_dir(
        args.set_id, args.config, args.run_id, config.repeat_experiment
    )
    result_agents = list()
    result_selectors = list()
    result_networks = list()
    result_last_agents = list()
    result_last_selectors = list()
    result_actionpairs = list()
    result_lasthistory = list()
    result_qvalues = list()

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
        result_agents += result[0]
        result_selectors += result[1]
        result_networks.append(result[2])
        num_agents = result_networks[-1].shape[0]
        result_last_agents.append(result[0][-(num_agents):])
        result_last_selectors.append(result[1][-(num_agents):])
        result_actionpairs.append(result[3])
        result_lasthistory.append(result[4])
        result_qvalues += result[5]

    path_network = results_dir + "/networks"
    path_nimage = path_network + "/imgs"

    # save actionpair information
    result_actionpairs = np.array(result_actionpairs)
    with open(results_dir + "/actionpairs.npy", "wb") as f:
        np.save(f, result_actionpairs)

    # save networks
    with open(path_network + "/nets.npy", "wb") as f:
        np.save(f, np.array(result_networks))

    collect_summary(result_agents, results_dir, config, "agent", "summary_agent_raw")
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
        # save partner selection information
        if config.select_partner:
            result_lasthistory = np.array(result_lasthistory)
            with open(results_dir + "/selections.npy", "wb") as f:
                np.save(f, result_lasthistory)
        collect_summary(
            result_selectors, results_dir, config, "selector", "summary_selector_raw"
        )
        process_summary(
            results_dir,
            config,
            "selector",
            "summary_selector_raw",
            "summary_selector_processed",
        )

    config_save = "_".join(args.config.split("/"))
    with open(results_dir + f"/{config_save}.yaml", "w") as fp:
        OmegaConf.save(config, fp)

    if config.profile_qvalues:
        collect_qvalues_agents(result_qvalues, results_dir, "summary_qvalues_agents")

    # visualize results
    time_plot = time.time()
    plot_dynamics(
        results_dir + "/summary_agent_raw.gz", results_dir + "/dynamics_agent"
    )
    if config.select_partner:
        plot_dynamics(
            results_dir + "/summary_selector_raw.gz",
            results_dir + "/dynamics_selector",
        )

    time_plot = time.time() - time_plot

    # record statistics
    with open(results_dir + "/experiment_summary.txt", "w") as fp:
        fp.write(f"[{title_run}]\n")
        s_time = f"{time_elapsed:.2f} seconds elapsed for the experiment\n"
        fp.write(s_time)
        print("    " + s_time)

        s_time_plot = f"{time_plot:.2f} seconds elapsed for plotting the dynamics\n"
        fp.write(s_time_plot)
        print("    " + s_time_plot)
    print(f"  [{title_run}] finished")
