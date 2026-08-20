"""
author: Seongho Son (seong.son.22@ucl.ac.uk)
various functions for processing experiment results.
"""

import os
import time

import numpy as np
import pandas as pd
import scipy.stats as st

# from models.agent import Agent

# from omegaconf import OmegaConf

# isort: off
from .analysis_net import (
    average_distances,
    centrality,
    clustering_coefficient,
    count_clusters,
    distance,
    assortativity,
)

# isort: on

COLS_BASIC = [
    "iteration",
    "step",
    "model",
    "idx_agent",
    "sum_reward",
]
COLS_ACTIONPAIR = [
    "(D,D)",
    "(D,C)",
    "(C,D)",
    "(C,C)",
]
COLS_ACTION = ["D", "C"]


def create_results_dir(name_exp, name_config, run_id, repeat_experiment=1):
    # make a directory for the result of a specific run
    name_dir = f"./results/{name_exp}/{name_config}/{run_id}"
    os.makedirs(name_dir, exist_ok=True)

    # make directories for network
    os.makedirs(name_dir + "/networks/imgs", exist_ok=True)

    # make directories for models and iteration-wise figures
    os.makedirs(name_dir + "/models", exist_ok=True)
    for i in range(repeat_experiment):
        os.makedirs(name_dir + f"/models/iter{i}", exist_ok=True)
        os.makedirs(name_dir + f"/subfigures/iter{i}", exist_ok=True)
    return name_dir


def summarise_qvalues(qvalues):
    """
    ** assumes 2 * 2 structure.

    inputs: (2, 2) numpy array
    return: (2) numpy array, consisting of integers
    """
    return qvalues.argmax(axis=1)


def rank_qvalues(qvalues):
    """
    arguments:
        qvalues: 1-dimensional array of Q-values.
    returns:
        ranking of qvalues, in descending order.
        (highest value being rank 1)
    """
    return np.argsort(np.argsort(qvalues)[::-1]) + 1


def summarise_agents(
    names,
    iteration=0,
    step=0,
    num_agents=0,
    sum_reward=None,
    nums_actionpairs=None,
):
    summary = list()
    for idx in range(num_agents):
        row = [iteration, step, names[idx], idx, sum_reward[idx]] + nums_actionpairs[
            idx
        ].sum(axis=0).tolist()

        summary.append(row)

    return summary


def summarise_selectors(
    names,
    iteration=0,
    step=0,
    num_agents=0,
    sum_reward=None,
    nums_selected_lasthistory=None,
):
    """
    summarise selectors' preference of strategies.
    arguments:
        selectors: list of selector models
        iteration
        step
        sum_reward
        nums_actionpairs
        len_history
    returns:
        summary of selectors' Q-values
    """
    summary = list()
    for idx in range(num_agents):
        summary.append(
            [
                iteration,
                step,
                names[idx],
                idx,
                sum_reward[idx],
            ]
            + nums_selected_lasthistory[idx].sum(axis=0).tolist()
        )

    return summary


def summarise_qvalues_agents(
    population,
    stm,
    iteration=0,
    step=0,
    num_agents=0,
):
    """
    summarise qvalues of agents at a specified iteration.
    """
    summary = list()
    states = stm.possible_states
    cols = [
        "iteration",
        "step",
        "name",
        "idx_agent",
    ] + columns_qvalues(stm)
    for agent in population[:num_agents]:
        if "dqn" not in agent.name:
            continue
        row = [
            int(iteration),
            step,
            agent.name,
            population.index(agent),
        ]

        qvalues = agent(states).reshape(-1).detach().numpy().tolist()

        summary.append(row + qvalues)

    summary = pd.DataFrame(summary, columns=cols)
    return summary


def collect_summary(
    summaries, path_results, config, target="agent", fname="summary_agent_raw"
):
    if target == "agent":
        cols_summary = COLS_BASIC + COLS_ACTIONPAIR
    elif target == "selector":
        if config.state.history_type == "single":
            cols_summary = COLS_BASIC + COLS_ACTION
        elif config.state.history_type == "pair":
            cols_summary = COLS_BASIC + COLS_ACTIONPAIR

    df_raw = pd.DataFrame(
        summaries,
        columns=cols_summary,
    )

    path_df = path_results + f"/{fname}.gz"
    if os.path.exists(path_df):
        # load df, append
        for i in range(100):
            try:
                df_saved = pd.read_csv(path_df, index_col=0)
                pd.concat([df_saved, df_raw]).to_csv(path_df)
                break
            except Exception as e:
                print(e)
                print(f"[collect_summary-{i}] waiting for the other process")
                time.sleep(3)
    else:
        # no prior data exists, create
        df_raw.to_csv(path_df)

    return True


def columns_qvalues(stm):
    """
    produce columns for qvalue profiling.
    """

    cols = list()
    possible_state_ids = stm.possible_state_ids
    for sid in possible_state_ids:
        for action in COLS_ACTION:
            cols.append(f"{action} | {sid}")

    return cols


def collect_qvalues_agents(summaries, path_results, fname="summary_qvalues_agents"):
    df_raw = pd.concat(summaries)
    df_raw.to_csv(path_results + f"/{fname}.gz")


def factorial_permutation(items):
    """
    return possible permutations of given items.
    arguments: list of items
    returns: list of possible permutations
    """

    if len(items) > 1:
        res = list()
        for item in items:
            l_next = items.copy()
            l_next.remove(item)
            for r_next in factorial_permutation(l_next):
                r_next.append(item)
                res.append(r_next)
        return res
    else:
        return [items]


def calculate_CIs(vs, alpha=0.95):
    """
    calculate confidence interval with the given alpha.
    """
    # NOTE: passed positionally on purpose. SciPy renamed this parameter
    # from `alpha` to `confidence` in 1.9; positional works on both.
    return st.t.interval(alpha, df=len(vs) - 1, loc=vs.mean(), scale=st.sem(vs))


def process_summary(
    path_results,
    config,
    target="agent",
    fname_open="summary_agent_raw",
    fname_save="summary_agent_processed",
    average_percentage=None,
):
    """
    process raw summary into readable form, with useful information

    arguments:
        path_results: directory of the raw summary
        path_net: directory of saved network in .npy format
        config
        target: whether the targeted information is regarding agent or something else
        fname
    """
    df_raw = pd.read_csv(
        path_results + f"/{fname_open}.gz",
        index_col=0,
    )

    # process df_raw to summarise across iterations
    if average_percentage is not None:
        cols_groupby = ["iteration", "model", "idx_agent"]
        df_raw = (
            df_raw.loc[df_raw.step >= (config.num_steps * (1 - average_percentage))]
            .groupby(cols_groupby)
            .mean()
            .reset_index(level=cols_groupby)
        )
    else:
        df_raw = df_raw.loc[df_raw.step == (config.num_steps - 1)]
    df_processed_0 = df_raw.drop(["step"], axis=1)
    # save df for individual analysis of agents with network structure
    path_df_peragent = path_results + f"/{fname_save}_peragent.csv"
    df_processed_0.to_csv(path_df_peragent)

    # exclude isolated nodes (hence never learned)
    # df_processed_0 = df_processed_0.loc[df_processed_0.centrality_degree > 0]
    df_processed_0 = df_processed_0.dropna()
    df_processed_0["count"] = 1
    num_runs = df_processed_0.iteration.value_counts().shape[0]
    # num_agents = df_processed_0.idx_agent.value_counts().shape[0]

    df_processed_1 = (
        df_processed_0.drop(["idx_agent"], axis=1).groupby(["iteration", "model"]).sum()
    )
    for k in ["sum_reward"]:
        df_processed_1[k + "_cache"] = df_processed_1[k]

    if target == "agent":  # interactions: only applicable to agents
        for k in ["(C,C)"]:
            df_processed_1[k + "_cache"] = df_processed_1[k]

    df_processed_1["proportion"] = (
        df_processed_1["count"]
        / df_processed_1.groupby(["iteration", "model"])["count"].sum()
    )

    groupby_dfp1 = df_processed_1.groupby(["model"])

    df_processed_2 = groupby_dfp1.sum()
    df_processed_2["proportion"] = (
        df_processed_2["count"] / df_processed_2["count"].sum()
    )
    proportions = df_processed_1["proportion"]
    CIs_proportions = calculate_CIs(proportions.values, 0.95)
    df_processed_2["proportion_CI95_low"] = CIs_proportions[0]
    df_processed_2["proportion_CI95_high"] = CIs_proportions[1]

    if target == "agent":
        CCs = df_processed_1["(C,C)_cache"]
        num_interactions = df_processed_2[COLS_ACTIONPAIR].sum(axis=1)
        df_processed_2["CC_prop"] = df_processed_2["(C,C)"] / df_processed_2[
            COLS_ACTIONPAIR
        ].sum(axis=1)
        CIs_CCs = calculate_CIs(CCs.values, 0.95)
        df_processed_2["CC_prop_CI95_low"] = CIs_CCs[0] / (num_interactions / num_runs)
        df_processed_2["CC_prop_CI95_high"] = CIs_CCs[1] / (num_interactions / num_runs)

    sum_rewards = df_processed_1["sum_reward_cache"]
    df_processed_2["sum_reward"] = df_processed_2["sum_reward"] / num_runs
    CIs_sum_rewards = calculate_CIs(sum_rewards.values, 0.95)
    df_processed_2["sum_reward_CI95_low"] = CIs_sum_rewards[0]
    df_processed_2["sum_reward_CI95_high"] = CIs_sum_rewards[1]
    num_denominator = (df_processed_2["count"] / num_runs) + 1e-4
    df_processed_2["avg_reward"] = df_processed_2["sum_reward"] / num_denominator
    df_processed_2["avg_reward_CI95_low"] = (
        df_processed_2["sum_reward_CI95_low"] / num_denominator
    )
    df_processed_2["avg_reward_CI95_high"] = (
        df_processed_2["sum_reward_CI95_high"] / num_denominator
    )

    df_processed_2 = df_processed_2.drop(
        [k + "_cache" for k in ["sum_reward"]],
        axis=1,
    )
    if target == "agent":
        df_processed_2 = df_processed_2.drop(["(C,C)_cache"], axis=1)

    df_total = pd.DataFrame(
        [[0] * (len(df_processed_2.columns))],
        columns=df_processed_2.columns,
        index=pd.MultiIndex.from_tuples(
            [tuple(["total"] + [0] * (len(df_processed_2.index.names) - 1))],
            names=tuple(df_processed_2.index.names),
        ),
    )
    if target == "agent":
        cols_actioninfo = COLS_ACTIONPAIR
    if target == "selector":
        if config.state.history_type == "single":
            cols_actioninfo = COLS_ACTION
        elif config.state.history_type == "pair":
            cols_actioninfo = COLS_ACTIONPAIR

    for k in ["sum_reward", "proportion", "count"] + cols_actioninfo:
        df_total[k] = df_processed_2[k].sum()

    df_processed_3 = df_processed_1.groupby(["iteration"]).sum()

    if target == "agent":
        CCs = df_processed_3["(C,C)"]
        num_interactions = df_total[COLS_ACTIONPAIR].sum().sum()
        df_total["CC_prop"] = CCs.sum() / num_interactions
        CIs_CCs = calculate_CIs(CCs.values, 0.95)
        df_total["CC_prop_CI95_low"] = CIs_CCs[0] / (num_interactions / num_runs)
        df_total["CC_prop_CI95_high"] = CIs_CCs[1] / (num_interactions / num_runs)

    sum_rewards = df_processed_3["sum_reward"]
    df_total["sum_reward"] = sum_rewards.sum() / num_runs
    CIs_sum_rewards = calculate_CIs(sum_rewards.values, 0.95)
    df_total["sum_reward_CI95_low"] = CIs_sum_rewards[0]
    df_total["sum_reward_CI95_high"] = CIs_sum_rewards[1]
    num_denominator = (df_total["count"] / num_runs) + 1e-4
    df_total["avg_reward"] = df_total["sum_reward"] / num_denominator
    df_total["avg_reward_CI95_low"] = df_total["sum_reward_CI95_low"] / num_denominator
    df_total["avg_reward_CI95_high"] = (
        df_total["sum_reward_CI95_high"] / num_denominator
    )

    df_processed_4 = pd.concat([df_processed_2, df_total])
    df_processed_4.to_csv(path_results + f"/{fname_save}.csv")


def calculate_expected_edges(config):
    """
    calculate expected number of edges given the configuration.

    arguments:
        config: configuration used for generating the network

    returns:
        num_edges: number of expected edges
    """

    # get number of agents involved
    num_agents = sum([config.population[k] for k in config.population.keys()])

    if config.agentnet.name == "ER":
        return int(config.agentnet.p * num_agents * (num_agents - 1) / 2)

    elif config.agentnet.name == "WS":
        return num_agents * config.agentnet.half_degree_regular

    elif config.agentnet.name == "BA":
        return 1 + ((num_agents - 2) * config.agentnet.m)

    else:  # assume the network is complete
        return num_agents * (num_agents - 1) / 2


def provide_network_analysis(
    path_df,
    path_nets,
):
    """
    provide measures for network analysis.

    arguments:
        path_df
        path_nets
    """
    df_agents = pd.read_csv(
        path_df,
        index_col=0,
    )

    with open(path_nets, "rb") as f:
        networks = np.load(f)

    # calculate distance
    # exclude nodes that are isolated
    mat_dis, mat_nsp = distance(networks)
    avg_dis = average_distances(mat_dis)

    df_agents["max_distance"] = mat_dis.max()
    df_agents["average_distance"] = avg_dis.mean()

    # calculate clustering coefficient
    df_agents["coef_clustering"] = clustering_coefficient(networks).reshape(-1)

    # calculate degree
    for mode in ["degree", "closeness", "betweenness"]:
        df_agents[f"centrality_{mode}"] = centrality(networks, mode).reshape(-1)

    # count clusters
    num_clusters = count_clusters(networks)
    df_agents["average_clusters"] = num_clusters.mean()

    # assortativity
    astt = assortativity(networks)
    df_agents["average_assortativity"] = astt.mean()

    # save results back to the same dataframe
    df_agents.to_csv(path_df)

    return df_agents


def get_dynamics(path_df):
    """
    get temporal dynamics of agents' actions and rewards
    by reading .csv file of experiment result pd.DataFrame

    arguments:
        path_df: pd.DataFrame containing strategy distribution of agents
    returns:
        steps: steps at which agents were recorded
        models: names of models used in the experiment
        rewards: average reward of strategies per iteration and recorded intervals
        actions: number of actions per iteration and recorded steps
        actionpairs: number of action pairs per iteration and recorded steps
    """
    df_raw = pd.read_csv(path_df, index_col=0)
    if COLS_ACTION[0] in df_raw.columns:
        cols_actioninfo = COLS_ACTION
    elif COLS_ACTIONPAIR[0] in df_raw.columns:
        cols_actioninfo = COLS_ACTIONPAIR

    cols_common = [
        "iteration",
        "step",
        "model",
        "idx_agent",
        "sum_reward",
    ] + cols_actioninfo
    cols_remove = list()
    for col in cols_common:
        if col not in df_raw.columns:
            cols_remove.append(col)
    for col in cols_remove:
        cols_common.remove(col)

    models = dict()
    rewards = dict()
    for k in df_raw["model"].value_counts().index:
        models[k] = list()
        rewards[k] = list()
    rewards["total"] = list()

    actions = {"D": list(), "C": list()}
    actioninfos = dict()
    for actioninfo in cols_actioninfo:
        actioninfos[actioninfo] = list()

    iterations = 0
    steps = list()
    for name1, group1 in df_raw.groupby("iteration"):
        for k in models:
            rewards[k].append(list())
        rewards["total"].append(list())
        for k in actions:
            actions[k].append(list())
        for k in actioninfos:
            actioninfos[k].append(list())

        for name2, group2 in group1.groupby("step"):
            if iterations == 0:
                steps.append(name2)
            for k in models:
                rewards[k][-1].append(0)

            # num_agents = len(group2)
            for name3, group3 in group2.groupby("model"):
                rewards[name3][-1][-1] = group3["sum_reward"].mean()
            rewards["total"][-1].append(group2["sum_reward"].mean())

            freq_actioninfos = group2[cols_actioninfo].sum()
            freq_actioninfos /= freq_actioninfos.sum()
            for k in actioninfos:
                actioninfos[k][-1].append(freq_actioninfos[k])
            if len(actioninfos.keys()) == 2:
                actions["D"][-1].append(freq_actioninfos[cols_actioninfo[0]].sum())
                actions["C"][-1].append(freq_actioninfos[cols_actioninfo[1]].sum())
            elif len(actioninfos.keys()) == 4:
                actions["D"][-1].append(freq_actioninfos[cols_actioninfo[0:2]].sum())
                actions["C"][-1].append(freq_actioninfos[cols_actioninfo[2:4]].sum())

        iterations += 1

    return steps, models, rewards, actions, actioninfos


def smooth_props(
    path_results,
    config,
    fname_open="summary_agent_raw.gz",
    fname_save="smoothed_props.csv",
):
    """
    calculate and save the smoothed stats of strategy proportions.

    argumensts:
        path_df
        config
        fname_save
    """
    steps, models, rewards, actions, actionpairs = get_dynamics(
        path_results + "/" + fname_open
    )

    d_results = dict()
    nums_stats = (config.num_steps // 4) // config.record_step
    for k in models:
        prop_s = np.array(models[k]).mean(axis=0)
        d_results[k] = dict()
        # results of last 20k steps
        d_results[k]["std"] = prop_s[-nums_stats:].std(axis=0)
        d_results[k]["mean"] = prop_s[-nums_stats:].mean(axis=0)

    df_prop_smoothed = list()
    cols_prop_smoothed = ["strategy", "prop_mean", "prop_std"]
    for k in d_results:
        df_prop_smoothed.append(
            [
                k,
                d_results[k]["mean"],
                d_results[k]["std"],
            ]
        )

    df_prop_smoothed = pd.DataFrame(
        df_prop_smoothed,
        columns=cols_prop_smoothed,
    )
    df_prop_smoothed.to_csv(path_results + "/" + fname_save)


def check_strategy(
    path_models,
    path_config,
):
    """
    load models, check strategy of agents.
    * assumes len_history=1
    """

    # config = OmegaConf.load(path_config)

    res = list()
    l_iters = os.listdir(path_models)
    for path_iter in l_iters:
        l_models = os.listdir(path_iter)
        for path_model in l_models:
            res.append([path_iter, path_model])

    print(res)
