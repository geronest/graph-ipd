"""
author: Seongho Son (seong.son.22@ucl.ac.uk)
various functions for visualization of experiment data.
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .colors import ColorRevolver
from .results import get_dynamics

COLOURS_ACTIONPAIRS = {
    "(D,D)": "tab:red",
    "(D,C)": "tab:green",
    "(C,D)": "tab:purple",
    "(C,C)": "tab:blue",
}
LABELS_ACTIONPAIRS = {
    "(D,D)": "mutual def.",
    "(D,C)": "temptation",
    "(C,D)": "sucker",
    "(C,C)": "mutual coop.",
}

COLOURS_ACTIONS = {"D": "tab:red", "C": "tab:blue"}
LABELS_ACTIONS = {"D": "defection", "C": "cooperation"}


def plot_values(dict_values, idxs, path_save):
    plt.figure()
    for k1 in dict_values.keys():
        for k2 in dict_values[k1].keys():
            plt.plot(idxs, dict_values[k1][k2], label=k1 + "_" + k2)
    plt.legend()
    plt.savefig(path_save)
    plt.close()


def plot_dynamics(path_df, path_save, max_values=5):
    """
    plot temporal dynamics of agents' strategies and rewards
    X axis: steps (episodes) passed
    Y axis: proportion of a given type of strategy
    * 23/Dec/2022: assumes history length 1

    arguments:
        path_df: pd.DataFrame containing strategy distribution of agents
        path_save: path for saving plotting result

    """

    steps, models, rewards, actions, actioninfos = get_dynamics(path_df)
    ks = list(models.keys())

    # parse experiment_id and run_id for plot title
    info_title = "_".join(path_df.split("/")[-3:-1])

    plot_dynamics_rewards(
        path_save, steps, rewards, ks + ["total"], info_title, max_values
    )
    plot_dynamics_actions(path_save, steps, actions, info_title)
    plot_dynamics_actioninfos(path_save, steps, actioninfos, info_title)


def plot_dynamics_strategies(
    path_save,
    steps,
    props_strategy,
    colours,
    ks,
    info_title,
    max_values=5,
):
    """
    plot dynamics of strategy proportion.
    """

    means = dict()
    stds = dict()
    last_means = list()
    for k in props_strategy:
        props_strategy[k] = np.array(props_strategy[k])
        means[k] = props_strategy[k].mean(axis=0)
        stds[k] = props_strategy[k].std(axis=0)
        last_means.append(means[k][-1])

    ranking_means = np.flip(np.argsort(last_means))

    num_subplots = 1
    if len(ks) > max_values:
        num_subplots = len(ks) // max_values
        num_subplots += 1 if (len(ks) % max_values > 0) else 0

    fig, axes = plt.subplots(num_subplots, 1, figsize=(5, 5 * num_subplots))
    fig.suptitle(f"[strategy dynamics] {info_title}")

    idx_value = 0
    for rank in ranking_means:
        idx_ax = idx_value // max_values
        k = ks[rank]
        ax = axes[idx_ax] if num_subplots > 1 else axes
        ax.set_ylim(-0.05, 1.05)
        color_strategy = colours[k.split("_")[-1]]
        ax.plot(steps, means[k], color=color_strategy, label=k)
        ax.fill_between(
            steps,
            means[k] - stds[k],
            means[k] + stds[k],
            color=color_strategy,
            alpha=0.2,
        )
        idx_value += 1

    if num_subplots > 1:
        for ax in axes:
            ax.set_xlabel("steps")
            ax.set_ylabel("proportion")
            ax.legend()
    else:
        axes.set_xlabel("steps")
        axes.set_ylabel("proportion")
        axes.legend()

    plt.savefig(path_save + "_strategies.png")
    plt.close()


def plot_dynamics_rewards(
    path_save,
    steps,
    rewards,
    ks,
    info_title,
    max_values=5,
):
    """
    plot dynamics of average rewards.
    """

    means = dict()
    stds = dict()
    last_means = list()
    for k in rewards:
        rewards[k] = np.array(rewards[k])
        means[k] = rewards[k].mean(axis=0)
        stds[k] = rewards[k].std(axis=0)
        last_means.append(means[k][-1])

    ranking_means = np.flip(np.argsort(last_means))

    num_subplots = 1
    if len(ks) > max_values:
        num_subplots = len(ks) // max_values
        num_subplots += 1 if (len(ks) % max_values > 0) else 0

    fig, axes = plt.subplots(num_subplots, 1, figsize=(5, 5 * num_subplots))
    fig.suptitle(f"[reward sum dynamics] {info_title}")

    crevolver = ColorRevolver()

    idx_value = 0
    for rank in ranking_means:
        idx_ax = idx_value // max_values
        k = ks[rank]
        ax = axes[idx_ax] if num_subplots > 1 else axes
        color_reward = crevolver.get_color() if k != "total" else "c"
        ax.plot(steps, means[k], color=color_reward, label=k)
        ax.fill_between(
            steps,
            means[k] - stds[k],
            means[k] + stds[k],
            color=color_reward,
            alpha=0.2,
        )
        idx_value += 1

    if num_subplots > 1:
        for ax in axes:
            ax.set_xlabel("steps")
            ax.set_ylabel("reward sum")
            ax.legend()
    else:
        axes.set_xlabel("steps")
        axes.set_ylabel("reward sum")
        axes.legend()

    plt.savefig(path_save + "_rewardsum.png")
    plt.close()


def plot_dynamics_actions(
    path_save,
    steps,
    props_actions,
    info_title,
):
    """
    plot dynamics of actions.
    """

    means = dict()
    stds = dict()
    last_means = list()
    for k in props_actions:
        props_actions[k] = np.array(props_actions[k])
        means[k] = props_actions[k].mean(axis=0)
        stds[k] = props_actions[k].std(axis=0)
        last_means.append(means[k][-1])

    num_subplots = 1
    fig, ax = plt.subplots(num_subplots, 1, figsize=(5, 5 * num_subplots))
    fig.suptitle(f"[action frequency dynamics] {info_title}")

    colours = {"D": "tab:red", "C": "tab:blue"}

    ax.set_ylim(-0.05, 1.05)
    for k in props_actions:
        colour_action = colours[k]
        ax.plot(steps, means[k], color=colour_action, label=k)
        ax.fill_between(
            steps,
            means[k] - stds[k],
            means[k] + stds[k],
            color=colour_action,
            alpha=0.2,
        )
    ax.set_xlabel("steps")
    ax.set_ylabel("action frequency")
    ax.legend()

    plt.savefig(path_save + "_actions.png")
    plt.close()


def plot_dynamics_actioninfos(
    path_save,
    steps,
    props_aps,
    info_title,
):
    """
    plot dynamics of action infos.
    """

    means = dict()
    stds = dict()
    last_means = list()
    for k in props_aps:
        props_aps[k] = np.array(props_aps[k])
        means[k] = props_aps[k].mean(axis=0)
        stds[k] = props_aps[k].std(axis=0)
        last_means.append(means[k][-1])
    if len(props_aps.keys()) == 4:
        colours = COLOURS_ACTIONPAIRS
        labels = LABELS_ACTIONPAIRS
    elif len(props_aps.keys()) == 2:
        colours = COLOURS_ACTIONS
        labels = LABELS_ACTIONS

    num_subplots = 1
    fig, ax = plt.subplots(num_subplots, 1, figsize=(5, 5 * num_subplots))
    fig.suptitle(f"[action info frequency dynamics] {info_title}")

    ax.set_ylim(-0.05, 1.05)
    for k in props_aps:
        colour_ap = colours[k]
        ax.plot(steps, means[k], color=colour_ap, label=labels[k])
        ax.fill_between(
            steps,
            means[k] - stds[k],
            means[k] + stds[k],
            color=colour_ap,
            alpha=0.2,
        )
    ax.set_xlabel("steps")
    ax.set_ylabel("action info frequency")
    ax.legend()

    plt.savefig(path_save + "_actioninfos.png")
    plt.close()


def list_to_str(lst):
    res = ""
    for s in lst:
        res += str(s)
    return res


def dec_to_bin(n, min_zeros=2):
    """
    convert decimal number to binary number.
    adds zeros when necessary.

    arguments:
        n
        min_zeros: number of minimum zeros.
    """

    res = ""
    min_zeros = max(min_zeros, int(np.ceil(np.log(max(n, 1)) / np.log(2))))
    for i in range(min_zeros):
        res = str(n % 2) + res
        n = n // 2
    return res


def plot_simple_network(network, figsize=(5, 5)):
    """
    visualise a network represented only by a numpy.ndarray.
    """

    # setup figure
    plt.figure(figsize=figsize)

    # make networkx graph
    # G = nx.from_numpy_matrix(network)
    G = nx.from_numpy_array(network)
    pos = nx.spring_layout(G, seed=1)
    nx.draw(G, pos)

    plt.legend()
    plt.show()
    plt.close()

def plot_ising_network(network, spins, figsize=(5, 5)):
    """
    visualise a network for Ising model represented only by a numpy.ndarray.
    """
    colours_ising = {
        1: "tab:blue",
        -1: "tab:red"
    }

    # setup figure
    plt.figure(figsize=figsize)

    # make networkx graph
    G = nx.from_numpy_array(network)
    pos = nx.spring_layout(G, seed=1)
    nx.draw(G, pos)

    nodes = np.arange(network.shape[0])
    labels_nodes = {
        1: [],
        -1: []
    }
    for i in range(network.shape[0]):
        if spins[i] > 0:
            labels_nodes[1].append(i)
        else:
            labels_nodes[-1].append(i)

    for spin in [1, -1]:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=labels_nodes[spin],
            node_color=colours_ising[spin],
        )

    plt.legend()
    plt.show()
    plt.close()


def plot_network_agents(
    idx_run,
    idx_step,
    names_agents,
    network,
    actionpairs,
    path_save,
    figsize=(7, 7),
):
    """
    visualise network structure with game history information.
    1) node colours represent the action pair with highest portion
       the agent has experienced in the last CONFIG.RECORD_STEP episodes.
    2) edge colours represent the action pair with highest portion
       the agent pair has experienced in the last CONFIG.RECORD_STEP episodes.

    arguments:
        network: adjacency matrix of the network.
        actionpairs: (num_agents, num_agents, num_actionspairs) tensor
        path_save
        figsize
    """

    # setup figure
    plt.figure(figsize=figsize)
    plt.title(f"[run{idx_run}-step{idx_step}] agent interaction network")

    # make networkx graph
    G = nx.from_numpy_array(network)
    pos = nx.spring_layout(G, seed=1)
    nx.draw(G, pos, alpha=0.0)
    nx.draw_networkx_edges(G, pos, G.edges)

    # aggregated interaction history of each node.
    # shape (num_agents, num_actionpairs)
    node_agg_record = np.zeros(actionpairs.shape[-2:])
    for i in range(node_agg_record.shape[0]):
        # record of me -> others
        node_agg_record[i] += actionpairs[i].sum(axis=0)
        # record of others -> me
        node_agg_record[i] += actionpairs[:, i].sum(axis=0)[[0, 2, 1, 3]]
    node_agg_alpha = node_agg_record.max(axis=1) / node_agg_record.sum(axis=1)

    labels = list(COLOURS_ACTIONPAIRS.keys())

    labels_nodes = dict()
    alphas_nodes = dict()
    labels_edges = dict()
    for k in COLOURS_ACTIONPAIRS.keys():
        labels_nodes[k] = list()
        alphas_nodes[k] = list()
        labels_edges[k] = list()

    # colour nodes based on their corresponding agents' action pairs
    # colour edges based on their corresponding agent pairs' action pairs
    for idx_a in range(actionpairs.shape[0]):
        idx_pair = actionpairs[idx_a].sum(axis=0).argmax()
        labels_nodes[labels[idx_pair]].append(idx_a)
        alphas_nodes[labels[idx_pair]].append(node_agg_alpha[idx_a])

        for idx_b in range(actionpairs.shape[1]):
            if actionpairs[idx_a, idx_b].max() > 0:
                idx_edge = actionpairs[idx_a, idx_b].argmax()
                labels_edges[labels[idx_edge]].append((idx_a, idx_b))

    for label in labels:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=labels_nodes[label],
            node_color=COLOURS_ACTIONPAIRS[label],
            alpha=alphas_nodes[label],
        )
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=labels_edges[label],
            edge_color=COLOURS_ACTIONPAIRS[label],
        )
        plt.scatter(
            [], [], color=COLOURS_ACTIONPAIRS[label], label=LABELS_ACTIONPAIRS[label]
        )

    name_nodes = {i: f"{i}_" + names_agents[i] for i in range(len(names_agents))}
    nx.draw_networkx_labels(
        G,
        pos,
        labels=name_nodes,
        font_size=8,
        verticalalignment="baseline",
    )

    plt.legend()

    # save figure
    plt.savefig(path_save)
    plt.close()


def plot_network_selection_actioninfo(
    idx_run,
    idx_step,
    names_selectors,
    network,
    actionpairs,
    selections,
    path_save,
    figsize=(7, 7),
):
    """
    visualise network structure with partner selection information.
    1) node colours represent the number of times it was selected
       during the last CONFIG.RECORD_STEP episodes?
    2) directed edges represent top N partners each node has selected.

    arguments:
        idx_run
        idx_step
        names_selectors
        network: adjacency matrix of the network.
        actionpairs
        selections
        path_save
        figsize
    """

    names_selectors = [f"{i}_{names_selectors[i]}" for i in range(len(names_selectors))]

    w_selection = selections.sum(axis=-1)
    w_selection = w_selection / w_selection.sum(axis=-1).reshape(-1, 1)
    mean_selection = network / network.sum(axis=-1).reshape(-1, 1)

    # aggregated interaction history of each node.
    # shape (num_agents, num_actionpairs)
    node_agg_record = np.zeros(actionpairs.shape[-2:])
    for i in range(node_agg_record.shape[0]):
        # record of me -> others
        node_agg_record[i] += actionpairs[i].sum(axis=0)
        # record of others -> me
        # need to switch (D,C) and (C,D) due to the opposite direction
        node_agg_record[i] += actionpairs[:, i].sum(axis=0)[[0, 2, 1, 3]]
    node_agg_alpha = node_agg_record.max(axis=1) / node_agg_record.sum(axis=1)

    colours = COLOURS_ACTIONPAIRS

    labels = list(colours.keys())

    labels_nodes = dict()
    alphas_nodes = dict()
    labels_edges = dict()
    for k in colours.keys():
        labels_nodes[k] = list()
        alphas_nodes[k] = list()
        labels_edges[k] = list()

    # colour nodes based on their corresponding agents' action pairs
    # colour edges based on their corresponding agent pairs' action pairs
    # edges plotted only when opponent was selected more than average
    for idx_a in range(actionpairs.shape[0]):
        # idx_pair = actionpairs[idx_a].sum(axis=0).argmax()
        idx_pair = node_agg_record[idx_a].argmax()
        labels_nodes[labels[idx_pair]].append(idx_a)
        alphas_nodes[labels[idx_pair]].append(node_agg_alpha[idx_a])

        for idx_b in range(actionpairs.shape[1]):
            mean_ab = mean_selection[idx_a, idx_b]
            if mean_ab > 0 and w_selection[idx_a, idx_b] >= mean_ab:
                idx_edge = actionpairs[idx_a, idx_b].argmax()
                labels_edges[labels[idx_edge]].append((idx_a, idx_b))
            else:
                # excluding the connection to affect the plot
                w_selection[idx_a, idx_b] = 0

    # setup figure
    plt.figure(figsize=figsize)
    # fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.title(f"[run{idx_run}-step{idx_step}] partner selection network - interaction")

    # make networkx graph
    G = nx.MultiDiGraph(network)
    pos = nx.spring_layout(G, seed=1)
    nx.draw(
        G,
        pos,
        # ax=ax,
        alpha=0.0,
    )

    for label in labels:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=labels_nodes[label],
            node_color=colours[label],
            # ax=ax,
            alpha=alphas_nodes[label],
        )

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=labels_edges[label],
            edge_color=colours[label],
            connectionstyle="arc3, rad=0.1",
            # ax=ax,
        )

        # ax.scatter(
        plt.scatter([], [], color=colours[label], label=LABELS_ACTIONPAIRS[label])

    name_nodes = {i: names_selectors[i] for i in range(len(names_selectors))}
    nx.draw_networkx_labels(
        G,
        pos,
        labels=name_nodes,
        font_size=8,
        verticalalignment="baseline",
    )

    plt.legend()

    # save figure
    plt.savefig(path_save)
    plt.close()


def plot_network_selection_lasthistory(
    idx_run,
    idx_step,
    names_selectors,
    network,
    actionpairs,
    selections,
    path_save,
    figsize=(7, 7),
):
    """
    visualise network structure with partner selection information.
    1) node colours represent the number of times it was selected
       during the last CONFIG.RECORD_STEP episodes?
    2) directed edges represent top N partners each node has selected.

    arguments:
        idx_run
        idx_step
        names_selectors
        network: adjacency matrix of the network.
        actionpairs
        selections
        path_save
        figsize
    """

    names_selectors = [f"{i}_{names_selectors[i]}" for i in range(len(names_selectors))]

    w_selection = selections.sum(axis=-1)
    w_selection = w_selection / w_selection.sum(axis=-1).reshape(-1, 1)
    mean_selection = network / network.sum(axis=-1).reshape(-1, 1)

    # aggregated interaction history of each node.
    # shape (num_agents, num_actionpairs)
    node_agg_record = np.zeros(actionpairs.shape[-2:])
    for i in range(node_agg_record.shape[0]):
        # record of me -> others
        node_agg_record[i] += actionpairs[i].sum(axis=0)
        # record of others -> me
        # need to switch (D,C) and (C,D) due to the opposite direction
        node_agg_record[i] += actionpairs[:, i].sum(axis=0)[[0, 2, 1, 3]]
    node_agg_alpha = node_agg_record.max(axis=1) / node_agg_record.sum(axis=1)

    colours_aps = COLOURS_ACTIONPAIRS
    if selections.shape[-1] == 4:
        colours_edges = COLOURS_ACTIONPAIRS
        desc_edges = LABELS_ACTIONPAIRS
    elif selections.shape[-1] == 2:
        colours_edges = COLOURS_ACTIONS
        desc_edges = LABELS_ACTIONS

    labels_aps = list(colours_aps.keys())
    labels_acs = list(colours_edges.keys())

    labels_nodes = dict()
    alphas_nodes = dict()
    labels_edges = dict()
    for k in colours_aps.keys():
        labels_nodes[k] = list()
        alphas_nodes[k] = list()
    for k in colours_edges.keys():
        labels_edges[k] = list()

    # colour nodes based on their corresponding agents' action pairs
    # colour edges based on their corresponding agent pairs' action pairs
    # edges plotted only when opponent was selected more than average
    for idx_a in range(actionpairs.shape[0]):
        # idx_pair = actionpairs[idx_a].sum(axis=0).argmax()
        idx_pair = node_agg_record[idx_a].argmax()
        labels_nodes[labels_aps[idx_pair]].append(idx_a)
        alphas_nodes[labels_aps[idx_pair]].append(node_agg_alpha[idx_a])

        for idx_b in range(actionpairs.shape[1]):
            mean_ab = mean_selection[idx_a, idx_b]
            if mean_ab > 0 and w_selection[idx_a, idx_b] >= mean_ab:
                idx_edge = selections[idx_a, idx_b].argmax()
                labels_edges[labels_acs[idx_edge]].append((idx_a, idx_b))
            else:
                # excluding the connection to affect the plot
                w_selection[idx_a, idx_b] = 0

    # setup figure
    plt.figure(figsize=figsize)
    # fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.title(f"[run{idx_run}-step{idx_step}] partner selection network - last history")

    # make networkx graph
    G = nx.MultiDiGraph(network)
    pos = nx.spring_layout(G, seed=1)
    nx.draw(
        G,
        pos,
        # ax=ax,
        alpha=0.0,
    )

    for label in labels_aps:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=labels_nodes[label],
            node_color=colours_aps[label],
            # ax=ax,
            alpha=alphas_nodes[label],
        )
        plt.scatter([], [], color=colours_aps[label], label=LABELS_ACTIONPAIRS[label])

    for label in labels_acs:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=labels_edges[label],
            edge_color=colours_edges[label],
            connectionstyle="arc3, rad=0.1",
            # ax=ax,
        )
        plt.plot([], [], color=colours_edges[label], label=desc_edges[label])

    name_nodes = {i: names_selectors[i] for i in range(len(names_selectors))}
    nx.draw_networkx_labels(
        G,
        pos,
        labels=name_nodes,
        font_size=8,
        verticalalignment="baseline",
    )

    plt.legend()

    # save figure
    plt.savefig(path_save)
    plt.close()


def get_qinfo(config):
    """
    get information on the collected Q-value information.
    * assumes (history length 1, single action history)
    """
    gamma = config["dqn"]["gamma"]
    payoff_agent0 = config["matgame"]["scores"][0]
    qinfo = dict()

    qinfo["qmax"] = max(payoff_agent0) / (1 - gamma)
    qinfo["qmin"] = min(payoff_agent0) / (1 - gamma)
    qinfo["CC"] = payoff_agent0[3] / (1 - gamma)
    qinfo["DD"] = payoff_agent0[0] / (1 - gamma)
    qinfo["CD"] = payoff_agent0[2] / (1 - gamma)
    qinfo["DC"] = payoff_agent0[1] / (1 - gamma)

    return qinfo


def infer_strategy(df):
    """
    infer strategy of the agent given Q-value information.
    assumes (len1, single) action history.
    """
    cols_qv = ["D | D", "C | D", "D | C", "C | C"]
    strategies = [
        "Coop",
        "TFT",
        "Def",
        "rTFT",
    ]

    df["Coop"] = (
        (df[cols_qv[1]] > df[cols_qv[0]]) * (df[cols_qv[3]] > df[cols_qv[2]]) * 1
    )
    df["TFT"] = (
        (df[cols_qv[1]] < df[cols_qv[0]]) * (df[cols_qv[3]] > df[cols_qv[2]]) * 1
    )
    df["Def"] = (
        (df[cols_qv[1]] < df[cols_qv[0]]) * (df[cols_qv[3]] < df[cols_qv[2]]) * 1
    )
    df["rTFT"] = (
        (df[cols_qv[1]] > df[cols_qv[0]]) * (df[cols_qv[3]] < df[cols_qv[2]]) * 1
    )
    return df, strategies


def plot_qvalues(
    config,
    df,
    cols_qv,
    iteration=0,
    idx_agent=0,
    path_save="sample_qv.png",
    figsize=(10, 10),
):
    df_target = df.loc[(df.iteration == iteration) & (df.idx_agent == idx_agent)]
    df_target, cols_strategy = infer_strategy(df_target)

    qinfo = get_qinfo(config)
    linestyles = ["-.", "-"]
    colors = ["tab:red", "tab:blue"]
    colors_strategy = {
        "Coop": "tab:blue",
        "Def": "tab:red",
        "TFT": "tab:orange",
        "rTFT": "tab:green",
    }

    plt.figure(figsize=figsize)
    plt.ylim(qinfo["qmin"] * 1.05, qinfo["qmax"] * 1.05)
    plt.xlabel("time step")
    plt.ylabel("Q values")
    plt.title(f"Q value difference | iteration {iteration} | idx_agent {idx_agent}")
    plt.axhline(0, linestyle="--", color="k")
    plt.axhline(qinfo["CC"], linestyle="--", color="tab:blue", label="mutual coop.")
    plt.axhline(qinfo["DD"], linestyle="--", color="tab:red", label="mutual def.")
    plt.axhline(qinfo["CD"], linestyle="--", color="tab:green", label="Sucker")
    plt.axhline(qinfo["DC"], linestyle="--", color="tab:purple", label="exploit")

    for idx_strategy, strategy in enumerate(cols_strategy):
        plt.fill_between(
            df_target["step"],
            df_target[strategy] * qinfo["qmin"] * 1.1,
            df_target[strategy] * qinfo["qmax"] * 1.1,
            color=colors_strategy[strategy],
            alpha=0.2,
        )

    for idx_col, col_qv in enumerate(cols_qv):
        lstyle = linestyles[(idx_col // 2)]
        color = colors[(idx_col % 2)]
        plt.plot(
            df_target["step"],
            df_target[col_qv],
            label=col_qv,
            linestyle=lstyle,
            color=color,
        )

    plt.legend()

    # save figure
    plt.savefig(path_save)
    plt.close()
