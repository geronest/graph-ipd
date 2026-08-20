"""
author: Seongho Son (seong.son.22@ucl.ac.uk)

implementation of network-based agent population
"""

import numpy as np
import torch


def check_connectedness(mat):
    """
    check connectedness of the given connection matrix.
    ** unidirectional connection is also valid
    input:
        mat: square matrix representation of nodes' connection to other nodes.
            - assumes 2 dimensional matrix
    return:
        True if all the nodes in the matrix are connected to a single network.
        False otherwise.
    """
    assert len(mat.shape) == 2
    assert mat.shape[0] == mat.shape[1]
    mat = mat + mat.T + np.eye(mat.shape[0])
    for i in range(int(np.log(mat.shape[0]) / np.log(2)) + 1):
        mat = np.dot(mat, mat)
    return (mat > 0).all()


def create_network_ER(num_agents, p, no_isolate=True, rng=None, verbose=True):
    """
    construct an Erdos-Renyi network.
    ** no self-link.
    ** if p > ln(n) / n, then usually no isolated node.
    inputs:
        p: probability of each possible link being realized.
        num_connections: number of links in the network.
    return:
        result network
    """
    assert p > 0 and p <= 1

    if rng is None:
        rng = np.random.default_rng()

    # print warning message if p is below connectivity threshold
    min_p = np.log(num_agents) / num_agents
    if p < min_p and verbose:
        print(
            f"[WARNING | AgentNetwork-create_network_ER] {p} < {min_p}, "
            + "likely to have isolated node"
        )

    network = np.zeros(num_agents**2)

    # connect every node to at least one node, eliminating isolation
    if no_isolate:
        # proportion of edges used for eliminating isolation
        num_minedges = (num_agents // 2) + (num_agents % 2)
        num_maxedges = (num_agents * (num_agents - 1)) / 2
        p_req = num_minedges / num_maxedges
        """
            (num_minedges + X) / all possible edges = p
            p * (all_possible_edges) = num_minedges + X
            p * (all_possible_edges) - num_minedges = X
            new_p * (all_possible_edges - num_minedges) = X
            new_p = (p * all_possible_edges - num_minedges) /
                    (all_possible_edges - num_minedges)
        """

        if p - p_req < 0:
            print(f"[ERROR] p({p}) - p_req({p_req}) < 0; ABORTING")
            raise ValueError
        else:
            p = (p * num_maxedges - num_minedges) / (num_maxedges - num_minedges)

        idxs_ni = list()
        for i in range(num_agents // 2):
            idxs_ni.append((2 * i + 1) * num_agents + (2 * i))
        if num_agents % 2 == 1:
            idxs_ni.append((num_agents - 1) * num_agents + 0)
        idxs_ni = np.array(idxs_ni)
        network[idxs_ni] = 1

    idxs = list()
    for i in range(num_agents - 1):
        for j in range(i + 1):
            idxs.append((i + 1) * num_agents + j)
    idxs = np.array(idxs)
    if no_isolate:
        idxs = np.setdiff1d(idxs, idxs_ni)
    samples = rng.random(idxs.shape)

    network[idxs[samples < p]] = 1
    network = network.reshape((num_agents, num_agents))
    return network + network.T


def create_network_WS(num_agents, half_degree_regular=2, p=0.1, rng=None):
    """
    construct an Watts-Strogatz network.
    ** no self-link.
    inputs:
        num_agents
        half_degree_regular:
            halved degree of each node when constructing regular lattice.
        p: probability of each edge changing connection.
    return:
        result network
    """
    assert isinstance(half_degree_regular, int)
    assert half_degree_regular > 0 and half_degree_regular < (num_agents - 1)
    assert (p >= 0) and (p <= 1)

    if rng is None:
        rng = np.random.default_rng()

    # create regular lattice
    row = [0] * num_agents
    for i in range(half_degree_regular):
        row[i + 1] = 1
    mat = list()
    for i in range(num_agents):
        mat.append(row.copy())
        row.insert(0, row.pop())
    network = np.array(mat)
    network = network + network.T

    # randomly change connections
    num_agents = network.shape[0]
    edges = np.where(network > 0)  # indexes of edges

    # randomly choose an edge with probability p
    change = np.where(rng.random(len(edges[0])) < p)

    for idx_change in change[0]:
        idx_x = edges[0][idx_change]
        idx_y = edges[1][idx_change]

        # don't change connections when node 'idx_y' could become isolated
        if (network[idx_y].sum() > 1) and idx_y > idx_x:
            zeros = np.where((network + np.eye(num_agents))[idx_x] == 0)[0]
            # choose another vertex to link
            nidx_y = zeros[rng.integers(0, len(zeros), 1)[0]]

            # change link
            network[idx_x][idx_y] = 0
            network[idx_y][idx_x] = 0
            network[idx_x][nidx_y] = 1
            network[nidx_y][idx_x] = 1

    return network


def create_network_BA(num_agents, m, rng=None):
    """
    construct an Barabasi-Albert network.
    ** no self-link.
    inputs:
        num_agents
        m: number of new edges each new node adds to the network.
    return:
        result network
    """
    assert num_agents > 2
    assert isinstance(m, int)
    assert m > 0 and m < (num_agents - 1)

    if rng is None:
        rng = np.random.default_rng()

    # create network, where initial m nodes are connected
    network = np.zeros((num_agents, num_agents))
    for i in range(m):
        idx_y = (i + 1) % m
        network[i, idx_y] = 1
        network[idx_y, i] = 1

    # starting from node 2, perform preferential attachment
    for idx_node in range(m, num_agents, 1):
        # calculate prob. of node idx_node linking to node i
        probs = network[:idx_node].sum(axis=1) / network[:idx_node].sum()
        idxs_y = rng.choice(idx_node, m, replace=False, p=probs)

        for idx_y in idxs_y:
            network[idx_node, idx_y] = 1
            network[idx_y, idx_node] = 1

    return network


def create_network_gBA(num_agents, m):
    """
    construct an growing Barabasi-Albert network.
    ** no self-link.
    inputs:
        num_agents
        m: number of new edges each new node adds to the network.
    return:
        result network
    """
    assert num_agents > 2
    assert isinstance(m, int)
    assert m > 0 and m < (num_agents - 1)

    # create network, where initial m nodes are connected
    network = np.zeros((num_agents, num_agents))
    for i in range(m):
        idx_y = (i + 1) % m
        network[i, idx_y] = 1
        network[idx_y, i] = 1

    return network


def create_network_line(num_agents):
    """
    construct a single-line network.
    """

    network = np.zeros((num_agents, num_agents))
    for i in range(num_agents - 1):
        network[i, i + 1] = 1

    network = network + network.T

    return network


def create_network_ring(num_agents):
    """
    construct a ring-shaped network.
    """

    network = np.zeros((num_agents, num_agents))
    for i in range(num_agents):
        idx_y = (i + 1) % num_agents
        network[i, idx_y] = 1

    network = network + network.T

    return network


def create_network_concentrate(num_agents):
    """
    construct a concentrated(to agent 0) network.
    """

    network = np.zeros((num_agents, num_agents))
    network[0, 1:] = 1

    network = network + network.T

    return network


def create_network_tree(num_agents, num_leaves=2):
    """
    construct a tree network.
    """

    network = np.zeros((num_agents, num_agents))

    idx_parent = 0
    curr_num_leaves = 0
    for i in range(1, num_agents):
        network[i, idx_parent] = 1
        curr_num_leaves += 1
        if curr_num_leaves == num_leaves:
            curr_num_leaves = 0
            idx_parent += 1

    network = network + network.T

    return network


class AgentNetwork:
    def __init__(self, config, num_agents, seed_modifier=0, verbose=False):
        connected = False
        num_trials = 1
        self.current_agents = num_agents
        self.config = config.agentnet
        self.rng = np.random.default_rng(
            seed=config.random_seeds.network + seed_modifier
        )
        self.gen_param = None

        # determine network structure if specified in config.
        # otherwise full connectivity
        if verbose:
            print(
                f"[AgentNetwork] {num_trials} th trial "
                + f"of forming {config.agentnet.name} network"
            )
        if config.agentnet.name == "ER":  # Erdos-Renyi
            if "p_min" in config.agentnet:
                range_p = config.agentnet.p_max - config.agentnet.p_min
                val_p = config.agentnet.p_min + (self.rng.random() * range_p)
            else:
                val_p = config.agentnet.p
            self.gen_param = val_p

            self.network = create_network_ER(
                num_agents,
                val_p,
                config.agentnet.no_isolate,
                rng=self.rng,
            )
        elif config.agentnet.name == "WS":  # Watts-Strogatz
            if "p_min" in config.agentnet:
                range_p = config.agentnet.p_max - config.agentnet.p_min
                val_p = config.agentnet.p_min + (self.rng.random() * range_p)
            else:
                val_p = config.agentnet.p
            self.gen_param = val_p

            self.network = create_network_WS(
                num_agents,
                config.agentnet.half_degree_regular,
                val_p,
                rng=self.rng,
            )
        elif config.agentnet.name == "BA":  # Barabasi-Albert
            if "m_min" in config.agentnet:
                range_m = config.agentnet.m_max - config.agentnet.m_min
                val_m = config.agentnet.m_min + int(self.rng.random() * range_m)
                if val_m > (num_agents - 2):  # self.rng.random() == 1.0?
                    val_m = num_agents - 2
            else:
                val_m = config.agentnet.m
            self.gen_param = val_m

            self.network = create_network_BA(
                num_agents,
                val_m,
                rng=self.rng,
            )
        elif config.agentnet.name == "gBA":  # Barabasi-Albert, growing over time
            self.network = create_network_gBA(num_agents, config.agentnet.m)
            self.current_agents = config.agentnet.m
        elif config.agentnet.name == "line":
            self.network = create_network_line(num_agents)
        elif config.agentnet.name == "ring":
            self.network = create_network_ring(num_agents)
        elif config.agentnet.name == "concentrate":
            self.network = create_network_concentrate(num_agents)
        elif config.agentnet.name == "tree":
            self.network = create_network_tree(num_agents, config.agentnet.num_leaves)
        else:  # fully connected network, without self link
            self.network = np.ones((num_agents, num_agents)) - np.eye(num_agents)
        connected = check_connectedness(
            self.network[: self.current_agents, : self.current_agents]
        )
        if not connected:
            print(f"[AgentNetwork] connectedness: {connected}")
        num_trials += 1
        if verbose:
            print(self.network)

        self.idx_sample = 0
        self.l_sample = np.arange(self.current_agents)

    def connected_agents(self, idx):
        """
        return the list of connected neighbours when an agent's index is given.

        inputs:
            idx: index of an agent in the network.
        return:
            neighbours: list of connected agents' indexes
        """
        return np.where(self.network[idx] > 0.0)[0]

    def sample_agents(self):
        """
        sample two agents, only those with existing connection
        (link weight bigger than zero) between them.
        ** assumes that every node has at least one connected neighbour
        return:
            idxs: indexes of each agent sampled
        """
        idxs = list()
        # each agent is sampled in a rotating order
        idxs.append(self.l_sample[self.idx_sample])
        if (self.idx_sample + 1) % self.l_sample.shape[0] == 0:
            self.idx_sample = 0
            self.rng.shuffle(self.l_sample)
        else:
            self.idx_sample += 1

        # choose another agent that is linked to the previous one
        candidates = np.arange(self.current_agents)[self.network[idxs[0]] > 0.0]
        idxs.append(candidates[int(torch.rand(1).item() * candidates.shape[0])])
        return idxs

    def add_agent(self):
        """
        add a single agent into the growing network.
        """

        # skip if the predefined network is already full
        if self.current_agents >= self.network.shape[0]:
            return

        if self.config.name == "gBA":
            # perform preferential attachment
            probs = (
                self.network[: self.current_agents].sum(axis=1)
                / self.network[: self.current_agents].sum()
            )
            idxs_y = self.rng.choice(
                self.current_agents, self.config.m, replace=False, p=probs
            )

            for idx_y in idxs_y:
                self.network[self.current_agents, idx_y] = 1
                self.network[idx_y, self.current_agents] = 1

        self.current_agents += 1
        self.l_sample = np.arange(self.current_agents)

    def set_edge(self, idx_a, idx_b, value):
        """
        set an edge in the network.
        * assumes undirected connection.
        """
        self.network[idx_a, idx_b] = value
        self.network[idx_b, idx_a] = value

    def filter_vector(
        self,
        idx_agent,
        exclude_me=True,
        exclude_nonneighbours=True,
        exclude_neighbours=False,
    ):
        """
        filter values in the given vector to zero
        when their indices correspond to the given criteria.

        arguments:
            vector: assumed shape self.network.shape
            include_neighbours
            exclude_neighbours
        return:
            idxs_remaining
        """
        idxs_exclude = list()
        neighbours = self.connected_agents(idx_agent)
        if exclude_me:
            idxs_exclude.append(idx_agent)
        if exclude_nonneighbours:
            cands = np.setdiff1d(
                np.arange(self.network.shape[0]),
                neighbours,
            )
            idxs_exclude += cands.tolist()
        if exclude_neighbours:
            idxs_exclude += neighbours.tolist()

        idxs_remaining = np.setdiff1d(
            np.arange(self.network.shape[0]),
            idxs_exclude,
        )

        return idxs_remaining

    def select_node(self, idxs_cand, vector_agent, select_high=True, select_prob=False):
        """
        choose an index of a node with given information.
        * assumes no self-connection.

        arguments:
            vector_agent: a vector with information for corresponding nodes.
            select_high: whether to select a node with high value in the given vector.
            select_prob:
                whether to select with probability
                proportional to the given values.
        """
        if select_prob:
            if select_high:
                probs = vector_agent / vector_agent.sum()
            else:
                newvec = vector_agent.max() - vector_agent
                probs = newvec / newvec.sum()
            target = self.rng.choice(vector_agent.shape[0], 1, replace=False, p=probs)
        else:
            indices = np.arange(vector_agent.shape[0])
            if select_high:
                cands_target = indices[vector_agent == vector_agent.max()]
            else:
                cands_target = indices[vector_agent == vector_agent.min()]
            target = self.rng.choice(cands_target, 1, replace=False)
        return idxs_cand[target]

    def preferential_detach(
        self,
        idx_agent,
        vector_agent,
        select_high=True,
        select_prob=False,
        min_neighbours=1,
    ):
        """
        perform preferential detachment with the given information.
        * assumes undirectional connections.

        arguments:
            idx_agent
            vector_agent
            select_high
            min_neighbours
        """
        if self.network[idx_agent].sum() > min_neighbours:
            idxs_remaining = self.filter_vector(
                idx_agent,
                exclude_me=True,
                exclude_nonneighbours=True,
                exclude_neighbours=False,
            )
            idx_target = self.select_node(
                idxs_remaining, vector_agent[idxs_remaining], select_high, select_prob
            )
            if self.network[idx_target].sum() > 1:
                self.set_edge(idx_agent, idx_target, 0)
        return self.network[idx_agent].copy()

    def preferential_attach(
        self,
        idx_agent,
        vector_agent,
        select_high=True,
        select_prob=False,
    ):
        """
        perform preferential attachment with the given information.
        * assumes undirectional connections.

        arguments:
            idx_agent
            vector_agent
            select_high
            min_neighbours
        """
        if self.network[idx_agent].sum() < self.network.shape[0] - 1:
            idxs_remaining = self.filter_vector(
                idx_agent,
                exclude_me=True,
                exclude_nonneighbours=False,
                exclude_neighbours=True,
            )
            idx_target = self.select_node(
                idxs_remaining, vector_agent[idxs_remaining], select_high, select_prob
            )
            self.set_edge(idx_agent, idx_target, 1)
        return self.network[idx_agent].copy()

    def flip_edges(self, rho):
        """
        flip edges based on flipping probability rho.
        * assumes each connection is represented by either 1 or 0.
        * assumes undirectional connections.
        """

        num_agents = self.network.shape[0]
        reverse_network = 1 - self.network

        for i in range(num_agents - 1):
            for j in range(i + 1):
                if self.rng.random() < rho:
                    self.network[i, j] = reverse_network[i, j]
                    self.network[j, i] = self.network[i, j]

    def modify_structure(
        self,
        current_step,
    ):
        """
        apply modification to the structure of network
        based on the given configuration.

        arguments:
            current_step: current step of the experiment run
        """
        if "modifications" in self.config:
            cfg_mod = self.config.modifications
            if (current_step + 1) % cfg_mod.modify_step == 0:
                if cfg_mod.randomly_choose_agent:
                    idx_agent = self.rng.choice(self.current_agents, 1)[0]
                else:
                    pass

                if cfg_mod.preference_criterion == "degree":
                    vector_agent = self.network[idx_agent].copy()
                elif cfg_mod.preference_criterion == "state":
                    pass

                if "preferential_detachment" in cfg_mod:
                    self.preferential_detach(
                        idx_agent,
                        vector_agent,
                        cfg_mod.preferential_detachment.select_high,
                        cfg_mod.preferential_detachment.select_prob,
                        cfg_mod.preferential_detachment.min_neighbours,
                    )

                if "preferential_attachment" in cfg_mod:
                    self.preferential_attach(
                        idx_agent,
                        vector_agent,
                        cfg_mod.preferential_attachment.select_high,
                        cfg_mod.preferential_attachment.select_prob,
                    )
        return True


if __name__ == "__main__":
    net_ER = create_network_ER(6, 0.3)
    net_WS = create_network_WS(6, 2, 0.2)
    net_BA = create_network_BA(6, 2)

    print("ER")
    print(net_ER)
    print("WS")
    print(net_WS)
    print("BA")
    print(net_BA)
