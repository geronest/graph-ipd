"""
author: Seongho Son (seong.son.22@ucl.ac.uk)
functions for analysing various properties of networks.

assumes networks are given in the shape of (iteration, num_agents, num_agents).
assumes networks consist of a single connected cluster.
does not assume self-connection.
assumes networks consist of undirected connections.
"""

import numpy as np


def mat_exp(nets, n=2):
    """
    perform repeated multiplication of a matrix.
    """

    pass


def distance(nets):
    """
    calculate distance matrix of nodes in the network.

    arguments:
        nets
    """

    mat_dis = np.zeros(nets.shape)
    mat_nsp = np.zeros(nets.shape)  # number of shortest paths

    # iterate until every node pair's distance is calculated
    net_acc = np.tile(np.eye(nets.shape[-1]), (nets.shape[0], 1, 1))
    eyes = net_acc.copy()

    for i in range(nets.shape[1]):
        net_acc = np.matmul(nets, net_acc)
        idx_update = (net_acc > 0) * (mat_dis == 0)
        mat_dis[idx_update] = i + 1
        mat_nsp[idx_update] = net_acc[idx_update]

        # if all element of mat_dis is above zero, break
        if mat_dis.all():
            break

    # exclude distance to self
    mat_dis -= mat_dis * eyes

    return mat_dis, mat_nsp


def edges_between_nodes(net, nodes):
    """
    calculate the number of edges between the specified ndoes.

    arguments:
        net (shape: num_agents * num_agents)
        nodes: list of integers that specify the nodes
    """

    pass


def clustering_coefficient(nets):
    """
    calculate clutering coefficient of each node.
    * assumes no self connection.
    * assumes undirected connection.
    * assumes weight 1 for every connection.

    arguments:
        nets (shape: iteration * num_agents * num_agents)

    """

    # exclude self-loops
    nets = np.clip(
        nets
        - np.tile(np.expand_dims(np.eye(nets.shape[-1]), 0), (nets.shape[0], 1, 1)),
        0,
        1,
    )
    degree = nets.sum(axis=-1)
    denominator = np.clip(degree * (degree - 1), 1, None)
    edges_neighbors = np.zeros(degree.shape)

    for idx_iteration in range(nets.shape[0]):
        net = nets[idx_iteration]
        # count edges between node i's neighbors
        for i in range(nets.shape[1]):
            idxs_neighbors = np.where(net[i] > 0.5)[0]
            edges_neighbors[idx_iteration, i] = net[
                idxs_neighbors.repeat(idxs_neighbors.shape[0]),
                np.tile(idxs_neighbors, idxs_neighbors.shape[0]),
            ].sum()

    coef_cls = edges_neighbors / denominator

    return coef_cls


def centrality(nets, mode="degree"):
    """
    calculate various types of centrality.

    arguments:
        nets
        mode
    """

    mat_dis, mat_nsp = distance(nets)
    degree = nets.sum(axis=-1)

    if mode == "degree":
        return degree
    elif mode == "closeness":
        sum_dis = mat_dis.sum(axis=1)
        return (nets.shape[-1] - 1) / sum_dis
    elif mode == "betweenness":
        # num. of shortest paths between vertex s and t
        sp_st = np.zeros(nets.shape[:-1])
        # num. of shortest paths between vertex s and t, with v in the middle
        sp_stv = np.zeros(nets.shape[:-1])

        # for a vertex v, calculate sp_st and sp_stv
        for i in range(nets.shape[1]):
            # calculate sp_st (s, t != v)
            for j in range(nets.shape[1] - 1):
                if j == i:
                    continue
                for k in range(j + 1, nets.shape[1]):
                    if k == i:
                        continue
                    sp_st[:, i] += mat_nsp[:, j, k]
                    idx_stv = (mat_dis[:, j, i] + mat_dis[:, i, k]) == mat_dis[:, j, k]
                    sp_stv[:, i] += mat_nsp[:, j, i] * mat_nsp[:, i, k] * idx_stv

        sp_st[sp_st == 0] += 0.0001

        return sp_stv / sp_st


def average_distances(mat_dis):
    """
    calculate the average distance of nodes that are *not isolated*.

    arguments:
        mat_dis: shape (iteration, num_agents, num_agents)
    return:
        distances: shape (iteration,)
    """

    distances = np.zeros(mat_dis.shape[0])
    for i in range(mat_dis.shape[0]):
        net = mat_dis[i]
        connections = net[net > 0]
        distances[i] = connections.mean()

    return distances


def count_clusters(nets):
    """
    count the number of clusters in the network.

    arguments:
        nets: shape (iteration, num_agents, num_agents)
    return:
        nums_clusters: shape (iteration,)
    """

    def find_cluster(net, idx_start):
        """
        find a cluster of connected nodes including idx_start.

        arguments:
            net
            idx_start
        return:
            indexes of nodes in the cluster
        """
        nodes = np.arange(net.shape[0])

        row = net[idx_start]
        neighbors = nodes[row > 0]
        row[idx_start] += 1

        continue_searching = True
        while continue_searching:
            new_neighbors = list()
            continue_searching = False
            for neighbor in neighbors:
                row_add = net[neighbor]
                idx_new_neighbors = (row == 0) * (row_add > 0)
                row[idx_new_neighbors] += row_add[idx_new_neighbors]

                if idx_new_neighbors.sum() > 0:
                    continue_searching = True
                new_neighbors += nodes[idx_new_neighbors].tolist()
            neighbors = new_neighbors

        return (row > 0) * 1.0

    nums_clusters = np.zeros(nets.shape[0])

    for i in range(nets.shape[0]):
        net = nets[i]
        num_clusters = 0
        in_cluster = np.zeros(net.shape[0])
        for j in range(net.shape[0]):
            if in_cluster[j] == 0:
                in_cluster += find_cluster(net, j)
                num_clusters += 1

        nums_clusters[i] = num_clusters

    return nums_clusters


def assortativity(nets):
    """
    calculate assortativity coefficient of the networks.
    """

    degrees = nets.sum(axis=-1)
    num_links = degrees.sum(axis=-1)
    assortativity = np.zeros(nets.shape[0])

    for i in range(nets.shape[0]):
        net = nets[i]
        idxs_link = np.where(net > 0)

        degree_prod = 0
        degree_sum = 0
        degree_squaresum = 0

        for j in range(idxs_link[0].shape[0]):
            d1 = degrees[i][idxs_link[0][j]]
            d2 = degrees[i][idxs_link[1][j]]

            degree_prod += d1 * d2
            degree_sum += d1 + d2
            degree_squaresum += d1**2 + d2**2

        v1 = degree_prod / num_links[i]
        v2 = (degree_sum / (2 * num_links[i])) ** 2
        v3 = degree_squaresum / (2 * num_links[i])
        assortativity[i] = (v1 - v2) / (v3 - v2 + 1e-9)

    return assortativity


def richclub_coefficient(nets):
    """
    calculate rich-club coefficient of the networks.
    """

    degrees = nets.sum(axis=-1)
    rc_coef = np.ones(nets.shape[:-1]) * -1

    for i in range(nets.shape[0]):
        net = nets[i]
        degrees_net = np.arange(degrees[i].max() + 1, dtype=np.int32)

        for k in degrees_net:
            nodes_k = np.where(degrees[i] > k)[0]
            nk = nodes_k.shape[0]

            subnet = net[nodes_k][:, nodes_k]
            ek = subnet.sum()

            if ek > 0:
                rc_coef[i][k] = ek / (nk * (nk - 1))
            else:
                rc_coef[i][k] = 0

    return rc_coef
