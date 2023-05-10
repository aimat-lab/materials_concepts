import numpy as np
from datetime import date
from scipy import sparse
import networkx as nx
import random
from tqdm import tqdm
import pickle

DAY_ORIGIN = date(1970, 1, 1)
NUM_OF_VERTICES = 125_019


def get_until(graph, day):
    return graph[graph[:, 2] < (day - DAY_ORIGIN).days]


def get_degrees(adjmatrix):
    return np.array(adjmatrix.sum(0))[0]


def normalize_degrees(degrees):
    return degrees / np.max(degrees) if np.max(degrees) > 0 else degrees


def build_adj_matrix(edge_list):
    """Build a symmetric adjacency matrix from edge list."""
    EDGE_WEIGHT = np.ones(len(edge_list) * 2)  # each connection weights the same

    row_ind = np.concatenate((edge_list[:, 0], edge_list[:, 1]))
    col_ind = np.concatenate((edge_list[:, 1], edge_list[:, 0]))

    return sparse.csr_matrix(
        (
            EDGE_WEIGHT,
            (row_ind, col_ind),
        ),
        shape=(NUM_OF_VERTICES, NUM_OF_VERTICES),
    )


def build_graphs(full_graph, until_year):
    print("Create Graph for: ", until_year)
    day_curr = date(until_year, 12, 31)

    edges_curr = get_until(full_graph, day_curr)
    adj_mat_sparse_curr = build_adj_matrix(edges_curr)

    graph_curr = nx.from_scipy_sparse_array(
        adj_mat_sparse_curr,
        parallel_edges=False,
        edge_attribute="weight",  # weight corresponds with number of edges between nodes
    )

    print("Done: Create Graph for ", until_year)
    print("Num of edges: ", graph_curr.number_of_edges())

    return graph_curr, adj_mat_sparse_curr


def get_draw_sample(n):
    bag = range(n)
    drawn = set()

    def draw_sample():
        while (comb := frozenset(random.sample(bag, 2))) in drawn:
            pass

        drawn.add(comb)
        return comb

    return draw_sample


def create_training_data(
    full_graph,
    year_start,
    years_delta,
    edges_used=500_000,
    vertex_degree_cutoff=10,
    ratio_connected=0.5,
):
    """
    :param full_graph: Full graph, numpy array dim(n,3) [vertex 1, vertex 2, time stamp]
    :param year_start: year of graph
    :param years_delta: distance for prediction in years (prediction on graph of year_start+years_delta)
    :param edges_used: optional filter to create a random subset of edges for rapid prototyping (default: 500,000)
    :param vertex_degree_cutoff: optional filter, for vertices in training set
                                 having a minimal degree of at least vertex_degree_cutoff  (default: 10)

    :return:
    unconnected_vertex_pairs: potential edges for year_start+years_delta
    unconnected_vertex_pairs_solution: numpy array with integers
        (0=unconnected, 1=connected), solution, length = len(unconnected_vertex_pairs)
    """

    year_end = year_start + years_delta

    train_graph, train_sparse_mat = build_graphs(full_graph, until_year=year_start)

    future_graph, _ = build_graphs(full_graph, until_year=year_end)

    edges = train_graph.edges()
    future_edges = future_graph.edges()

    new_edges = list(set(future_edges) - set(edges))
    threshold = int(edges_used * ratio_connected)
    new_edges = new_edges[:threshold]

    vertices = np.array(range(NUM_OF_VERTICES))

    ## Create all edges to be predicted
    unconnected_vertex_pairs = list(new_edges)
    unconnected_vertex_pairs_solution = [1] * len(new_edges)

    draw_sample = get_draw_sample(len(vertices))
    pbar = tqdm(total=int(edges_used * (1 - ratio_connected)))
    while len(unconnected_vertex_pairs) < edges_used:
        i1, i2 = draw_sample()
        v1, v2 = vertices[i1], vertices[i2]

        # graph is undirected: == has_edge(v2,v1)
        if (
            v1 != v2
            and not train_graph.has_edge(v1, v2)
            and not future_graph.has_edge(v1, v2)
        ):
            unconnected_vertex_pairs.append((v1, v2))
            unconnected_vertex_pairs_solution.append(0)
            pbar.update(1)

    pbar.close()

    print(
        "Number of unconnected vertex pairs for prediction: ",
        len(unconnected_vertex_pairs_solution),
    )
    print(
        "Number of vertex pairs that will be connected: ",
        sum(unconnected_vertex_pairs_solution),
    )
    print(
        "Ratio of vertex pairs that will be connected: ",
        sum(unconnected_vertex_pairs_solution) / len(unconnected_vertex_pairs_solution),
    )

    unconnected_vertex_pairs = np.array(unconnected_vertex_pairs)
    unconnected_vertex_pairs_solution = np.array(
        list(map(int, unconnected_vertex_pairs_solution))
    )

    perm = np.random.permutation(len(unconnected_vertex_pairs))

    return unconnected_vertex_pairs[perm], unconnected_vertex_pairs_solution[perm]


def retrieve_properties_for_vertex_pair(
    AA02,
    AA12,
    AA22,
    all_degs0,
    all_degs1,
    all_degs2,
    all_degs02,
    all_degs12,
    all_degs22,
    v1,
    v2,
):
    """
    Computes hand-crafted properties for one vertex in vlist
    """
    all_properties = []

    all_properties.append(all_degs0[v1])  # 0
    all_properties.append(all_degs0[v2])  # 1
    all_properties.append(all_degs1[v1])  # 2
    all_properties.append(all_degs1[v2])  # 3
    all_properties.append(all_degs2[v1])  # 4
    all_properties.append(all_degs2[v2])  # 5
    all_properties.append(all_degs02[v1])  # 6
    all_properties.append(all_degs02[v2])  # 7
    all_properties.append(all_degs12[v1])  # 8
    all_properties.append(all_degs12[v2])  # 9
    all_properties.append(all_degs22[v1])  # 10
    all_properties.append(all_degs22[v2])  # 11

    all_properties.append(AA02[v1, v2])  # 12
    all_properties.append(AA12[v1, v2])  # 13
    all_properties.append(AA22[v1, v2])  # 14

    return all_properties


def compute_all_properties_of_list(all_sparse, vlist):
    """
    Computes hand-crafted properties for all vertices in vlist

    all_sparse: list of adjacency matrices.
    vlist: ?
    """

    print("Computing all matrix squares...")
    # compute matrix squares
    AA02 = all_sparse[0] ** 2
    AA02 = AA02 / AA02.max()
    print("1...")
    AA12 = all_sparse[1] ** 2
    AA12 = AA12 / AA12.max()
    print("2...")
    AA22 = all_sparse[2] ** 2
    AA22 = AA02 / AA22.max()
    print("3")

    print("done")

    all_degs0 = normalize_degrees(get_degrees(all_sparse[0]))
    all_degs1 = normalize_degrees(get_degrees(all_sparse[1]))
    all_degs2 = normalize_degrees(get_degrees(all_sparse[2]))

    all_degs02 = normalize_degrees(get_degrees(AA02[0]))
    all_degs12 = normalize_degrees(get_degrees(AA12[1]))
    all_degs22 = normalize_degrees(get_degrees(AA22[2]))

    print("Computed all degrees")

    all_properties = []

    for ii in tqdm(range(len(vlist))):
        vals = retrieve_properties_for_vertex_pair(
            AA02,
            AA12,
            AA22,
            all_degs0,
            all_degs1,
            all_degs2,
            all_degs02,
            all_degs12,
            all_degs22,
            vlist[ii][0],
            vlist[ii][1],
        )

        all_properties.append(vals)

    return all_properties


def split_train_test(X, y, test_size=0.2):
    indices = list(range(len(X)))  # random shuffle input
    random.shuffle(indices)

    rel_train_size = 1 - test_size
    abs_train_size = int(len(indices) * rel_train_size)
    train_indices = indices[:abs_train_size]
    test_indices = indices[abs_train_size:]

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    print("Training, connected  : ", sum(y_train == 1))
    print("Training, unconnected: ", sum(y_train == 0))

    return X_train, y_train, X_test, y_test


def undersample_data(X, y, ratio=0.01):
    X_small = []
    y_small = []

    for i in range(len(X)):
        if (y[i] == 0 and random.random() < ratio) or y[i] == 1:
            X_small.append(X[i])
            y_small.append(y[i])

    return X_small, y_small


def get_adj_matrices(full_graph, years):
    return [
        build_adj_matrix(get_until(full_graph, day=date(year, 12, 31)))
        for year in years
    ]


def split_based_on_class(X, y):
    X_0 = []
    X_1 = []

    for i in range(len(X)):
        if y[i] == 1:
            X_1.append(X[i])
        else:
            X_0.append(X[i])

    return X_0, X_1


def main():
    graph = np.load("graph/edges.npz", allow_pickle=True)["arr_0"]
    edges_used = 500_000
    vertex_degree_cutoff = 10  # not used currently
    year_start = 2014
    train_years = [2012, 2013, 2014]
    years_delta = 3

    # train graph is until 2011
    train_edges_for_checking, train_edges_solution = create_training_data(
        graph,
        year_start=year_start,
        years_delta=years_delta,
        edges_used=edges_used,
        vertex_degree_cutoff=vertex_degree_cutoff,
        ratio_connected=0.1,
    )

    X_train, solution_train, X_test, solution_test = split_train_test(
        train_edges_for_checking, train_edges_solution, test_size=0.1
    )

    print("Length of training data: ", len(X_train))
    print("Length of test data: ", len(X_test))

    # Compute train sparse matrices

    training_matrices = get_adj_matrices(graph, years=train_years)
    data_train = compute_all_properties_of_list(training_matrices, X_train)
    data_test = compute_all_properties_of_list(training_matrices, X_test)

    storage = {}
    storage["data_train"] = data_train
    storage["data_test"] = data_test
    storage["solution_train"] = solution_train
    storage["solution_test"] = solution_test

    with open("graph/data.pkl", "wb") as f:
        pickle.dump(storage, f)


if __name__ == "__main__":
    main()
