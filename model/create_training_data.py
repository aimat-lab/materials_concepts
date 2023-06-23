import numpy as np
from datetime import date
from scipy import sparse
import networkx as nx
import random
from tqdm import tqdm
import pickle
from collections import Counter

DAY_ORIGIN = date(1970, 1, 1)


def get_draw_sample(n):
    bag = range(n)
    drawn = set()

    def draw_sample():
        while (comb := frozenset(random.sample(bag, 2))) in drawn:
            pass

        drawn.add(comb)
        return comb

    return draw_sample


class Graph:
    def __init__(self, path, num_of_vertices):
        self.num_of_vertices = num_of_vertices
        self.edges = Graph.load(path)  # todo load from pickled file
        self.adj_mat = Graph.build_adj_matrix(self.edges)
        print(self.adj_mat.shape)
        self.degrees = Graph.calc_degrees(self.adj_mat)

    @staticmethod
    def load(path):
        return np.load(path, allow_pickle=True)["arr_0"]

    @staticmethod
    def build_adj_matrix(edge_list):
        """Build a symmetric adjacency matrix from edge list."""

        symmetric_edges = np.vstack((edge_list, edge_list[:, [1, 0, 2]]))
        rows = symmetric_edges[:, 0]
        cols = symmetric_edges[:, 1]
        data = np.ones(symmetric_edges.shape[0])

        return sparse.coo_matrix(
            (
                data,
                (rows, cols),
            ),
        )

    @staticmethod
    def build_nx_graph(adj_mat):
        return nx.from_scipy_sparse_array(
            adj_mat,
            parallel_edges=False,
            edge_attribute="weight",  # weight = number of edges between nodes
        )

    @staticmethod
    def calc_degrees(adj_mat):
        return np.array(adj_mat.sum(0))[0]

    def get_until(self, day):
        return self.edges[self.edges[:, 2] < (day - DAY_ORIGIN).days]

    def get_adj_mat(self, until_year):
        cutoff_date = date(until_year, 12, 31)

        edges = self.get_until(cutoff_date)
        adj_mat = Graph.build_adj_matrix(edges)
        return adj_mat

    def get_nx_graph(self, until_year):
        return Graph.build_nx_graph(self.get_adj_mat(until_year))

    def degree(self, vertex):
        return self.degrees[vertex]

    def vertices(self):
        return np.array(range(self.num_of_vertices))

    def get_vertices(self, max_degree):
        return np.where(self.degrees <= max_degree)[0]


def random_generator(n):
    for idx in np.random.permutation(n * n):
        yield divmod(idx, n)


def split_array(array, train_split=0.1):
    n = len(array)
    split_index = int(n * train_split)  # Calculate the index to split at

    part_1 = array[:split_index]  # First 90% of elements
    part_2 = array[split_index:]  # Last 10% of elements

    return part_1, part_2


def generate_vertex_pairs(
    vertices,
    edges_used,
    past_graph,
    future_graph,
    train_split=0.9,
    pos_samples_ratio=0.4,
):
    new_edges = list(set(future_graph.edges()) - set(past_graph.edges()))
    new_edges = new_edges[
        : int(len(new_edges) * pos_samples_ratio)
    ]  # take only a fraction of the new edges

    print("Found pos samples:", len(new_edges))
    no_edges_train = []

    draw_sample = get_draw_sample(len(vertices))
    print("Drawing neg samples for training:", edges_used)

    pbar = tqdm(total=int(edges_used))
    while len(no_edges_train) < edges_used:
        i1, i2 = draw_sample()
        v1, v2 = vertices[i1], vertices[i2]

        # graph is undirected: == has_edge(v2,v1)
        if (
            v1 != v2
            and not past_graph.has_edge(v1, v2)
            and not future_graph.has_edge(v1, v2)
        ):
            no_edges_train.append((v1, v2))
            pbar.update(1)

    pbar.close()

    amt_new_edges_test = len(new_edges) / pos_samples_ratio * (1 - train_split)
    amt_no_edges_test = int(
        100 / (amt_new_edges_test / (len(vertices) ** 2)) * pos_samples_ratio
    )

    print("Drawing neg samples for test: ", amt_no_edges_test)
    pbar = tqdm(total=int(amt_no_edges_test))
    no_edges_test = []
    while len(no_edges_test) < amt_no_edges_test:
        i1, i2 = draw_sample()
        v1, v2 = vertices[i1], vertices[i2]

        # graph is undirected: == has_edge(v2,v1)
        if (
            v1 != v2
            and not past_graph.has_edge(v1, v2)
            and not future_graph.has_edge(v1, v2)
        ):
            no_edges_test.append((v1, v2))
            pbar.update(1)

    pbar.close()

    new_edges_train, new_edges_test = split_array(new_edges, train_split=train_split)

    return {
        "new_edges_train": new_edges_train,
        "no_edges_train": no_edges_train,
        "new_edges_test": new_edges_test,
        "no_edges_test": no_edges_test,
    }


def create_training_data(
    path_to_graph,
    year_start,
    years_delta,  # delta
    max_vertex_degree,  # c: 25
    amount_links,  # w
    edges_used=10_000_000,  # ?
):
    year_end = year_start + years_delta

    graph = Graph(path_to_graph, 57.460)

    past_graph = graph.get_nx_graph(until_year=year_start)
    future_graph = graph.get_nx_graph(until_year=year_end)

    vertices = graph.get_vertices(max_degree=max_vertex_degree)

    data = generate_vertex_pairs(vertices, edges_used, past_graph, future_graph)

    X_train = np.array(data["new_edges_train"] + data["no_edges_train"])
    y_train = np.array(
        [1] * len(data["new_edges_train"]) + [0] * len(data["no_edges_train"])
    )

    permutation_index = np.random.permutation(len(X_train))

    X_train = X_train[permutation_index]
    y_train = y_train[permutation_index]

    X_test = np.array(data["new_edges_test"] + data["no_edges_test"])
    y_test = np.array(
        [1] * len(data["new_edges_test"]) + [0] * len(data["no_edges_test"])
    )

    permutation_index = np.random.permutation(len(X_test))

    X_test = X_test[permutation_index]
    y_test = y_test[permutation_index]

    store = {
        "year": year_start,
        "delta": years_delta,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }

    # pickle store
    with open("data.pkl", "wb") as f:
        pickle.dump(store, f)


if __name__ == "__main__":
    create_training_data(
        "graph/edges.npz",
        year_start=2019,
        years_delta=3,
        edges_used=1_000_000,
        max_vertex_degree=25,
        amount_links=1,  # >= ?
    )
