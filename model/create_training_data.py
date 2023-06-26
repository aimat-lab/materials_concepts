import numpy as np
from datetime import date
from scipy import sparse
import networkx as nx
import random
from tqdm import tqdm
import pickle
from collections import Counter

DAY_ORIGIN = date(1970, 1, 1)


class Graph:
    def __init__(self, path):
        self.num_of_vertices, self.edges = Graph.load(path)
        self.adj_mat = Graph.build_adj_matrix(self.edges)
        self.degrees = Graph.calc_degrees(self.adj_mat)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data["num_of_vertices"], data["edges"]

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
            edge_attribute="links",
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
        if max_degree is None:
            return self.vertices()

        return np.where(self.degrees <= max_degree)[0]


class TrainingDataGenerator:
    def __init__(self, path_to_graph: str, year_start: int, year_delta: int):
        self.year_start = year_start
        self.year_delta = year_delta
        year_end = year_start + year_delta

        self.graph = Graph(path_to_graph)
        self.past_graph = self.graph.get_nx_graph(until_year=year_start)
        self.future_graph = self.graph.get_nx_graph(until_year=year_end)

    def generate(self, edges_used: int, min_links: int = 1, max_v_degree: int = 25):
        """Generate training data by taking all positive samples and randomly drawing negative samples until the desired number of samples is reached.

        Parameters:
        edges_used: number of vertex pairs to generate
        min_links: minimum number of links (at time = year_start + delta) between vertex pairs to be counted as positive sample
        max_v_degree: maximum degree of vertices which are considered for sampling

        Returns:
        X: array of vertex pairs with vertex [0, len(self.graph.num_of_vertices)]
        y: array of labels (1 = connected, 0 = unconnected)
        """
        filtered_vertices = self.graph.get_vertices(max_degree=max_v_degree)

        pos_samples = self._get_pos_samples(filtered_vertices, min_links)
        to_draw_neg = edges_used - len(pos_samples)
        neg_samples = self._get_neg_samples(to_draw_neg, filtered_vertices)

        X = np.array(pos_samples + neg_samples)
        y = np.array([1] * len(pos_samples) + [0] * len(neg_samples))
        return self.shuffle(X, y)

    def _get_pos_samples(self, filtered_vertices, min_links):
        """Positive samples of vertex pairs: {year_start} unconnected, {year_start + delta} connected

        Aproach:
        All edges which exist at {year_start + delta} and didn't exist at {year_start} are candidates.
        Filter out all edges which have less than {min_links} links at {year_start + delta}
        and check if the vertices are in the filtered set of vertices.
        """
        pos_samples = set(self.future_graph.edges()) - set(self.past_graph.edges())

        lookup = {elem: True for elem in filtered_vertices}
        pos_samples = [
            (v1, v2) for v1, v2 in pos_samples if lookup.get(v1) and lookup.get(v2)
        ]

        pos_samples = [
            (v1, v2)
            for v1, v2 in pos_samples
            if self.future_graph.edges[v1, v2]["links"] >= min_links
        ]

        return pos_samples

    def _get_neg_samples(self, to_draw, vertices):
        """Negative samples of vertex pairs: {year_start} unconnected, {year_start + delta} unconnected"""
        X = []

        draw_sample = self.get_draw_sample(len(vertices))
        pbar = tqdm(total=to_draw)
        while len(X) < to_draw:
            i1, i2 = draw_sample()
            v1, v2 = vertices[i1], vertices[i2]

            if (
                v1 != v2
                and not self.past_graph.has_edge(v1, v2)
                and not self.future_graph.has_edge(v1, v2)
                # order of vertices in call doesn't matter as networkx Graph is undirected
            ):
                X.append((v1, v2))
                pbar.update(1)

        pbar.close()

        return X

    @staticmethod
    def get_draw_sample(n):
        bag = range(n)
        drawn = set()

        def draw_sample():
            while (comb := frozenset(random.sample(bag, 2))) in drawn:
                pass

            drawn.add(comb)
            return comb

        return draw_sample

    @staticmethod
    def shuffle(X, y):
        """Shuffle X and y in unison"""
        assert len(X) == len(y)
        p = np.random.permutation(len(X))
        return X[p], y[p]


if __name__ == "__main__":
    generator = TrainingDataGenerator("graph/edges.pkl", year_start=2019, year_delta=3)
    X, y = generator.generate(edges_used=1_000_000, min_links=1, max_v_degree=None)
