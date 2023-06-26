from datetime import date
import numpy as np
import pickle
from scipy import sparse
import networkx as nx


class Graph:
    DAY_ORIGIN = date(1970, 1, 1)

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
        return self.edges[self.edges[:, 2] < (day - Graph.DAY_ORIGIN).days]

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

    def get_adj_matrices(self, years):
        return [
            self.build_adj_matrix(self.get_until(date(year, 12, 31))) for year in years
        ]
