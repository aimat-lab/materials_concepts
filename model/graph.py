from datetime import date
import numpy as np
import pickle
from scipy import sparse
import networkx as nx


class Graph:
    DAY_ORIGIN = date(1970, 1, 1)

    def __init__(self, path=None, edge_list=None):
        if path is None and edge_list is None:
            raise ValueError("Either path or edge_list must be provided.")

        if path is not None:
            self.edges = Graph.load(path)
            self._vertices = np.array(sorted(self.get_nodes_from_edge_list(self.edges)))
            self.adj_mat = Graph.build_adj_matrix(self.edges)
            self.degrees = Graph.calc_degrees(self.adj_mat)
        else:
            self.edges = edge_list
            self._vertices = np.array(sorted(self.get_nodes_from_edge_list(self.edges)))
            self.adj_mat = Graph.build_adj_matrix(self.edges)
            self.degrees = Graph.calc_degrees(self.adj_mat)[
                self._vertices
            ]  # only select degrees of vertices that are in the edge list as all other vertices have degree 0

    @classmethod
    def from_edge_list(cls, edge_list):
        return cls(edge_list=edge_list)

    @classmethod
    def from_path(cls, path):
        return cls(path=path)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data["edges"]

    @staticmethod
    def get_nodes_from_edge_list(edges) -> set:
        return set(edges[:, 0]).union(edges[:, 1])

    @staticmethod
    def build_adj_matrix(edge_list, binary=False, dim=None):
        """Build a symmetric adjacency matrix from edge list."""

        symmetric_edges = np.vstack((edge_list, edge_list[:, [1, 0, 2]]))
        rows = symmetric_edges[:, 0]
        cols = symmetric_edges[:, 1]
        data = np.ones(symmetric_edges.shape[0])

        if dim:
            adj_mat = sparse.csr_matrix(
                (
                    data,
                    (rows, cols),
                ),
                shape=(dim, dim),
            ).astype(np.int16)
        else:
            adj_mat = sparse.csr_matrix(
                (
                    data,
                    (rows, cols),
                ),
            ).astype(np.int16)

        return adj_mat if not binary else (adj_mat > 0).astype(np.int16)

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

    def get_until(self, date):
        return self.edges[self.edges[:, 2] < (date - Graph.DAY_ORIGIN).days]

    def get_until_year(self, year):
        return self.get_until(date(year + 1, 1, 1))

    def get_adj_mat(self, until_year):
        cutoff_date = date(until_year + 1, 1, 1)

        edges = self.get_until(cutoff_date)
        adj_mat = Graph.build_adj_matrix(edges)
        return adj_mat

    def get_nx_graph(self, until_year):
        return Graph.build_nx_graph(self.get_adj_mat(until_year))

    def degree(self, vertex):
        return self.degrees[vertex]

    @property
    def vertices(self):
        return self._vertices

    def get_vertices(self, until_year, min_degree=0, max_degree=None):
        g = Graph.from_edge_list(self.get_until_year(until_year))

        if max_degree is None:
            vs = g.vertices[g.degrees >= min_degree]
        else:
            vs = g.vertices[(g.degrees >= min_degree) & (g.degrees <= max_degree)]

        return vs

    def get_adj_matrices(self, years, binary=False, full=False):
        return [
            self.build_adj_matrix(
                self.get_until(date(year, 12, 31)),
                binary=binary,
                dim=len(self.vertices) if full else None,
            )
            for year in years
        ]
