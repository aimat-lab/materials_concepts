from datetime import date, timedelta
import numpy as np
import pickle
from scipy import sparse
import networkx as nx

from materials_concepts.utils.constants import ORIGIN_DATE
from tqdm import tqdm
from materials_concepts.utils.utils import save_pickle, load_pickle


class Graph:
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
        return self.edges[self.edges[:, 2] < (date - ORIGIN_DATE).days]

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
                dim=max(self.vertices) + 1 if full else None,
            )
            for year in years
        ]

    def compute_event_years_for_pairs(
        self,
        pairs: np.ndarray,
        start_year: int,
        end_year: int | None = None,
        cache_path: str | None = None,
    ) -> np.ndarray:
        """
        For each (u,v) in pairs, return the first calendar year when an edge (u,v)
        appears strictly AFTER start_year (i.e., event_year > start_year). If end_year is
        provided, only consider events up to and including end_year; otherwise later events
        are ignored (treated as no event in window). Returns -1 when no event within the window.

        If cache_path is provided and exists, loads and returns cached years.
        """
        if cache_path is not None:
            try:
                cached = load_pickle(cache_path)
                if isinstance(cached, dict) and "years" in cached:
                    return np.asarray(cached["years"], dtype=int)
                # Backward compatibility if plain array was cached
                if isinstance(cached, (list, np.ndarray)):
                    return np.asarray(cached, dtype=int)
            except Exception:
                pass

        # Build mapping from undirected pair -> earliest event year in (start_year, end_year]
        pair_min_year: dict[tuple[int, int], int] = {}

        # Convert edge offsets to calendar years once
        # edges is expected to be of shape (m, 3) with columns [u, v, day_offset]
        for u, v, day_offset in tqdm(self.edges, desc="Scanning edges for event years"):
            try:
                u = int(u)
                v = int(v)
                day_offset = int(day_offset)
            except Exception:
                # skip malformed row
                continue

            year = (ORIGIN_DATE + timedelta(days=day_offset)).year

            if year <= start_year:
                continue
            if end_year is not None and year > end_year:
                continue

            key = (u, v) if u <= v else (v, u)
            prev = pair_min_year.get(key)
            if prev is None or year < prev:
                pair_min_year[key] = year

        # Lookup for requested pairs
        years_out = np.full(len(pairs), -1, dtype=int)
        for i, (u, v) in enumerate(tqdm(pairs, desc="Assigning event years to pairs")):
            try:
                u = int(u)
                v = int(v)
            except Exception:
                years_out[i] = -1
                continue
            key = (u, v) if u <= v else (v, u)
            years_out[i] = int(pair_min_year.get(key, -1))

        if cache_path is not None:
            try:
                save_pickle({"years": years_out.tolist(), "start_year": start_year, "end_year": end_year}, cache_path)
            except Exception:
                pass

        return years_out
