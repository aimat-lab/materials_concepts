from datetime import date
import networkx as nx
from scipy import sparse
from tqdm import tqdm
import numpy as np

DAY_ORIGIN = date(1970, 1, 1)


class Graph:
    def __init__(self, path, num_of_vertices):
        self.num_of_vertices = num_of_vertices
        self.edges = Graph.load(path)  # todo load from pickled file
        self.adj_mat = Graph.build_adj_matrix(self.edges)
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

    def get_adj_matrices(self, years):
        return [
            Graph.build_adj_matrix(self.get_until(date(year, 12, 31))) for year in years
        ]


def get_degrees(adjmatrix):
    return np.array(adjmatrix.sum(0))[0]


def normalize_degrees(degrees):
    return degrees / np.max(degrees) if np.max(degrees) > 0 else degrees


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

    return np.array(all_properties).astype(np.float32)


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


def calculate_embeddings(X_train, X_test):
    graph = Graph("graph/edges.npz", 57.460)

    matrices = graph.get_adj_matrices([2010, 2013, 2016])
    embeddings = compute_all_properties_of_list(
        matrices, np.concatenate([X_train, X_test])
    )

    return embeddings[: len(X_train)], embeddings[len(X_train) :]


if __name__ == "__main__":
    import pickle

    print("Loading data...")
    with open("model/data_2019.pkl", "rb") as f:
        data = pickle.load(f)

    print("Calculating embeddings...")
    X_train, X_test = calculate_embeddings(data["X_train"], data["X_test"])

    print("Saving embeddings...")
    with open("model/baseline/embeddings_2019.pkl", "wb") as f:
        pickle.dump({"X_train": X_train, "X_test": X_test}, f)
