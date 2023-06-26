from tqdm import tqdm
import numpy as np

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from graph import Graph


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
    graph = Graph("graph/edges.pkl")

    matrices = graph.get_adj_matrices([2010, 2013, 2016])
    embeddings = compute_all_properties_of_list(
        matrices, np.concatenate([X_train, X_test])
    )

    return embeddings[: len(X_train)], embeddings[len(X_train) :]


if __name__ == "__main__":
    import pickle

    print("Loading data...")
    with open("model/data.pkl", "rb") as f:
        data = pickle.load(f)

    print("Calculating embeddings...")
    X_train, X_test = calculate_embeddings(data["X_train"], data["X_test"])

    print("Saving embeddings...")
    with open("model/baseline/embeddings.pkl", "wb") as f:
        pickle.dump({"X_train": X_train, "X_test": X_test}, f)
