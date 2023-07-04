from tqdm import tqdm
import numpy as np
import fire
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from graph import Graph


logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s", "%m-%d-%Y %H:%M:%S"
)

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

logger.addHandler(stdout_handler)


def get_degrees(adjmatrix):
    return np.array(adjmatrix.sum(0))[0]


def normalize_degrees(degrees):
    return degrees / np.max(degrees) if np.max(degrees) > 0 else degrees


def get_jaccard_coefficient(
    u: int, v: int, num_neighbors: np.ndarray, adjacency_matrix_squared: np.ndarray
):
    num_neighbors_intersection = adjacency_matrix_squared[u, v]
    num_neighbors_union = (
        num_neighbors[u] + num_neighbors[v] - num_neighbors_intersection
    )
    return (
        num_neighbors_intersection / num_neighbors_union
        if num_neighbors_union > 0
        else 0
    )


def retrieve_properties_for_vertex_pair(
    AA02,
    AA12,
    AA22,
    all_degs0,
    all_degs1,
    all_degs2,
    all_degs0_n,
    all_degs1_n,
    all_degs2_n,
    all_degs02,
    all_degs12,
    all_degs22,
    v1,
    v2,
    include_jaccard=True,
):
    """
    Computes hand-crafted properties for one vertex in vlist
    """
    all_properties = []

    all_properties.append(all_degs0_n[v1])  # 0
    all_properties.append(all_degs0_n[v2])  # 1
    all_properties.append(all_degs1_n[v1])  # 2
    all_properties.append(all_degs1_n[v2])  # 3
    all_properties.append(all_degs2_n[v1])  # 4
    all_properties.append(all_degs2_n[v2])  # 5
    all_properties.append(all_degs02[v1])  # 6
    all_properties.append(all_degs02[v2])  # 7
    all_properties.append(all_degs12[v1])  # 8
    all_properties.append(all_degs12[v2])  # 9
    all_properties.append(all_degs22[v1])  # 10
    all_properties.append(all_degs22[v2])  # 11
    all_properties.append(AA02[v1, v2])  # 12
    all_properties.append(AA12[v1, v2])  # 13
    all_properties.append(AA22[v1, v2])  # 14

    if include_jaccard:
        logging.debug("Computing jaccard")
        all_properties.append(
            get_jaccard_coefficient(
                u=v1, v=v2, num_neighbors=all_degs0, adjacency_matrix_squared=AA02
            )
        )
        all_properties.append(
            get_jaccard_coefficient(
                u=v1, v=v2, num_neighbors=all_degs1, adjacency_matrix_squared=AA12
            )
        )
        all_properties.append(
            get_jaccard_coefficient(
                u=v1, v=v2, num_neighbors=all_degs2, adjacency_matrix_squared=AA22
            )
        )

    return np.array(all_properties).astype(np.float32)


def compute_all_properties_of_list(all_sparse, vlist, include_jaccard=True):
    """
    Computes hand-crafted properties for all vertices in vlist

    all_sparse: list of adjacency matrices.
    vlist: ?
    """

    logging.info("Computing all matrix squares")
    # compute matrix squares
    AA02 = all_sparse[0] ** 2
    # AA02 = AA02 / AA02.max()  # AA02.astype(np.float16)
    logging.info("1...")
    AA12 = all_sparse[1] ** 2
    # AA12 = AA12 / AA12.max()
    logging.info("2...")
    AA22 = all_sparse[2] ** 2
    # AA22 = AA02 / AA22.max()
    logging.info("3")

    logging.info("Computing degrees")

    all_degs0 = get_degrees(all_sparse[0])
    all_degs0_n = normalize_degrees(all_degs0)
    all_degs1 = get_degrees(all_sparse[1])
    all_degs1_n = normalize_degrees(all_degs1)
    all_degs2 = get_degrees(all_sparse[2])
    all_degs2_n = normalize_degrees(all_degs2)

    all_degs02 = normalize_degrees(get_degrees(AA02))
    all_degs12 = normalize_degrees(get_degrees(AA12))
    all_degs22 = normalize_degrees(get_degrees(AA22))

    logging.info("Computed all degrees")

    all_properties = []

    for ii in tqdm(range(len(vlist))):
        vals = retrieve_properties_for_vertex_pair(
            AA02,
            AA12,
            AA22,
            all_degs0,
            all_degs1,
            all_degs2,
            all_degs0_n,
            all_degs1_n,
            all_degs2_n,
            all_degs02,
            all_degs12,
            all_degs22,
            vlist[ii][0],
            vlist[ii][1],
            include_jaccard=include_jaccard,
        )

        all_properties.append(vals)

    return all_properties


def calculate_embeddings(graph_path, X_train, X_test, include_jaccard=True):
    logging.debug("Building graph")
    graph = Graph(graph_path)

    logging.debug("Calculating adjacency matrices")
    matrices = graph.get_adj_matrices([2010, 2013, 2016])

    embeddings = compute_all_properties_of_list(
        matrices, np.concatenate([X_train, X_test]), include_jaccard=include_jaccard
    )

    return embeddings[: len(X_train)], embeddings[len(X_train) :]


def main(
    graph_path="data/graph/edges.pkl",
    data_path="data/model/data.pkl",
    output_path="data/model/baseline/embeddings.pkl",
    include_jaccard=True,
    log_path="logs/baseline_embeddings.log",
):
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    import pickle

    logging.info("Loading data...")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    logging.info("Calculating embeddings...")
    X_train, X_test = calculate_embeddings(
        graph_path, data["X_train"], data["X_test"], include_jaccard=include_jaccard
    )

    logging.info("Saving embeddings...")
    with open(output_path, "wb") as f:
        pickle.dump({"X_train": X_train, "X_test": X_test}, f)


if __name__ == "__main__":
    fire.Fire(main)
