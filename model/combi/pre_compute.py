from tqdm import tqdm
import pickle, gzip
import fire
import logging
import os
import sys
import numpy as np

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

file_handler = logging.FileHandler("logs.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(stdout_handler)
logger.addHandler(file_handler)


def get_node_features(graph_path, years, binary):
    logging.debug("Building graph")
    graph = Graph(graph_path)

    logging.debug("Calculating adjacency matrices")
    matrices = graph.get_adj_matrices(years, binary=binary, full=True)
    degrees = [calc_degs(m) for m in matrices]

    logging.debug("Squaring matrices")
    squares = [m**2 for m in tqdm(matrices)]
    degrees_squared = [calc_degs(m) for m in squares]

    v_features = []
    for v in graph.vertices:
        v_features.append(
            [degrees[i][v] for i in range(len(degrees))]
            + [degrees_squared[i][v] for i in range(len(degrees_squared))]
        )

    v_features = np.array(v_features)
    print(v_features.shape)

    return v_features


def calc_degs(adj):
    return np.array(adj.sum(0))[0]


def save_compressed(obj, path):
    compressed = gzip.compress(pickle.dumps(obj))
    with open(path, "wb") as f:
        f.write(compressed)


def main(
    graph_path="data/graph/edges_medium.pkl",
    output_path="data/model/combi/matrices_2016.pkl.gz",
    binary=True,
    years=[2010, 2013, 2016],
):
    logging.info("Calculating embeddings...")
    v_features = get_node_features(graph_path, years=years, binary=binary)

    logging.info("Saving matrices...")

    store = {
        "binary": binary,
        "years": years,
        "v_features": v_features,
    }

    logging.info(f"Saving features of nodes ({years}) to {output_path}")
    save_compressed(store, output_path)


if __name__ == "__main__":
    fire.Fire(main)
