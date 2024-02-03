import logging

import fire
import numpy as np
from tqdm import tqdm

from materials_concepts.model.graph import Graph
from materials_concepts.utils.utils import save_compressed, setup_logger

logger = setup_logger(
    logging.getLogger(__name__), file="logs/pre_compute.log", level=logging.DEBUG
)


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
