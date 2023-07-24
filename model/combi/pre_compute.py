from tqdm import tqdm
import pickle, gzip
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

file_handler = logging.FileHandler("logs.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(stdout_handler)
logger.addHandler(file_handler)


def get_square_matrices(graph_path, years, binary):
    logging.debug("Building graph")
    graph = Graph(graph_path)

    logging.debug("Calculating adjacency matrices")
    matrices = graph.get_adj_matrices(years, binary=binary)

    logging.debug("Squaring matrices")
    squares = [m**2 for m in tqdm(matrices)]
    return {year: square for year, square in zip(years, squares)}


def save_compressed(obj, path):
    compressed = gzip.compress(pickle.dumps(obj))
    with open(path, "wb") as f:
        f.write(compressed)


def main(
    graph_path="data/graph/edges_medium.pkl",
    output_path="data/model/combi/embeddings.pkl.gz",
    binary=True,
    years=[2010, 2013, 2016],
):
    logging.info("Calculating embeddings...")
    matrices = get_square_matrices(graph_path, years=years, binary=binary)

    logging.info("Saving matrices...")

    matrices.update(
        {
            "binary": binary,
        }
    )

    logging.info(f"Saving Matrices: {matrices.keys()} to {output_path}")
    save_compressed(matrices, output_path)


if __name__ == "__main__":
    fire.Fire(main)
