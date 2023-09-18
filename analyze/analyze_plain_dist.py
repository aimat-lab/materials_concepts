import numpy as np
from utils import load
from collections import Counter
from graph import Graph
from tqdm import tqdm
import bfs
import fire


def main(
    N=10000,
    data_path="data/model/data.M.pkl",
    graph_path="data/graph/edges.M.pkl",
    period="train",
):
    print("Loading data")
    data = load(data_path)

    pos_index = np.where(data[f"y_{period}"] == 1)
    pos_edges = data[f"X_{period}"][pos_index]

    c = Counter()

    for edge in pos_edges:
        c.update(edge)

    print("Loading graph")
    g = Graph.from_path(graph_path).get_nx_graph(data[f"year_{period}"])

    depthCounter = Counter()

    # sample N edges
    sample = np.random.choice(len(pos_edges), N, replace=False)
    for i, (u, v) in tqdm(enumerate(pos_edges[sample]), total=N):
        distance = bfs.distance(g, u, v)
        depthCounter.update([distance])

        if (i + 1) % 1000 == 0:
            print(f"Depth Counter: {depthCounter}")

    print("Finished")
    print(f"Depth Counter: {depthCounter}")


if __name__ == "__main__":
    fire.Fire(main)
