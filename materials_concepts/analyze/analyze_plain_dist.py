from collections import Counter

import fire
import numpy as np
from tqdm import tqdm

from materials_concepts.analyze.bfs import distance as bfs_distance
from materials_concepts.model.graph import Graph


def main(
    N=10000,
    graph_path="data/graph/edges.M.pkl",
    year=2016,
    year_end=2019,
):
    print("Loading graph")
    G = Graph.from_path(graph_path)
    g = G.get_nx_graph(year)
    g_future = G.get_nx_graph(year_end)

    pos_edges = np.array(list(set(g_future.edges()) - set(g.edges())))
    print(f"Amount of positive edges ({year}-{year_end}): {len(pos_edges)}")

    depthCounter = Counter()

    # sample N edges
    sample = np.random.choice(len(pos_edges), N, replace=False)
    for i, (u, v) in tqdm(enumerate(pos_edges[sample]), total=N):
        distance = bfs_distance(g, u, v)
        depthCounter.update([distance])

        if (i + 1) % 1000 == 0:
            print(f"Depth Counter: {depthCounter}")

    print("Finished")
    print(f"Depth Counter: {depthCounter}")


if __name__ == "__main__":
    fire.Fire(main)
