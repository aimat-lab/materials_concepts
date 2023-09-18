import numpy as np
from utils import load, load_compressed
from collections import Counter
from graph import Graph
from tqdm import tqdm
import bfs
import fire


TN = 1
FN = 2
FP = 3
TP = 4


def main(
    sample_size=3000,
    data_path="data/model/data.M.pkl",
    graph_path="data/graph/edges.M.pkl",
    predictions_path="data/model/combi/predictions.pkl.gz",
):
    print("Loading data")
    data = load(data_path)
    truth = data["y_test"]
    scores = load_compressed(predictions_path)

    assert len(data["y_test"]) == len(scores)

    classes = np.zeros_like(scores)

    classes[(scores < 0.5) & (truth == 0)] = TN
    classes[(scores < 0.5) & (truth == 1)] = FN
    classes[(scores >= 0.5) & (truth == 0)] = FP
    classes[(scores >= 0.5) & (truth == 1)] = TP

    class_dist = Counter(classes)
    print(f"Class distribution: {class_dist}")

    index = {}
    index[TN] = np.where(classes == TN)[0]
    index[FN] = np.where(classes == FN)[0]
    index[FP] = np.where(classes == FP)[0]
    index[TP] = np.where(classes == TP)[0]

    print("Loading graph")

    g = Graph.from_path(graph_path).get_nx_graph(data["year_test"])

    edges_depth = {}
    for _class in (TN, FN, FP, TP):
        print(f"Class: {_class}")

        if len(index[_class]) < sample_size:
            index_sample = index[_class]
        else:
            index_sample = np.random.choice(index[_class], sample_size, replace=False)

        depthCounter = Counter()

        for ind in tqdm(index_sample):
            u, v = data["X_test"][ind]

            distance = bfs.distance(g, u, v)
            depthCounter.update([distance])
            edges_depth[ind] = distance

        print(f"Depth Counter: {depthCounter}")
        print("-" * 80)


if __name__ == "__main__":
    fire.Fire(main)
