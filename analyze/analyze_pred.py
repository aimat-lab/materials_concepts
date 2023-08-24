import numpy as np
from utils import load, load_compressed, save
from collections import Counter
from graph import Graph
from tqdm import tqdm
import bfs


TN = 1
FN = 2
FP = 3
TP = 4

SAMPLE_SIZE = 3000

print("Loading data")
data = load("data/model/data.M.pkl")
truth = data["y_test"]
scores = load_compressed("data/model/combi/predictions.pkl.gz")

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

g = Graph.from_path("data/graph/edges.M.pkl").get_nx_graph(data["year_test"])

edges_depth = {}
for _class in (TN, FN, FP, TP):
    print(f"Class: {_class}")

    if len(index[_class]) < SAMPLE_SIZE:
        index_sample = index[_class]
    else:
        index_sample = np.random.choice(index[_class], SAMPLE_SIZE, replace=False)

    depthCounter = Counter()

    for ind in tqdm(index_sample):
        u, v = data["X_test"][ind]

        distance = bfs.distance(g, u, v)
        depthCounter.update([distance])
        edges_depth[ind] = distance

    print(f"Depth Counter: {depthCounter}")
    print("-" * 80)

save(edges_depth, "data/model/combi/edges_depth.pkl")

# First Run:
# TN (1): Counter({3: 638, 2: 360, 4: 2})
# FN (2): Counter({2: 31, 3: 15})
# FP (3): Counter({2: 806, 3: 194})
# TP (4): Counter({2: 232, 3: 12})
