import numpy as np
from utils import load
from collections import Counter
from graph import Graph
from tqdm import tqdm
import bfs
import sys

N = int(sys.argv[1]) or 10000

print("Loading data")
data = load("data/model/data.M.pkl")

pos_index = np.where(data["y_train"] == 1)
pos_edges = data["X_train"][pos_index]


c = Counter()

for edge in pos_edges:
    c.update(edge)

print("Loading graph")
g = Graph.from_path("data/graph/edges.M.pkl").get_nx_graph(data["year_train"])


depthCounter = Counter()

# sample N edges
sample = np.random.choice(len(pos_edges), N, replace=False)
for i, (u, v) in tqdm(enumerate(pos_edges[sample]), total=N):
    distance = bfs.distance(g, u, v)
    depthCounter.update([distance])

    if i % 1000 == 0:
        print(f"Depth Counter: {depthCounter}")


print("Finished")
print(f"Depth Counter: {depthCounter}")
