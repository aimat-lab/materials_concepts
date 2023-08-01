import numpy as np
import pickle
import gzip
from collections import Counter, deque
from graph import Graph
from tqdm import tqdm


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_compressed(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def bfs_to_depth(graph, start_node, max_depth=None):
    # Initialize all nodes as not visited
    for node in graph:
        graph.nodes[node]["visited"] = False
        graph.nodes[node]["depth"] = float("inf")

    # Create a deque for BFS
    queue = deque([(start_node, 0)])

    # Mark the source node as visited and set its depth as 0
    graph.nodes[start_node]["visited"] = True
    graph.nodes[start_node]["depth"] = 0

    while queue:
        # Dequeue a vertex from queue and print it
        node, depth = queue.popleft()

        # If depth reached is equal to max_depth, stop exploring its neighbors
        if max_depth and depth == max_depth:
            break

        # Get all neighbors of the dequeued vertex node
        # If a neighbor hasn't been visited, then mark it visited and enqueue it
        for i in graph.neighbors(node):
            if not graph.nodes[i]["visited"]:
                queue.append((i, depth + 1))
                graph.nodes[i]["visited"] = True
                graph.nodes[i]["depth"] = depth + 1


TN = 1
FN = 2
FP = 3
TP = 4

SAMPLE_SIZE = 1000

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
for _class in (FN, FP, TP):
    print(f"Class: {_class}")

    if len(index[_class]) < SAMPLE_SIZE:
        index_sample = index[_class]
    else:
        index_sample = np.random.choice(index[_class], SAMPLE_SIZE, replace=False)

    depthCounter = Counter()

    for ind in tqdm(index_sample):
        u, v = data["X_test"][ind]

        bfs_to_depth(g, u)
        depthCounter.update([g.nodes[v]["depth"]])
        edges_depth[ind] = g.nodes[v]["depth"]

    print(f"Depth Counter: {depthCounter}")
    print("-" * 80)

save(edges_depth, "data/model/combi/edges_depth.pkl")

# TN (1): Counter({3: 638, 2: 360, 4: 2})
