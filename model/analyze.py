import numpy as np
import pickle
from collections import Counter, deque
from graph import Graph
from tqdm import tqdm


def load(path):
    with open(path, "rb") as f:
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

# sample 5000 edges
sample = np.random.choice(len(pos_edges), 5000, replace=False)
for i, (u, v) in tqdm(enumerate(pos_edges[sample])):
    bfs_to_depth(g, u)
    depthCounter.update([g.nodes[v]["depth"]])

    if i % 1000 == 0:
        print(f"Depth Counter: {depthCounter}")


print("Finished")
print(f"Depth Counter: {depthCounter}")
