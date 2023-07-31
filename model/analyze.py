import numpy as np
import pickle
from collections import Counter, deque
from graph import Graph


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


class EdgeHolder:
    def __init__(self, edges):
        self.edges = edges

    def split_edges(self, contained_node):
        return_edges = []
        new_edges = []

        for edge in self.edges:
            if contained_node in edge:
                return_edges.append(edge)
            else:
                new_edges.append(edge)

        self.edges = new_edges
        return return_edges

    def __len__(self):
        return len(self.edges)

    def empty(self):
        return len(self) == 0


print("Loading data")
data = load("data/model/data.M.pkl")

pos_index = np.where(data["y_train"] == 1)
pos_edges = data["X_train"][pos_index]


c = Counter()

for edge in pos_edges:
    c.update(edge)

print("Loading graph")
g = Graph.from_path("data/graph/edges.M.pkl").get_nx_graph(data["year_train"])


edgeHolder = EdgeHolder(pos_edges)
depthCounter = Counter()
for node, count in c.most_common():
    if edgeHolder.empty():
        break

    print("Applying BFS")
    bfs_to_depth(g, node)  # in-place

    print(f"Node: {node}, Count: {count}")

    # Get all edges that contain the node
    edges = edgeHolder.split_edges(node)
    print(f"# Edges: {len(edges)} | # Edges left: {len(edgeHolder)}")

    other_nodes = set([i for edge in edges for i in edge if i != node])
    assert len(other_nodes) == len(edges)

    # Get the depths of the other nodes, based on start_node 'node'
    depths = [g.nodes[i]["depth"] for i in other_nodes]
    depthCounter.update(depths)

print(f"Depth Counter: {depthCounter}")
