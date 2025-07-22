from collections import deque


def bfs(graph, start_node):
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

        # Get all neighbors of the dequeued vertex node
        # If a neighbor hasn't been visited, then mark it visited and enqueue it
        for i in graph.neighbors(node):
            if not graph.nodes[i]["visited"]:
                queue.append((i, depth + 1))
                graph.nodes[i]["visited"] = True
                graph.nodes[i]["depth"] = depth + 1
