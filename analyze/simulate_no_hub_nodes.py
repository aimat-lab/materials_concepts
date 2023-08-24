import numpy as np
from collections import Counter
from graph import Graph
from tqdm import tqdm
import pandas as pd
import bfs
import sys

N = sys.argv[1] or 50
POWER_HUBS_CUTOFF = sys.argv[2] or 500
SAVE_PATH = sys.argv[3] or "data/analzye/depth_distribution.csv"

print("Loading graph")
g = Graph.from_path("data/graph/edges.M.pkl").get_nx_graph(2023)


hub_nodes = sorted(g.degree, key=lambda x: x[1], reverse=True)[:POWER_HUBS_CUTOFF]

print("Removing hub nodes")
g.remove_nodes_from(hub_nodes)

sample = np.random.choice(g.nodes, N, replace=False)
sample_data = []
for node in tqdm(sample):
    bfs.graph_depth(g, node)

    depth_counter = Counter()
    for vertex in g:
        depth_counter.update([g.nodes[vertex]["depth"]])

    data = dict(depth_counter)
    data.update({"origin": node})
    sample_data.append(data)

df = pd.DataFrame(sample_data)


df.to_csv(SAVE_PATH, index=False)

pd.set_option("display.float_format", "{:.0f}".format)
print(df.describe().round(0).drop("count").drop("origin", axis=1))
