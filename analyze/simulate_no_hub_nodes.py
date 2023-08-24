import numpy as np
from collections import Counter
from graph import Graph
from tqdm import tqdm
import pandas as pd
import bfs

N = 50
POWER_HUBS_CUTOFF = 2000

print("Loading graph")
g = Graph.from_path("data/graph/edges.M.pkl").get_nx_graph(2023)


degs = [deg for _, deg in g.degree]
degs = sorted(degs, reverse=True)

cut_off_degree = degs[POWER_HUBS_CUTOFF]
print("Cut-off degree:", cut_off_degree)

hub_nodes = [node for node, deg in g.degree if deg > cut_off_degree]

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


df.to_csv("data/depth_distribution_wo_2000_hubs.csv", index=False)

pd.set_option("display.float_format", "{:.0f}".format)
print(df.describe().round(0).drop("count").drop("origin", axis=1))
