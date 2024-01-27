import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import date

DAY_ORIGIN = date(1970, 1, 1)


def get_until(graph, day):
    return graph[graph[:, 2] < (day - DAY_ORIGIN).days]


with open("data/graph/edges.M.pkl", "rb") as f:
    edge_holder = pickle.load(f)

edges = edge_holder["edges"]

years = [1980, 1990, 2000, 2010, 2020]
colors = ["blue", "orange", "green", "red", "purple"]
percentiles = np.linspace(0, 1, 100)

plt.figure(figsize=(10, 7))

for year, color in zip(years, colors):
    print(year)
    cutoff_day = date(year, 1, 1)
    edges_until_year = get_until(edges, cutoff_day)

    # Compute node degrees
    _, counts = np.unique(
        np.concatenate((edges_until_year[:, 0], edges_until_year[:, 1])),
        return_counts=True,
    )

    # Sort counts (degrees) and calculate cumulative sum
    sorted_counts = np.sort(counts)
    cumulative_edges = np.cumsum(sorted_counts)
    cumulative_percentage = cumulative_edges / cumulative_edges[-1] * 100

    # Generate x-values as the percentage of nodes
    x_values = np.linspace(0, 100, len(cumulative_percentage))

    plt.plot(x_values, cumulative_percentage, label=str(year), color=color)


plt.xlabel("Percentage of nodes (from low to high node degree)")
plt.ylabel("Percentage of edges")
plt.title("Concept Centralization", fontsize=20, y=1.05)
plt.legend()
plt.grid()
plt.ylim(0, 100)  # To ensure Y-axis doesn't exceed 100%
plt.xlim(0, 100)  # To ensure X-axis starts at 0
plt.show()
