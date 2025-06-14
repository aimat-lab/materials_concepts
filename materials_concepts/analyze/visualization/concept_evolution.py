from ast import literal_eval

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter
from tqdm import tqdm

from materials_concepts.model.graph import Graph

df = pd.read_csv("data/table/materials-science.llama2.works.csv")
df.llama_concepts = df.llama_concepts.apply(literal_eval)
df["publication_date"] = pd.to_datetime(df["publication_date"])
df["year"] = df["publication_date"].dt.year

df = df[df["year"] <= 2022]


G = Graph.from_path("data/graph/edges.L.pkl")
connections = pd.Series(index=range(df.year.min(), df.year.max() + 1), dtype=int)
for year in tqdm(range(df.year.min(), df.year.max() + 1)):
    connections[year] = len(G.get_until_year(year))


# Group by 'year' and for each group extract unique concepts
unique_concepts_per_year = df.groupby("year").apply(
    lambda group: len(set.union(*group["llama_concepts"].map(set)))
)

cumulative_unique_concepts = unique_concepts_per_year.cumsum()
possible_connections = cumulative_unique_concepts**2


plt.figure(figsize=(10, 6))
cumulative_unique_concepts.plot(kind="line", marker="o", color="b")
plt.title("Accumulated Number of Unique Concepts per Year", fontsize=20, y=1.05)
plt.ylabel("Accumulated Number of Unique Concepts")
plt.xlabel("Year")
plt.xticks(rotation=45)
ax = plt.gca()  # Get the current axis instance
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False, useOffset=False))
plt.show()

# Plotting Possible and Actual Number of Connections with Two Axes

possible_connections = possible_connections.loc[1965:]

plt.plot(
    possible_connections.index,
    possible_connections,
    "r-o",
    label="Possible Connections",
)

# Plotting Actual Connections
connections = connections.loc[possible_connections.index]
plt.plot(possible_connections.index, connections, "g-o", label="Actual Connections")

# Setting labels, title, and legends
plt.xlabel("Year")
plt.ylabel("Number of Connections")
plt.title("Possible and Actual Connections per Year", fontsize=20, y=1.05)

# Applying symlog scale to y-axis
plt.yscale("log")

# Optionally, setting the rotation of x-axis labels and adjusting the layout
plt.xticks(rotation=45)
plt.legend(loc="upper left")
plt.tight_layout()

# Displaying the plot
plt.show()
