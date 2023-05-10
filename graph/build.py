import pandas as pd
import numpy as np
from tqdm import tqdm

# build lookup dict: concept -> id
lookup = {}
with open("data/lists/all_concepts_filtered.txt") as f:
    for index, line in enumerate(f.readlines()):
        concept = line.strip()
        lookup[concept] = index

df = pd.read_csv("data/materials-science.rake.works.csv")
df = df.dropna(subset=["rake_concepts"])  # drop rows with no concepts

origin_day = pd.to_datetime("1970-01-01")
df["pub_date_days"] = pd.to_datetime(df.publication_date).apply(
    lambda ts: (ts - origin_day).days
)


def get_pairs(items):
    pairs = []
    for i1 in items:
        for i2 in items:
            if i1 == i2:
                # this ensures that we don't get to the diagonal line in the pairing matrix
                # as order doesn't matter, this yields just half of the matrix (excluding the diagonal)
                break
            pairs.append((i1, i2))
    return pairs


all_edges = []
for concept_list, pub_date in tqdm(list(zip(df.rake_concepts, df.pub_date_days))):
    concept_ids = {
        lookup[c] for c in concept_list.split(",") if lookup.get(c) is not None
    }  # set comprehension because rake doesn't filter out duplicates

    for v1, v2 in get_pairs(concept_ids):
        all_edges.append(np.array((v1, v2, pub_date)))


all_edges = np.array(all_edges)
np.savez_compressed("graph/edges.npz", all_edges)

# load edges
# edges = np.load("graph/edges.npz", allow_pickle=True)["arr_0"]
