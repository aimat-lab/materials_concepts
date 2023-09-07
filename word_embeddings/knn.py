import numpy as np
import pickle
import gzip
from sklearn.neighbors import NearestNeighbors
import tabulate
import pandas as pd

lookup = pd.read_csv("data/table/lookup/lookup.M.2.csv")


def id_to_concept(id):
    return lookup[lookup["id"] == id]["concept"].values[0]


def concept_to_id(concept):
    return lookup[lookup["concept"] == concept]["id"].values[0]


def load_compressed(path):
    with open(path, "rb") as f:
        compressed = f.read()
    return pickle.loads(gzip.decompress(compressed))


k = 6
embeddings_path = "data/model/concept_embs/hq.word-embs.2016.M.pkl.gz"
data = load_compressed(embeddings_path)

keys = sorted(data.keys())
values = [np.array(data[k]) for k in keys]

nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(values)


def get_knn(concept):
    if concept not in keys:
        print(f"Concept {concept} not found")
        return []
    else:
        distances, indices = nbrs.kneighbors([np.array(data[concept])])
        return [
            (keys[i], round(d, 3)) for i, d in zip(indices[0][1:], distances[0][1:])
        ]


def print_knn(concept, translate=False):
    """Print tabulated knn for concept"""
    concept = concept_to_id(concept) if translate else concept

    knn = get_knn(concept)
    if translate:
        knn = [(id_to_concept(c), d) for c, d in knn]

    print(tabulate.tabulate(knn))


TRANSLATE = True

print("pyrocarbon matrix")
print_knn("pyrocarbon matrix", translate=TRANSLATE)
print("\n\n")

print("adiabatic temperature change")
print_knn("adiabatic temperature change", translate=TRANSLATE)
print("\n\n")

print("pure titanium sheet")
print_knn("pure titanium sheet", translate=TRANSLATE)
