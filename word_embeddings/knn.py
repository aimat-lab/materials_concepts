import numpy as np
import pickle
import gzip
from sklearn.neighbors import NearestNeighbors
import tabulate


def load_compressed(path):
    with open(path, "rb") as f:
        compressed = f.read()
    return pickle.loads(gzip.decompress(compressed))


k = 6
embeddings_path = "data/model/concept_embs/word-embs.full.M.pkl.gz"
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


def print_knn(concept):
    """Print tabulated knn for concept"""
    knn = get_knn(concept)
    print(tabulate.tabulate(knn))


print("pyrocarbon matrix")
print_knn("pyrocarbon matrix")
print("\n\n")

print("adiabatic temperature change")
print_knn("adiabatic temperature change")
print("\n\n")

print("pure titanium sheet")
print_knn("pure titanium sheet")
