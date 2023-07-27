from torch import nn
import torch
import pandas as pd
import os, sys
import gzip, pickle
from collections import namedtuple
import numpy as np

parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_directory)

from graph import Graph

Data = namedtuple(
    "Data", ["pairs", "feature_embeddings", "concept_embeddings", "labels"]
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_compressed(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def get_embeddings(pairs, feature_embeddings, concept_embeddings):
    l = []
    for v1, v2 in pairs:
        i1 = int(v1.item())
        i2 = int(v2.item())

        emb1_f = np.array(feature_embeddings[i1])
        emb2_f = np.array(feature_embeddings[i2])

        emb1_c = np.array(concept_embeddings[i1])
        emb2_c = np.array(concept_embeddings[i2])

        l.append(np.concatenate([emb1_f, emb2_f, emb1_c, emb2_c]))
    return torch.tensor(np.array(l)).float()


class BaselineNetwork(nn.Module):
    def __init__(self, layer_dims: list):
        """
        Fully Connected layers
        """
        super(BaselineNetwork, self).__init__()

        layers = []
        for in_, out_ in zip(layer_dims[:-1], layer_dims[1:]):
            layers.append(nn.Linear(in_, out_))
            layers.append(nn.BatchNorm1d(out_))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.1))  # TODO: try out 0.3

        layers.pop()  # remove last dropout layer
        layers.pop()  # remove last relu layer
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Pass throught network
        """
        res = self.net(x)

        return res


print("Loading model...")
layers = [1556, 1024, 1024, 512, 256, 32, 1]
model = BaselineNetwork(layers).to(device)
model.load_state_dict(torch.load("data/model/combi/model.pt"))

search = "znonanorod array"


lookup = pd.read_csv("data/table/lookup/lookup_medium.csv")
lookup_c_id = {concept: id for concept, id in zip(lookup["concept"], lookup["id"])}
lookup_id_c = {id: concept for concept, id in zip(lookup["concept"], lookup["id"])}

concept_id = lookup_c_id[search]

print("Loading graph...")
prediction_since = 2019
graph = Graph("data/graph/edges_medium.pkl")
g_pred = graph.get_nx_graph(prediction_since)

print("Finding unconnected nodes...")
unconnected = []
for n in g_pred.nodes:
    if n == concept_id:
        continue

    if g_pred.has_edge(concept_id, n):
        continue

    unconnected.append(n)


pairs = [(concept_id, n) for n in unconnected]

# load embeddings
emb_f_test_path = "data/model/combi/features_2019.M.pkl.gz"
emb_c_test_path = "data/model/concept_embs/av_embs_2019.M.pkl.gz"

data = Data(
    pairs=pairs,
    feature_embeddings=load_compressed(emb_f_test_path),
    concept_embeddings=load_compressed(emb_c_test_path),
)

inputs = get_embeddings(
    data.pairs, data.feature_embeddings, data.concept_embeddings
).to(device)
outs = model(inputs)

with open("data/model/combi/results.pkl") as f:
    pickle.dump(
        {
            "concepts": [lookup_id_c[id] for id in unconnected],
            "predictions": outs,
        }
    )
