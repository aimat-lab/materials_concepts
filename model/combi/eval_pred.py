from torch import nn
import torch
import pandas as pd
import os, sys
import gzip, pickle
from collections import namedtuple
import numpy as np
import logging

parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_directory)

from graph import Graph

Data = namedtuple(
    "Data", ["pairs", "feature_embeddings", "concept_embeddings", "labels"]
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_logger(level=logging.INFO, log_to_stdout=True):
    logger = logging.getLogger()
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"
    )

    if log_to_stdout:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)

    file_handler = logging.FileHandler("logs.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


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


logger = setup_logger(level=logging.INFO, log_to_stdout=True)

logger.info("Loading model")
layers = [1556, 1024, 1024, 512, 256, 32, 1]
model = BaselineNetwork(layers).to(device)
model.load_state_dict(torch.load("data/model/combi/model.pt"))

search = "znonanorod array"


lookup = pd.read_csv("data/table/lookup/lookup_medium.csv")
lookup_c_id = {concept: id for concept, id in zip(lookup["concept"], lookup["id"])}
lookup_id_c = {id: concept for concept, id in zip(lookup["concept"], lookup["id"])}

concept_id = lookup_c_id[search]

logger.info("Loading graph")
prediction_since = 2019
graph = Graph("data/graph/edges_medium.pkl")
g = Graph.from_edge_list(graph.get_until_year(prediction_since))
g_nx = graph.get_nx_graph(prediction_since)

logger.info("Finding unconnected nodes")
unconnected = []
for n in g.vertices:
    if n == concept_id:
        continue

    if g_nx.has_edge(concept_id, n):
        continue

    unconnected.append(n)

pairs = torch.tensor([(concept_id, n) for n in unconnected])

# load embeddings
emb_f_test_path = "data/model/combi/features_2019.M.pkl.gz"
emb_c_test_path = "data/model/concept_embs/av_embs_2019.M.pkl.gz"

data = Data(
    pairs=pairs,
    feature_embeddings=load_compressed(emb_f_test_path),
    concept_embeddings=load_compressed(emb_c_test_path),
    labels=None,
)

logger.info("Retrieving embeddings")
inputs = get_embeddings(
    data.pairs, data.feature_embeddings, data.concept_embeddings
).to(device)

logger.info("Predicting")
outs = model(inputs)
outs = outs.detach().cpu().numpy().flatten()

logger.info("Saving results")
with open("data/model/combi/results.pkl") as f:
    pickle.dump(
        {
            "concepts": [lookup_id_c[id] for id in unconnected],
            "predictions": outs,
        }
    )
