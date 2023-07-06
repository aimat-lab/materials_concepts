import pickle
import torch
from torch_geometric.data import Data
import numpy as np

import sys, os

parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_directory)

from graph import Graph
from metrics import print_metrics


def shuffle(X, y):
    """Shuffle X and y in unison"""
    assert len(X) == len(y)
    p = np.random.permutation(len(X))
    return X[p], y[p]


def sample(X: np.ndarray, y: np.ndarray, pos_to_neg_ratio: float):
    """Sample the data to have a given ratio of positive to negative samples"""
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]

    curr_pos_ratio = len(pos_indices) / len(y)
    if curr_pos_ratio > pos_to_neg_ratio:
        num_neg = len(neg_indices)
        num_pos = int(pos_to_neg_ratio * num_neg)
    else:
        num_pos = len(pos_indices)
        num_neg = int(num_pos / pos_to_neg_ratio)

    pos_indices = np.random.choice(pos_indices, size=num_pos, replace=False)
    neg_indices = np.random.choice(neg_indices, size=num_neg, replace=False)

    X = np.concatenate([X[pos_indices], X[neg_indices]])
    y = np.concatenate([y[pos_indices], y[neg_indices]])

    return shuffle(X, y)


graph = Graph("data/graph/edges.pkl")


def calc_degs(adj):
    return np.array(adj.sum(0))[0]


embs = [
    calc_degs(adj) for adj in graph.get_adj_matrices([2014, 2015, 2016], binary=False)
]

test_embs = [
    calc_degs(adj) for adj in graph.get_adj_matrices([2017, 2018, 2019], binary=False)
]

x = torch.tensor(np.array(embs), dtype=torch.float).T

test_x = torch.tensor(np.array(test_embs), dtype=torch.float).T


edge_index = torch.tensor(graph.get_until_year(2016), dtype=torch.long).t().contiguous()

data = Data(x=x, edge_index=edge_index)

with open("data/model/data.pkl", "rb") as f:
    future_data = pickle.load(f)

X_train, y_train = sample(future_data["X_train"], future_data["y_train"], 0.5)

data.future_edge_index = torch.tensor(X_train, dtype=torch.long).t().contiguous()
data.future_edge_labels = torch.tensor(y_train, dtype=torch.float)

test_edge_index = (
    torch.tensor(graph.get_until_year(2019), dtype=torch.long).t().contiguous()
)
test_data = Data(x=test_x, edge_index=test_edge_index)
test_data.future_edge_index = (
    torch.tensor(future_data["X_test"], dtype=torch.long).t().contiguous()
)
test_data.future_edge_labels = torch.tensor(future_data["y_test"], dtype=torch.float)
