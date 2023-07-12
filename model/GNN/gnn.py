from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
import torch.nn.functional as F
import torch
from torch_geometric.data import Data
import fire
import pickle
import numpy as np
import sys, os

parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_directory)

from graph import Graph
from metrics import print_metrics


class GCN(torch.nn.Module):
    def __init__(self, input_channel, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_channel, hidden_channels)
        # self.conv2 = GATConv(hidden_channels, hidden_channels)
        # self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        self.relu1 = torch.nn.ReLU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu1(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # x = self.conv2(x, edge_index)
        # x = self.bn2(x)
        # x = self.prelu2(x)
        # x = F.dropout(x, p=0.5, training=self.training)

        # x = self.conv3(x, edge_index)
        # x = self.bn3(x)
        # x = torch.nn.PReLU(x)
        # x = F.dropout(x, p=0.5, training=self.training)

        return x


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, gnn, mlp):
        super(LinkPredictor, self).__init__()
        self.gnn = gnn
        self.mlp = mlp

    def forward(self, data):
        x, future_edge_index = data.x, data.future_edge_index
        x = self.gnn(x, future_edge_index)

        # Link prediction based on concatenated node embeddings
        x_i = torch.index_select(x, 0, future_edge_index[0])
        x_j = torch.index_select(x, 0, future_edge_index[1])
        x_concat = torch.cat([x_i, x_j], dim=-1)

        return self.mlp(x_concat)


def shuffle(X, y):
    """Shuffle X and y in unison"""
    assert len(X) == len(y)
    p = np.random.permutation(len(X))
    return X[p], y[p]


def sample(
    X: np.ndarray, y: np.ndarray, pos_to_neg_ratio: float
):  # problem with *_medium dataset
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


def calc_degs(adj):
    return np.array(adj.sum(0))[0]


def prepare_data(
    graph_path="data/graph/edges.pkl",
    data_path="data/model/data.pkl",
):
    graph = Graph(graph_path)

    embs = [
        calc_degs(adj)
        for adj in graph.get_adj_matrices([2014, 2015, 2016], binary=False)
    ]

    test_embs = [
        calc_degs(adj)
        for adj in graph.get_adj_matrices([2017, 2018, 2019], binary=False)
    ]

    x = torch.tensor(np.array(embs), dtype=torch.float).T

    test_x = torch.tensor(np.array(test_embs), dtype=torch.float).T

    edge_index = (
        torch.tensor(graph.get_until_year(2016), dtype=torch.long).t().contiguous()
    )

    data = Data(x=x, edge_index=edge_index)

    with open(data_path, "rb") as f:
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
    test_data.future_edge_labels = torch.tensor(
        future_data["y_test"], dtype=torch.float
    )

    return data, test_data


def main(
    graph_data="data/graph/edges.pkl",
    data_path="data/model/data.pkl",
    input_channel=3,
    hidden_channels=8,
    hidden_mlp=16,
    output_dim=1,
    epochs=50,
    lr=0.005,
):
    data, test_data = prepare_data(graph_data, data_path)

    model = LinkPredictor(
        GCN(input_channel=input_channel, hidden_channels=hidden_channels),
        MLP(
            input_dim=2 * hidden_channels, hidden_dim=hidden_mlp, output_dim=output_dim
        ),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        out = model(data)

        loss = F.binary_cross_entropy(out, data.future_edge_labels.unsqueeze(1))

        # Backward pass
        loss.backward()
        optimizer.step()

        # Print loss
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

    # Evaluate on the test set
    model.eval()
    with torch.no_grad():
        pred = model(test_data)

    print_metrics(
        test_data.future_edge_labels,
        pred,
    )


if __name__ == "__main__":
    fire.Fire(main)
