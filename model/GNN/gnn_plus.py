import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score
import pickle, gzip
import os, sys
import logging

parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_directory)

from graph import Graph

NODE_DIM = 768


def setup_logger(file, level=logging.INFO, log_to_stdout=True):
    logger = logging.getLogger()
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"
    )

    if log_to_stdout:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)

    file_handler = logging.FileHandler(file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_compressed(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def load_node_features(cut_off_year, num_vertices):
    path = f"data/model/GNN/av_embs_{cut_off_year}.M.pkl.gz"
    data_dict = load_compressed(path)

    data = []
    for i in range(num_vertices):
        data.append(data_dict.get(i, torch.zeros(NODE_DIM)))

    return torch.stack(data)


def create_pyg_dataset(data_dict, graph, dataset_type):
    cut_off_year = data_dict[f"year_{dataset_type}"]
    edge_list = graph.get_until_year(cut_off_year)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()[:2, :]
    edge_attr = torch.tensor(edge_list, dtype=torch.float).t().contiguous()[2, :]

    vertex_pairs = torch.tensor(data_dict[f"X_{dataset_type}"], dtype=torch.int)
    labels = torch.tensor(data_dict[f"y_{dataset_type}"], dtype=torch.float)

    logger.info("Loading node features")
    x = load_node_features(cut_off_year, len(graph.vertices))

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pair=vertex_pairs,
        y=labels,
    )

    return data


# vertices (2016): 141748
# vertices (2019): 146764
# vertices (full): 148032


class GCN(nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        return x


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        return torch.sigmoid(x)


class Net(nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(Net, self).__init__()
        self.gcn = GCN(num_features, hidden_channels)
        self.mlp = MLP(2 * hidden_channels, hidden_channels, 1)

    def forward(self, data):
        x, edge_index, pairs = data.x, data.edge_index, data.pair
        logger.debug("Applying GCN")
        x = self.gcn(x, edge_index)

        # Concatenate the embeddings of the two nodes in each pair
        logger.debug("Concatenating embeddings")
        pair_embeddings = torch.cat([x[pairs[:, 0]], x[pairs[:, 1]]], dim=-1)
        logger.debug(pair_embeddings.shape)

        logger.debug("Applying MLP")
        return self.mlp(pair_embeddings)


def train(model, train_data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()

    out = model(train_data)
    loss = criterion(out.view(-1), train_data.y)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(model, test_data):
    model.eval()

    with torch.no_grad():
        out = model(test_data)

    auc = roc_auc_score(test_data.y.cpu(), out.view(-1).cpu())

    return auc


def main():
    global logger
    logger = setup_logger("logs/gnn_plus.log", level=logging.DEBUG, log_to_stdout=True)

    hidden_channels = NODE_DIM  # number of hidden channels in GCN and MLP

    logger.info("Loading data")
    data_dict = load_pkl("data/model/data.M.pkl")
    logger.info("Loading graph")
    graph = Graph("data/graph/edges_medium.pkl")

    logger.info("Creating PyG dataset 'train'")
    pyg_graph_train = create_pyg_dataset(data_dict, graph, "train")
    logger.info("Creating PyG dataset 'test'")
    pyg_graph_test = create_pyg_dataset(data_dict, graph, "test")

    model = Net(NODE_DIM, hidden_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    train_data = pyg_graph_train
    test_data = pyg_graph_test

    logger.info("Training")
    for epoch in range(100):
        logger.debug(f"Epoch {epoch}")
        loss = train(model, train_data, optimizer, criterion)
        if epoch % 10 == 0:
            auc = test(model, test_data)
            print(f"Epoch: {epoch}, Loss: {loss:.4f}, AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
