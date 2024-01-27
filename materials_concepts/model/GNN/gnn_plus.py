import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, confusion_matrix
import pickle, gzip
import os, sys
import logging
import fire

parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_directory)

from graph import Graph

NODE_DIM = 5


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


def calc_degs(adj):
    return torch.tensor(adj.sum(0))[0]


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_compressed(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def load_node_features(graph, cutoff_year, amt_years=5):
    years = list(range(cutoff_year - amt_years + 1, cutoff_year + 1))
    embs = [calc_degs(adj) for adj in graph.get_adj_matrices(years, binary=False)]
    return torch.stack(embs).to(torch.float).T


def create_pyg_dataset(data_dict, graph, dataset_type):
    cut_off_year = data_dict[f"year_{dataset_type}"]
    edge_list = graph.get_until_year(cut_off_year)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()[:2, :]
    edge_attr = torch.tensor(edge_list, dtype=torch.float).t().contiguous()[2, :]

    vertex_pairs = torch.tensor(data_dict[f"X_{dataset_type}"], dtype=torch.long)
    labels = torch.tensor(data_dict[f"y_{dataset_type}"], dtype=torch.float)

    logger.info("Loading node features")
    x = load_node_features(graph, cut_off_year, amt_years=5)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pair=vertex_pairs,
        y=labels,
    )

    return data


class GCN(nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GATv2Conv(num_features, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        # x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        return x


class MLP(nn.Module):
    def __init__(self, layer_dims):
        super(MLP, self).__init__()

        layers = []
        for in_, out_ in zip(layer_dims[:-1], layer_dims[1:]):
            layers.append(nn.Linear(in_, out_))
            layers.append(nn.ReLU())

        layers.pop()
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)
        print(self.net)

    def forward(self, x):
        return self.net(x)


class Net(nn.Module):
    def __init__(self, num_features, hidden_channels, mlp_layer_dims):
        super(Net, self).__init__()
        self.gcn = GCN(num_features, hidden_channels)
        self.mlp = MLP(mlp_layer_dims)

    def forward(self, data):
        x, edge_index, pairs = data.x, data.edge_index, data.batch_pair
        logger.debug("Applying GCN")
        x = self.gcn(x, edge_index)

        # Concatenate the embeddings of the two nodes in each pair
        logger.debug("Concatenating embeddings")
        pair_embeddings = torch.cat([x[pairs[:, 0]], x[pairs[:, 1]]], dim=-1)
        logger.debug(pair_embeddings.shape)

        logger.debug("Applying MLP")
        return self.mlp(pair_embeddings)


def sample_batch(y, batch_size):
    # sample (batch_size / 2) positive and (batch_size / 2) negative samples
    pos_indices = torch.where(y == 1)[0]
    neg_indices = torch.where(y == 0)[0]

    i_pos = torch.randint(0, len(pos_indices), (batch_size // 2,))
    i_neg = torch.randint(0, len(neg_indices), (batch_size // 2,))
    batch_indices = torch.cat([pos_indices[i_pos], neg_indices[i_neg]])

    # shuffle batch
    batch_indices = batch_indices[torch.randperm(len(batch_indices))]
    return batch_indices


def train(model, train_data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()

    batch_indices = sample_batch(train_data.y, batch_size=10_000)
    train_data.batch_pair = train_data.pair[batch_indices]
    batch_y = train_data.y[batch_indices]

    out = model(train_data)
    loss = criterion(out.view(-1), batch_y)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(model, test_data):
    model.eval()

    test_data.batch_pair = test_data.pair
    with torch.no_grad():
        out = model(test_data)

    auc = roc_auc_score(test_data.y.cpu(), out.view(-1).cpu())
    tn, fp, fn, tp = confusion_matrix(
        test_data.y.cpu(), out.view(-1).cpu() > 0.5
    ).ravel()

    return auc, (tn, fp, fn, tp)


def main(
    log_file="logs/gnn_plus.log",
    num_epochs=300,
    lr=0.01,
    log_interval=1,
):
    global logger
    logger = setup_logger(log_file, level=logging.INFO, log_to_stdout=True)

    logger.info("Loading data")
    data_dict = load_pkl("data/model/data.M.pkl")
    logger.info("Loading graph")
    graph = Graph("data/graph/edges_medium.pkl")

    logger.info("Creating PyG dataset 'train'")
    pyg_graph_train = create_pyg_dataset(data_dict, graph, "train")
    logger.info("Creating PyG dataset 'test'")
    pyg_graph_test = create_pyg_dataset(data_dict, graph, "test")

    model = Net(5, hidden_channels=10, mlp_layer_dims=[20, 10, 5, 1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    train_data = pyg_graph_train
    test_data = pyg_graph_test

    logger.info("Training")
    for epoch in range(num_epochs):
        logger.debug(f"Epoch {epoch}")
        loss = train(model, train_data, optimizer, criterion)
        if (epoch + 1) % log_interval == 0:
            auc, (tn, fp, fn, tp) = test(model, test_data)
            logger.info(
                f"Epoch: {epoch}, Loss: {loss:.4f}, AUC: {auc:.4f}, TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}"
            )


if __name__ == "__main__":
    fire.Fire(main)
