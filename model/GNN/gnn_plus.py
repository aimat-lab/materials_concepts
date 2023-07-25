import torch
from torch_geometric.data import Data
import pickle, gzip
import os, sys

parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_directory)

from graph import Graph

NODE_DIM = 768


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

    x = load_node_features(cut_off_year, len(graph.vertices))

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pair=vertex_pairs,
        y=labels,
    )

    return data


data_dict = load_pkl("data/model/data.M.pkl")
graph = Graph("data/graph/edges_medium.pkl")


pyg_graph_train = create_pyg_dataset(data_dict, graph, "train")

# vertices (2016): 141748
# vertices (2019): 146764
# vertices (full): 148032
