from torch import nn
import torch
import numpy as np
import pickle
import fire
import sys, os
import gzip
import logging
from collections import namedtuple
from importlib import reload


Data = namedtuple(
    "Data", ["pairs", "feature_embeddings", "concept_embeddings", "labels"]
)

parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_directory)

from metrics import test, print_metrics

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def flatten(t):
    return [item for sublist in t for item in sublist]


def load_data(data_path):
    logger.info("Loading data")
    with open(data_path, "rb") as f:
        return pickle.load(f)


def load_compressed(path):
    logger.info(f"Loading compressed file {path}")
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


class BaselineNetwork(nn.Module):
    def __init__(self, layer_dims: list, dropout: float):
        """
        Fully Connected layers
        """
        super(BaselineNetwork, self).__init__()

        layers = []
        for in_, out_ in zip(layer_dims[:-1], layer_dims[1:]):
            layers.append(nn.Linear(in_, out_))
            layers.append(nn.BatchNorm1d(out_))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))

        layers.pop()  # remove last dropout layer
        layers.pop()  # remove last relu layer
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)
        logger.debug(self.net)

    def forward(self, x):
        """
        Pass throught network
        """
        res = self.net(x)

        return res


def get_embeddings(pairs, feature_embeddings, concept_embeddings):
    logger.debug(f"Getting embeddings for {len(pairs)} samples")

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


def eval(model, data: Data):
    """Load the pytorch model and evaluate it on the test set"""
    model.eval()

    inputs = get_embeddings(
        data.pairs, data.feature_embeddings, data.concept_embeddings
    ).to(device)

    predictions = np.array(flatten(model(inputs).detach().cpu().numpy()))

    auc, _, confusion_matrix = test(data.labels, predictions, threshold=0.5)
    return auc, confusion_matrix


def main(
    data_path="data/model/data.M.pkl",
    emb_f_test_path="data/model/combi/features_2019.M.pkl.gz",
    emb_c_test_path="data/model/concept_embs/av_embs_2019.M.pkl.gz",
    layers=[1556, 1556, 933, 10, 1],
    dropout=0.25,
    model_path="data/model/combi/pos_rate-dropout-tuned.pt",
):
    reload(logging)
    global logger
    logger = setup_logger(file="logs/logs.log", level=logging.INFO, log_to_stdout=True)
    data = load_data(data_path)

    d_test = Data(
        pairs=torch.tensor(data["X_test"]),
        feature_embeddings=load_compressed(emb_f_test_path),
        concept_embeddings=load_compressed(emb_c_test_path),
        labels=torch.tensor(data["y_test"], dtype=torch.float),
    )

    model = BaselineNetwork(layers, dropout).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    inputs = get_embeddings(
        d_test.pairs, d_test.feature_embeddings, d_test.concept_embeddings
    ).to(device)

    predictions = np.array(flatten(model(inputs).detach().cpu().numpy()))

    for threshold in np.arange(0.5, 1, 0.05):
        print_metrics(d_test.labels, predictions, threshold=threshold)


if __name__ == "__main__":
    fire.Fire(main)
