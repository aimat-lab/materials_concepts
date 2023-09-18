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


def save_compressed(obj, path):
    logger.info(f"Saving compressed file {path}")
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f)


def load_compressed(path):
    if not path:
        return None

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


def euclidean_distance(vec1, vec2):
    """Compute the Euclidean distance between two vectors."""
    return np.linalg.norm(vec1 - vec2)


def cosine_similarity(vec1, vec2):
    """Compute the cosine similarity between two vectors."""
    dot_prod = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_prod / (norm1 * norm2)


def concat_embs(emb1, emb2):
    return np.concatenate([emb1, emb2])


def handcrafted_features(emb1, emb2):
    a = euclidean_distance(emb1, emb2)
    b = cosine_similarity(emb1, emb2)

    return np.array([a, b])


def get_embeddings(
    pairs, feature_embeddings, concept_embeddings, feature_func=concat_embs
):
    logger.debug(f"Getting embeddings for {len(pairs)} samples")

    l = []
    for v1, v2 in pairs:
        i1 = int(v1.item())
        i2 = int(v2.item())

        feature_vector = []

        if feature_embeddings is not None:
            emb1_f = np.array(feature_embeddings[i1])
            emb2_f = np.array(feature_embeddings[i2])

            feature_vector.extend([emb1_f, emb2_f])

        if concept_embeddings is not None:
            emb1_c = np.array(concept_embeddings[i1])
            emb2_c = np.array(concept_embeddings[i2])

            feature_vector.append(feature_func(emb1_c, emb2_c))

        l.append(np.concatenate(feature_vector))
    return torch.tensor(np.array(l)).float()


def eval(model, data: Data, feature_func):
    """Load the pytorch model and evaluate it on the test set"""
    model.eval()

    inputs = get_embeddings(
        data.pairs, data.feature_embeddings, data.concept_embeddings, feature_func
    ).to(device)

    predictions = np.array(flatten(model(inputs).detach().cpu().numpy()))

    auc, _, confusion_matrix = test(data.labels, predictions, threshold=0.5)
    return auc, confusion_matrix


def predict(model, data: Data, feature_func, mode):
    model.eval()

    feature_embs = data.feature_embeddings if mode != "pure_embs" else None
    concept_embs = data.concept_embeddings if mode != "baseline" else None

    print(
        f"Predicting with {mode} mode, feature_embs: {feature_embs is not None} and concept_embs: {concept_embs is not None}"
    )

    inputs = get_embeddings(data.pairs, feature_embs, concept_embs, feature_func).to(
        device
    )

    return np.array(flatten(model(inputs).detach().cpu().numpy()))


emb_strategies = {
    "concat": concat_embs,
    "handcrafted": handcrafted_features,
}

architectures_map = {
    "baseline": [20, 300, 180, 108, 64, 10, 1],
    "combi": [1556, 1556, 933, 559, 335, 10, 1],
    "pure_embs": [1536, 1024, 819, 10, 1],
}


def main(
    data_path="data/model/data.pkl",
    emb_f_test_path=None,
    emb_c_test_path=None,
    emb_comb_strategy="concat",
    model_path="data/model/combi/model.pt",
    architecture="combi",
    predict_path="data/model/combi/predictions.pkl.gz",
    log_file="logs.log",
):
    reload(logging)
    global logger
    logger = setup_logger(file=log_file, level=logging.INFO, log_to_stdout=True)

    logger.info("Running with parameters:")
    logger.info(f"emb_comb_strategy: {emb_comb_strategy}")

    data = load_data(data_path)

    features_test = load_compressed(emb_f_test_path)
    d_test = Data(
        pairs=torch.tensor(data["X_test"]),
        feature_embeddings=features_test["v_features"] if features_test else None,
        concept_embeddings=load_compressed(emb_c_test_path),
        labels=torch.tensor(data["y_test"], dtype=torch.float),
    )

    model = BaselineNetwork(architectures_map[architecture], dropout=0).to(
        device
    )  # in eval mode, dropout is not used

    model.load_state_dict(torch.load(model_path, map_location=device))

    predictions = predict(
        model, d_test, emb_strategies[emb_comb_strategy], architecture
    )

    print_metrics(d_test.labels, predictions, threshold=0.5)

    save_compressed(predictions, predict_path)


if __name__ == "__main__":
    fire.Fire(main)
