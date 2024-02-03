import gzip
import logging
import pickle
from collections import namedtuple

import fire
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn

from materials_concepts.model.metrics import print_metrics, test
from materials_concepts.utils.utils import (
    flatten,
    load_pickle,
    save_compressed,
    setup_logger,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger = setup_logger(
    logging.getLogger(__name__), file="logs/eval.log", level=logging.DEBUG
)

Data = namedtuple(
    "Data", ["pairs", "feature_embeddings", "concept_embeddings", "labels"]
)


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
        for in_, out_ in zip(layer_dims[:-1], layer_dims[1:], strict=False):
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
    auc_curve_path="data/model/combi/auc_curve.pkl.gz",
):
    logger.info("Running with parameters:")
    logger.info(f"emb_comb_strategy: {emb_comb_strategy}")

    data = load_pickle(data_path)

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

    auc = roc_auc_score(d_test.labels, predictions)
    fpr, tpr, _ = roc_curve(d_test.labels, predictions)

    save_compressed(predictions, predict_path)
    save_compressed({"fpr": fpr, "tpr": tpr, "auc": auc}, auc_curve_path)


if __name__ == "__main__":
    fire.Fire(main)
