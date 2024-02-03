import logging
from collections import namedtuple

import fire
import numpy as np
import pandas as pd
import torch
from torch import nn

from materials_concepts.model.metrics import test
from materials_concepts.utils.utils import (
    load_compressed,
    load_pickle,
    save_compressed,
    setup_logger,
    flatten,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger = setup_logger(
    logging.getLogger(__name__), file="logs/eval.log", level=logging.DEBUG
)

Data = namedtuple(
    "Data", ["pairs", "feature_embeddings", "concept_embeddings", "labels"]
)


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
    csv_path="data/model/combi/threshold_tuning.csv",
    pred_path="data/model/combi/predictions.pkl.gz",
):
    data = load_pickle(data_path)

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

    stats = []
    for threshold in np.arange(0.5, 1, 0.05):
        auc, (precision, recall, fscore, _), (tn, fp, fn, tp) = test(
            d_test.labels, predictions, threshold=threshold
        )

        metrics = dict(
            threshold=threshold,
            auc=auc,
            precision=precision,
            recall=recall,
            fscore=fscore,
            tn=tn,
            fp=fp,
            fn=fn,
            tp=tp,
        )
        stats.append(metrics)

    df = pd.DataFrame(stats)
    df = df.round(4)
    df.to_csv(csv_path, index=False)

    save_compressed(
        predictions,
        pred_path,
    )


if __name__ == "__main__":
    fire.Fire(main)
