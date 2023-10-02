from torch import nn
import torch
import numpy as np
import pandas as pd
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

from metrics import test

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


def save_compressed(obj, path):
    logger.info(f"Saving compressed file {path}")
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f)


def main(
    data_path="data/model/val.data.M.pkl",
    pred_path="data/model/mixture/predictions.pkl.gz",
    csv_path="data/model/mixture/eval.csv",
):
    reload(logging)
    global logger
    logger = setup_logger(file="logs/logs.log", level=logging.INFO, log_to_stdout=True)
    data = load_data(data_path)

    labels = data["y_test"]
    predictions = load_compressed(pred_path)

    stats = []
    for threshold in [
        0.5,
        0.9,
        0.95,
        0.98,
        0.985,
        0.989,
        0.99,
        0.991,
        0.992,
        0.993,
        0.994,
        0.995,
    ]:
        auc, (precision, recall, fscore, _), (tn, fp, fn, tp) = test(
            labels, predictions, threshold=threshold
        )

        metrics = dict(
            threshold=threshold,
            auc=auc,
            precision=precision,
            recall=recall,
            tn=tn,
            fp=fp,
            fn=fn,
            tp=tp,
        )
        stats.append(metrics)

    df = pd.DataFrame(stats)
    df = df.round(4)

    print(df)
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    fire.Fire(main)
