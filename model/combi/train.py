from torch import nn
import torch
import numpy as np
import pickle
import fire
import sys, os
import gzip
import logging
from collections import namedtuple

Data = namedtuple(
    "Data", ["pairs", "feature_embeddings", "concept_embeddings", "labels"]
)

parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_directory)

from metrics import test

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_logger(level=logging.INFO, log_to_stdout=True):
    logger = logging.getLogger()
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"
    )

    if log_to_stdout:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)

    file_handler = logging.FileHandler("logs.log")
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
    def __init__(self, layer_dims: list):
        """
        Fully Connected layers
        """
        super(BaselineNetwork, self).__init__()

        layers = []
        for in_, out_ in zip(layer_dims[:-1], layer_dims[1:]):
            layers.append(nn.Linear(in_, out_))
            # TODO: nn.Dropout
            layers.append(nn.ReLU())

        layers.pop()
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


class Trainer:
    def __init__(
        self,
        model,
        train_data,
        eval_data,
        optimizer,
        criterion,
        batch_size,
        pos_ratio,
        log_interval,
    ):
        self.model = model
        self.train_data = train_data
        self.eval_data = eval_data
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.pos_ratio = pos_ratio
        self.log_interval = log_interval

    def train(self, num_epochs):
        logger.info("Training model")

        for epoch in range(1, num_epochs + 1):
            loss = self._train_epoch()
            if epoch % self.log_interval == 0:
                auc, (tn, fp, fn, tp) = eval(self.model, self.eval_data)
                logger.info(
                    f"Epoch: {epoch}, Loss: {loss:.4f}, AUC: {auc:.4f}, TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}"
                )
            else:
                logger.info(f"Epoch {epoch}, Loss: {loss:.4f}")

    def _train_epoch(self):
        data = self.train_data

        self.model.train()
        self.optimizer.zero_grad()

        # get batch
        batch_indices = sample_batch(data.labels, self.batch_size, self.pos_ratio)

        inputs = get_embeddings(
            data.pairs[batch_indices],
            feature_embeddings=data.feature_embeddings,
            concept_embeddings=data.concept_embeddings,
        ).to(device)
        labels = data.labels[batch_indices].to(device)

        # Forward pass
        outputs = self.model(inputs)
        loss = self.criterion(outputs.view(-1), labels)

        # Backward and optimize
        loss.backward()
        self.optimizer.step()

        return loss.item()


def sample_batch(y, batch_size, pos_ratio=0.5):
    pos_indices = torch.where(y == 1)[0]
    neg_indices = torch.where(y == 0)[0]

    amt_pos = int(batch_size * pos_ratio)
    amt_neg = batch_size - amt_pos

    i_pos = torch.randint(0, len(pos_indices), (amt_pos,))
    i_neg = torch.randint(0, len(neg_indices), (amt_neg,))

    batch_indices = torch.cat([pos_indices[i_pos], neg_indices[i_neg]])

    # shuffle batch
    batch_indices = batch_indices[torch.randperm(len(batch_indices))]
    return batch_indices


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
    data_path="data/model/data.pkl",
    emb_f_train_path="data/model/combi/features_2016.M.pkl.gz",
    emb_f_test_path="data/model/combi/features_2019.M.pkl.gz",
    emb_c_train_path="data/model/concept_embs/av_embs_2016.M.pkl.gz",
    emb_c_test_path="data/model/concept_embs/av_embs_2019.M.pkl.gz",
    lr=0.001,
    batch_size=100,
    num_epochs=1000,
    pos_ratio=0.3,
    layers=[1556, 1024, 512, 256, 64, 32, 16, 8, 4, 1],
    log_interval=10,
):
    global logger
    logger = setup_logger(level=logging.INFO, log_to_stdout=True)

    data = load_data(data_path)

    d_train = Data(
        pairs=torch.tensor(data["X_train"]),
        feature_embeddings=load_compressed(emb_f_train_path),
        concept_embeddings=load_compressed(emb_c_train_path),
        labels=torch.tensor(data["y_train"], dtype=torch.float),
    )

    d_test = Data(
        pairs=torch.tensor(data["X_test"]),
        feature_embeddings=load_compressed(emb_f_test_path),
        concept_embeddings=load_compressed(emb_c_test_path),
        labels=torch.tensor(data["y_test"], dtype=torch.float),
    )

    model = BaselineNetwork(layers).to(device)

    trainer = Trainer(
        model=model,
        train_data=d_train,
        eval_data=d_test,
        optimizer=torch.optim.Adam(model.parameters(), lr=lr),
        criterion=nn.BCELoss(),
        batch_size=batch_size,
        pos_ratio=pos_ratio,
        log_interval=log_interval,
    )
    trainer.train(num_epochs)


if __name__ == "__main__":
    fire.Fire(main)
