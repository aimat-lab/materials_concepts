import gzip
import logging
import pickle
import sys
from collections import namedtuple
from importlib import reload

import fire
import numpy as np
import torch
from torch import nn

from materials_concepts.model.metrics import test

Data = namedtuple(
    "Data", ["pairs", "feature_embeddings", "concept_embeddings", "labels"]
)


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


def dot_product(vec1, vec2):
    """Compute the dot product between two vectors."""
    return np.dot(vec1, vec2)


def angle_between(vec1, vec2):
    """Compute the angle (in radians) between two vectors."""
    cosine_sim = cosine_similarity(vec1, vec2)
    return np.arccos(np.clip(cosine_sim, -1.0, 1.0))


def concat_embs(emb1, emb2):
    return np.concatenate([emb1, emb2])


def handcrafted_features(emb1, emb2):
    a = euclidean_distance(emb1, emb2)
    b = cosine_similarity(emb1, emb2)

    return np.array([a, b])


# other approach: PCA on embeddings => 4D embedding


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


class EarlyStopping:
    def __init__(self, sliding_window=5, eta=1e-4):
        self.sliding_window = sliding_window
        self.eta = eta
        self.losses = []
        self.aucs = []

    def append(self, loss, auc):
        self.losses.append(loss)
        self.aucs.append(auc)

    @property
    def current_auc_sliding(self):
        return np.mean(self.aucs[-self.sliding_window :])

    @property
    def previous_auc_sliding(self):
        return np.mean(self.aucs[-self.sliding_window * 2 : -self.sliding_window])

    @property
    def current_loss_sliding(self):
        return np.mean(self.losses[-self.sliding_window :])

    @property
    def previous_loss_sliding(self):
        return np.mean(self.losses[-self.sliding_window * 2 : -self.sliding_window])

    def should_stop_early(self):
        if not self.sliding_window:
            return False

        if len(self.aucs) < self.sliding_window * 2:
            return False

        # current averaged auc is higher than previous averaged auc
        if (self.current_auc_sliding - self.previous_auc_sliding) > self.eta:
            return False

        # previous averaged loss is higher than current averaged loss
        if (self.previous_loss_sliding - self.current_loss_sliding) > self.eta:
            return False

        return True


class Loader:
    def __init__(self, y):
        self.indices = torch.randperm(len(y))
        self.count = 0

    def __call__(self, batch_size):
        if self.count > len(self.indices) - batch_size:
            self.indices = torch.randperm(len(self.indices))
            self.count = 0

        batch = self.indices[self.count : self.count + batch_size]
        self.count += 1
        return batch


class Trainer:
    def __init__(
        self,
        model,
        train_data,
        eval_data,
        optimizer,
        scheduler,
        criterion,
        batch_size,
        pos_ratio,
        early_stopping,
        log_interval,
        emb_strategy,
        use_loader=False,
    ):
        self.model = model
        self.train_data = train_data
        self.eval_data = eval_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.batch_size = batch_size
        self.pos_ratio = pos_ratio
        self.log_interval = log_interval
        self.early_stopping = early_stopping
        self.use_loader = use_loader
        self.data_loader = Loader(train_data.labels)
        self.emb_strategy = emb_strategy

    def train(self, num_epochs):
        logger.info("Training model")

        for epoch in range(1, num_epochs + 1):
            loss = self._train_epoch()

            if epoch % self.log_interval == 0:
                auc, (tn, fp, fn, tp) = eval(
                    self.model, self.eval_data, feature_func=self.emb_strategy
                )

                self.early_stopping.append(loss=loss, auc=auc)

                logger.info(
                    f"Epoch: {epoch}, Loss: {loss:.4f}, AUC: {auc:.4f}, TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}"
                )

                if self.early_stopping.should_stop_early():
                    logger.info("Early stopping triggered")
                    break

    def _train_epoch(self):
        data = self.train_data

        self.model.train()
        self.optimizer.zero_grad()

        # get batch
        if self.use_loader:
            batch_indices = self.data_loader(self.batch_size)
        else:
            batch_indices = sample_batch(data.labels, self.batch_size, self.pos_ratio)

        inputs = get_embeddings(
            data.pairs[batch_indices],
            feature_embeddings=data.feature_embeddings,
            concept_embeddings=data.concept_embeddings,
            feature_func=self.emb_strategy,
        ).to(device)
        labels = data.labels[batch_indices].to(device)

        # Forward pass
        outputs = self.model(inputs)
        loss = self.criterion(outputs.view(-1), labels)

        # Backward and optimize
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

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


def eval(model, data: Data, feature_func):
    """Load the pytorch model and evaluate it on the test set"""
    model.eval()

    inputs = get_embeddings(
        data.pairs, data.feature_embeddings, data.concept_embeddings, feature_func
    ).to(device)

    predictions = np.array(flatten(model(inputs).detach().cpu().numpy()))

    auc, _, confusion_matrix = test(data.labels, predictions, threshold=0.5)
    return auc, confusion_matrix


emb_strategies = {
    "concat": concat_embs,
    "handcrafted": handcrafted_features,
}


def main(
    data_path="data/model/data.pkl",
    emb_f_train_path=None,
    emb_f_test_path=None,
    emb_c_train_path=None,
    emb_c_test_path=None,
    emb_comb_strategy="concat",
    lr=0.001,
    gamma=0.8,
    batch_size=100,
    num_epochs=1000,
    pos_ratio=0.3,
    dropout=0.1,
    layers=[1556, 1024, 512, 256, 64, 32, 16, 8, 4, 1],
    step_size=40,
    log_interval=10,
    log_file="logs.log",
    save_model=False,
    sliding_window=5,
    use_loader=False,
):
    reload(logging)
    global logger
    logger = setup_logger(file=log_file, level=logging.INFO, log_to_stdout=True)

    logger.info("Running with parameters:")
    logger.info(f"lr: {lr}")
    logger.info(f"batch_size: {batch_size}")
    logger.info(f"num_epochs: {num_epochs}")
    logger.info(f"pos_ratio: {pos_ratio}")
    logger.info(f"dropout: {dropout}")
    logger.info(f"layers: {layers}")
    logger.info(f"step_size: {step_size}")
    logger.info(f"gamma: {gamma}")
    logger.info(f"log_interval: {log_interval}")
    logger.info(f"sliding_window: {sliding_window}")
    logger.info(f"use_loader: {use_loader}")
    logger.info(f"emb_comb_strategy: {emb_comb_strategy}")

    data = load_data(data_path)

    features_train = load_compressed(emb_f_train_path)
    d_train = Data(
        pairs=torch.tensor(data["X_train"]),
        feature_embeddings=features_train["v_features"] if features_train else None,
        concept_embeddings=load_compressed(emb_c_train_path),
        labels=torch.tensor(data["y_train"], dtype=torch.float),
    )

    features_test = load_compressed(emb_f_test_path)
    d_test = Data(
        pairs=torch.tensor(data["X_test"]),
        feature_embeddings=features_test["v_features"] if features_test else None,
        concept_embeddings=load_compressed(emb_c_test_path),
        labels=torch.tensor(data["y_test"], dtype=torch.float),
    )

    model = BaselineNetwork(layers, dropout).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma
    )

    trainer = Trainer(
        model=model,
        train_data=d_train,
        eval_data=d_test,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=nn.BCELoss(),
        batch_size=batch_size,
        pos_ratio=pos_ratio,
        early_stopping=EarlyStopping(sliding_window=sliding_window),
        log_interval=log_interval,
        use_loader=use_loader,
        emb_strategy=emb_strategies[emb_comb_strategy],
    )
    trainer.train(num_epochs)

    if save_model:
        torch.save(model.state_dict(), save_model)


if __name__ == "__main__":
    fire.Fire(main)
