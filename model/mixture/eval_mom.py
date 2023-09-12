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


def concat_embs(emb1, emb2):
    return np.concatenate([emb1, emb2])


# other approach: PCA on embeddings => 4D embedding


def get_embeddings(
    pairs, feature_embeddings, concept_embeddings, feature_func=concat_embs
):
    logger.debug(
        f"Getting embeddings for {len(pairs)} samples with {type(feature_embeddings)}, {type(concept_embeddings)}"
    )

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


def eval_predictions(y, predictions):
    auc, _, confusion_matrix = test(y, predictions, threshold=0.5)
    return auc, confusion_matrix


def predict(model, data: Data, feature_func, mode):
    model.eval()

    feature_embs = data.feature_embeddings if mode == "baseline" else None
    concept_embs = data.concept_embeddings if mode != "baseline" else None

    print(
        f"Predicting with {mode} mode, feature_embs: {feature_embs is not None} and concept_embs: {concept_embs is not None}"
    )

    inputs = get_embeddings(data.pairs, feature_embs, concept_embs, feature_func).to(
        device
    )

    return np.array(flatten(model(inputs).detach().cpu().numpy()))


def blend(predictions, blending):
    assert len(predictions) == len(blending)

    new_predictions = []
    for p, b in zip(predictions, blending):
        p *= b
        new_predictions.append(p)

    return np.sum(new_predictions, axis=0)


emb_strategies = {
    "concat": concat_embs,
}

architectures_map = {
    "baseline": [20, 300, 180, 108, 64, 10, 1],
    "combi": [1556, 1556, 933, 559, 335, 10, 1],
    "pure_embs": [1536, 1024, 819, 10, 1],
}


def load_model(path, architecture):
    model = BaselineNetwork(layer_dims=architectures_map[architecture], dropout=0).to(
        device
    )
    model.load_state_dict(torch.load(path))
    return model


def main(
    data_path="data/model/data.pkl",
    emb_f_test_path=None,
    emb_c_test_path=None,
    emb_comb_strategy="concat",
    model_path_1="data/model/baseline/gridsearch/74d59ea45398f5ea629137dda6b6ad70.pt",
    architecture1="baseline",
    model_path_2="data/model/baseline/gridsearch/aeddf108ae5a2e0b3fec8e1222ac0710.pt",
    architecture2="baseline",
    log_file="logs/log.log",
    sliding_window=5,
    use_loader=False,
):
    reload(logging)
    global logger
    logger = setup_logger(file=log_file, level=logging.DEBUG, log_to_stdout=True)

    logger.info("Running with parameters:")
    logger.info(f"sliding_window: {sliding_window}")
    logger.info(f"emb_comb_strategy: {emb_comb_strategy}")

    data = load_data(data_path)

    features_test = load_compressed(emb_f_test_path)
    d_test = Data(
        pairs=torch.tensor(data["X_test"]),
        feature_embeddings=features_test["v_features"] if features_test else None,
        concept_embeddings=load_compressed(emb_c_test_path),
        labels=torch.tensor(data["y_test"], dtype=torch.float),
    )

    models = [
        load_model(model_path_1, architecture1),
        load_model(model_path_2, architecture2),
    ]

    architectures = [architecture1, architecture2]

    predictions = np.array(
        [
            predict(
                model,
                d_test,
                feature_func=emb_strategies[emb_comb_strategy],
                mode=architecture,
            )
            for model, architecture in zip(models, architectures)
        ]
    )

    blending = [0.5, 0.5]
    blended_preds = blend(predictions, blending)

    auc, (tn, fp, fn, tp) = eval_predictions(d_test.labels, blended_preds)
    logger.info(
        "Evaluation on test set AUC: {:.4f} with blending: {}".format(auc, blending)
    )
    logger.info(
        "Evaluation on test set: TP: {}, FN: {}, FP: {}, TN: {}".format(tp, fn, fp, tn)
    )


if __name__ == "__main__":
    fire.Fire(main)
