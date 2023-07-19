from torch import nn
import torch
import numpy as np
import pickle
import fire
import sys, os
import gzip
import logging

parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_directory)

from metrics import print_metrics

TENSOR_DIM = 768

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

        layer_dims.append(1)

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


def get_embeddings(X, X_embs):
    logger.debug(f"Getting embeddings for {len(X)} samples")

    l = []
    for v1, v2 in X:
        i1 = int(v1.item())
        i2 = int(v2.item())
        l.append(
            np.concatenate(
                [
                    np.array(X_embs.get(i1, torch.zeros(TENSOR_DIM))),
                    np.array(X_embs.get(i2, torch.zeros(TENSOR_DIM))),
                ]
            )
        )
    return torch.tensor(np.array(l))


def train(
    model,
    X_train,
    X_train_embs,
    y_train,
    learning_rate,
    batch_size,
    num_epochs,
):
    logger.info(f"Training model... ({len(X_train):,})")
    X_train = torch.Tensor(np.array(X_train))
    y_train = torch.Tensor(np.array(y_train))

    # Create a PyTorch dataset
    dataset = torch.utils.data.TensorDataset(X_train, y_train)

    # Create a data loader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    criterion = nn.BCELoss()

    # Define your optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    accuracy_values = []

    for epoch in range(num_epochs):
        running_loss = 0.0

        total = 0
        correct = 0

        model.train()
        for i, (inputs, labels) in enumerate(data_loader):
            # Zero the gradients
            optimizer.zero_grad()

            inputs = get_embeddings(inputs, X_train_embs)

            # Forward pass
            outputs = model(inputs)  # load embeddings here
            loss = criterion(outputs, labels.unsqueeze(1))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track the loss
            running_loss += loss.item()

            # Track accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Print or log information
            if (i + 1) % 1000 == 0:
                batch_loss = running_loss / 10
                batch_accuracy = correct / total
                logger.info(
                    f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(data_loader)}], Loss: {batch_loss:.4f}, Accuracy: {batch_accuracy:.4f}"
                )
                running_loss = 0.0
                correct = 0
                total = 0

        correct = 0
        total = 0

        model.eval()
        with torch.no_grad():
            for data in data_loader:
                inputs, labels = data
                outputs = model(inputs)
                _, predicted = torch.max(
                    outputs.data, 1
                )  # Get the index of the maximum logit value
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        accuracy_values.append(accuracy)

        # Print the average loss for the epoch
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Accuracy: {accuracy:.4f}")


def eval(model, data, embeddings, metrics_path):
    """Load the pytorch model and evaluate it on the test set"""
    logger.info("Evaluating")
    test_inputs = get_embeddings(data["X_test"], embeddings["X_test"])
    X_test = torch.tensor(test_inputs, dtype=torch.float).to(device)
    predictions = np.array(flatten(model(X_test).detach().cpu().numpy()))

    print_metrics(data["y_test"], predictions, threshold=0.5, save_path=metrics_path)


def shuffle(X, y):
    """Shuffle X and y in unison"""
    assert len(X) == len(y)
    p = np.random.permutation(len(X))
    return X[p], y[p]


def sample(X: np.ndarray, y: np.ndarray, pos_to_neg_ratio: float):
    """Sample the data to have a given ratio of positive to negative samples"""
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]

    curr_pos_ratio = len(pos_indices) / len(y)
    if curr_pos_ratio > pos_to_neg_ratio:
        num_neg = len(neg_indices)
        num_pos = int(pos_to_neg_ratio * num_neg)
    else:
        num_pos = len(pos_indices)
        num_neg = int(num_pos / pos_to_neg_ratio)

    pos_indices = np.random.choice(pos_indices, size=num_pos, replace=False)
    neg_indices = np.random.choice(neg_indices, size=num_neg, replace=False)

    X = np.concatenate([X[pos_indices], X[neg_indices]])
    y = np.concatenate([y[pos_indices], y[neg_indices]])

    return shuffle(X, y)


def random_sample(X, y, n=1000):
    """Sample n random samples from X and y"""
    indices = np.random.choice(len(X), size=n, replace=False)
    return X[indices], y[indices]


def main(
    data_path="data/model/data.pkl",
    emb_train_path="data/model/concept_embs/av_embs_2016.pkl.gz",
    emb_test_path="data/model/concept_embs/av_embs_2019.pkl.gz",
    lr=0.001,
    batch_size=100,
    num_epochs=1,
    train_model=False,
    save_model=False,
    eval_mode=False,
    metrics_path=None,
    pos_to_neg_ratio=0.03,
    input_dim=1536,
):
    global logger
    logger = setup_logger(level=logging.INFO, log_to_stdout=True)

    data = load_data(data_path)

    embeddings = {
        "X_train": load_compressed(emb_train_path),
        "X_test": load_compressed(emb_test_path),
    }

    X_train, y_train = random_sample(
        *sample(data["X_train"], data["y_train"], pos_to_neg_ratio), n=10000
    )

    model = BaselineNetwork([input_dim, 1024, 512, 256, 64, 16]).to(device)

    if train_model:
        model.train()
        train(
            model,
            X_train=X_train,
            X_train_embs=embeddings["X_train"],
            y_train=y_train,
            learning_rate=lr,
            batch_size=batch_size,
            num_epochs=num_epochs,
        )
        logger.info("Saving model")
        if save_model:
            torch.save(model.state_dict(), save_model)
    elif eval_mode:
        logger.info("Loading model")
        model.load_state_dict(torch.load(eval_mode))
    else:
        logger.info("Please specify either --train_model or --eval_model")
        return

    eval(model, data, embeddings, metrics_path)


if __name__ == "__main__":
    fire.Fire(main)
