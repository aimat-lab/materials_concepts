import logging

import fire
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

from materials_concepts.model.metrics import print_metrics
from materials_concepts.utils.utils import flatten, load_pickle, setup_logger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger = setup_logger(
    logging.getLogger(__name__), file="logs/baseline.log", level=logging.DEBUG
)


def load_data(data_path, embeddings_path):
    logger.info("Loading data...")
    data = load_pickle(data_path)

    embeddings = load_pickle(embeddings_path)

    return data, embeddings


class BaselineNetwork(nn.Module):
    def __init__(self, layer_dims: list):
        """
        Fully Connected layers
        """
        super().__init__()

        layer_dims.append(1)

        layers = []
        for in_, out_ in zip(layer_dims[:-1], layer_dims[1:], strict=False):
            layers.append(nn.Linear(in_, out_))
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


def train(
    model: BaselineNetwork, X_train, y_train, learning_rate, batch_size, num_epochs
):
    logger.info(f"Training model... ({len(X_train):,})")
    X_train = torch.Tensor(np.array(X_train))
    y_train = torch.Tensor(np.array(y_train))

    # Create a PyTorch dataset
    dataset = TensorDataset(X_train, y_train)

    # Create a data loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

            # Forward pass
            outputs = model(inputs)
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

            # logger.info or log information
            if (i + 1) % 1000 == 0:
                batch_loss = running_loss / 10
                batch_accuracy = correct / total
                logger.debug(
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
    logger.info("Evaluating...")
    X_test = torch.tensor(np.array(embeddings["X_test"]), dtype=torch.float).to(device)
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


def main(
    data_path="model/data.pkl",
    embeddings_path="model/baseline/embeddings.pkl",
    lr=0.001,
    batch_size=100,
    num_epochs=1,
    train_model=False,
    save_model=False,
    eval_mode=False,
    metrics_path=None,
    pos_to_neg_ratio=0.03,
    input_dim=15,
):
    data, embeddings = load_data(data_path, embeddings_path)

    X_train, y_train = sample(embeddings["X_train"], data["y_train"], pos_to_neg_ratio)

    model = BaselineNetwork([input_dim, 100, 100, 10]).to(device)

    if train_model:
        logger.info("Training...")
        model.train()
        train(
            model,
            X_train=X_train,
            y_train=y_train,
            learning_rate=lr,
            batch_size=batch_size,
            num_epochs=num_epochs,
        )
        logger.info("Saving model...")
        if save_model:
            torch.save(model.state_dict(), save_model)
    elif eval_mode:
        logger.info("Loading model...")
        model.load_state_dict(torch.load(eval_mode))
    else:
        logger.info("Please specify either --train_model or --eval_model")
        return

    eval(model, data, embeddings, metrics_path)


if __name__ == "__main__":
    fire.Fire(main)
