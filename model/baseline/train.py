from torch import nn
import torch
import numpy as np
import pickle
import fire
import sys, os

parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_directory)

from metrics import print_metrics


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def flatten(t):
    return [item for sublist in t for item in sublist]


def load_data(data_path, embeddings_path):
    print("Loading data...")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    with open(embeddings_path, "rb") as f:
        embeddings = pickle.load(f)

    return data, embeddings


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
            layers.append(nn.ReLU())

        layers.pop()
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)
        print(self.net)

    def forward(self, x):
        """
        Pass throught network
        """
        res = self.net(x)

        return res


def train(model, X_train, y_train, learning_rate, batch_size, num_epochs):
    print(f"Training model... ({len(X_train):,})")
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

            # Print or log information
            if (i + 1) % 1000 == 0:
                batch_loss = running_loss / 10
                batch_accuracy = correct / total
                print(
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
        print(f"Epoch [{epoch+1}/{num_epochs}], Accuracy: {accuracy:.4f}")


def eval(model, data, embeddings, metrics_path):
    """Load the pytorch model and evaluate it on the test set"""
    print("Evaluating...")
    X_test = torch.tensor(np.array(embeddings["X_test"]), dtype=torch.float).to(device)
    predictions = np.array(flatten(model(X_test).detach().cpu().numpy()))

    print_metrics(data["y_test"], predictions, threshold=0.5, save_path=metrics_path)


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
):
    data, embeddings = load_data(data_path, embeddings_path)

    model = BaselineNetwork([15, 100, 100, 10]).to(device)

    if train_model:
        print("Training...")
        model.train()
        train(
            model,
            X_train=np.array(embeddings["X_train"]),
            y_train=data["y_train"],
            learning_rate=lr,
            batch_size=batch_size,
            num_epochs=num_epochs,
        )
        print("Saving model...")
        if save_model:
            torch.save(model.state_dict(), save_model)
    elif eval_mode:
        print("Loading model...")
        model.load_state_dict(torch.load(eval_mode))
    else:
        print("Please specify either --train_model or --eval_model")
        return

    eval(model, data, embeddings, metrics_path)


if __name__ == "__main__":
    fire.Fire(main)

# AUC 0.8659
# Precision 0.0075
# Recall 0.2190
# F1 0.0145
# Confusion matrix:
# TN: 996852, FP: 3043, FN: 82, TP: 23
