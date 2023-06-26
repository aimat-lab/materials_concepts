from torch import nn
import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def flatten(t):
    return [item for sublist in t for item in sublist]


def load_data():
    print("Loading data...")
    with open("model/data.pkl", "rb") as f:
        data = pickle.load(f)

    print("Loading embeddings...")
    with open("model/baseline/embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)

    return data, embeddings


class BaselineNetwork(nn.Module):
    def __init__(self):
        """
        Fully Connected layers
        """
        super(BaselineNetwork, self).__init__()

        self.net = nn.Sequential(  # very small network for tests
            nn.Linear(15, 100),  # 15 properties
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid(),  # For Classification
        )

    def forward(self, x):
        """
        Pass throught network
        """
        res = self.net(x)

        return res


def train(
    model, X_train, y_train, X_test, y_test, learning_rate, batch_size, num_epochs
):
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


def eval(model, data, embeddings):
    """Load the pytorch model and evaluate it on the test set"""
    print("Evaluating...")
    X_test = torch.tensor(np.array(embeddings["X_test"]), dtype=torch.float).to(device)
    predictions = np.array(flatten(model(X_test).detach().cpu().numpy()))

    auc = roc_auc_score(data["y_test"], predictions)

    THRESHOLD = 0.5
    precision, recall, fscore, _ = precision_recall_fscore_support(
        data["y_test"], predictions > THRESHOLD, average="binary"
    )

    print("AUC", f"{auc:.4f}")
    print("Precision", f"{precision:.4f}")
    print("Recall", f"{recall:.4f}")
    print("F1", f"{fscore:.4f}")

    print("Confusion matrix:")
    tn, fp, fn, tp = confusion_matrix(data["y_test"], predictions > THRESHOLD).ravel()
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")


def main():
    data, embeddings = load_data()

    model = BaselineNetwork().to(device)

    print("Training...")
    model.train()
    train(
        model,
        X_train=embeddings["X_train"],
        y_train=data["y_train"],
        X_test=embeddings["X_test"],
        y_test=data["y_test"],
        learning_rate=0.001,
        batch_size=100,
        num_epochs=1,
    )

    eval(model, data, embeddings)

    torch.save(model.state_dict(), "model/baseline/model.pt")


def main_eval():
    data, embeddings = load_data()
    print("Loading model...")
    model = BaselineNetwork().to(device)
    model.load_state_dict(torch.load("model/baseline/model.pt"))

    eval(model, data, embeddings)


if __name__ == "__main__":
    main_eval()
