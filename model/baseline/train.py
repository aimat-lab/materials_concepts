from torch import nn
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def train_model(model, data_train, solution_train, lr_enc, batch_size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    optimizer_predictor = torch.optim.Adam(model.parameters(), lr=lr_enc)

    data_train = torch.tensor(data_train, dtype=torch.float).to(device)
    solution_train = torch.tensor(solution_train, dtype=torch.float).to(device)

    criterion = torch.nn.BCELoss()

    print("Starting Training...")
    running_loss = 0.0
    for i in range(10000):  # should be much larger, with good early stopping criteria
        model.train()

        idx = torch.randint(0, len(data_train), (batch_size,))
        data_train_samples = data_train[idx]
        output = model(data_train_samples)
        target = torch.tensor(solution_train[idx], dtype=torch.float).to(device)

        target = target.unsqueeze(1)
        loss = criterion(output, target)  # unsqueeze to match dimensions
        loss = torch.clamp(loss, min=0.0, max=50000.0).double()  # is this needed?

        optimizer_predictor.zero_grad()
        loss.backward()
        optimizer_predictor.step()
        running_loss += loss.item()

        if i % 50 == 0:
            print("Iteration: ", i, " Loss: ", running_loss / 50)
            running_loss = 0

    return True


def train_model_x(model, X_train, y_train, learning_rate, batch_size, num_epochs):
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


def flatten(t):
    return [item for sublist in t for item in sublist]


def eval_model(model, data, solution, name="Metrics"):
    data = torch.tensor(np.array(data), dtype=torch.float).to(device)
    predictions = np.array(flatten(model(data).detach().cpu().numpy()))

    auc = roc_auc_score(solution, predictions)

    THRESHOLD = 0.5
    precision, recall, fscore, _ = precision_recall_fscore_support(
        solution, predictions > THRESHOLD, average="binary"
    )

    print("\n" + name)
    print("AUC", f"{auc:.4f}")
    print("Precision", f"{precision:.4f}")
    print("Recall", f"{recall:.4f}")
    print("F1", f"{fscore:.4f}")


def main():
    import pickle

    print("Loading data...")
    with open("graph/data_training.pkl", "rb") as f:
        data = pickle.load(f)

    model = BaselineNetwork().to(device)

    print("Training...")
    model.train()
    train_model_x(
        model,
        data["data_train"],
        data["solution_train"],
        learning_rate=0.001,
        batch_size=100,
        num_epochs=2,
    )

    eval_model(
        model,
        data=data["data_test"],
        solution=data["solution_test"],
        name="Training Dist",
    )

    # evaluate model: currently non-sense
    with open("graph/data_test.pkl", "rb") as f:
        test = pickle.load(f)

    eval_model(
        model, data=test["data_test"], solution=test["solution_test"], name="Real Dist"
    )

    torch.save(model.state_dict(), "model/baseline/model.pt")


if __name__ == "__main__":
    main()
