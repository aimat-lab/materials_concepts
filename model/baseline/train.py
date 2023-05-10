from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt

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

        if i % 49 == 0:
            print("Iteration: ", i, " Loss: ", running_loss / 50)
            running_loss = 0

    return True


def main():
    import pickle

    with open("graph/data.pkl", "rb") as f:
        data = pickle.load(f)

    model = BaselineNetwork().to(device)

    batch_size = 100  # Large batch_size seems to be important
    lr_enc = 5 * 10**-4

    model.train()
    train_model(
        model,
        data["data_train"],
        data["solution_train"],
        lr_enc,
        batch_size,
    )
    torch.save(model.state_dict(), "model/baseline/model.pt")


if __name__ == "__main__":
    main()
