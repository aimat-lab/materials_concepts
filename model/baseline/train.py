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
        )

    def forward(self, x):
        """
        Pass throught network
        """
        res = self.net(x)

        return res


def train_model(
    model, data_train0, data_train1, data_test0, data_test1, lr_enc, batch_size
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    size_of_loss_check = 2000

    optimizer_predictor = torch.optim.Adam(model.parameters(), lr=lr_enc)

    data_train0 = torch.tensor(data_train0, dtype=torch.float).to(device)
    data_test0 = torch.tensor(data_test0, dtype=torch.float).to(device)

    data_train1 = torch.tensor(data_train1, dtype=torch.float).to(device)
    data_test1 = torch.tensor(data_test1, dtype=torch.float).to(device)

    test_loss_total = []
    moving_avg = []
    criterion = torch.nn.MSELoss()

    # There are much more vertex pairs that wont be connected (0) rather than ones
    # that will be connected (1). However, we observed that training with an equally weighted
    # training set (same number of examples for (0) and (1)) results in more stable training.
    # (Imaging we have 1.000.000 nonconnected and 10.000 connected)
    #
    # For that reason, we dont have true 'episodes' (where each example from the training set
    # has been used in the training). Rather, in each of our iteration, we sample batch_size
    # random training examples from data_train0 and from data_train1.

    print("Starting Training...")
    for iteration in range(
        10000
    ):  # should be much larger, with good early stopping criteria
        model.train()
        data_sets = [data_train0, data_train1]
        total_loss = 0
        for idx_dataset in range(len(data_sets)):
            idx = torch.randint(0, len(data_sets[idx_dataset]), (batch_size,))
            data_train_samples = data_sets[idx_dataset][idx]
            calc_properties = model(data_train_samples)
            curr_pred = torch.tensor([idx_dataset] * batch_size, dtype=torch.float).to(
                device
            )

            curr_pred = curr_pred.unsqueeze(1)
            real_loss = criterion(
                calc_properties, curr_pred
            )  # unsqueeze to match dimensions
            total_loss += torch.clamp(real_loss, min=0.0, max=50000.0).double()

        optimizer_predictor.zero_grad()
        total_loss.backward()
        optimizer_predictor.step()

        # Evaluating the current quality.
        with torch.no_grad():
            model.eval()
            # calculate train set
            eval_datasets = [data_train0, data_train1, data_test0, data_test1]
            all_real_loss = []
            for idx_dataset in range(len(eval_datasets)):
                calc_properties = model(
                    eval_datasets[idx_dataset][0:size_of_loss_check]
                )
                curr_pred = torch.tensor(
                    [idx_dataset % 2]
                    * len(eval_datasets[idx_dataset][0:size_of_loss_check]),
                    dtype=torch.float,
                ).to(device)
                curr_pred = curr_pred.unsqueeze(1)
                real_loss = criterion(calc_properties, curr_pred)
                all_real_loss.append(real_loss.detach().cpu().numpy())

            test_loss_total.append(
                np.mean(all_real_loss[2]) + np.mean(all_real_loss[3])
            )

            if iteration % 50 == 0:
                print(
                    str(iteration) + " - train: ",
                    np.mean(all_real_loss[0]) + np.mean(all_real_loss[1]),
                    "; test: ",
                    np.mean(all_real_loss[2]) + np.mean(all_real_loss[3]),
                )

            if len(test_loss_total) > 200:  # early stopping
                test_loss_moving_avg = sum(test_loss_total[-50:])
                moving_avg.append(test_loss_moving_avg)
                if len(moving_avg) > 10:
                    if (
                        moving_avg[-1] > moving_avg[-2]
                        and moving_avg[-1] > moving_avg[-10]
                    ):
                        print("Early stopping kicked in")
                        break

    plt.plot(test_loss_total)
    plt.title("Loss")
    plt.show()

    plt.plot(test_loss_total[200:])
    plt.title("Loss (zoomed))")
    plt.show()

    plt.plot(moving_avg)
    plt.title("Moving average")
    plt.show()

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
        data["data_train0"],
        data["data_train1"],
        data["data_test0"],
        data["data_test1"],
        lr_enc,
        batch_size,
    )
    torch.save(model.state_dict(), "model/baseline/model.pt")


if __name__ == "__main__":
    main()
