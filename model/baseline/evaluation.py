import numpy as np
import matplotlib.pyplot as plt
import torch
from train import BaselineNetwork
import pickle
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def flatten(t):
    return [item for sublist in t for item in sublist]


THRESHOLD = 0.5


def eval_model(model, data, solution):
    data = torch.tensor(np.array(data), dtype=torch.float).to(DEVICE)
    predictions = np.array(flatten(model(data).detach().cpu().numpy()))

    auc = roc_auc_score(solution, predictions)
    precision, recall, fscore, _ = precision_recall_fscore_support(
        solution, predictions > THRESHOLD, average="binary"
    )

    print("AUC", f"{auc:.4f}")
    print("Precision", f"{precision:.4f}")
    print("Recall", f"{recall:.4f}")
    print("F1", f"{fscore:.4f}")


model = BaselineNetwork().to(DEVICE)
model.load_state_dict(torch.load("model/baseline/model.pt"))

with open("graph/data_training.pkl", "rb") as f:
    data = pickle.load(f)

data_test = data["data_test"]
solution_test = data["solution_test"]


np.set_printoptions(precision=3)
eval_model(model, data=data_test, solution=solution_test, name="AUC_test")
