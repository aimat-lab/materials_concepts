import numpy as np
import matplotlib.pyplot as plt
import torch
from train import BaselineNetwork
import pickle

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def flatten(t):
    return [item for sublist in t for item in sublist]


def calculate_roc(data_vertex_pairs, data_solution):
    data_solution = np.array(data_solution)
    data_vertex_pairs_sorted = data_solution[data_vertex_pairs]

    xpos = [0]
    ypos = [0]
    roc_vals = []
    for i in range(len(data_vertex_pairs_sorted)):
        if data_vertex_pairs_sorted[i] == 1:
            xpos.append(xpos[-1])
            ypos.append(ypos[-1] + 1)
        if data_vertex_pairs_sorted[i] == 0:
            xpos.append(xpos[-1] + 1)
            ypos.append(ypos[-1])
            roc_vals.append(ypos[-1])

    roc_vals = np.array(roc_vals) / max(ypos)
    ypos = np.array(ypos) / max(ypos)
    xpos = np.array(xpos) / max(xpos)

    auc = sum(roc_vals) / len(roc_vals)
    return (tuple(xpos), tuple(ypos)), auc


def eval_model(model, data, solution, name="AUC", plot=False):
    data = torch.tensor(data, dtype=torch.float).to(DEVICE)
    predictions = flatten(model(data).detach().cpu().numpy())
    predictions = np.flip(np.argsort(predictions, axis=0))
    curve, auc = calculate_roc(predictions, solution)
    print(f"{name}: ", auc)

    if plot:
        plt.plot(curve[0], curve[1])
        plt.show()


model = BaselineNetwork().to(DEVICE)
model.load_state_dict(torch.load("model/baseline/model.pt"))

with open("graph/data_2017_10p.pkl", "rb") as f:
    data = pickle.load(f)

data_train = data["data_train"]
solution_train = data["solution_train"]

data_test = data["data_test"]
solution_test = data["solution_test"]

eval_model(model, data=data_train, solution=solution_train, name="AUC_train", plot=True)
eval_model(model, data=data_test, solution=solution_test, name="AUC_test", plot=True)
