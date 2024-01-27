import matplotlib.pyplot as plt
import gzip, pickle


def load_compressed(path):
    with open(path, "rb") as f:
        compressed = f.read()
    return pickle.loads(gzip.decompress(compressed))


models = ["baseline", "pure_embs", "combi", "mixture"]
model_name = {
    "baseline": "Baseline",
    "pure_embs": "Concept Embeddings",
    "combi": "Combination of features",
    "mixture": "Combination of models",
}


colors = ["red", "green", "blue", "orange"]

plt.figure(figsize=(10, 7))
for model, color in zip(models, colors):
    data = load_compressed(f"data/analyze/{model}.auc_curve.pkl.gz")

    plt.plot(
        data["fpr"],
        data["tpr"],
        color=color,
        lw=2,
        label=f"{model_name[model]} - AUC: {data.get('auc', -1):.4f}",
    )

plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic", fontsize=20, y=1.05)
plt.legend(loc="lower right")
plt.show()
