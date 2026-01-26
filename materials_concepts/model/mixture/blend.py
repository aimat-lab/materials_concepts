import fire
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from materials_concepts.model.metrics import print_metrics, test
from materials_concepts.utils.utils import (
    load_compressed,
    load_pickle,
    save_compressed
)


def main(
    data_path: str,
    predictions_path_1: str,
    predictions_path_2: str,
    save_path: str,
    blend: list[float] = (0.6, 0.4),
    metrics_path: str | None = None,
    plot_path: str | None = None,
):
    """
    Blend two sets of predictions and evaluate the result.
    """
    data = load_pickle(data_path)
    labels = data["y_test"]

    preds1 = load_compressed(predictions_path_1)
    preds2 = load_compressed(predictions_path_2)

    weights = np.linspace(0.0, 1.0, 11)
    aucs: list[float] = []

    for w_1 in weights:
        w_2 = 1.0 - w_1
        blended = preds1 * w_1 + preds2 * w_2

        auc, *_ = test(labels, blended, threshold=0.5)
        aucs.append(float(auc))

        print(f"Blend weights: {w_1:.1f}, {w_2:.1f}")
        print_metrics(labels, blended, threshold=0.5)
        print("-" * 40 + "\n")

    best_idx = int(np.argmax(aucs))
    best_w1 = float(weights[best_idx])
    best_auc = float(aucs[best_idx])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(weights, aucs, marker="o", linewidth=2)
    ax.set_xlabel(f"w_1 (weight for {predictions_path_1})")
    ax.set_ylabel("AUC")
    ax.set_title(
        "Blended ROC-AUC vs blend weight (w_1)\n"
        f"best: w_1={best_w1:.1f}, w_2={1.0 - best_w1:.1f} (AUC={best_auc:.4f})"
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.0, 1.0)

    if plot_path is None:
        plot_path = f"{save_path}.blend_auc_vs_w1.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


    blended_predictions = preds1 * blend[0] + preds2 * blend[1]

    if metrics_path:
        print_metrics(labels, blended_predictions, threshold=0.5, save_path=metrics_path)

    save_compressed(blended_predictions, save_path)


if __name__ == "__main__":
    fire.Fire(main) 