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
    metrics_path: str | None = None,
    details_path: str | None = None,
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

        if details_path:
            # write to a details file
            with open(details_path, "a") as f:
                f.write(f"Blend weights: {w_1:.1f}, {w_2:.1f}\n")
                f.write(f"AUC: {auc:.4f}\n")
                f.write("-" * 40 + "\n")
        else:
            print(f"Blend weights: {w_1:.1f}, {w_2:.1f}")
            print_metrics(labels, blended, threshold=0.5)
            print("-" * 40 + "\n")

    best_idx = int(np.argmax(aucs))
    best_w1 = float(weights[best_idx])
    best_auc = float(aucs[best_idx])

    print(f"Best blend weight w1: {best_w1:.2f} with AUC: {best_auc:.4f}")
    blended_predictions = preds1 * best_w1 + preds2 * (1.0 - best_w1)

    if metrics_path:
        print_metrics(labels, blended_predictions, threshold=0.5, save_path=metrics_path)

    save_compressed(blended_predictions, save_path)


if __name__ == "__main__":
    fire.Fire(main) 