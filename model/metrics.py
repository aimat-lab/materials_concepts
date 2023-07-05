from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import json


def print_metrics(y_test, y_pred, threshold=0.5, save_path=None):
    auc = roc_auc_score(y_test, y_pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_test, y_pred > threshold, average="binary"
    )

    print("AUC", f"{auc:.4f}")
    print("Precision", f"{precision:.4f}")
    print("Recall", f"{recall:.4f}")
    print("F1", f"{fscore:.4f}")

    print("Confusion matrix:")
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred > threshold).ravel()
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

    if save_path:
        with open(save_path, "w") as f:
            json.dump(
                {
                    "auc": auc,
                    "precision": precision,
                    "recall": recall,
                    "fscore": fscore,
                    "confusion_matrix": {
                        "tn": int(tn),
                        "fp": int(fp),
                        "fn": int(fn),
                        "tp": int(tp),
                    },
                },
                f,
                indent=4,
            )
