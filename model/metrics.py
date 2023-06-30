from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def print_metrics(y_test, y_pred, threshold=0.5):
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
