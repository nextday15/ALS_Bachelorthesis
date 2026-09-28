import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    recall_score,
)

CLASSES = [1, 2, 3, 4, 5]


def classification_scores(y_true, y_pred, labels=None):
    if labels is None:
        labels = CLASSES
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rec = recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    f1s = f1_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    scores = {
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_recall": float(
            recall_score(
                y_true, y_pred, labels=labels, average="macro", zero_division=0
            )
        ),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }
    for i, c in enumerate(labels):
        scores[f"recall_{c}"] = float(rec[i])
        scores[f"f1_{c}"] = float(f1s[i])
    return scores


def confusion(y_true, y_pred, labels=None):
    if labels is None:
        labels = CLASSES
    return confusion_matrix(y_true, y_pred, labels=labels)
