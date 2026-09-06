from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(truth, pred))),
        "mae": float(mean_absolute_error(truth, pred)),
        "me": float(np.mean(pred - truth)),
        "r2": float(r2_score(truth, pred)),
    }


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
