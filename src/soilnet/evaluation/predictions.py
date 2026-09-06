from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

from soilnet.evaluation.metrics import classification_metrics, regression_metrics
from soilnet.io import write_json


@torch.inference_mode()
def prediction_rows_and_metrics(model, loader, device, config: dict[str, Any]):
    model.eval()
    rows, true_regression, predicted_regression, true_class, predicted_class = [], [], [], [], []
    cursor = 0
    for image, light, regression, classification in loader:
        prediction_regression, prediction_classification = model(
            image.to(device), light.to(device) if config["use_li"] else None
        )
        truth_reg = regression.numpy() * 100.0
        pred_reg = prediction_regression.cpu().numpy() * 100.0
        truth_cls = classification.numpy()
        probabilities = torch.softmax(prediction_classification, dim=1).cpu().numpy()
        pred_cls = probabilities.argmax(axis=1)
        for index in range(len(truth_cls)):
            source = loader.dataset.rows[cursor + index]
            rows.append({
                "sample_id": source["sample_id"], "relative_path": source["relative_path"],
                "SM_0_true": truth_reg[index, 0], "SM_0_pred": pred_reg[index, 0],
                "SM_20_true": truth_reg[index, 1], "SM_20_pred": pred_reg[index, 1],
                "class_true": int(truth_cls[index]), "class_pred": int(pred_cls[index]),
                "class_probabilities": ";".join(f"{value:.10g}" for value in probabilities[index]),
            })
        cursor += len(truth_cls)
        true_regression.append(truth_reg)
        predicted_regression.append(pred_reg)
        true_class.append(truth_cls)
        predicted_class.append(pred_cls)
    truth_reg = np.concatenate(true_regression)
    pred_reg = np.concatenate(predicted_regression)
    truth_cls = np.concatenate(true_class)
    pred_cls = np.concatenate(predicted_class)
    metrics = {
        "regression": {
            target: regression_metrics(truth_reg[:, index], pred_reg[:, index])
            for index, target in enumerate(("SM_0", "SM_20"))
        },
        "classification": classification_metrics(truth_cls, pred_cls),
        "n": len(rows),
        "regression_scale": "original_0_to_100_percentage_points",
    }
    return rows, metrics


def write_prediction_artifacts(prediction_path: Path, metric_path: Path, rows, metrics) -> None:
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "sample_id", "relative_path", "SM_0_true", "SM_0_pred", "SM_20_true", "SM_20_pred",
            "class_true", "class_pred", "class_probabilities",
        ])
        writer.writeheader()
        writer.writerows(rows)
    write_json(metric_path, metrics)
