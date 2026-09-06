#!/usr/bin/env python3
"""Explicit, guarded, exactly-once final test evaluation command."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms

from soilnet.data import SoilNetDataset
from soilnet.evaluation import classification_metrics, regression_metrics
from soilnet.final_protocol import refuse_if_test_completed, verify_locked_inputs, write_test_completion_marker
from soilnet.io import resolve_paths, sha256_file, write_json
from soilnet.models import SoilNetDualHead
from soilnet.reproducibility import set_determinism


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-test-once", action="store_true")
    parser.add_argument("--config", type=Path, default=REPO / "config/experiments/final_revalidation_v2.yaml")
    parser.add_argument("--lock", type=Path, default=REPO / "results/audit/final_run_lock_v2.json")
    args = parser.parse_args()
    if not args.confirm_test_once:
        raise RuntimeError("Explicit --confirm-test-once is required")
    locked = verify_locked_inputs(REPO, args.config, args.lock)
    config = locked["config"]
    marker = REPO / config["test_firewall"]["completion_marker"]
    refuse_if_test_completed(marker)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU_BLOCKED: final evaluation requires CUDA")

    output_dir = REPO / "results/final/FINAL_LEAKAGE_CONTROLLED_REVALIDATION"
    training_marker = output_dir / "training_completed.json"
    if not training_marker.is_file():
        raise RuntimeError("Training-completion evidence is missing")
    training = json.loads(training_marker.read_text(encoding="utf-8"))
    model_path = REPO / training["primary_checkpoint"]
    if sha256_file(model_path) != training["model_sha256"]:
        raise RuntimeError("Primary epoch-15 checkpoint SHA256 changed")

    paths = resolve_paths()
    split_path = REPO / config["data"]["final_split_manifest"]
    set_determinism(int(config["reproducibility"]["seed"]))
    transform = transforms.Compose([
        transforms.Resize(tuple(config["input"]["image_size"])), transforms.ToTensor(),
        transforms.Normalize(mean=config["input"]["image_normalization_mean"], std=config["input"]["image_normalization_std"]),
    ])
    dataset = SoilNetDataset(split_path, paths["data_root"], transform=transform, split="test")
    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=0)
    device = torch.device("cuda")
    model = SoilNetDualHead(num_classes=int(config["model"]["num_classes"]), use_light=True, backbone_pretrained=False)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    true_regression, predicted_regression, true_class, predicted_class = [], [], [], []
    with torch.inference_mode():
        for image, light, regression, classification in loader:
            regression_output, classification_output = model(image.to(device), light.to(device))
            true_regression.append(regression.numpy() * 100.0)
            predicted_regression.append(regression_output.cpu().numpy() * 100.0)
            true_class.append(classification.numpy())
            predicted_class.append(classification_output.argmax(dim=1).cpu().numpy())
    truth_reg = np.concatenate(true_regression)
    pred_reg = np.concatenate(predicted_regression)
    truth_cls = np.concatenate(true_class)
    pred_cls = np.concatenate(predicted_class)
    rows = []
    for index, target in enumerate(("SM_0", "SM_20")):
        for metric, value in regression_metrics(truth_reg[:, index], pred_reg[:, index]).items():
            rows.append({"task": "regression", "target": target, "metric": metric, "value": value,
                         "unit": "dimensionless" if metric == "r2" else "percentage_point", "n": len(dataset)})
    for metric, value in classification_metrics(truth_cls, pred_cls).items():
        rows.append({"task": "classification", "target": "moisture_class", "metric": metric,
                     "value": value, "unit": "ratio", "n": len(dataset)})
    metric_path = output_dir / "final_test_metrics.csv"
    with metric_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "target", "metric", "value", "unit", "n"])
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_dir / "final_test_confusion_matrix.json", {
        "labels": list(range(int(config["model"]["num_classes"]))),
        "counts": confusion_matrix(truth_cls, pred_cls, labels=list(range(int(config["model"]["num_classes"])))).tolist(),
    })
    write_test_completion_marker(
        marker, repo=REPO, model_path=model_path, split_path=split_path, config_path=args.config,
    )
    print(json.dumps({"status": "TEST_EVALUATION_COMPLETED_ONCE", "marker": marker.relative_to(REPO).as_posix()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
