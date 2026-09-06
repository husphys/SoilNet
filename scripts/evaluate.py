#!/usr/bin/env python3
"""Evaluation-only entry point. It never trains or updates a checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from soilnet.data import SoilNetDataset
from soilnet.evaluation import classification_metrics, regression_metrics
from soilnet.io import resolve_paths, sha256_file, write_json
from soilnet.models import SoilNetDualHead
from soilnet.reproducibility import result_metadata, set_determinism


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint is not a dictionary/state_dict")
    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(checkpoint.get(key), dict):
            return checkpoint[key]
    if checkpoint and all(hasattr(value, "shape") for value in checkpoint.values()):
        return checkpoint
    raise ValueError("No recognized model state_dict in checkpoint")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-relative", required=True, help="Path relative to CHECKPOINT_ROOT")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output-id", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--allow-nonstrict", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.output_id):
        raise ValueError("output-id may contain only letters, digits, dot, underscore, and hyphen")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    paths = resolve_paths()
    checkpoint_path = (paths["checkpoint_root"] / args.checkpoint_relative).resolve()
    if paths["checkpoint_root"] not in checkpoint_path.parents:
        raise ValueError("checkpoint-relative escapes CHECKPOINT_ROOT")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    manifest = REPO / "data" / "splits" / "final_split.csv"
    dataset_manifest = REPO / "data" / "manifests" / "soilnet_samples.csv"
    set_determinism(args.seed)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = SoilNetDataset(manifest, paths["data_root"], transform=transform, split=args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device(args.device)
    model = SoilNetDualHead(num_classes=args.num_classes, use_light=True, backbone_pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = {str(key).removeprefix("module."): value for key, value in extract_state_dict(checkpoint).items()}
    incompatible = model.load_state_dict(state, strict=not args.allow_nonstrict)
    model.to(device).eval()
    true_regression, predicted_regression, true_classes, predicted_classes = [], [], [], []
    with torch.inference_mode():
        for image, light, regression, classification in loader:
            regression_output, classification_output = model(image.to(device), light.to(device))
            true_regression.append(regression.numpy() * 100.0)
            predicted_regression.append(regression_output.cpu().numpy() * 100.0)
            true_classes.append(classification.numpy())
            predicted_classes.append(classification_output.argmax(dim=1).cpu().numpy())
    y_reg = np.concatenate(true_regression)
    p_reg = np.concatenate(predicted_regression)
    y_cls = np.concatenate(true_classes)
    p_cls = np.concatenate(predicted_classes)
    metric_rows = []
    for index, target in enumerate(("SM_0", "SM_20")):
        for metric, value in regression_metrics(y_reg[:, index], p_reg[:, index]).items():
            metric_rows.append({"task": "regression", "target": target, "metric": metric, "value": value, "unit": "percentage_point" if metric != "r2" else "dimensionless", "split": args.split, "n": len(dataset)})
    for metric, value in classification_metrics(y_cls, p_cls).items():
        metric_rows.append({"task": "classification", "target": "moisture_class", "metric": metric, "value": value, "unit": "ratio", "split": args.split, "n": len(dataset)})
    output_csv = REPO / "results" / "metrics" / f"{args.output_id}.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "target", "metric", "value", "unit", "split", "n"])
        writer.writeheader()
        writer.writerows(metric_rows)
    metadata = result_metadata(
        seed=args.seed, config="config/audit.yaml", dataset_manifest_sha256=sha256_file(dataset_manifest),
        split_manifest_sha256=sha256_file(manifest), checkpoint_sha256=sha256_file(checkpoint_path),
    )
    metadata.update({
        "experiment_id": args.output_id,
        "evaluation_split": args.split,
        "checkpoint_relative": args.checkpoint_relative,
        "strict_load": not args.allow_nonstrict,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "sample_count": len(dataset),
        "metric_csv": output_csv.relative_to(REPO).as_posix(),
    })
    write_json(output_csv.with_suffix(".json"), metadata)
    print(json.dumps({"metrics": metric_rows, "metadata": metadata}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
