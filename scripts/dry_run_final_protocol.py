#!/usr/bin/env python3
"""Bounded CPU-only engineering dry run; outputs no research metric values."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from soilnet.data import SoilNetDataset
from soilnet.evaluation import classification_metrics, regression_metrics
from soilnet.final_protocol import normalize_historical_soilnet_state
from soilnet.io import resolve_paths, sha256_file, write_json
from soilnet.models import SoilNetDualHead
from soilnet.reproducibility import set_determinism

SELECTED_RELATIVE = "checkpoints_VicReg_original_NEW/vicreg_model_final_mu_27.0.pth"
SELECTED_SHA256 = "42599ac025b8d8c8c5b8cca36624665f155e0ff588cd90de2fd47b08cd2d60d8"


def main() -> int:
    paths = resolve_paths()
    manifest = REPO / "data/manifests/soilnet_samples.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        total_rows = sum(1 for _ in csv.DictReader(handle))
    if total_rows != 2057:
        raise RuntimeError("Source manifest row count changed")
    set_determinism(20260905)
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    source_dataset = SoilNetDataset(manifest, paths["data_root"], transform=transform)
    subset = Subset(source_dataset, list(range(8)))
    loader = DataLoader(subset, batch_size=2, shuffle=False, num_workers=0)
    model = SoilNetDualHead(num_classes=10, use_light=True, backbone_pretrained=False)
    ssl_path = paths["checkpoint_root"] / SELECTED_RELATIVE
    if sha256_file(ssl_path) != SELECTED_SHA256:
        raise RuntimeError("Selected SSL checkpoint SHA256 changed")
    state = torch.load(ssl_path, map_location="cpu", weights_only=True)
    state, normalization = normalize_historical_soilnet_state(state, model.state_dict())
    model.load_state_dict(state, strict=True)
    model.train()
    image, light, regression, classification = next(iter(loader))
    predicted_regression, predicted_classification = model(image, light)
    loss = nn.MSELoss()(predicted_regression, regression)
    loss = loss + nn.CrossEntropyLoss()(predicted_classification, classification)
    loss.backward()
    gradients_present = any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
    with tempfile.TemporaryDirectory(prefix="soilnet-phase3a-dry-") as directory:
        checkpoint = Path(directory) / "dry_run.pth"
        torch.save({"model_state_dict": model.state_dict(), "dry_run": True}, checkpoint)
        restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
        reloaded = SoilNetDualHead(num_classes=10, use_light=True, backbone_pretrained=False)
        reloaded.load_state_dict(restored["model_state_dict"], strict=True)
        checkpoint_roundtrip = True
    # Exercise metric functions with two in-memory samples, but intentionally
    # discard values so this cannot be mistaken for scientific evaluation.
    regression_function_ok = set(regression_metrics([[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]])) == {"rmse", "mae", "me", "r2"}
    classification_function_ok = set(classification_metrics([0, 1], [0, 1])) == {
        "accuracy", "macro_f1", "macro_precision", "macro_recall",
    }
    loss_finite = bool(torch.isfinite(loss))
    report = {
        "status": "PASS" if all((loss_finite, gradients_present, checkpoint_roundtrip, regression_function_ok, classification_function_ok)) else "FAIL",
        "scope": "ENGINEERING_DRY_RUN_ONLY_NOT_A_RESEARCH_RESULT",
        "device": "cpu", "source_manifest_rows": total_rows, "samples_loaded": len(subset),
        "forward_batch_size": len(image), "forward_pass": True, "loss_finite": loss_finite,
        "backward_gradients_present": gradients_present, "checkpoint_safe_roundtrip": checkpoint_roundtrip,
        "regression_metric_function_exercised": regression_function_ok,
        "classification_metric_function_exercised": classification_function_ok,
        "checkpoint_normalization": normalization,
        "metric_values_recorded": False, "prospective_test_split_used": False,
        "training_run": False, "temporary_checkpoint_deleted": True,
    }
    write_json(REPO / "results/audit/phase3a_dry_run.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
