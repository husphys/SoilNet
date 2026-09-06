#!/usr/bin/env python3
"""Bounded P0 preflight: temporary model, no optimizer step or metrics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch

from soilnet.io import write_json
from soilnet.training import run_one_batch_smoke
from soilnet.utils import load_experiment_context


def main() -> int:
    if (REPO / "results/final/test_evaluation_completed.json").exists():
        raise RuntimeError("Unexpected final-test marker before bounded smoke")
    before = sorted(str(path) for path in REPO.rglob("*.pth") if path.is_file())
    context = load_experiment_context(REPO / "config/experiments/P0_final_soilnet.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = run_one_batch_smoke(context, device=device)
    report.update({
        "experiment_id": context.config["experiment_id"],
        "config_sha256": context.config_sha256,
        "manifest_sha256": context.manifest_sha256,
        "split_sha256": context.split_sha256,
        "gpu_available": torch.cuda.is_available(),
        "full_training_run": False,
        "test_samples_loaded": 0,
        "test_metrics_computed": False,
        "repository_pth_before": before,
        "repository_pth_after": sorted(str(path) for path in REPO.rglob("*.pth") if path.is_file()),
    })
    write_json(REPO / "results/audit/p0_one_batch_smoke.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
