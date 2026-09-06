#!/usr/bin/env python3
"""Read-only P0 v3 environment/model/checkpoint compatibility check."""
from __future__ import annotations

import json
import platform
import sys
from importlib import metadata
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from soilnet.io import write_json
from soilnet.models import build_model
from soilnet.utils import load_experiment_context


def version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def main() -> int:
    errors: list[str] = []
    context = load_experiment_context(REPO / "config/experiments/P0_final_soilnet.yaml")
    try:
        model, load_report = build_model(context.config, context.checkpoint_root)
    except Exception as error:
        model, load_report = None, {}
        errors.append(f"model/checkpoint compatibility failed: {error}")
    if load_report.get("matched_keys") != 577 or load_report.get("canonical_keys") != 577:
        errors.append("strict μ27 checkpoint load is not 577/577")
    for key in ("missing_after_normalization", "unexpected_after_normalization", "shape_mismatches"):
        if load_report.get(key) != []:
            errors.append(f"checkpoint compatibility mismatch: {key}")
    earlier_smoke = json.loads((REPO / "results/audit/p0_one_batch_smoke.json").read_text(encoding="utf-8"))
    if earlier_smoke.get("status") != "PASS" or earlier_smoke.get("backward_batches") != 1:
        errors.append("previous bounded forward/backward smoke evidence is not PASS")
    report = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "environment_mutated": False,
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "torchvision": version("torchvision"),
            "torchaudio": version("torchaudio"),
            "timm": version("timm"),
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU_BLOCKED",
        },
        "p0_config_sha256": context.config_sha256,
        "manifest_sha256": context.manifest_sha256,
        "split_sha256": context.split_sha256,
        "checkpoint_load": load_report,
        "model_instantiated": model is not None,
        "prior_bounded_forward_backward_smoke": {
            "status": earlier_smoke.get("status"),
            "train_batches": earlier_smoke.get("train_batches"),
            "validation_batches": earlier_smoke.get("validation_batches"),
            "backward_batches": earlier_smoke.get("backward_batches"),
            "test_loader_instantiated": earlier_smoke.get("test_loader_instantiated"),
        },
        "full_training_run": False,
        "test_loader_instantiated": False,
        "compatibility_interpretation": "tested compatible reproduction environment; not claimed historical-exact",
    }
    write_json(REPO / "results/audit/p0_v3_compatibility.json", report)
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
