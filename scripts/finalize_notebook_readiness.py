#!/usr/bin/env python3
"""Freeze the P0–P2 implementation inventory and refresh tracked checksums."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from soilnet.io import sha256_file, write_json


CONFIGS = [
    "config/experiments/P0_final_soilnet.yaml",
    "config/experiments/P1_no_li.yaml",
    "config/experiments/P1_no_ssl.yaml",
    "config/experiments/P1_mobilenetv2.yaml",
    "config/experiments/P1_mobilevitv2.yaml",
    "config/experiments/P2_mobilenetv3.yaml",
    "config/experiments/P2_efficientnet_b0.yaml",
    "config/experiments/P2_classical_ml.yaml",
]
NOTEBOOKS = [
    f"notebooks/final_experiments/{name}"
    for name in (
        "00_gpu_environment_check.ipynb",
        "01_P0_final_soilnet.ipynb",
        "02_P1_ablation_no_li.ipynb",
        "03_P1_ablation_no_ssl.ipynb",
        "04_P1_mobilenetv2.ipynb",
        "05_P1_mobilevitv2.ipynb",
        "06_P2_mobilenetv3.ipynb",
        "07_P2_efficientnet_b0.ipynb",
        "08_P2_classical_ml.ipynb",
        "09_final_test_evaluation.ipynb",
        "10_results_summary.ipynb",
    )
]
IMPLEMENTATION = [
    "src/soilnet/io.py",
    "src/soilnet/models/soilnet.py",
    "src/soilnet/models/experiment_soilnet.py",
    "src/soilnet/models/baselines.py",
    "src/soilnet/models/factory.py",
    "src/soilnet/training/engine.py",
    "src/soilnet/training/classical.py",
    "src/soilnet/evaluation/metrics.py",
    "src/soilnet/evaluation/predictions.py",
    "src/soilnet/evaluation/final_test.py",
    "src/soilnet/utils/experiment.py",
    "scripts/generate_final_notebooks.py",
    "scripts/check_final_notebooks.py",
    "scripts/smoke_p0_notebook_pipeline.py",
    "scripts/report_run_disk_usage.py",
    "scripts/check_p0_v3_compatibility.py",
    "scripts/finalize_notebook_readiness.py",
    "scripts/verify_artifacts.py",
]
EVIDENCE = [
    "docs/EXPERIMENT_EXECUTION_PLAN.md",
    "docs/FINAL_REVALIDATION_PROTOCOL_V3.md",
    "docs/MANUAL_RUN_GUIDE.md",
    "results/audit/notebook_generation_unresolved.md",
    "results/audit/notebook_static_check.json",
    "results/audit/p0_one_batch_smoke.json",
    "results/audit/p0_v3_compatibility.json",
]
PORTABLE_FILES = ["README.md", "scripts/README.md", "config/paths.example.yaml"]
LOCK_PATH = "results/audit/notebook_generation_lock.json"
V3_LOCK_PATH = "results/audit/final_run_lock_v3.json"


def hash_map(paths: list[str]) -> dict[str, str]:
    missing = [relative for relative in paths if not (REPO / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot finalize; missing files: {missing}")
    return {relative: sha256_file(REPO / relative) for relative in paths}


def refresh_registry(additional: list[str]) -> int:
    registry = REPO / "data/checksums/manifest_sha256.csv"
    with registry.open(newline="", encoding="utf-8") as handle:
        existing = [row["relative_path"] for row in csv.DictReader(handle)]
    paths = sorted(set(existing + additional))
    missing = [relative for relative in paths if not (REPO / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Checksummed files are missing: {missing}")
    with registry.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(
            {
                "relative_path": relative,
                "size_bytes": (REPO / relative).stat().st_size,
                "sha256": sha256_file(REPO / relative),
            }
            for relative in paths
        )
    return len(paths)


def main() -> int:
    compatibility = json.loads((REPO / "results/audit/p0_v3_compatibility.json").read_text(encoding="utf-8"))
    if compatibility.get("status") != "PASS":
        raise RuntimeError("Cannot freeze P0 v3: compatibility check is not PASS")
    gpu_notebook = json.loads(
        (REPO / "notebooks/final_experiments/00_gpu_environment_check.ipynb").read_text(encoding="utf-8")
    )
    gpu_output = "\n".join(
        "".join(output.get("text", []))
        for cell in gpu_notebook["cells"] if cell.get("cell_type") == "code"
        for output in cell.get("outputs", []) if output.get("output_type") == "stream"
    )
    manual_gpu_pass = all(marker in gpu_output for marker in (
        "cuda_available: True", "gpu: NVIDIA GeForce RTX 3050", "'GPU_SMOKE': 'PASS'",
    ))
    v3_implementation = [
        "src/soilnet/training/engine.py", "src/soilnet/models/factory.py",
        "src/soilnet/utils/experiment.py", "src/soilnet/evaluation/final_test.py",
        "scripts/generate_final_notebooks.py", "scripts/check_final_notebooks.py",
        "notebooks/final_experiments/00_gpu_environment_check.ipynb",
        "notebooks/final_experiments/01_P0_final_soilnet.ipynb",
    ]
    v3_lock = {
        "lock_revision": 3,
        "status": "PROTOCOL_REVISED_PRE_TEST",
        "current_implementation_status": "READY_FOR_MANUAL_P0_RUN",
        "reason": "Protocol revised before held-out test evaluation to match the strongest surviving historical fine-tuning implementation evidence.",
        "old_manuscript_protocol": {
            "epochs": 15, "batch_size": 32, "optimizer": "Adam",
            "learning_rate": 0.0005, "weight_decay": 0.0,
        },
        "historical_code_evidence": {
            "source": "audited GitHub fine-tuning notebook mu=27",
            "epochs": 60, "batch_size": 32, "optimizer": "Adam", "learning_rate": 0.0001,
        },
        "final_revalidation_v3": {
            "epochs": 60, "batch_size": 32, "optimizer": "Adam",
            "learning_rate": 0.0001, "weight_decay": 0.0,
            "primary_checkpoint": "epoch_60_final.pth",
        },
        "test_loader_used_before_revision": False,
        "test_metrics_read_or_computed_before_revision": False,
        "historical_labeled_records": 2057,
        "excluded_records": 130,
        "final_clean_records": 1927,
        "split_counts": {"train": 1407, "validation": 289, "test": 231},
        "final_clean_manifest_sha256": "8ff45054d4b8e3df9758d0c112dc16719572b2267906fb3a5ed5b3262a6732bd",
        "final_split_sha256": "8927b8223b8c4c234d264ad9ea62ac2df6161124a79787a2e71eeb5cd23eae2f",
        "ssl_checkpoint_relative_path": "checkpoints_VicReg_original_NEW/vicreg_model_final_mu_27.0.pth",
        "ssl_checkpoint_sha256": "42599ac025b8d8c8c5b8cca36624665f155e0ff588cd90de2fd47b08cd2d60d8",
        "ssl_lineage": "VERIFIED_LINEAGE",
        "seed": 20260905,
        "seed_provenance": "retained from Phase 3B v2 pre-test lock; not claimed as a recovered historical seed",
        "p0_config_relative_path": "config/experiments/P0_final_soilnet.yaml",
        "p0_config_sha256": sha256_file(REPO / "config/experiments/P0_final_soilnet.yaml"),
        "protocol_document_sha256": sha256_file(REPO / "docs/FINAL_REVALIDATION_PROTOCOL_V3.md"),
        "compatibility_report_sha256": sha256_file(REPO / "results/audit/p0_v3_compatibility.json"),
        "compatibility_checks": compatibility,
        "manual_jupyter_gpu_check": {
            "status": "PASS" if manual_gpu_pass else "NOT_RUN",
            "cuda_available": True if manual_gpu_pass else None,
            "gpu_name": "NVIDIA GeForce RTX 3050" if manual_gpu_pass else None,
            "forward_backward": "PASS" if manual_gpu_pass else None,
            "research_metrics_created": False,
            "test_evaluated": False,
        },
        "implementation_sha256": hash_map(v3_implementation),
        "p1_status": "OPTIONAL_NOT_YET_REQUIRED",
        "p2_status": {
            "MobileNetV2": "OPTIONAL", "MobileViTv2": "OPTIONAL",
            "MobileNetV3": "SKIPPED_VARIANT_UNRESOLVED",
            "EfficientNet-B0": "OPTIONAL", "Classical ML": "OPTIONAL",
        },
        "full_training_run": False,
        "test_evaluated": False,
    }
    write_json(REPO / V3_LOCK_PATH, v3_lock)
    lock = {
        "status": "FROZEN_P0_V3_P2_OPTIONAL",
        "full_training_run": False,
        "test_loader_instantiated": False,
        "research_metrics_from_smoke": False,
        "seed": 20260905,
        "manifest_sha256": "8ff45054d4b8e3df9758d0c112dc16719572b2267906fb3a5ed5b3262a6732bd",
        "split_sha256": "8927b8223b8c4c234d264ad9ea62ac2df6161124a79787a2e71eeb5cd23eae2f",
        "ssl_checkpoint_sha256": "42599ac025b8d8c8c5b8cca36624665f155e0ff588cd90de2fd47b08cd2d60d8",
        "p0_status": "READY_FOR_MANUAL_P0_RUN",
        "p2_mobilenetv3_status": "SKIPPED_VARIANT_UNRESOLVED",
        "config_sha256": hash_map(CONFIGS),
        "notebook_sha256": hash_map(NOTEBOOKS),
        "implementation_sha256": hash_map(IMPLEMENTATION),
        "evidence_sha256": hash_map(EVIDENCE + [V3_LOCK_PATH]),
    }
    write_json(REPO / LOCK_PATH, lock)
    registry_count = refresh_registry(
        CONFIGS + NOTEBOOKS + IMPLEMENTATION + EVIDENCE + PORTABLE_FILES + [V3_LOCK_PATH, LOCK_PATH]
    )
    print(json.dumps({"status": "PASS", "lock": LOCK_PATH, "registry_rows": registry_count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
