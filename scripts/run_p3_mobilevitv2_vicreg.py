from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from soilnet.io import load_yaml, resolve_paths, resolve_run_artifact_root, sha256_file, write_json
from soilnet.models import build_model
from soilnet.reproducibility import set_determinism
from soilnet.training.engine import run_one_batch_preflight, train_experiment
from soilnet.training.vicreg import (
    pretrain_mobilevitv2_vicreg,
    reconcile_resolved_p3_before_ssl,
    reconcile_resolved_p3_config,
    validate_or_archive_stale_supervised_state,
)
from soilnet.utils import load_experiment_context


P2_CONFIG = REPO / "config/experiments/P2_mobilevitv2_imagenet_li_bestreg.yaml"
P3_CONFIG = REPO / "config/experiments/P3_mobilevitv2_vicreg_li_bestreg.yaml"
PROTECTED_EXPERIMENTS = (
    "P0_FINAL_SOILNET_VICREG_MU27_LI_V4_BESTREG",
    "P1_SOILNET_VICREG_MU27_NO_LI_BESTREG",
    "P1_SOILNET_IMAGENET_LI_NO_SSL_BESTREG",
    "P2_MOBILEVITV2_IMAGENET_LI_BESTREG",
)
SUPERVISED_MATCH_FIELDS = (
    "architecture", "timm_model_name", "use_li", "li_ablation_strategy", "li_fusion",
    "num_classes", "class_mapping", "dataset_manifest", "dataset_manifest_sha256",
    "split", "split_sha256", "split_counts", "seed", "epochs", "batch_size",
    "optimizer", "learning_rate", "weight_decay", "loss", "metrics", "image_size",
    "normalization_mean", "normalization_std", "train_augmentation",
    "validation_transform", "use_amp", "num_workers", "primary_task",
    "save_validation_best", "checkpoint_selection", "selection_formula",
    "primary_checkpoint", "epoch_60_checkpoint", "test_evaluation",
)


def hash_tree(path: Path) -> dict[str, str]:
    if not path.is_dir():
        raise RuntimeError(f"STOP: protected experiment directory is missing: {path}")
    return {
        file.relative_to(path).as_posix(): sha256_file(file)
        for file in sorted(path.rglob("*")) if file.is_file()
    }


def preflight_protocol(p2: dict[str, Any], p3: dict[str, Any], checkpoint_root: Path) -> dict[str, Any]:
    mismatches = {
        key: {"P2": p2.get(key), "P3": p3.get(key)}
        for key in SUPERVISED_MATCH_FIELDS if p2.get(key) != p3.get(key)
    }
    if mismatches:
        raise RuntimeError(f"STOP: P2/P3 supervised protocol mismatch: {mismatches}")
    if p2.get("ssl_checkpoint") is not None or p3.get("ssl_checkpoint") is not None:
        raise RuntimeError("STOP: source configs must not point to any existing SSL checkpoint")
    if p3["initialization_provenance"].get("soilnet_vicreg_checkpoint_loaded") is not False:
        raise RuntimeError("STOP: P3 must explicitly prohibit the SoilNet VICReg checkpoint")
    set_determinism(int(p3["seed"]))
    p2_model, _ = build_model(p2, checkpoint_root, pretrained_override=False)
    p2_state = p2_model.state_dict()
    set_determinism(int(p3["seed"]))
    p3_model, _ = build_model(p3, checkpoint_root, pretrained_override=False)
    p3_state = p3_model.state_dict()
    same_keys = p2_state.keys() == p3_state.keys()
    same_shapes = same_keys and all(p2_state[key].shape == p3_state[key].shape for key in p2_state)
    identical_initial_state = same_shapes and all(torch.equal(p2_state[key], p3_state[key]) for key in p2_state)
    if not identical_initial_state:
        raise RuntimeError("STOP: P2/P3 model graph or pre-VICReg initialization differs")
    return {
        "status": "PASS",
        "checked_before_any_training": True,
        "supervised_match_fields": list(SUPERVISED_MATCH_FIELDS),
        "supervised_mismatches": mismatches,
        "model_state_keys_identical": same_keys,
        "model_state_shapes_identical": same_shapes,
        "pre_vicreg_model_state_identical_at_locked_seed": identical_initial_state,
        "only_intended_treatment_difference": "MobileViTv2 backbone receives VICReg pretraining before supervised fine-tuning",
        "soilnet_vicreg_checkpoint_loaded": False,
        "test_loader_instantiated": False,
        "test_images_loaded": 0,
    }


def comparison_rows(
    p2_metrics: dict[str, Any], p3_metrics: dict[str, Any],
    p2_complexity: tuple[int, int], p3_complexity: tuple[int, int],
):
    rows = []
    for label, metrics, (parameters, best_epoch) in (
        ("P2 MobileViTv2 ImageNet-only", p2_metrics, p2_complexity),
        ("P3 MobileViTv2 + VICReg", p3_metrics, p3_complexity),
    ):
        regression = metrics["regression"]
        rows.append({
            "experiment": label,
            "best_epoch": best_epoch,
            "SM_0_RMSE": regression["SM_0"]["rmse"],
            "SM_0_MAE": regression["SM_0"]["mae"],
            "SM_0_R2": regression["SM_0"]["r2"],
            "SM_20_RMSE": regression["SM_20"]["rmse"],
            "SM_20_MAE": regression["SM_20"]["mae"],
            "SM_20_R2": regression["SM_20"]["r2"],
            "mean_RMSE": (regression["SM_0"]["rmse"] + regression["SM_20"]["rmse"]) / 2,
            "mean_MAE": (regression["SM_0"]["mae"] + regression["SM_20"]["mae"]) / 2,
            "parameters": parameters,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume only from identity-checked rolling checkpoints")
    parser.add_argument(
        "--run-ssl-pretraining", action="store_true",
        help="Explicitly authorize VICReg only when no verified final SSL checkpoint exists",
    )
    parser.add_argument("--preflight-only", action="store_true", help="Verify P2/P3 parity without writing or training")
    args = parser.parse_args()
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    p2, p3 = load_yaml(P2_CONFIG), load_yaml(P3_CONFIG)
    paths = resolve_paths()
    run_root = resolve_run_artifact_root()
    experiment_root = run_root / "P3_MOBILEVITV2_VICREG_LI_BESTREG"
    supervised_dir = experiment_root / "supervised"
    if (supervised_dir / "run_metadata.json").is_file():
        raise RuntimeError("STOP: P3 is already complete; refusing to overwrite it")

    protected_before = {
        name: hash_tree(run_root / name) for name in PROTECTED_EXPERIMENTS
    }
    protocol_report = preflight_protocol(p2, p3, paths["checkpoint_root"])
    print(json.dumps({"P2_P3_PREFLIGHT": protocol_report}, indent=2), flush=True)
    if args.preflight_only:
        return
    experiment_root.mkdir(parents=True, exist_ok=True)
    source_snapshot = experiment_root / "source_config.yaml"
    if source_snapshot.exists() and source_snapshot.read_bytes() != P3_CONFIG.read_bytes():
        raise RuntimeError("STOP: existing P3 source-config snapshot differs")
    if not source_snapshot.exists():
        shutil.copyfile(P3_CONFIG, source_snapshot)
    write_json(experiment_root / "p2_p3_preflight_report.json", protocol_report)
    write_json(experiment_root / "protected_artifacts_before.json", protected_before)

    pre_ssl_reconciliation = reconcile_resolved_p3_before_ssl(
        config=p3,
        config_sha256=sha256_file(P3_CONFIG),
        experiment_root=experiment_root,
        stale_archive_root=run_root.parent / "soilnet_stale_p3",
    )
    print(json.dumps({"pre_ssl_resolved_config": pre_ssl_reconciliation}, indent=2))

    pretraining = pretrain_mobilevitv2_vicreg(
        config=p3, config_sha256=sha256_file(P3_CONFIG), repo=REPO,
        data_root=paths["data_root"], experiment_root=experiment_root,
        run_ssl_pretraining=args.run_ssl_pretraining,
    )
    reconciliation = reconcile_resolved_p3_config(
        config=p3,
        config_sha256=sha256_file(P3_CONFIG),
        experiment_root=experiment_root,
        stale_archive_root=run_root.parent / "soilnet_stale_p3",
    )
    resolved_config = Path(reconciliation["resolved_config_path"])

    context = load_experiment_context(resolved_config)
    supervised_state = validate_or_archive_stale_supervised_state(
        resolved_config=context.config,
        resolved_config_sha256=context.config_sha256,
        experiment_root=experiment_root,
        stale_archive_root=run_root.parent / "soilnet_stale_p3",
    )
    print(json.dumps({"resolved_config": reconciliation, "supervised_state": supervised_state}, indent=2))
    supervised_preflight = run_one_batch_preflight(context, device="cuda")
    if supervised_preflight.get("status") != "PASS" or not supervised_preflight.get("vicreg_checkpoint_loaded"):
        raise RuntimeError("STOP: supervised P3 preflight did not strictly load VICReg")
    write_json(experiment_root / "supervised_preflight_report.json", supervised_preflight)
    metadata = train_experiment(context, resume_if_available=args.resume)

    p2_dir = run_root / "P2_MOBILEVITV2_IMAGENET_LI_BESTREG"
    p2_metrics = json.loads((p2_dir / "validation_metrics.json").read_text(encoding="utf-8"))
    p2_metadata = json.loads((p2_dir / "run_metadata.json").read_text(encoding="utf-8"))
    p3_metrics = json.loads((supervised_dir / "validation_metrics.json").read_text(encoding="utf-8"))
    rows = comparison_rows(
        p2_metrics,
        p3_metrics,
        (int(p2_metadata["complexity"]["parameters"]), int(p2_metadata["best_epoch"])),
        (int(metadata["complexity"]["parameters"]), int(metadata["best_epoch"])),
    )
    comparison_csv = experiment_root / "validation_comparison.csv"
    with comparison_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(experiment_root / "validation_comparison.json", rows)
    header = "| Experiment | Best epoch | SM-0 RMSE | SM-0 MAE | SM-0 R2 | SM-20 RMSE | SM-20 MAE | SM-20 R2 | Mean RMSE | Mean MAE | Parameters |"
    separator = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    body = [
        "| {experiment} | {best_epoch} | {SM_0_RMSE:.6f} | {SM_0_MAE:.6f} | {SM_0_R2:.6f} | "
        "{SM_20_RMSE:.6f} | {SM_20_MAE:.6f} | {SM_20_R2:.6f} | {mean_RMSE:.6f} | {mean_MAE:.6f} | {parameters} |".format(**row)
        for row in rows
    ]
    (experiment_root / "VALIDATION_COMPARISON.md").write_text(
        "# P2 vs P3 validation comparison\n\n" + "\n".join([header, separator, *body]) + "\n",
        encoding="utf-8",
    )

    protected_after = {name: hash_tree(run_root / name) for name in PROTECTED_EXPERIMENTS}
    if protected_before != protected_after:
        raise RuntimeError("STOP: a protected P0/P1/P2 artifact changed during P3")
    write_json(experiment_root / "protected_artifacts_after.json", protected_after)
    execution = {
        "status": "COMPLETE",
        "command": "conda run -n soilnet python scripts/run_p3_mobilevitv2_vicreg.py",
        "resume_requested": args.resume,
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "total_duration_seconds": time.monotonic() - started,
        "pretraining_duration_seconds": pretraining["training_duration_seconds"],
        "supervised_duration_seconds": metadata["training_duration_seconds"],
        "best_epoch": metadata["best_epoch"],
        "best_checkpoint": metadata["primary_checkpoint_path"],
        "best_checkpoint_sha256": metadata["primary_checkpoint_sha256"],
        "test_evaluated": "NO",
        "test_loader_instantiated": False,
        "artifact_paths": sorted(str(path) for path in experiment_root.rglob("*") if path.is_file()),
    }
    write_json(experiment_root / "execution_report.json", execution)
    print(json.dumps({"P3_COMPLETE": execution, "VALIDATION_COMPARISON": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
