#!/usr/bin/env python3
from pathlib import Path

import nbformat as nbf


REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "notebooks/final_experiments/13_P3_mobilevitv2_vicreg_bestreg.ipynb"


def code(source: str):
    return nbf.v4.new_code_cell(source.strip() + "\n")


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


book = nbf.v4.new_notebook()
book.metadata = {
    "kernelspec": {"display_name": "Python (soilnet)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
book.cells = [
    markdown(r"""
# P3 — pure MobileViTv2 + VICReg + LI (validation BESTREG)

This notebook is a fresh-kernel, sequential **Run All** workflow for
`P3_MOBILEVITV2_VICREG_LI_BESTREG`. It never constructs a test dataset or test
loader. The held-out test set remains closed unless P3 is later frozen into the
separate final-test workflow.

**Scientific question:** whether VICReg changes MobileViTv2 validation results
under the fixed P2 supervised protocol. **Configuration:** the frozen P3 YAML.
**Dataset:** the 11,995-image unlabeled manifest plus labeled training/validation
records. **Split:** 1,407 train and 289 validation; test is metadata only.
**Initialization checkpoint:** experiment-specific epoch-80 VICReg encoder.
**Expected outputs:** frozen pretraining history/metadata, supervised history,
validation predictions/metrics, and validation-best checkpoint.

The supervised selection score is

$$R_{\mathrm{val}}=\frac{RMSE_{\mathrm{SM0}}+RMSE_{\mathrm{SM20}}}{2}.$$

All 60 supervised epochs run; validation-best selection is not early stopping.
"""),
    markdown("## 1. Environment check and Run-All controls"),
    code(r"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from pathlib import Path
import csv
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

def find_repo(start: Path) -> Path:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src/soilnet").is_dir():
            return candidate
    raise RuntimeError("STOP: run this notebook from inside the SoilNet checkout")

REPO = find_repo(Path.cwd())
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import pandas as pd
import timm
import torch
import torchvision
import yaml
from IPython.display import display

from soilnet.evaluation.predictions import prediction_rows_and_metrics
from soilnet.final_protocol import current_git_commit
from soilnet.io import load_yaml, resolve_paths, resolve_run_artifact_root, sha256_file, write_json
from soilnet.models import build_frozen_model, build_model
from soilnet.training.engine import build_val_loader, run_one_batch_preflight, train_experiment
from soilnet.training.vicreg import (
    reconcile_resolved_p3_before_ssl,
    reconcile_resolved_p3_config,
    pretrain_mobilevitv2_vicreg,
    run_vicreg_one_batch_preflight,
    validate_or_archive_stale_supervised_state,
    verify_mobilevitv2_vicreg,
)
from soilnet.utils import load_experiment_context
from scripts.run_p3_mobilevitv2_vicreg import (
    P2_CONFIG, P3_CONFIG, PROTECTED_EXPERIMENTS, SUPERVISED_MATCH_FIELDS,
    comparison_rows, hash_tree, preflight_protocol,
)

# The completed epoch-80 SSL checkpoint is now frozen.  Keep this False for
# subsequent Run All executions so the notebook can only verify and reuse it.
RUN_SSL_PRETRAINING = False
RESUME_SUPERVISED_IF_AVAILABLE = True

if not torch.cuda.is_available():
    raise RuntimeError("GPU_BLOCKED: P3 requires CUDA; no CPU fallback is authorized")

print({
    "repo": str(REPO),
    "git_commit": current_git_commit(REPO),
    "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip()),
    "python": platform.python_version(),
    "pytorch": torch.__version__,
    "torchvision": torchvision.__version__,
    "timm": timm.__version__,
    "cuda_available": torch.cuda.is_available(),
    "gpu_name": torch.cuda.get_device_name(0),
    "RUN_SSL_PRETRAINING": RUN_SSL_PRETRAINING,
    "TEST_LOADER_INSTANTIATED": False,
})
"""),
    markdown("## 2. Path and locked-config check"),
    code(r"""
p2 = load_yaml(P2_CONFIG)
p3 = load_yaml(P3_CONFIG)
p3_source_config_sha256 = sha256_file(P3_CONFIG)
paths = resolve_paths()
run_root = resolve_run_artifact_root()
experiment_root = run_root / p3["experiment_id"]
supervised_dir = experiment_root / "supervised"
p2_dir = run_root / p2["output_dir"]
stale_archive_root = run_root.parent / "soilnet_stale_p3"

required_p2 = (p2_dir / "run_metadata.json", p2_dir / "validation_metrics.json")
missing_p2 = [str(path) for path in required_p2 if not path.is_file()]
if missing_p2:
    raise RuntimeError(f"STOP: required frozen P2 validation artifacts are missing: {missing_p2}")
if p3["experiment_id"] != "P3_MOBILEVITV2_VICREG_LI_BESTREG":
    raise RuntimeError("STOP: unexpected P3 experiment ID")
if p3["timm_model_name"] != "mobilevitv2_050.cvnets_in1k":
    raise RuntimeError("STOP: MobileViTv2 model tag changed")
if p3.get("ssl_checkpoint") is not None:
    raise RuntimeError("STOP: source config must not point to a pre-existing SSL checkpoint")
if p3["initialization_provenance"].get("soilnet_vicreg_checkpoint_loaded") is not False:
    raise RuntimeError("STOP: SoilNet VICReg checkpoints are prohibited for P3")

source_context = load_experiment_context(P3_CONFIG)
pre_ssl_resolved_reconciliation = reconcile_resolved_p3_before_ssl(
    config=p3,
    config_sha256=p3_source_config_sha256,
    experiment_root=experiment_root,
    stale_archive_root=stale_archive_root,
)
print({
    "DATA_ROOT": str(paths["data_root"]),
    "CHECKPOINT_ROOT": str(paths["checkpoint_root"]),
    "RUN_ARTIFACT_ROOT": str(run_root),
    "experiment_root": str(experiment_root),
    "experiment_id": p3["experiment_id"],
    "model_name": p3["timm_model_name"],
    "initialization_source": p3["initialization_provenance"]["source"],
    "source_config_sha256": p3_source_config_sha256,
    "seed": p3["seed"],
    "epochs": p3["epochs"],
    "batch_size": p3["batch_size"],
    "optimizer": p3["optimizer"],
    "learning_rate": p3["learning_rate"],
    "LI_enabled": p3["use_li"],
    "TEST_LOADER_INSTANTIATED": False,
    "PRE_SSL_RESOLVED_CONFIG_STATUS": pre_ssl_resolved_reconciliation["status"],
})
"""),
    markdown("## 3. Dataset and split verification (metadata only; no test dataset)"),
    code(r"""
print({
    "dataset_manifest": str(source_context.manifest_path),
    "dataset_manifest_sha256": source_context.manifest_sha256,
    "split_manifest": str(source_context.split_path),
    "split_sha256": source_context.split_sha256,
    "train_count": p3["split_counts"]["train"],
    "validation_count": p3["split_counts"]["validation"],
    "held_out_test_count_metadata_only": p3["split_counts"]["test"],
    "unlabeled_image_count": p3["vicreg_pretraining"]["expected_image_count"],
    "TEST_LOADER_INSTANTIATED": False,
})
"""),
    markdown("## 4. P2/P3 protocol comparison — only initialization/VICReg fields may differ"),
    code(r"""
protocol_report = preflight_protocol(p2, p3, paths["checkpoint_root"])
allowed_difference_fields = {
    "experiment_id", "protocol_revision", "protocol_version", "ablation_type",
    "scientific_question", "architecture_role", "initialization", "ssl", "mu",
    "initialization_provenance", "vicreg_pretraining", "output_dir",
}
display_fields = list(dict.fromkeys([
    "experiment_id", "architecture", "timm_model_name", "initialization", "ssl", "mu",
    *SUPERVISED_MATCH_FIELDS, "vicreg_pretraining", "output_dir",
]))
diff_rows = []
for field in display_fields:
    same = p2.get(field) == p3.get(field)
    status = "MATCH" if same else ("ALLOWED_INITIALIZATION_DIFF" if field in allowed_difference_fields else "FORBIDDEN_DIFF")
    diff_rows.append({"field": field, "P2": p2.get(field), "P3": p3.get(field), "status": status})
diff_table = pd.DataFrame(diff_rows)
display(diff_table)
forbidden = diff_table[diff_table["status"] == "FORBIDDEN_DIFF"]
if not forbidden.empty:
    raise RuntimeError(f"STOP: forbidden P2/P3 differences: {forbidden['field'].tolist()}")
print(json.dumps(protocol_report, indent=2))
"""),
    markdown("## 5. Model construction and protected-artifact snapshot"),
    code(r"""
protected_before = {name: hash_tree(run_root / name) for name in PROTECTED_EXPERIMENTS}
torch.manual_seed(int(p3["seed"]))
architecture_probe, _ = build_model(p3, paths["checkpoint_root"], pretrained_override=False)
parameter_count = sum(parameter.numel() for parameter in architecture_probe.parameters())
if parameter_count != 1189189:
    raise RuntimeError(f"STOP: P3 parameter count changed: {parameter_count}")
print({
    "model_class": architecture_probe.__class__.__name__,
    "model_name": p3["timm_model_name"],
    "parameters": parameter_count,
    "LI_enabled": architecture_probe.use_light,
    "regression_outputs": architecture_probe.reg_head[-1].out_features,
    "classification_outputs": architecture_probe.cls_head[-1].out_features,
    "protected_experiments_snapshotted": list(protected_before),
    "TEST_LOADER_INSTANTIATED": False,
})
del architecture_probe
"""),
    markdown("## 6. Engineering preflight: VICReg loss/backward and supervised train/validation batches"),
    code(r"""
vicreg_preflight = run_vicreg_one_batch_preflight(
    config=p3, repo=REPO, data_root=paths["data_root"], device="cuda"
)
p2_context = load_experiment_context(P2_CONFIG)
supervised_pipeline_preflight = run_one_batch_preflight(p2_context, device="cuda")
if not vicreg_preflight["loss_finite"] or supervised_pipeline_preflight["status"] != "PASS":
    raise RuntimeError("STOP: engineering preflight failed")
print(json.dumps({
    "VICREG_ONE_BATCH_PREFLIGHT": vicreg_preflight,
    "SUPERVISED_TRAIN_VALIDATION_PREFLIGHT": supervised_pipeline_preflight,
    "TEST_LOADER_INSTANTIATED": False,
}, indent=2))
"""),
    markdown("## 7. VICReg pretraining or exact verified reuse"),
    code(r"""
experiment_root.mkdir(parents=True, exist_ok=True)
source_snapshot = experiment_root / "source_config.yaml"
if source_snapshot.exists() and source_snapshot.read_bytes() != P3_CONFIG.read_bytes():
    raise RuntimeError("STOP: existing P3 source-config snapshot differs")
if not source_snapshot.exists():
    shutil.copyfile(P3_CONFIG, source_snapshot)
write_json(experiment_root / "p2_p3_preflight_report.json", protocol_report)
write_json(experiment_root / "protected_artifacts_before_notebook.json", protected_before)

if RUN_SSL_PRETRAINING:
    ssl_metadata = pretrain_mobilevitv2_vicreg(
        config=p3,
        config_sha256=p3_source_config_sha256,
        repo=REPO,
        data_root=paths["data_root"],
        experiment_root=experiment_root,
        run_ssl_pretraining=RUN_SSL_PRETRAINING,
    )
    ssl_action = "PRETRAINED_OR_VERIFIED_COMPLETED_REUSE"
else:
    ssl_metadata = verify_mobilevitv2_vicreg(
        config=p3,
        config_sha256=p3_source_config_sha256,
        experiment_root=experiment_root,
    )
    ssl_action = "STRICT_VERIFIED_REUSE"

print({
    "EXPERIMENT_ID": p3["experiment_id"],
    "CONFIG_SHA256": p3_source_config_sha256,
    "UNLABELED_MANIFEST_SHA256": ssl_metadata["unlabeled_manifest_sha256"],
    "SPLIT_SHA256": p3["split_sha256"],
    "TIMM_MODEL_NAME": p3["timm_model_name"],
    "RUN_SSL_PRETRAINING": RUN_SSL_PRETRAINING,
    "SSL_RESUME_STATUS": ssl_metadata.get("resume_resolution", {}).get("status", ssl_action),
    "SSL_START_EPOCH": ssl_metadata.get("resume_resolution", {}).get("start_epoch"),
    "TRAIN_COUNT": p3["split_counts"]["train"],
    "VALIDATION_COUNT": p3["split_counts"]["validation"],
    "SSL_ACTION": ssl_action,
    "SSL_CHECKPOINT": ssl_metadata["encoder_checkpoint_path"],
    "SSL_CHECKPOINT_SHA256": ssl_metadata["encoder_checkpoint_sha256"],
    "SSL_EPOCH": ssl_metadata["epochs"],
    "SSL_CONFIG_SHA256": ssl_metadata["source_config_sha256"],
    "UNLABELED_MANIFEST_SHA256": ssl_metadata["unlabeled_manifest_sha256"],
    "TEST_LOADER_INSTANTIATED": False,
})
"""),
    markdown("## 8. Reconcile resolved config, supervised state, and strict-load the P3 encoder"),
    code(r"""
# This cell is also the supported recovery entry point in an already-running
# kernel. Reload edited local modules so cell 8 onward cannot retain the stale
# implementations imported before the resolved-config reconciliation fix.
import importlib
import soilnet.models.factory as _factory_module
import soilnet.models as _models_module
import soilnet.training.engine as _engine_module
import soilnet.training.vicreg as _vicreg_module
import soilnet.utils.experiment as _experiment_module

importlib.invalidate_caches()
_factory_module = importlib.reload(_factory_module)
_models_module = importlib.reload(_models_module)
_experiment_module = importlib.reload(_experiment_module)
_vicreg_module = importlib.reload(_vicreg_module)
_engine_module = importlib.reload(_engine_module)

build_frozen_model = _models_module.build_frozen_model
build_model = _models_module.build_model
build_val_loader = _engine_module.build_val_loader
run_one_batch_preflight = _engine_module.run_one_batch_preflight
train_experiment = _engine_module.train_experiment
load_experiment_context = _experiment_module.load_experiment_context
reconcile_resolved_p3_config = _vicreg_module.reconcile_resolved_p3_config
validate_or_archive_stale_supervised_state = _vicreg_module.validate_or_archive_stale_supervised_state

stale_archive_root = run_root.parent / "soilnet_stale_p3"
resolved_reconciliation = reconcile_resolved_p3_config(
    config=p3,
    config_sha256=p3_source_config_sha256,
    experiment_root=experiment_root,
    stale_archive_root=stale_archive_root,
)
resolved_config = Path(resolved_reconciliation["resolved_config_path"])
print("resolved_config diff")
display(pd.DataFrame(resolved_reconciliation["field_diff"], columns=[
    "key", "existing_value", "current_expected_value"
]))
print(json.dumps(resolved_reconciliation, indent=2))
context = load_experiment_context(resolved_config)
resolved = context.config
if context.config["ssl_checkpoint"]["sha256"] != ssl_metadata["encoder_checkpoint_sha256"]:
    raise RuntimeError("STOP: resolved context does not reference the verified epoch-80 SSL checkpoint")

supervised_state_resolution = validate_or_archive_stale_supervised_state(
    resolved_config=context.config,
    resolved_config_sha256=context.config_sha256,
    experiment_root=experiment_root,
    stale_archive_root=stale_archive_root,
)
print(json.dumps({"SUPERVISED_STATE_RESOLUTION": supervised_state_resolution}, indent=2))

strict_preflight = run_one_batch_preflight(context, device="cuda")
if strict_preflight["status"] != "PASS" or strict_preflight["vicreg_checkpoint_loaded"] is not True:
    raise RuntimeError("STOP: strict P3 encoder load or supervised one-batch preflight failed")
write_json(experiment_root / "supervised_preflight_report.json", strict_preflight)
print({
    "experiment_id": context.config["experiment_id"],
    "model_name": context.config["timm_model_name"],
    "initialization_source": strict_preflight["load_report"]["source_kind"],
    "VICREG_CHECKPOINT_SHA256": strict_preflight["load_report"]["checkpoint_sha256"],
    "matched_backbone_keys": strict_preflight["load_report"]["matched_keys"],
    "parameters": parameter_count,
    "seed": context.config["seed"],
    "epochs": context.config["epochs"],
    "batch_size": context.config["batch_size"],
    "optimizer": context.config["optimizer"],
    "learning_rate": context.config["learning_rate"],
    "LI_enabled": context.config["use_li"],
    "TEST_LOADER_INSTANTIATED": False,
})
"""),
    markdown("## 9. Supervised fine-tuning — 60 epochs, validation BESTREG, no early stopping"),
    code(r"""
supervised_started = time.monotonic()
completed_metadata_path = supervised_dir / "run_metadata.json"
if completed_metadata_path.is_file():
    run_metadata = json.loads(completed_metadata_path.read_text(encoding="utf-8"))
    if run_metadata.get("training_completed") is not True:
        raise RuntimeError("STOP: invalid existing supervised metadata")
    if sha256_file(Path(run_metadata["primary_checkpoint_path"])) != run_metadata["primary_checkpoint_sha256"]:
        raise RuntimeError("STOP: existing P3 best-checkpoint SHA256 mismatch")
    supervised_action = "STRICT_VERIFIED_COMPLETED_REUSE"
else:
    run_metadata = train_experiment(context, resume_if_available=RESUME_SUPERVISED_IF_AVAILABLE)
    supervised_action = "TRAINED_OR_RESUMED_TO_60_EPOCHS"
supervised_elapsed_this_run = time.monotonic() - supervised_started
if run_metadata["epochs_completed"] != 60:
    raise RuntimeError("STOP: P3 supervised training did not complete all 60 epochs")
print({
    "SUPERVISED_ACTION": supervised_action,
    "epochs_completed": run_metadata["epochs_completed"],
    "best_epoch": run_metadata["best_epoch"],
    "best_mean_validation_RMSE": run_metadata["best_mean_validation_RMSE"],
    "best_checkpoint": run_metadata["primary_checkpoint_path"],
    "best_checkpoint_sha256": run_metadata["primary_checkpoint_sha256"],
    "TEST_LOADER_INSTANTIATED": False,
})
"""),
    markdown("## 10. Validation history and best-checkpoint selection audit"),
    code(r"""
history = pd.read_csv(supervised_dir / "training_history.csv")
if len(history) != 60 or history["epoch"].tolist() != list(range(1, 61)):
    raise RuntimeError("STOP: supervised history is not exactly 60 sequential epochs")
minimum_index = history["mean_regression_RMSE"].idxmin()
minimum_row = history.loc[minimum_index]
if int(minimum_row["epoch"]) != int(run_metadata["best_epoch"]):
    raise RuntimeError("STOP: saved best epoch is not the strict minimum validation mean RMSE")
display(history[[
    "epoch", "total_loss", "regression_loss", "classification_loss",
    "validation_SM_0_rmse", "validation_SM_0_mae", "validation_SM_0_r2",
    "validation_SM_20_rmse", "validation_SM_20_mae", "validation_SM_20_r2",
    "mean_regression_RMSE", "is_best_regression",
]])
print({
    "best_epoch": int(minimum_row["epoch"]),
    "minimum_validation_mean_RMSE": float(minimum_row["mean_regression_RMSE"]),
    "selection_formula": context.config["selection_formula"],
    "early_stopping_used": False,
})
"""),
    markdown("## 11. Reload best checkpoint from disk and reproduce validation metrics"),
    code(r"""
best_path = Path(run_metadata["primary_checkpoint_path"])
observed_best_sha = sha256_file(best_path)
if observed_best_sha != run_metadata["primary_checkpoint_sha256"]:
    raise RuntimeError("STOP: validation-best checkpoint SHA256 mismatch")
payload = torch.load(best_path, map_location="cuda", weights_only=True)
verified_model = build_frozen_model(context.config).to("cuda")
verified_model.load_state_dict(payload["model_state_dict"], strict=True)
validation_loader = build_val_loader(context)
_, reproduced_metrics = prediction_rows_and_metrics(
    verified_model, validation_loader, torch.device("cuda"), context.config
)
saved_metrics = json.loads((supervised_dir / "validation_metrics.json").read_text(encoding="utf-8"))
metric_checks = {}
for target in ("SM_0", "SM_20"):
    for metric in ("rmse", "mae", "r2"):
        key = f"{target}_{metric}"
        difference = abs(reproduced_metrics["regression"][target][metric] - saved_metrics["regression"][target][metric])
        metric_checks[key] = difference
        if difference > 1e-10:
            raise RuntimeError(f"STOP: reproduced validation metric mismatch for {key}: {difference}")
artifact_verification = {
    "status": "PASS",
    "checkpoint_path": str(best_path),
    "checkpoint_sha256": observed_best_sha,
    "checkpoint_epoch": int(payload["epoch"]),
    "validation_metric_absolute_differences": metric_checks,
    "validation_sample_count": reproduced_metrics["n"],
    "test_loader_instantiated": False,
    "test_images_loaded": 0,
}
write_json(experiment_root / "artifact_verification.json", artifact_verification)
print(json.dumps(artifact_verification, indent=2))
del verified_model, validation_loader, payload
torch.cuda.empty_cache()
"""),
    markdown("## 12. P2 ImageNet-only vs P3 MobileViTv2 + VICReg on validation"),
    code(r"""
p2_metrics = json.loads((p2_dir / "validation_metrics.json").read_text(encoding="utf-8"))
p2_metadata = json.loads((p2_dir / "run_metadata.json").read_text(encoding="utf-8"))
p3_metrics = json.loads((supervised_dir / "validation_metrics.json").read_text(encoding="utf-8"))
comparison = comparison_rows(
    p2_metrics,
    p3_metrics,
    (int(p2_metadata["complexity"]["parameters"]), int(p2_metadata["best_epoch"])),
    (int(run_metadata["complexity"]["parameters"]), int(run_metadata["best_epoch"])),
)
comparison_table = pd.DataFrame(comparison)
display(comparison_table)
comparison_table.to_csv(experiment_root / "validation_comparison.csv", index=False)
write_json(experiment_root / "validation_comparison.json", comparison)
p2_mean = comparison[0]["mean_RMSE"]
p3_mean = comparison[1]["mean_RMSE"]
if p3_mean < p2_mean:
    comparison_statement = "P3 has lower validation mean RMSE than P2 in this run."
elif p3_mean > p2_mean:
    comparison_statement = "P3 does not have lower validation mean RMSE than P2 in this run."
else:
    comparison_statement = "P2 and P3 have equal validation mean RMSE in this run."
print(comparison_statement)
"""),
    markdown("## 13. Artifact preservation check and FINAL RUN SUMMARY"),
    code(r"""
protected_after = {name: hash_tree(run_root / name) for name in PROTECTED_EXPERIMENTS}
if protected_before != protected_after:
    raise RuntimeError("STOP: a protected P0/P1/P2 artifact changed during P3")
write_json(experiment_root / "protected_artifacts_after_notebook.json", protected_after)

all_artifacts = sorted(str(path) for path in experiment_root.rglob("*") if path.is_file())
final_summary = {
    "status": "COMPLETE",
    "experiment_id": p3["experiment_id"],
    "SSL_checkpoint_path": ssl_metadata["encoder_checkpoint_path"],
    "SSL_checkpoint_SHA256": ssl_metadata["encoder_checkpoint_sha256"],
    "supervised_best_checkpoint_path": run_metadata["primary_checkpoint_path"],
    "supervised_best_checkpoint_SHA256": run_metadata["primary_checkpoint_sha256"],
    "best_epoch": run_metadata["best_epoch"],
    "minimum_validation_mean_RMSE": run_metadata["best_mean_validation_RMSE"],
    "parameters": run_metadata["complexity"]["parameters"],
    "SSL_training_seconds": ssl_metadata["training_duration_seconds"],
    "supervised_training_seconds": run_metadata["training_duration_seconds"],
    "git_commit": current_git_commit(REPO),
    "artifacts": all_artifacts,
    "P0_P1_P2_ARTIFACTS_UNCHANGED": True,
    "TEST_SET_OPENED": False,
    "TEST_LOADER_INSTANTIATED": False,
}
write_json(experiment_root / "FINAL_RUN_SUMMARY.json", final_summary)
print("=" * 80)
print("FINAL RUN SUMMARY")
print(json.dumps(final_summary, indent=2))
"""),
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(book, OUTPUT)
print(OUTPUT)
