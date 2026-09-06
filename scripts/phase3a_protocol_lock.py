#!/usr/bin/env python3
"""Build the PHASE 3A evidence lock without training or reading test metrics."""
from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import timm
import torch

from soilnet.io import resolve_paths, sha256_file, write_csv, write_json
from soilnet.final_protocol import normalize_historical_soilnet_state
from soilnet.models import SoilNetDualHead

SELECTED_RELATIVE = "checkpoints_VicReg_original_NEW/vicreg_model_final_mu_27.0.pth"
SELECTED_SHA256 = "42599ac025b8d8c8c5b8cca36624665f155e0ff588cd90de2fd47b08cd2d60d8"
EPOCH80_RELATIVE = "checkpoints_VicReg_original_NEW/vicreg_mu_27.0_epoch_80.pth"
METRICS_RELATIVE = "checkpoints_VicReg_original_NEW/vicreg_metrics_original_soilnet_mu_27.csv"
NOTEBOOK_RELATIVE = "checkpoints_Vicreg_original_NEW/VicReg_SoiNet_optimized_mu_27.ipynb"
DATA_NOTEBOOK_RELATIVE = (
    "Soil_Labeled_Data/soilNet_di chuyen tu o C/Pretrain_VicReg_New/"
    "VicReg_SoiNet_optimized_mu_27.ipynb"
)
SEED = 20260905


def read_csv(relative: str) -> list[dict[str, str]]:
    with (REPO / relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def truth(value: object) -> bool:
    return str(value).casefold() == "true"


def verify_selected_lineage(paths: dict[str, Path]) -> dict[str, Any]:
    final_path = paths["checkpoint_root"] / SELECTED_RELATIVE
    epoch_path = paths["checkpoint_root"] / EPOCH80_RELATIVE
    metrics_path = paths["checkpoint_root"] / METRICS_RELATIVE
    legacy_notebook = paths["legacy_root"] / NOTEBOOK_RELATIVE
    data_notebook = paths["data_root"] / DATA_NOTEBOOK_RELATIVE
    for path in (final_path, epoch_path, metrics_path, legacy_notebook, data_notebook):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(final_path) != SELECTED_SHA256:
        raise RuntimeError("Selected SSL checkpoint SHA256 changed")
    final_state = torch.load(final_path, map_location="cpu", weights_only=True)
    epoch_state = torch.load(epoch_path, map_location="cpu", weights_only=True)
    embedded_state = epoch_state.get("model_state_dict") if isinstance(epoch_state, dict) else None
    if not isinstance(final_state, dict) or not isinstance(embedded_state, dict):
        raise RuntimeError("Selected/final epoch checkpoint state is not a safe state_dict")
    tensor_identity = set(final_state) == set(embedded_state) and all(
        torch.equal(final_state[key], embedded_state[key]) for key in final_state
    )
    canonical = SoilNetDualHead(num_classes=10, use_light=True, backbone_pretrained=False).state_dict()
    _, normalization = normalize_historical_soilnet_state(final_state, canonical)
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        metrics = list(csv.DictReader(handle))
    final_metric = metrics[-1] if metrics else {}
    metric_identity = (
        len(metrics) == 80
        and str(epoch_state.get("epoch")) == "80"
        and final_metric.get("mu") == "27.0"
        and final_metric.get("epoch") == "80"
        and float(final_metric.get("vicreg_loss", "nan")) == float(epoch_state.get("vicreg_loss", "nan"))
    )
    notebook_identity = sha256_file(legacy_notebook) == sha256_file(data_notebook)
    if not (tensor_identity and metric_identity and notebook_identity):
        raise RuntimeError("Independent SSL lineage checks did not all pass")
    return {
        "final_equals_epoch80_model_state": tensor_identity,
        "epoch80_metric_matches_csv": metric_identity,
        "legacy_and_data_notebook_sha_match": notebook_identity,
        "epoch": 80,
        "metric_rows": 80,
        "notebook_sha256": sha256_file(legacy_notebook),
        "metrics_sha256": sha256_file(metrics_path),
        "epoch80_checkpoint_sha256": sha256_file(epoch_path),
        "canonical_compatibility_adapter": normalization,
    }


def build_ssl_candidates(paths: dict[str, Path], lineage: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    classified = read_csv("results/audit/checkpoint_classification.csv")
    for row in classified:
        combined = f"{row['relative_path']}/{row['filename']}".casefold()
        if row.get("suspected_mu") not in {"27", "27.0"} or "vicreg" not in combined:
            continue
        scope = row["scope"]
        relative = row["relative_path"]
        category = row["primary_category"]
        tiny = "tiny" in combined
        original_generation = "original_new" in combined or (
            scope == "DATA_ROOT_EMBEDDED_ARTIFACTS" and "checkpoints_vicreg/" in combined
        ) or (scope == "DATA_ROOT_EMBEDDED_ARTIFACTS" and "finetune_update/" in combined and not tiny)
        encoder = truth(row.get("model_compatible"))
        projector = category == "VICREG_PROJECTOR" or "projector_state_dict" in row.get("top_level_keys", "")
        selected = scope == "CHECKPOINT_ROOT" and relative == SELECTED_RELATIVE
        if selected:
            selection = "SELECTED_FOR_FINAL_REVALIDATION"
            reason = (
                "Safe compatible raw SoilNet state; identical tensor-for-tensor to epoch-80 model_state_dict; "
                "epoch/loss matches the 80-row mu=27 CSV; save notebook is byte-identical across DATA_ROOT and pinned legacy; "
                "269 verified legacy mobilevit_full alias keys are removed before strict canonical loading."
            )
            lineage_status = "VERIFIED_LINEAGE"
            epoch = 80
        elif tiny:
            selection = "REJECTED_INITIALIZATION_MISMATCH"
            reason = "TinyImageNet generation does not match the manuscript-preselected ImageNet-initialized SoilNet VICReg chain."
            lineage_status = "HIGH_CONFIDENCE_LINEAGE"
            epoch = row.get("suspected_epoch", "")
        elif category == "FINETUNED_SOILNET_DUALHEAD":
            selection = "REJECTED_POST_FINETUNE_ARTIFACT"
            reason = "Artifact is downstream fine-tuned state, not the pre-finetuning SSL encoder."
            lineage_status = "HIGH_CONFIDENCE_LINEAGE"
            epoch = row.get("suspected_epoch", "")
        elif category == "VICREG_PROJECTOR" or not encoder:
            selection = "REJECTED_NO_ENCODER"
            reason = "Projector/head-only state cannot initialize the final SoilNet encoder."
            lineage_status = "VERIFIED_LINEAGE" if "MATCHES_GITHUB" in row.get("additional_flags", "") else "HIGH_CONFIDENCE_LINEAGE"
            epoch = row.get("suspected_epoch", "")
        elif category == "PARTIAL_STATE":
            selection = "REJECTED_INTERMEDIATE_SSL_EPOCH"
            reason = "Intermediate SSL training state is superseded by the tensor-identical final epoch-80 raw model state."
            lineage_status = "HIGH_CONFIDENCE_LINEAGE"
            epoch = row.get("suspected_epoch", "")
        elif original_generation:
            selection = "REJECTED_ALTERNATE_GENERATION"
            reason = "Compatible original/ImageNet VICReg generation, but lacks the selected Scope-A epoch-80 plus exact metrics chain."
            lineage_status = "HIGH_CONFIDENCE_LINEAGE"
            epoch = row.get("suspected_epoch", "")
        else:
            selection = "REJECTED_INSUFFICIENT_EVIDENCE"
            reason = "Artifact does not satisfy the complete pre-finetuning encoder lineage gate."
            lineage_status = "AMBIGUOUS"
            epoch = row.get("suspected_epoch", "")
        if scope == "CHECKPOINT_ROOT" and "original_new" in combined:
            notebook = f"LEGACY_GITHUB:{NOTEBOOK_RELATIVE}"
            metrics = f"CHECKPOINT_ROOT:{METRICS_RELATIVE}"
        elif tiny:
            notebook = "LEGACY_GITHUB:Vicreg_soiNet_tinyImagenet_mu=27.ipynb"
            metrics = "UNMAPPED_OR_TRAINING_ONLY"
        else:
            notebook = "UNMAPPED_GENERATION"
            metrics = "UNMAPPED_GENERATION"
        rows.append({
            "candidate_id": f"{scope}:{row['sha256'][:20]}", "path": f"{scope}:{relative}",
            "scope": scope, "sha256": row["sha256"], "category": category, "mu": "27",
            "imagenet_initialized": bool(original_generation and not tiny), "ssl_method": "VICReg",
            "epoch_if_known": epoch, "encoder_present": encoder, "projector_present": projector,
            "compatible_with_final_soilnet": encoder, "lineage_status": lineage_status,
            "linked_notebook": notebook, "linked_metrics": metrics,
            "selection_status": selection, "reason": reason,
        })

    phase1 = read_csv("results/audit/checkpoint_inventory.csv")
    for row in phase1:
        combined = f"{row['relative_path']}/{row['filename']}".casefold()
        if row["source"] != "GITHUB" or row.get("mu_if_available") not in {"27", "27.0"} or "vicreg" not in combined:
            continue
        projector = row.get("compatible_model") == "VICREG_PROJECTOR_KEY_PATTERN"
        encoder = row.get("compatible_model") == "SOILNET_KEY_PATTERN"
        tiny = "tiny" in combined
        rows.append({
            "candidate_id": f"LEGACY_GITHUB:{row['sha256'][:20]}",
            "path": f"LEGACY_GITHUB:{row['relative_path']}", "scope": "LEGACY_GITHUB",
            "sha256": row["sha256"], "category": "VICREG_PROJECTOR" if projector else "SSL_BACKBONE",
            "mu": "27", "imagenet_initialized": False if tiny else True, "ssl_method": "VICReg",
            "epoch_if_known": row.get("epoch_if_available", ""), "encoder_present": encoder,
            "projector_present": projector, "compatible_with_final_soilnet": encoder,
            "lineage_status": "VERIFIED_LINEAGE" if row["status"] == "VERIFIED_IDENTICAL" else "HIGH_CONFIDENCE_LINEAGE",
            "linked_notebook": "LEGACY_GITHUB:Vicreg_soiNet_tinyImagenet_mu=27.ipynb" if tiny else f"LEGACY_GITHUB:{NOTEBOOK_RELATIVE}",
            "linked_metrics": "UNMAPPED_OR_TRAINING_ONLY" if tiny else f"LEGACY_GITHUB:{METRICS_RELATIVE}",
            "selection_status": "REJECTED_INITIALIZATION_MISMATCH" if tiny else "REJECTED_NO_ENCODER",
            "reason": "TinyImageNet initialization mismatch." if tiny else "Projector-only artifact; no encoder state.",
        })
    selected = [row for row in rows if row["selection_status"] == "SELECTED_FOR_FINAL_REVALIDATION"]
    if len(selected) != 1 or selected[0]["sha256"] != SELECTED_SHA256:
        raise RuntimeError("Exactly one selected SSL checkpoint was not established")
    rows.sort(key=lambda row: (row["selection_status"] != "SELECTED_FOR_FINAL_REVALIDATION", row["scope"], row["path"]))
    write_csv(REPO / "results/audit/final_ssl_checkpoint_candidates.csv", rows, [
        "candidate_id", "path", "scope", "sha256", "category", "mu", "imagenet_initialized",
        "ssl_method", "epoch_if_known", "encoder_present", "projector_present",
        "compatible_with_final_soilnet", "lineage_status", "linked_notebook", "linked_metrics",
        "selection_status", "reason",
    ])
    return rows


def audit_duplicate_labels() -> dict[str, Any]:
    samples = read_csv("data/manifests/soilnet_samples.csv")
    splits = read_csv("data/splits/final_split.csv")
    split_by_id = {row["sample_id"]: row for row in splits}
    phase2_exact = read_csv("results/audit/labeled_exact_duplicates.csv")
    exact_id_by_sha = {row["sha256"]: row["duplicate_group_id"] for row in phase2_exact}
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in samples:
        groups[row["sha256"]].append(row)
    output = []
    label_fields = ("SM_0", "SM_20", "light_value", "moisture_class", "moisture_bin", "light_bin")
    for digest, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        values = {field: sorted({row.get(field, "").strip() for row in members}) for field in label_fields}
        soil_types = sorted({
            parts[3] if len(parts := row["relative_path"].split("/")) > 4 else "UNAVAILABLE"
            for row in members
        })
        group_ids = sorted({split_by_id[row["sample_id"]]["group_id"] for row in members})
        conflicting = [field for field, field_values in values.items() if len(field_values) > 1]
        if len(soil_types) > 1:
            conflicting.append("soil_type")
        missing = [field for field, field_values in values.items() if "" in field_values]
        missing.extend(["capture_session", "source_group"])
        status = "CONFLICTING_LABEL" if conflicting else ("MISSING_METADATA" if missing else "CONSISTENT")
        output.append({
            "duplicate_group_id": exact_id_by_sha.get(digest, f"sha256:{digest[:16]}"),
            "sha256": digest, "group_size": len(members), "classification": status,
            "conflicting_fields": ";".join(conflicting), "missing_metadata_fields": ";".join(missing),
            "SM_0_values": ";".join(values["SM_0"]), "SM_20_values": ";".join(values["SM_20"]),
            "light_value_values": ";".join(values["light_value"]),
            "moisture_class_values": ";".join(values["moisture_class"]),
            "moisture_bin_values": ";".join(values["moisture_bin"]),
            "light_bin_values": ";".join(values["light_bin"]), "soil_type_values": ";".join(soil_types),
            "capture_session_values": "UNAVAILABLE_NOT_IN_LABELS",
            "source_group_values": "UNAVAILABLE_NOT_IN_LABELS",
            "phase2_near_group_ids": ";".join(group_ids),
            "member_sample_ids": ";".join(sorted(row["sample_id"] for row in members)),
            "member_relative_paths": ";".join(sorted(row["relative_path"] for row in members)),
            "resolution": "DO_NOT_AUTO_RESOLVE" if status == "CONFLICTING_LABEL" else "NONE_REQUIRED",
        })
    if len(output) != 46:
        raise RuntimeError(f"Expected 46 exact duplicate groups, found {len(output)}")
    write_csv(REPO / "results/audit/exact_duplicate_label_consistency.csv", output, [
        "duplicate_group_id", "sha256", "group_size", "classification", "conflicting_fields",
        "missing_metadata_fields", "SM_0_values", "SM_20_values", "light_value_values",
        "moisture_class_values", "moisture_bin_values", "light_bin_values", "soil_type_values",
        "capture_session_values", "source_group_values", "phase2_near_group_ids", "member_sample_ids",
        "member_relative_paths", "resolution",
    ])
    counts = Counter(row["classification"] for row in output)
    conflict_fields = Counter(
        field for row in output for field in str(row["conflicting_fields"]).split(";") if field
    )
    return {
        "groups": len(output), "classification_counts": dict(counts),
        "conflicts": counts["CONFLICTING_LABEL"], "conflict_field_counts": dict(conflict_fields),
    }


def write_protocol_files(paths: dict[str, Path], duplicates: dict[str, Any], gpu_status: str) -> dict[str, Any]:
    preselection = {
        "architecture": "SoilNet",
        "architecture_detail": "modified MobileViTV2 + MobileNetV2 components + light-intensity input",
        "ssl_method": "VICReg", "imagenet_initialization": True, "mu": 27,
        "selection_source": "historical_manuscript",
        "selection_source_availability": "USER_SUPPLIED_PHASE3A_ASSERTION; manuscript file not present in audited repository",
        "selection_time_relation": "before prospective split evaluation",
        "prospective_metrics_consulted": False,
        "status": "PRESELECTION_RESOLVED_FROM_HISTORICAL_MANUSCRIPT",
    }
    write_json(REPO / "results/audit/final_preselection.json", preselection)

    config = f"""experiment_id: FINAL_LEAKAGE_CONTROLLED_REVALIDATION
protocol_status: BLOCKED_DATA_CONFLICT
training_allowed: false
preselection:
  status: PRESELECTION_RESOLVED_FROM_HISTORICAL_MANUSCRIPT
  selection_source: historical_manuscript
  selection_time_relation: before prospective split evaluation
  prospective_metrics_consulted: false
  architecture: SoilNet
  ssl_method: VICReg
  imagenet_initialization: true
  mu: 27
initialization:
  scope: CHECKPOINT_ROOT
  relative_path: {SELECTED_RELATIVE}
  sha256: {SELECTED_SHA256}
  lineage_status: VERIFIED_LINEAGE
  checkpoint_load_policy: all_577_canonical_keys_strict_after_verified_alias_removal
  head_initialization_note: regression_classification_and_light_weights_are_loaded_from_the_historical_state
  ssl_pretraining_data: SOIL_IMAGES_UNLABELED
  protocol_disclosure: TRANSDUCTIVE_PRETRAINING
data:
  historical_labeled_records: 2057
  expected_sha256_unique_images: 1973
  exact_duplicate_groups: 46
  conflicting_exact_duplicate_groups: {duplicates['conflicts']}
  final_unique_manifest: null
  final_unique_manifest_status: NOT_GENERATED_BLOCKED_DATA_CONFLICT
  final_split_manifest: null
  final_split_status: NOT_GENERATED_BLOCKED_DATA_CONFLICT
  representative_rule: lexicographically_first_normalized_relative_path
split:
  seed: {SEED}
  target_ratios:
    train: 0.73
    validation: 0.15
    test: 0.12
  grouping: exact_sha256_plus_near_duplicate_connected_components_plus_documented_source_or_session
  metric_free_assignment: true
model:
  regression_heads: [SM_0, SM_20]
  classification_head: moisture_class
  num_classes: 10
  light_branch: true
  regression_output_dimension: 2
training:
  epochs: 15
  primary_checkpoint: epoch_15_final
  secondary_validation_best_checkpoint: disabled
  batch_size: 32
  learning_rate: 0.0005
  optimizer: Adam
  weight_decay: 0.0
  weight_decay_source: historical_mu27_code_Adam_default
  regression_loss: MSELoss
  classification_loss: CrossEntropyLoss
  regression_loss_weight: 1.0
  classification_loss_weight: 1.0
  total_loss: regression_loss_plus_classification_loss
  loss_weight_source: historical_mu27_code
input:
  image_size: [224, 224]
  image_normalization_mean: [0.485, 0.456, 0.406]
  image_normalization_std: [0.229, 0.224, 0.225]
  light_internal_scale: divide_by_100
  regression_target_internal_scale: divide_by_100
metrics:
  regression: [RMSE, MAE, R2, ME]
  classification: [Accuracy, Macro-F1, Macro-Precision, Macro-Recall, confusion_matrix]
  regression_report_scale: percentage_points_0_to_100
  classification_report_scale: ratio_0_to_1
test_firewall:
  test_loader_in_training: false
  test_metrics_during_training: false
  final_test_command_separate: true
  evaluate_test_exactly_once: true
  completion_marker: results/final/test_evaluation_completed.json
reproducibility:
  seed: {SEED}
  pythonhashseed: {SEED}
  cudnn_benchmark: false
  cudnn_deterministic: true
  deterministic_algorithms: true
runtime_gate:
  gpu_status: {gpu_status}
  require_cuda_for_training: true
"""
    config_path = REPO / "config/experiments/final_revalidation.yaml"
    config_path.write_text(config, encoding="utf-8")

    protocol = f"""# Final revalidation protocol

Status: **BLOCKED_DATA_CONFLICT**. This document locks choices but does not authorize training.

## Preselection

- Architecture: SoilNet modified MobileViTV2 + MobileNetV2 with light-intensity input.
- SSL: VICReg, ImageNet-initialized chain, μ=27.
- Source: historical manuscript selection supplied before prospective evaluation.
- Prospective validation/test metrics consulted: no.

## Selected SSL initialization

- Scope/path: `CHECKPOINT_ROOT:{SELECTED_RELATIVE}`
- SHA256: `{SELECTED_SHA256}`
- Lineage: `VERIFIED_LINEAGE` for the SSL generation. The raw final state is tensor-identical to the epoch-80 model state; its epoch/loss matches the 80-row μ=27 CSV; the generating notebook is byte-identical in DATA_ROOT and the pinned legacy checkout.
- Compatibility adapter: all 577 canonical keys and shapes are required. Exactly 269 historical `mobilevit_full.*` registration aliases are removed; 261 duplicated stage tensors must equal their `mobilevit_encoder.*` counterparts. Loading remains strict after normalization.
- Load policy: reproduce the historical fine-tune behavior by loading all 577 canonical matching keys, including the regression, classification, and LI branches. These heads were present in the SSL file but were not optimized by the VICReg objective; this limitation is explicit rather than silently reinitializing them.
- Pretraining exposure: soil unlabeled pool; disclose as `TRANSDUCTIVE_PRETRAINING`, not completely held-out inductive pretraining.

## Locked training protocol

- 15 epochs; batch 32; Adam; learning rate 5e-4; weight decay 0.0.
- Loss = MSE(SM_0, SM_20) + CrossEntropy(moisture_class), weights 1.0 and 1.0 from historical μ=27 code.
- Primary checkpoint is the fixed epoch-15 state. No test-based selection and no μ/architecture/SSL search.
- Images resize to 224×224 and use ImageNet normalization. LI and regression targets are divided by 100 internally.

## Data and test firewall

All 46 exact-SHA duplicate groups have conflicting labels. Therefore `final_unique_manifest_v1.csv` and `final_unique_split_v1.csv` are intentionally not generated. Conflicts require an external, evidence-backed adjudication; lexicographic representative selection cannot resolve labels.

After a new experiment ID and valid locked manifests exist, training may instantiate only train and validation loaders. The test loader exists only in a separate explicit final-evaluation command. Test evaluation must refuse to run when `results/final/test_evaluation_completed.json` already exists and must write that marker with timestamp, Git commit, model SHA256, split SHA256, and config SHA256 after the one permitted evaluation.
"""
    (REPO / "docs/FINAL_REVALIDATION_PROTOCOL.md").write_text(protocol, encoding="utf-8")

    metric_spec = """# Final metric specification

This specification is locked before any final training or prospective metric access.

| Task | Target | Metrics | Reported scale |
|---|---|---|---|
| Regression | SM_0 | RMSE, MAE, R2, ME/bias | percentage points, original 0–100 target scale |
| Regression | SM_20 | RMSE, MAE, R2, ME/bias | percentage points, original 0–100 target scale |
| Classification | moisture_class (10 classes, 0–9) | Accuracy, Macro-F1, Macro-Precision, Macro-Recall, confusion matrix | ratios 0–1; confusion matrix as counts |

The model receives regression targets as `value / 100` and LI as `light_value / 100`. Regression predictions and truths are multiplied by 100 exactly once before reporting RMSE, MAE, and ME. Thus an internal error of 0.086 corresponds to 8.6 percentage points; the two values must never be reported interchangeably. R2 is dimensionless. ME is `mean(prediction - truth)` and retains sign.

Metrics are computed separately for SM_0 and SM_20. No combined regression/classification score is a primary manuscript metric. Validation metrics may monitor the fixed protocol but do not change the epoch-15 primary checkpoint. Test metrics are produced once, only by the separate final evaluation command.
"""
    (REPO / "docs/FINAL_METRIC_SPEC.md").write_text(metric_spec, encoding="utf-8")

    split_report = f"""# Final unique split report

Status: **NOT_GENERATED — BLOCKED_DATA_CONFLICT**

- Historical labeled records: 2,057
- SHA256-unique image bytes: 1,973
- Exact duplicate groups audited: {duplicates['groups']}
- Conflicting exact duplicate groups: {duplicates['conflicts']}
- Conflict fields by group: {json.dumps(duplicates['conflict_field_counts'], sort_keys=True)}
- Canonical unique manifest: not generated
- Final train/validation/test counts: unavailable
- Group leakage: not testable because the split was not generated
- Near-duplicate cross-split leakage: not testable because the split was not generated
- Distribution audit: deferred until label conflicts are resolved with external evidence

The source files were not modified. No representative was selected because byte-identical images carry conflicting SM_0, SM_20, LI and, in many groups, moisture-class labels. Resolving this by path order would silently choose scientific labels and is forbidden.
"""
    (REPO / "results/audit/final_split_report.md").write_text(split_report, encoding="utf-8")
    return preselection


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNCOMMITTED"


def write_run_lock(duplicates: dict[str, Any], lineage: dict[str, Any], gpu_status: str) -> dict[str, Any]:
    config_path = REPO / "config/experiments/final_revalidation.yaml"
    dry_run_path = REPO / "results/audit/phase3a_dry_run.json"
    lock = {
        "experiment_id": "FINAL_LEAKAGE_CONTROLLED_REVALIDATION",
        "lock_state": "INACTIVE_BLOCKED_DATA_CONFLICT", "training_allowed": False,
        "git_commit": git_commit(), "python": platform.python_version(), "pytorch": torch.__version__,
        "timm": timm.__version__, "seed": SEED,
        "deterministic_settings": {
            "PYTHONHASHSEED": str(SEED), "torch_manual_seed": SEED,
            "cudnn_benchmark": False, "cudnn_deterministic": True,
            "torch_use_deterministic_algorithms": True,
        },
        "source_labeled_manifest_sha256": sha256_file(REPO / "data/manifests/soilnet_samples.csv"),
        "dataset_manifest_sha256": "NOT_GENERATED_BLOCKED_DATA_CONFLICT",
        "split_manifest_sha256": "NOT_GENERATED_BLOCKED_DATA_CONFLICT",
        "ssl_checkpoint_scope": "CHECKPOINT_ROOT", "ssl_checkpoint_relative_path": SELECTED_RELATIVE,
        "ssl_checkpoint_sha256": SELECTED_SHA256, "config_sha256": sha256_file(config_path),
        "duplicate_consistency_sha256": sha256_file(REPO / "results/audit/exact_duplicate_label_consistency.csv"),
        "code_sha256": {
            "train_entrypoint": sha256_file(REPO / "scripts/train_final_revalidation.py"),
            "test_entrypoint": sha256_file(REPO / "scripts/evaluate_final_test.py"),
            "model": sha256_file(REPO / "src/soilnet/models/soilnet.py"),
            "dataset": sha256_file(REPO / "src/soilnet/data/dataset.py"),
            "metrics": sha256_file(REPO / "src/soilnet/evaluation/metrics.py"),
            "firewall": sha256_file(REPO / "src/soilnet/final_protocol.py"),
        },
        "conflicting_exact_duplicate_groups": duplicates["conflicts"], "gpu_gate": gpu_status,
        "dry_run_status": json.loads(dry_run_path.read_text(encoding="utf-8")).get("status") if dry_run_path.exists() else "NOT_RUN",
        "ssl_lineage_checks": lineage,
        "change_policy": "Resolve data conflicts with evidence, create a new experiment ID, then generate and lock new manifest/split/config artifacts before training.",
    }
    write_json(REPO / "results/audit/final_run_lock.json", lock)
    return lock


def write_readiness(duplicates: dict[str, Any], lock: dict[str, Any], gpu_status: str, candidate_count: int) -> None:
    lines = [
        "# PHASE 3A RUN READINESS", "",
        "| Item | Answer | Evidence / action |", "|---|---|---|",
        "| 1. μ preselection resolved? | YES — `PRESELECTION_RESOLVED_FROM_HISTORICAL_MANUSCRIPT` | Locked before prospective evaluation |",
        "| 2. Evidence source? | Historical manuscript assertion supplied for PHASE 3A | Manuscript file itself is not in audited roots |",
        f"| 3. Selected SSL μ=27 checkpoint? | YES — `CHECKPOINT_ROOT:{SELECTED_RELATIVE}` | One selected among {candidate_count} μ=27 VICReg-related candidates |",
        f"| 4. SHA256? | `{SELECTED_SHA256}` | Safe-load plus independent lineage checks |",
        f"| 5. 2,057 duplicate labels consistent? | NO — {duplicates['conflicts']}/46 exact groups conflict | Every group conflicts on SM_0 and SM_20; do not auto-resolve |",
        "| 6. Unique final count? | NOT GENERATED (1,973 byte-unique was expected) | Data conflict gate stopped generation |",
        "| 7. Final train/val/test counts? | NOT AVAILABLE | Split not generated |",
        "| 8. Group leakage? | NOT TESTABLE | Split not generated |",
        "| 9. Near-duplicate leakage? | NOT TESTABLE | Split not generated |",
        "| 10. Final epochs? | 15 | Historical manuscript protocol |",
        "| 11. Batch? | 32 | Historical manuscript protocol |",
        "| 12. LR? | 5e-4 | Historical manuscript protocol |",
        "| 13. Optimizer? | Adam, weight_decay=0.0 | Adam default in historical μ=27 code |",
        "| 14. Model outputs? | SM_0, SM_20, moisture_class; LI branch enabled | Code plus historical classification claim |",
        "| 15. Metric scale? | Regression percentage points 0–100; classification ratios 0–1 | Internal regression/LI divide by 100 |",
        f"| 16. Config SHA256? | `{lock['config_sha256']}` | Locked blocked-state config |",
        "| 17. Manifest SHA256? | NOT GENERATED — DATA CONFLICT | No canonical manifest created |",
        "| 18. Split SHA256? | NOT GENERATED — DATA CONFLICT | No split created or opened for metrics |",
        f"| 19. GPU ready? | {gpu_status} | CUDA diagnostic only; no driver changes |",
        "| 20. Safe to run final short fine-tune? | NO | Resolve all exact-byte label conflicts with external evidence, issue new experiment ID, then lock manifests; GPU must also be ready |",
        "", "## Final status", "", "**BLOCKED_DATA_CONFLICT**", "",
        "Secondary gates: GPU is blocked in the current environment and Git state is `UNCOMMITTED`. Neither changes the primary data-conflict stop.",
        "No training ran. No prospective test metric was read. No raw data/checkpoint was modified. No GitHub push occurred.", "",
    ]
    (REPO / "results/audit/PHASE3A_RUN_READINESS.md").write_text("\n".join(lines), encoding="utf-8")


def update_checksums() -> None:
    targets = [
        "README.md", "scripts/README.md",
        "config/experiments/final_revalidation.yaml", "docs/FINAL_REVALIDATION_PROTOCOL.md",
        "docs/FINAL_METRIC_SPEC.md", "results/audit/final_preselection.json",
        "results/audit/final_ssl_checkpoint_candidates.csv",
        "results/audit/exact_duplicate_label_consistency.csv", "results/audit/final_split_report.md",
        "results/audit/final_run_lock.json", "results/audit/PHASE3A_RUN_READINESS.md",
        "results/audit/gpu_diagnostic.json", "results/audit/gpu_diagnostic.txt",
        "scripts/phase3a_protocol_lock.py", "scripts/dry_run_final_protocol.py",
        "scripts/train_final_revalidation.py", "scripts/evaluate_final_test.py",
        "src/soilnet/final_protocol.py",
    ]
    dry_run = "results/audit/phase3a_dry_run.json"
    if (REPO / dry_run).exists():
        targets.append(dry_run)
    registry_path = REPO / "data/checksums/manifest_sha256.csv"
    existing = {row["relative_path"]: row for row in read_csv("data/checksums/manifest_sha256.csv")}
    for relative in targets:
        path = REPO / relative
        existing[relative] = {"relative_path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    write_csv(registry_path, [existing[key] for key in sorted(existing)], ["relative_path", "size_bytes", "sha256"])


def main() -> int:
    forbidden_outputs = [
        REPO / "data/manifests/final_unique_manifest_v1.csv",
        REPO / "data/splits/final_unique_split_v1.csv",
    ]
    if any(path.exists() for path in forbidden_outputs):
        raise RuntimeError("Blocked final manifest/split already exists; refusing to overwrite or silently accept it")
    paths = resolve_paths()
    lineage = verify_selected_lineage(paths)
    candidates = build_ssl_candidates(paths, lineage)
    duplicates = audit_duplicate_labels()
    if not duplicates["conflicts"]:
        raise RuntimeError("This blocked-state PHASE 3A script must be reviewed before any manifest generation")
    gpu = json.loads((REPO / "results/audit/gpu_diagnostic.json").read_text(encoding="utf-8"))
    gpu_status = "GPU_READY" if gpu.get("status") in {"CUDA_WORKING", "CUDA_AVAILABLE_NVML_LIMITED"} else "GPU_BLOCKED"
    write_protocol_files(paths, duplicates, gpu_status)
    lock = write_run_lock(duplicates, lineage, gpu_status)
    write_readiness(duplicates, lock, gpu_status, len(candidates))
    update_checksums()
    print(json.dumps({
        "preselection": "PRESELECTION_RESOLVED_FROM_HISTORICAL_MANUSCRIPT",
        "ssl_checkpoint": "SELECTED_FOR_FINAL_REVALIDATION", "ssl_sha256": SELECTED_SHA256,
        "duplicate_groups": duplicates["groups"], "conflicting_groups": duplicates["conflicts"],
        "manifest_generated": False, "split_generated": False, "gpu": gpu_status,
        "status": "BLOCKED_DATA_CONFLICT", "training_run": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
