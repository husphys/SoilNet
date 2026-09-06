#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from soilnet.io import resolve_paths, sha256_file, write_json


REQUIRED = [
    "config/audit.yaml", "config/paths.example.yaml", "data/manifests/dataset_inventory.csv",
    "data/manifests/soilnet_samples.csv", "data/splits/final_split.csv",
    "results/audit/legacy_file_inventory.csv", "results/audit/legacy_path_references.csv",
    "results/audit/checkpoint_inventory.csv", "results/audit/experiment_graph.csv",
    "results/audit/claims_evidence_matrix.csv", "docs/REPRODUCIBILITY.md",
    "results/metrics/result_registry.csv",
    "data/checksums/manifest_sha256.csv",
    "results/audit/historical_result_schema.csv",
    "results/audit/labeled_dataset_duplicate_summary.json",
    "results/audit/labeled_exact_duplicates.csv", "results/audit/labeled_near_duplicates.csv",
    "results/audit/checkpoint_classification.csv", "results/audit/data_root_checkpoint_inventory.csv",
    "results/audit/checkpoint_cross_scope_matches.csv", "results/audit/phase2_checkpoint_summary.json",
    "results/audit/dataset_count_provenance.md", "results/audit/fine_tuned_checkpoint_lineage.csv",
    "results/audit/sota_checkpoint_lineage.csv", "results/audit/historical_split_inventory.csv",
    "results/audit/unseen_labeled_candidates.csv", "results/audit/ssl_checkpoint_provenance.csv",
    "results/audit/classical_ml_reproduction_plan.md", "results/audit/public_release_scan.json",
    "results/audit/PHASE2_DECISION.md", "results/audit/phase2_summary.json",
    "config/experiments/final_revalidation.yaml", "docs/FINAL_REVALIDATION_PROTOCOL.md",
    "docs/FINAL_METRIC_SPEC.md", "results/audit/final_preselection.json",
    "results/audit/final_ssl_checkpoint_candidates.csv",
    "results/audit/exact_duplicate_label_consistency.csv", "results/audit/final_split_report.md",
    "results/audit/final_run_lock.json", "results/audit/phase3a_dry_run.json",
    "results/audit/PHASE3A_RUN_READINESS.md",
    "docs/FINAL_DATA_QC_POLICY.md", "results/audit/final_data_qc_policy.json",
    "data/manifests/excluded_conflicting_duplicates_v1.csv",
    "data/manifests/final_clean_manifest_v1.csv", "data/splits/final_clean_split_v1.csv",
    "results/audit/final_dataset_flow.md",
    "results/audit/conflicting_duplicate_characterization.csv",
    "results/audit/final_clean_split_report.md",
    "config/experiments/final_revalidation_v2.yaml",
    "results/audit/final_run_lock_v2.json", "results/audit/PHASE3B_DATA_READINESS.md",
    "docs/EXPERIMENT_EXECUTION_PLAN.md",
    "results/audit/notebook_generation_unresolved.md",
    "results/audit/notebook_static_check.json",
    "results/audit/p0_one_batch_smoke.json",
    "results/audit/notebook_generation_lock.json",
    "results/audit/P0_P2_NOTEBOOK_READINESS.md",
    "scripts/generate_final_notebooks.py", "scripts/check_final_notebooks.py",
    "scripts/smoke_p0_notebook_pipeline.py", "scripts/report_run_disk_usage.py",
    "scripts/finalize_notebook_readiness.py",
    "config/experiments/P0_final_soilnet.yaml",
    "config/experiments/P1_no_li.yaml", "config/experiments/P1_no_ssl.yaml",
    "config/experiments/P1_mobilenetv2.yaml", "config/experiments/P1_mobilevitv2.yaml",
    "config/experiments/P2_mobilenetv3.yaml", "config/experiments/P2_efficientnet_b0.yaml",
    "config/experiments/P2_classical_ml.yaml",
    "notebooks/final_experiments/00_gpu_environment_check.ipynb",
    "notebooks/final_experiments/01_P0_final_soilnet.ipynb",
    "notebooks/final_experiments/02_P1_ablation_no_li.ipynb",
    "notebooks/final_experiments/03_P1_ablation_no_ssl.ipynb",
    "notebooks/final_experiments/04_P1_mobilenetv2.ipynb",
    "notebooks/final_experiments/05_P1_mobilevitv2.ipynb",
    "notebooks/final_experiments/06_P2_mobilenetv3.ipynb",
    "notebooks/final_experiments/07_P2_efficientnet_b0.ipynb",
    "notebooks/final_experiments/08_P2_classical_ml.ipynb",
    "notebooks/final_experiments/09_final_test_evaluation.ipynb",
    "notebooks/final_experiments/10_results_summary.ipynb",
    "docs/FINAL_REVALIDATION_PROTOCOL_V3.md", "docs/MANUAL_RUN_GUIDE.md",
    "results/audit/final_run_lock_v3.json", "results/audit/P0_MANUAL_RUN_READINESS.md",
    "scripts/check_p0_v3_compatibility.py", "results/audit/p0_v3_compatibility.json",
]


def main() -> int:
    paths = resolve_paths()
    errors, warnings = [], []
    final_clean_split_rows = 0
    for relative in REQUIRED:
        if not (REPO / relative).is_file():
            errors.append(f"missing required artifact: {relative}")
    split_path = REPO / "data" / "splits" / "final_split.csv"
    rows = []
    if split_path.exists():
        with split_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        ids = defaultdict(set)
        groups = defaultdict(set)
        hashes = defaultdict(set)
        observed_hashes = {}
        for row in rows:
            ids[row["sample_id"]].add(row["split"])
            groups[row["group_id"]].add(row["split"])
            if row.get("sha256"):
                hashes[row["sha256"]].add(row["split"])
            path = paths["data_root"] / (row.get("effective_relative_path") or row["relative_path"])
            if not path.is_file():
                errors.append(f"referenced sample missing: {row['relative_path']}")
            elif row.get("sha256"):
                observed = observed_hashes.get(str(path))
                if observed is None:
                    observed = sha256_file(path)
                    observed_hashes[str(path)] = observed
                if observed != row["sha256"]:
                    errors.append(f"referenced sample SHA256 changed: {row['relative_path']}")
        if any(len(value) > 1 for value in ids.values()):
            errors.append("sample_id occurs across splits")
        if any(len(value) > 1 for value in groups.values()):
            errors.append("duplicate/near-duplicate group occurs across splits")
        if any(len(value) > 1 for value in hashes.values()):
            errors.append("identical SHA256 occurs across splits")
    checkpoint_inventory = REPO / "results" / "audit" / "checkpoint_inventory.csv"
    legacy_snapshot_available = paths["legacy_root"].is_dir()
    if checkpoint_inventory.exists():
        with checkpoint_inventory.open(newline="", encoding="utf-8") as handle:
            checkpoint_rows = list(csv.DictReader(handle))
        if sum(row["source"] == "LOCAL" for row in checkpoint_rows) != 969:
            errors.append("Phase 1 CHECKPOINT_ROOT scope is not exactly 969 files")
        if sum(row["source"] == "GITHUB" for row in checkpoint_rows) != 10:
            errors.append("Phase 1 LEGACY_GITHUB scope is not exactly 10 files")
        if any("DATA_ROOT_EMBEDDED_ARTIFACTS" in row["source"] for row in checkpoint_rows):
            errors.append("DATA_ROOT checkpoint scope was merged into Phase 1 checkpoint inventory")
        if not legacy_snapshot_available:
            warnings.append(
                "Historical LEGACY_GITHUB snapshot is unavailable for live re-hashing; "
                "its frozen 10-file inventory remains checksummed, and it is not required by P0 v3"
            )
        for row in checkpoint_rows:
            if row["source"] == "GITHUB" and not legacy_snapshot_available:
                continue
            root = paths["checkpoint_root"] if row["source"] == "LOCAL" else paths["legacy_root"]
            path = root / row["relative_path"]
            if not path.is_file():
                errors.append(f"checkpoint missing: {row['source']}:{row['relative_path']}")
            elif path.stat().st_size != int(row["size_bytes"]):
                errors.append(f"checkpoint size changed: {row['source']}:{row['relative_path']}")
            elif sha256_file(path) != row["sha256"]:
                errors.append(f"checkpoint SHA256 changed: {row['source']}:{row['relative_path']}")
    dataset_inventory = REPO / "data/manifests/dataset_inventory.csv"
    if dataset_inventory.exists():
        with dataset_inventory.open(newline="", encoding="utf-8") as handle:
            if any(row["relative_path"].casefold().endswith(".pth") for row in csv.DictReader(handle)):
                errors.append(".pth file was included in the image manifest")
    embedded_inventory = REPO / "results/audit/data_root_checkpoint_inventory.csv"
    if embedded_inventory.exists():
        with embedded_inventory.open(newline="", encoding="utf-8") as handle:
            embedded_rows = list(csv.DictReader(handle))
        if len(embedded_rows) != 108:
            errors.append(f"DATA_ROOT embedded checkpoint inventory has {len(embedded_rows)} rows, expected 108")
        if len({row["relative_path_from_data_root"] for row in embedded_rows}) != len(embedded_rows):
            errors.append("duplicate paths in DATA_ROOT embedded checkpoint inventory")
        for row in embedded_rows:
            path = Path(row["absolute_source_path"])
            try:
                path.relative_to(paths["data_root"])
            except ValueError:
                errors.append(f"embedded checkpoint outside DATA_ROOT: {row['relative_path_from_data_root']}")
                continue
            if not path.is_file():
                errors.append(f"embedded checkpoint missing: {row['relative_path_from_data_root']}")
            elif path.stat().st_size != int(row["size_bytes"]):
                errors.append(f"embedded checkpoint size changed: {row['relative_path_from_data_root']}")
            elif sha256_file(path) != row["sha256"]:
                errors.append(f"embedded checkpoint SHA256 changed: {row['relative_path_from_data_root']}")
    classification_path = REPO / "results/audit/checkpoint_classification.csv"
    if classification_path.exists():
        with classification_path.open(newline="", encoding="utf-8") as handle:
            classified = list(csv.DictReader(handle))
        scope_counts = {scope: sum(row["scope"] == scope for row in classified) for scope in {row["scope"] for row in classified}}
        if scope_counts.get("CHECKPOINT_ROOT") != 969 or scope_counts.get("DATA_ROOT_EMBEDDED_ARTIFACTS") != 108:
            errors.append(f"unexpected Phase 2 checkpoint classification scopes: {scope_counts}")
    lineage_path = REPO / "results/audit/fine_tuned_checkpoint_lineage.csv"
    if lineage_path.exists():
        with lineage_path.open(newline="", encoding="utf-8") as handle:
            lineages = list(csv.DictReader(handle))
        if len(lineages) != 30:
            errors.append(f"fine-tuned lineage inventory has {len(lineages)} rows, expected 30")
        if any(row["lineage_status"] == "VERIFIED_LINEAGE" for row in lineages):
            errors.append("fine-tuned lineage marked VERIFIED without save-time hash evidence")
    historical_split_path = REPO / "results/audit/historical_split_inventory.csv"
    if historical_split_path.exists():
        with historical_split_path.open(newline="", encoding="utf-8") as handle:
            historical_splits = list(csv.DictReader(handle))
        if any(row["genuine_heldout"].casefold() == "true" for row in historical_splits):
            errors.append("historical split incorrectly claims genuine held-out evidence")
    unseen_path = REPO / "results/audit/unseen_labeled_candidates.csv"
    if unseen_path.exists():
        with unseen_path.open(newline="", encoding="utf-8") as handle:
            unseen_rows = list(csv.DictReader(handle))
        if len(unseen_rows) != 2057:
            errors.append(f"unseen-candidate inventory has {len(unseen_rows)} rows, expected 2057")
        if any(row["status"] == "CONFIRMED_UNSEEN" for row in unseen_rows):
            errors.append("a labeled row was incorrectly asserted to be confirmed unseen")
    preselection_path = REPO / "results/audit/final_preselection.json"
    if preselection_path.exists():
        preselection = json.loads(preselection_path.read_text(encoding="utf-8"))
        expected = {
            "architecture": "SoilNet", "ssl_method": "VICReg", "imagenet_initialization": True,
            "mu": 27, "selection_source": "historical_manuscript",
            "selection_time_relation": "before prospective split evaluation",
            "status": "PRESELECTION_RESOLVED_FROM_HISTORICAL_MANUSCRIPT",
        }
        for key, value in expected.items():
            if preselection.get(key) != value:
                errors.append(f"final preselection mismatch for {key}: {preselection.get(key)!r}")
        if preselection.get("prospective_metrics_consulted") is not False:
            errors.append("final preselection claims prospective metrics were consulted")
    candidates_path = REPO / "results/audit/final_ssl_checkpoint_candidates.csv"
    if candidates_path.exists():
        with candidates_path.open(newline="", encoding="utf-8") as handle:
            candidates = list(csv.DictReader(handle))
        selected = [row for row in candidates if row["selection_status"] == "SELECTED_FOR_FINAL_REVALIDATION"]
        if len(selected) != 1:
            errors.append(f"expected exactly one selected final SSL checkpoint, found {len(selected)}")
        elif selected[0]["sha256"] != "42599ac025b8d8c8c5b8cca36624665f155e0ff588cd90de2fd47b08cd2d60d8":
            errors.append("selected final SSL checkpoint SHA256 changed")
    consistency_path = REPO / "results/audit/exact_duplicate_label_consistency.csv"
    conflict_count = 0
    if consistency_path.exists():
        with consistency_path.open(newline="", encoding="utf-8") as handle:
            consistency = list(csv.DictReader(handle))
        conflict_count = sum(row["classification"] == "CONFLICTING_LABEL" for row in consistency)
        if len(consistency) != 46 or conflict_count != 46:
            errors.append(f"exact duplicate consistency changed: rows={len(consistency)}, conflicts={conflict_count}")
    final_manifest = REPO / "data/manifests/final_unique_manifest_v1.csv"
    final_split = REPO / "data/splits/final_unique_split_v1.csv"
    final_test_marker = REPO / "results/final/test_evaluation_completed.json"
    if conflict_count and (final_manifest.exists() or final_split.exists()):
        errors.append("final manifest/split exists despite unresolved exact-label conflicts")
    if final_test_marker.exists():
        errors.append("final test completion marker exists during blocked PHASE 3A")
    final_config = REPO / "config/experiments/final_revalidation.yaml"
    final_lock_path = REPO / "results/audit/final_run_lock.json"
    if final_config.exists() and final_lock_path.exists():
        import yaml

        with final_config.open(encoding="utf-8") as handle:
            final_config_values = yaml.safe_load(handle)
        final_lock = json.loads(final_lock_path.read_text(encoding="utf-8"))
        if final_config_values.get("protocol_status") != "BLOCKED_DATA_CONFLICT" or final_config_values.get("training_allowed") is not False:
            errors.append("PHASE 3A config does not enforce the data-conflict training block")
        if final_lock.get("lock_state") != "INACTIVE_BLOCKED_DATA_CONFLICT" or final_lock.get("training_allowed") is not False:
            errors.append("PHASE 3A run lock is incorrectly active")
        if sha256_file(final_config) != final_lock.get("config_sha256"):
            errors.append("PHASE 3A config SHA256 does not match final run lock")
        if final_lock.get("dataset_manifest_sha256") != "NOT_GENERATED_BLOCKED_DATA_CONFLICT":
            errors.append("PHASE 3A lock incorrectly claims a canonical dataset manifest")
        if final_lock.get("split_manifest_sha256") != "NOT_GENERATED_BLOCKED_DATA_CONFLICT":
            errors.append("PHASE 3A lock incorrectly claims a final split manifest")
    phase3b_policy_path = REPO / "results/audit/final_data_qc_policy.json"
    phase3b_exclusion_path = REPO / "data/manifests/excluded_conflicting_duplicates_v1.csv"
    phase3b_manifest_path = REPO / "data/manifests/final_clean_manifest_v1.csv"
    phase3b_split_path = REPO / "data/splits/final_clean_split_v1.csv"
    phase3b_config_path = REPO / "config/experiments/final_revalidation_v2.yaml"
    phase3b_lock_path = REPO / "results/audit/final_run_lock_v2.json"
    if all(path.exists() for path in (
        phase3b_policy_path, phase3b_exclusion_path, phase3b_manifest_path,
        phase3b_split_path, phase3b_config_path, phase3b_lock_path,
    )):
        import yaml

        with phase3b_exclusion_path.open(newline="", encoding="utf-8") as handle:
            excluded = list(csv.DictReader(handle))
        with phase3b_manifest_path.open(newline="", encoding="utf-8") as handle:
            clean = list(csv.DictReader(handle))
        with phase3b_split_path.open(newline="", encoding="utf-8") as handle:
            clean_split = list(csv.DictReader(handle))
        final_clean_split_rows = len(clean_split)
        policy = json.loads(phase3b_policy_path.read_text(encoding="utf-8"))
        lock_v2 = json.loads(phase3b_lock_path.read_text(encoding="utf-8"))
        with phase3b_config_path.open(encoding="utf-8") as handle:
            config_v2 = yaml.safe_load(handle)
        excluded_hashes = {row["sha256"] for row in excluded}
        clean_hashes = [row["sha256"] for row in clean]
        if len(excluded) != 130 or len(excluded_hashes) != 46:
            errors.append(f"PHASE 3B exclusion count changed: rows={len(excluded)}, groups={len(excluded_hashes)}")
        if {row.get("reason") for row in excluded} != {"EXACT_DUPLICATE_CONFLICT"}:
            errors.append("PHASE 3B exclusion manifest has an invalid reason")
        if len(clean) != 1927 or len(clean_hashes) != len(set(clean_hashes)):
            errors.append(f"PHASE 3B final manifest is not 1927 unique SHA256 rows: rows={len(clean)}")
        if excluded_hashes.intersection(clean_hashes):
            errors.append("An excluded exact-conflict SHA256 remains in the final clean manifest")
        if len(clean_split) != len(clean):
            errors.append("PHASE 3B final clean split does not cover the final clean manifest")
        phase3b_groups = defaultdict(set)
        phase3b_near = defaultdict(set)
        phase3b_samples = defaultdict(set)
        phase3b_hashes = defaultdict(set)
        for row in clean_split:
            phase3b_groups[row["group_id"]].add(row["split"])
            phase3b_near[row["near_duplicate_group_id"]].add(row["split"])
            phase3b_samples[row["sample_id"]].add(row["split"])
            phase3b_hashes[row["sha256"]].add(row["split"])
        if any(len(value) > 1 for value in phase3b_groups.values()):
            errors.append("PHASE 3B connected component crosses splits")
        if any(len(value) > 1 for value in phase3b_near.values()):
            errors.append("PHASE 3B near-duplicate group crosses splits")
        if any(len(value) > 1 for value in phase3b_samples.values()):
            errors.append("PHASE 3B sample ID occurs across splits")
        if any(len(value) > 1 for value in phase3b_hashes.values()):
            errors.append("PHASE 3B exact SHA256 occurs across splits")
        if policy.get("policy_name") != "EXACT_DUPLICATE_CONFLICT_EXCLUSION":
            errors.append("PHASE 3B data QC policy name changed")
        if policy.get("representative_retained_from_conflict_group") is not False:
            errors.append("PHASE 3B policy retained a conflict-group representative")
        if config_v2.get("protocol_status") != "READY_TO_RUN_FINAL_REVALIDATION" or config_v2.get("training_allowed") is not True:
            errors.append("PHASE 3B scientific config is not ready/locked")
        if lock_v2.get("lock_state") != "ACTIVE" or lock_v2.get("current_execution_status") not in {
            "READY_TO_RUN_FINAL_REVALIDATION", "BLOCKED_GPU_ONLY",
        }:
            errors.append("PHASE 3B v2 run lock has an invalid state")
        expected_hashes = {
            "data_qc_policy_sha256": sha256_file(phase3b_policy_path),
            "exclusion_manifest_sha256": sha256_file(phase3b_exclusion_path),
            "final_clean_manifest_sha256": sha256_file(phase3b_manifest_path),
            "final_split_sha256": sha256_file(phase3b_split_path),
            "config_sha256": sha256_file(phase3b_config_path),
        }
        for key, expected in expected_hashes.items():
            if lock_v2.get(key) != expected:
                errors.append(f"PHASE 3B v2 lock hash mismatch: {key}")
        if lock_v2.get("ssl_checkpoint_sha256") != "42599ac025b8d8c8c5b8cca36624665f155e0ff588cd90de2fd47b08cd2d60d8":
            errors.append("PHASE 3B selected SSL checkpoint hash changed")
        for relative, expected in lock_v2.get("code_sha256", {}).items():
            path = REPO / relative
            if not path.is_file() or sha256_file(path) != expected:
                errors.append(f"PHASE 3B locked code changed: {relative}")
    v3_lock_path = REPO / "results/audit/final_run_lock_v3.json"
    if v3_lock_path.exists():
        v3 = json.loads(v3_lock_path.read_text(encoding="utf-8"))
        p0_config_path = REPO / "config/experiments/P0_final_soilnet.yaml"
        with p0_config_path.open(encoding="utf-8") as handle:
            p0_config = yaml.safe_load(handle)
        expected_v3 = {
            "lock_revision": 3,
            "status": "PROTOCOL_REVISED_PRE_TEST",
            "current_implementation_status": "READY_FOR_MANUAL_P0_RUN",
            "test_loader_used_before_revision": False,
            "test_metrics_read_or_computed_before_revision": False,
            "final_clean_manifest_sha256": "8ff45054d4b8e3df9758d0c112dc16719572b2267906fb3a5ed5b3262a6732bd",
            "final_split_sha256": "8927b8223b8c4c234d264ad9ea62ac2df6161124a79787a2e71eeb5cd23eae2f",
            "ssl_checkpoint_sha256": "42599ac025b8d8c8c5b8cca36624665f155e0ff588cd90de2fd47b08cd2d60d8",
            "seed": 20260905,
        }
        for key, expected in expected_v3.items():
            if v3.get(key) != expected:
                errors.append(f"P0 v3 lock mismatch for {key}: {v3.get(key)!r}")
        protocol = v3.get("final_revalidation_v3", {})
        if protocol != {
            "epochs": 60, "batch_size": 32, "optimizer": "Adam",
            "learning_rate": 0.0001, "weight_decay": 0.0,
            "primary_checkpoint": "epoch_60_final.pth",
        }:
            errors.append("P0 v3 final protocol values changed")
        if p0_config.get("protocol_status") != "PROTOCOL_REVISED_PRE_TEST":
            errors.append("P0 config is not marked PROTOCOL_REVISED_PRE_TEST")
        if sha256_file(p0_config_path) != v3.get("p0_config_sha256"):
            errors.append("P0 v3 config hash mismatch")
        for relative, expected in v3.get("implementation_sha256", {}).items():
            path = REPO / relative
            if not path.is_file() or sha256_file(path) != expected:
                errors.append(f"P0 v3 locked implementation changed: {relative}")
        if v3.get("compatibility_checks", {}).get("status") != "PASS":
            errors.append("P0 v3 compatibility checks are not PASS")
        if v3.get("manual_jupyter_gpu_check", {}).get("status") != "PASS":
            errors.append("P0 v3 manual Jupyter GPU diagnostic is not PASS")
    if any(path.is_file() for path in REPO.rglob("*.pth")):
        errors.append("raw .pth file was copied into the repository")
    dry_run_path = REPO / "results/audit/phase3a_dry_run.json"
    if dry_run_path.exists():
        dry_run = json.loads(dry_run_path.read_text(encoding="utf-8"))
        if dry_run.get("status") != "PASS" or dry_run.get("training_run") is not False:
            errors.append("bounded PHASE 3A dry run did not pass safely")
        if dry_run.get("prospective_test_split_used") is not False or dry_run.get("metric_values_recorded") is not False:
            errors.append("PHASE 3A dry run used prospective test data or recorded metric values")
    notebook_static_path = REPO / "results/audit/notebook_static_check.json"
    if notebook_static_path.exists():
        notebook_static = json.loads(notebook_static_path.read_text(encoding="utf-8"))
        expected_static = {
            "status": "PASS", "notebooks_checked": 11, "configs_checked": 8,
            "training_notebook_cells_executed": 0,
            "gpu_notebook_execution_status": "PASS", "test_loader_instantiated": False,
        }
        for key, expected in expected_static.items():
            if notebook_static.get(key) != expected:
                errors.append(f"P0–P2 static notebook check mismatch for {key}")
    p0_smoke_path = REPO / "results/audit/p0_one_batch_smoke.json"
    if p0_smoke_path.exists():
        p0_smoke = json.loads(p0_smoke_path.read_text(encoding="utf-8"))
        expected_smoke = {
            "status": "PASS", "train_batches": 1, "validation_batches": 1,
            "backward_batches": 1, "full_training_run": False,
            "test_loader_instantiated": False, "test_samples_loaded": 0,
            "test_metrics_computed": False, "research_metrics_created": False,
            "train_samples": 1407, "validation_samples": 289,
        }
        for key, expected in expected_smoke.items():
            if p0_smoke.get(key) != expected:
                errors.append(f"bounded P0 smoke mismatch for {key}: {p0_smoke.get(key)!r}")
        if p0_smoke.get("manifest_sha256") != "8ff45054d4b8e3df9758d0c112dc16719572b2267906fb3a5ed5b3262a6732bd":
            errors.append("bounded P0 smoke manifest hash mismatch")
        if p0_smoke.get("split_sha256") != "8927b8223b8c4c234d264ad9ea62ac2df6161124a79787a2e71eeb5cd23eae2f":
            errors.append("bounded P0 smoke split hash mismatch")
        load_report = p0_smoke.get("load_report", {})
        if load_report.get("checkpoint_sha256") != "42599ac025b8d8c8c5b8cca36624665f155e0ff588cd90de2fd47b08cd2d60d8":
            errors.append("bounded P0 smoke SSL checkpoint hash mismatch")
        if load_report.get("canonical_keys") != 577 or load_report.get("matched_keys") != 577:
            errors.append("bounded P0 smoke did not strictly match all 577 canonical keys")
        for key in ("missing_after_normalization", "unexpected_after_normalization", "shape_mismatches"):
            if load_report.get(key) != []:
                errors.append(f"bounded P0 smoke has checkpoint incompatibility: {key}")
        if p0_smoke.get("repository_pth_before") != [] or p0_smoke.get("repository_pth_after") != []:
            errors.append("bounded P0 smoke found or created a repository .pth file")
    notebook_lock_path = REPO / "results/audit/notebook_generation_lock.json"
    if notebook_lock_path.exists():
        notebook_lock = json.loads(notebook_lock_path.read_text(encoding="utf-8"))
        if notebook_lock.get("status") != "FROZEN_P0_V3_P2_OPTIONAL":
            errors.append("P0–P2 implementation lock status changed")
        if notebook_lock.get("full_training_run") is not False:
            errors.append("P0–P2 implementation lock claims full training")
        if notebook_lock.get("test_loader_instantiated") is not False:
            errors.append("P0–P2 implementation lock claims test access")
        for section in ("config_sha256", "notebook_sha256", "implementation_sha256", "evidence_sha256"):
            entries = notebook_lock.get(section, {})
            if not isinstance(entries, dict) or not entries:
                errors.append(f"P0–P2 implementation lock lacks {section}")
                continue
            for relative, expected in entries.items():
                path = REPO / relative
                if not path.is_file() or sha256_file(path) != expected:
                    errors.append(f"P0–P2 locked artifact changed: {relative}")
    checksum_registry = REPO / "data" / "checksums" / "manifest_sha256.csv"
    if checksum_registry.exists():
        with checksum_registry.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                path = REPO / row["relative_path"]
                if not path.exists():
                    errors.append(f"checksummed artifact missing: {row['relative_path']}")
                elif sha256_file(path) != row["sha256"]:
                    errors.append(f"checksummed artifact changed: {row['relative_path']}")
    # Assemble forbidden markers so this verifier does not flag its own source.
    drive_markers = [letter + ":" + chr(92) for letter in ("F", "C")]
    forbidden_markers = drive_markers + [drive_markers[1] + "Users", "/content/" + "drive", "/content/" + "working"]
    for directory in (REPO / "src", REPO / "scripts", REPO / "notebooks"):
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.casefold() in {".py", ".ipynb"}:
                source = path.read_text(encoding="utf-8", errors="replace")
                match = next((marker for marker in forbidden_markers if marker.casefold() in source.casefold()), None)
                if match:
                    errors.append(f"hardcoded legacy path in reproduction code: {path.relative_to(REPO)}")
    for required_dir in (REPO / "results" / "metrics", REPO / "config"):
        if not required_dir.exists():
            errors.append(f"missing directory: {required_dir.relative_to(REPO)}")
    report = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": warnings,
        "historical_split_rows": len(rows),
        "final_clean_split_rows": final_clean_split_rows,
        "legacy_github_live_verification": "AVAILABLE" if legacy_snapshot_available else "UNAVAILABLE_WARNING_ONLY",
    }
    write_json(REPO / "results" / "audit" / "verification.json", report)
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
