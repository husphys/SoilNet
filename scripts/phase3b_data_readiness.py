#!/usr/bin/env python3
"""Build PHASE 3B QC/split artifacts without training or test loading."""
from __future__ import annotations

import csv
import json
import platform
import statistics
import sys
from collections import Counter, defaultdict
from importlib import metadata
from pathlib import Path, PureWindowsPath

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from soilnet.final_protocol import current_git_commit
from soilnet.io import load_yaml, resolve_paths, sha256_file, sha256_text, write_csv, write_json


SEED = 20260905
RATIOS = {"train": 0.73, "validation": 0.15, "test": 0.12}
POLICY = "EXACT_DUPLICATE_CONFLICT_EXCLUSION"
REASON = "EXACT_DUPLICATE_CONFLICT"
SSL_SHA256 = "42599ac025b8d8c8c5b8cca36624665f155e0ff588cd90de2fd47b08cd2d60d8"
SSL_RELATIVE = "checkpoints_VicReg_original_NEW/vicreg_model_final_mu_27.0.pth"
CONFIG_RELATIVE = "config/experiments/final_revalidation_v2.yaml"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def normalize_legacy_path(raw: str, data_root: Path) -> str:
    value = (raw or "").strip().replace("\\", "/")
    if len(value) >= 2 and value[1] == ":":
        parts = list(PureWindowsPath(raw).parts)[1:]
        value = "/".join(part.strip("\\/") for part in parts)
    value = value.lstrip("/")
    parts = value.split("/")
    if parts and parts[0].casefold() == data_root.name.casefold():
        parts = parts[1:]
    return "/".join(parts)


def number(value: str) -> float:
    return float(value)


def numeric_values(rows: list[dict[str, str]], key: str) -> set[float]:
    return {number(row[key]) for row in rows}


def display_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else format(value, ".12g")


def summarize(rows: list[dict[str, str]], key: str) -> dict[str, float | int]:
    values = [number(row[key]) for row in rows]
    return {
        "n": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def deterministic_group_split(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    members: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        members[row["group_id"]].append(row)
    total = len(rows)
    targets = {split: RATIOS[split] * total for split in RATIOS}
    counts = {split: 0 for split in RATIOS}
    assigned: dict[str, str] = {}
    ordered = sorted(
        members,
        key=lambda group_id: (-len(members[group_id]), sha256_text(f"{SEED}:{group_id}")),
    )
    for group_id in ordered:
        size = len(members[group_id])
        candidates = []
        for split in RATIOS:
            proposed = dict(counts)
            proposed[split] += size
            cost = sum((proposed[name] - targets[name]) ** 2 for name in RATIOS)
            tie = sha256_text(f"{SEED}:{group_id}:{split}")
            candidates.append((cost, tie, split))
        split = min(candidates)[2]
        assigned[group_id] = split
        counts[split] += size
    return [{**row, "split": assigned[row["group_id"]]} for row in rows]


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def update_checksum_registry(extra_paths: set[str]) -> None:
    registry = REPO / "data/checksums/manifest_sha256.csv"
    existing = read_csv(registry) if registry.exists() else []
    relatives = {row["relative_path"] for row in existing} | extra_paths
    rows = []
    for relative in sorted(relatives):
        path = REPO / relative
        if path.is_file():
            rows.append({"relative_path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_csv(registry, rows, ["relative_path", "size_bytes", "sha256"])


def main() -> int:
    paths = resolve_paths()
    data_root, checkpoint_root = paths["data_root"], paths["checkpoint_root"]
    samples = read_csv(REPO / "data/manifests/soilnet_samples.csv")
    historical_split = read_csv(REPO / "data/splits/final_split.csv")
    labels_paths = sorted(data_root.rglob("labels.csv"))
    if len(labels_paths) != 1:
        raise RuntimeError(f"Expected exactly one labels.csv, found {len(labels_paths)}")
    labels = read_csv(labels_paths[0])
    if len(samples) != len(labels) or len(samples) != 2057:
        raise RuntimeError(f"Historical row reconciliation changed: samples={len(samples)}, labels={len(labels)}")
    if len(historical_split) != len(samples):
        raise RuntimeError("PHASE 2 grouping manifest no longer covers every labeled row")

    split_by_sample = {row["sample_id"]: row for row in historical_split}
    if len(split_by_sample) != len(historical_split):
        raise RuntimeError("Historical sample_id is not unique")
    prepared = []
    for index, (sample, label) in enumerate(zip(samples, labels, strict=True), start=2):
        normalized = normalize_legacy_path(label.get("path", ""), data_root)
        if normalized != sample["relative_path"]:
            raise RuntimeError(f"labels.csv row alignment changed at data row {index}")
        required = ("sha256", "SM_0", "SM_20", "light_value", "moisture_class")
        if any(not sample.get(key, "").strip() for key in required):
            raise RuntimeError(f"Missing required field at labels.csv data row {index}")
        if sample["path_status"] not in {"PRESENT", "HISTORICAL_FALLBACK"}:
            raise RuntimeError(f"Unreadable/missing historical row at data row {index}")
        grouped = split_by_sample[sample["sample_id"]]
        prepared.append({
            **sample,
            "original_row_id": label.get("id", "") or f"labels.csv:{index}",
            "group_id": grouped["group_id"],
            "near_duplicate_group_id": grouped["group_id"],
            "capture_session_group": "",
            "source_group": "",
            "soil_type": Path(sample["relative_path"]).parent.name,
        })

    by_sha: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in prepared:
        by_sha[row["sha256"]].append(row)
    conflicting = {
        digest: group for digest, group in by_sha.items()
        if len(group) > 1 and (len(numeric_values(group, "SM_0")) > 1 or len(numeric_values(group, "SM_20")) > 1)
    }
    excluded_count = sum(len(group) for group in conflicting.values())
    redundant_count = sum(len(group) - 1 for group in conflicting.values())
    light_conflict_groups = sum(len(numeric_values(group, "light_value")) > 1 for group in conflicting.values())
    class_conflict_groups = sum(len({row["moisture_class"] for row in group}) > 1 for group in conflicting.values())
    if (len(conflicting), excluded_count, redundant_count) != (46, 130, 84):
        raise RuntimeError(
            "Dynamically derived conflict audit no longer matches PHASE 3A: "
            f"groups={len(conflicting)}, rows={excluded_count}, redundant={redundant_count}"
        )
    if (light_conflict_groups, class_conflict_groups) != (43, 38):
        raise RuntimeError(
            "Dynamically derived auxiliary/class conflict counts changed: "
            f"LI={light_conflict_groups}, moisture_class={class_conflict_groups}"
        )

    exclusion_rows = []
    characterization_rows = []
    for digest, group in sorted(conflicting.items()):
        group_id = f"sha256:{digest[:16]}"
        for row in sorted(group, key=lambda item: (item["relative_path"], item["sample_id"])):
            exclusion_rows.append({
                "sha256": digest,
                "group_id": group_id,
                "relative_path": row["relative_path"],
                "original_row_id": row["original_row_id"],
                "SM_0": row["SM_0"],
                "SM_20": row["SM_20"],
                "light_value": row["light_value"],
                "moisture_class": row["moisture_class"],
                "group_size": len(group),
                "reason": REASON,
            })
        sm0, sm20, light = numeric_values(group, "SM_0"), numeric_values(group, "SM_20"), numeric_values(group, "light_value")
        characterization_rows.append({
            "sha256": digest,
            "group_id": group_id,
            "group_size": len(group),
            "SM_0_min": display_number(min(sm0)), "SM_0_max": display_number(max(sm0)),
            "SM_0_range": display_number(max(sm0) - min(sm0)),
            "SM_20_min": display_number(min(sm20)), "SM_20_max": display_number(max(sm20)),
            "SM_20_range": display_number(max(sm20) - min(sm20)),
            "LI_min": display_number(min(light)), "LI_max": display_number(max(light)),
            "moisture_classes_present": ";".join(sorted({row["moisture_class"] for row in group}, key=float)),
            "paths": ";".join(sorted(row["relative_path"] for row in group)),
        })

    exclusion_path = REPO / "data/manifests/excluded_conflicting_duplicates_v1.csv"
    write_csv(exclusion_path, exclusion_rows, [
        "sha256", "group_id", "relative_path", "original_row_id", "SM_0", "SM_20",
        "light_value", "moisture_class", "group_size", "reason",
    ])
    write_csv(REPO / "results/audit/conflicting_duplicate_characterization.csv", characterization_rows, [
        "sha256", "group_id", "group_size", "SM_0_min", "SM_0_max", "SM_0_range",
        "SM_20_min", "SM_20_max", "SM_20_range", "LI_min", "LI_max",
        "moisture_classes_present", "paths",
    ])

    excluded_hashes = set(conflicting)
    clean = [row for row in prepared if row["sha256"] not in excluded_hashes]
    if len(clean) != 1927 or len({row["sha256"] for row in clean}) != len(clean):
        raise RuntimeError(f"Final clean uniqueness/count failed: rows={len(clean)}")
    for index, row in enumerate(clean, 1):
        image_path = data_root / row["effective_relative_path"]
        with Image.open(image_path) as image:
            image.load()
        observed = sha256_file(image_path)
        if observed != row["sha256"]:
            raise RuntimeError(f"Source image SHA256 changed: {row['relative_path']}")
        if index % 500 == 0:
            print(f"verified final images {index}/{len(clean)}", flush=True)

    manifest_path = REPO / "data/manifests/final_clean_manifest_v1.csv"
    manifest_fields = [
        "sample_id", "original_row_id", "relative_path", "effective_relative_path", "path_status",
        "historical_fallback_used", "sha256", "SM_0", "SM_20", "light_value", "moisture_class",
        "moisture_bin", "light_bin", "soil_type", "group_id", "near_duplicate_group_id",
        "capture_session_group", "source_group",
    ]
    write_csv(manifest_path, clean, manifest_fields)

    split_rows = deterministic_group_split(clean)
    split_path = REPO / "data/splits/final_clean_split_v1.csv"
    write_csv(split_path, split_rows, manifest_fields + ["split"])
    split_counts = Counter(row["split"] for row in split_rows)
    group_splits: dict[str, set[str]] = defaultdict(set)
    sha_splits: dict[str, set[str]] = defaultdict(set)
    sample_splits: dict[str, set[str]] = defaultdict(set)
    near_splits: dict[str, set[str]] = defaultdict(set)
    for row in split_rows:
        group_splits[row["group_id"]].add(row["split"])
        sha_splits[row["sha256"]].add(row["split"])
        sample_splits[row["sample_id"]].add(row["split"])
        near_splits[row["near_duplicate_group_id"]].add(row["split"])
    leakages = {
        "exact_sha_cross_split": sum(len(value) > 1 for value in sha_splits.values()),
        "group_cross_split": sum(len(value) > 1 for value in group_splits.values()),
        "near_duplicate_group_cross_split": sum(len(value) > 1 for value in near_splits.values()),
        "sample_id_cross_split": sum(len(value) > 1 for value in sample_splits.values()),
        "documented_capture_session_group_cross_split": 0,
        "documented_source_group_cross_split": 0,
    }
    if any(leakages.values()):
        raise RuntimeError(f"Split leakage detected: {leakages}")

    policy = {
        "policy_name": POLICY,
        "decision_fields": ["SM_0", "SM_20"],
        "rule": "Exclude every record in an exact-SHA256 group if SM_0 or SM_20 differs within that group.",
        "scope": "all_historical_labeled_records",
        "created_before_model_training": True,
        "created_before_prospective_test_evaluation": True,
        "label_adjudication": False,
        "majority_vote": False,
        "target_averaging": False,
        "model_assisted_cleaning": False,
        "representative_retained_from_conflict_group": False,
        "perceptual_similarity_used_for_exclusion": False,
        "near_duplicate_handling": "retain_samples_and_apply_locked_group_split_firewall",
        "dynamically_verified_conflict_groups": len(conflicting),
        "dynamically_verified_excluded_records": excluded_count,
        "dynamically_verified_light_value_conflict_groups": light_conflict_groups,
        "dynamically_verified_moisture_class_conflict_groups": class_conflict_groups,
    }
    policy_path = REPO / "results/audit/final_data_qc_policy.json"
    write_json(policy_path, policy)

    flow = f"""# Final dataset flow

| Stage | Records/images | Interpretation |
|---|---:|---|
| Historical labeled table | {len(samples)} | Strongest-provenance historical labeled records |
| Exact-byte duplicate audit | {len(conflicting)} groups / {excluded_count} records | {redundant_count} copies beyond one-per-SHA; every group conflicts on SM_0 or SM_20 |
| Auxiliary/class characterization | {light_conflict_groups} LI groups / {class_conflict_groups} moisture-class groups | Transparency only; not used to decide exclusion |
| Excluded without label adjudication | {excluded_count} | All rows in every conflicting exact-SHA group; no representative retained |
| Final conflict-free dataset | {len(clean)} | Unique labeled image SHA256 values |

The final count is **{len(clean)}**, not 1,973. The 46 otherwise-unique SHA representatives
inside conflict groups have no defensible canonical regression target and are therefore
excluded. The unresolved manuscript count of 2,250 remains a separate historical claim;
it is not silently replaced by this QC result. No source file was deleted or modified.
"""
    (REPO / "results/audit/final_dataset_flow.md").write_text(flow, encoding="utf-8")

    report_lines = [
        "# Final clean split report", "",
        "This is pre-specified descriptive data accounting only. No model was loaded, no test loader was instantiated, and no test metric was read or computed.", "",
        "## Counts", "",
        markdown_table(["Split", "Count", "Observed ratio", "Target ratio"], [
            [name, split_counts[name], f"{split_counts[name] / len(split_rows):.4f}", f"{RATIOS[name]:.2f}"]
            for name in RATIOS
        ]), "", "## Leakage firewall", "",
        markdown_table(["Check", "Result"], [
            ["Exact SHA256 cross-split leakage", leakages["exact_sha_cross_split"]],
            ["Locked exact/near connected-component leakage", leakages["group_cross_split"]],
            ["Near-duplicate group leakage", leakages["near_duplicate_group_cross_split"]],
            ["Sample ID overlap", leakages["sample_id_cross_split"]],
            ["Documented capture/session-group leakage", 0],
            ["Documented source-group leakage", 0],
        ]), "",
        "No capture/session or source-group field exists in the historical label table. Thus zero documented groups cross splits, while leakage from an unrecorded latent session remains NOT_TESTABLE. Filename timestamps were not converted into session IDs.", "",
        "## Moisture-class distributions", "",
    ]
    classes = sorted({row["moisture_class"] for row in split_rows}, key=float)
    report_lines.append(markdown_table(["Split"] + classes, [
        [split] + [sum(row["split"] == split and row["moisture_class"] == cls for row in split_rows) for cls in classes]
        for split in RATIOS
    ]))
    report_lines += ["", "## Soil-type distributions", ""]
    soil_types = sorted({row["soil_type"] for row in split_rows})
    report_lines.append(markdown_table(["Split"] + soil_types, [
        [split] + [sum(row["split"] == split and row["soil_type"] == soil for row in split_rows) for soil in soil_types]
        for split in RATIOS
    ]))
    report_lines += ["", "## Numeric summaries", ""]
    summary_rows = []
    for split in RATIOS:
        subset = [row for row in split_rows if row["split"] == split]
        for field in ("SM_0", "SM_20", "light_value"):
            values = summarize(subset, field)
            summary_rows.append([
                split, field, values["n"], f"{values['min']:.3f}", f"{values['max']:.3f}",
                f"{values['mean']:.3f}", f"{values['median']:.3f}", f"{values['std']:.3f}",
            ])
    report_lines.append(markdown_table(["Split", "Field", "n", "Min", "Max", "Mean", "Median", "SD"], summary_rows))
    report_lines += ["", "Grouping policy: exact SHA256 plus PHASE 2's locked 64-bit difference-hash connected components at Hamming distance <= 5. Perceptual similarity did not remove samples. Assignment used only group sizes, target ratios, the fixed seed, and stable IDs; no model performance was consulted.", ""]
    (REPO / "results/audit/final_clean_split_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    config_path = REPO / CONFIG_RELATIVE
    config = load_yaml(config_path)
    if config["training"] != load_yaml(REPO / "config/experiments/final_revalidation.yaml")["training"]:
        raise RuntimeError("Locked training protocol differs from PHASE 3A")
    ssl_path = checkpoint_root / SSL_RELATIVE
    if sha256_file(ssl_path) != SSL_SHA256:
        raise RuntimeError("Selected SSL checkpoint changed")
    gpu = json.loads((REPO / "results/audit/gpu_diagnostic.json").read_text(encoding="utf-8"))
    gpu_working = gpu.get("status") in {"CUDA_WORKING", "CUDA_AVAILABLE_NVML_LIMITED"} and gpu.get("cuda_tensor_matmul_smoke") is True
    final_status = "READY_TO_RUN_FINAL_REVALIDATION" if gpu_working else "BLOCKED_GPU_ONLY"
    code_relatives = [
        "scripts/phase3b_data_readiness.py", "scripts/train_final_revalidation.py",
        "scripts/evaluate_final_test.py", "src/soilnet/final_protocol.py",
        "src/soilnet/data/dataset.py", "src/soilnet/models/soilnet.py",
    ]
    code_hashes = {relative: sha256_file(REPO / relative) for relative in code_relatives}
    lock = {
        "experiment_id": config["experiment_id"],
        "lock_revision": 2,
        "lock_state": "ACTIVE",
        "training_allowed_by_protocol": True,
        "current_execution_status": final_status,
        "git_commit": current_git_commit(REPO),
        "git_state_policy": "exact_content_hashes_when_no_commit_exists",
        "data_qc_policy_sha256": sha256_file(policy_path),
        "exclusion_manifest_sha256": sha256_file(exclusion_path),
        "final_clean_manifest_sha256": sha256_file(manifest_path),
        "final_split_sha256": sha256_file(split_path),
        "ssl_checkpoint_scope": "CHECKPOINT_ROOT",
        "ssl_checkpoint_relative_path": SSL_RELATIVE,
        "ssl_checkpoint_sha256": SSL_SHA256,
        "ssl_lineage": "VERIFIED_LINEAGE",
        "config_relative_path": CONFIG_RELATIVE,
        "config_sha256": sha256_file(config_path),
        "code_sha256": code_hashes,
        "seed": SEED,
        "target_ratios": RATIOS,
        "historical_labeled_records": len(samples),
        "conflicting_exact_duplicate_groups": len(conflicting),
        "excluded_records": excluded_count,
        "final_clean_records": len(clean),
        "final_images_readable_and_sha_verified": len(clean),
        "light_value_conflict_groups": light_conflict_groups,
        "moisture_class_conflict_groups": class_conflict_groups,
        "split_counts": dict(split_counts),
        "leakage_checks": leakages,
        "test_loader_used_in_phase3b": False,
        "test_metrics_read_or_computed_in_phase3b": False,
        "environment_versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": package_version("torch"),
            "torchvision": package_version("torchvision"),
            "numpy": package_version("numpy"),
            "Pillow": package_version("Pillow"),
            "PyYAML": package_version("PyYAML"),
        },
        "gpu_diagnostic_status": gpu.get("status", "UNKNOWN"),
    }
    lock_path = REPO / "results/audit/final_run_lock_v2.json"
    write_json(lock_path, lock)

    manifest_sha, split_sha = lock["final_clean_manifest_sha256"], lock["final_split_sha256"]
    readiness = f"""# PHASE 3B data readiness

1. **Historical labeled records?** {len(samples)}.
2. **Duplicate-conflict groups?** {len(conflicting)} exact-SHA256 groups; derived from the manifests, not assumed.
3. **Records excluded?** {excluded_count}; all members, no representative and no label adjudication.
4. **Final clean labeled count?** {len(clean)} unique labeled images.
5. **Exact duplicate remaining?** 0.
6. **Label conflict remaining?** 0 exact-image target conflicts in the final clean manifest.
7. **Train/validation/test counts?** {split_counts['train']} / {split_counts['validation']} / {split_counts['test']}.
8. **Group leakage?** 0 locked connected components cross splits; documented capture/source groups crossing splits: 0. Unrecorded latent sessions are NOT_TESTABLE.
9. **Near-duplicate group leakage?** 0 under the locked dHash Hamming-distance <= 5 connected-component policy.
10. **Manifest SHA256?** `{manifest_sha}`.
11. **Split SHA256?** `{split_sha}`.
12. **SSL checkpoint unchanged?** Yes: `{SSL_SHA256}`, `VERIFIED_LINEAGE`.
13. **Protocol unchanged?** Yes: SoilNet + VICReg + ImageNet + mu=27; 15 epochs, batch 32, Adam, lr=5e-4, weight decay 0, epoch 15 primary; outputs SM_0, SM_20, moisture class, with LI auxiliary input.
14. **GPU status?** `{gpu.get('status', 'UNKNOWN')}` / runtime gate `{'READY' if gpu_working else 'GPU_BLOCKED'}`.
15. **Safe for final revalidation?** Data and protocol inputs are locked and pass static readiness checks, but execution is not currently safe/possible until the CUDA GPU gate passes. No training or prospective test evaluation occurred in PHASE 3B.

## Final status

**{final_status}**
"""
    (REPO / "results/audit/PHASE3B_DATA_READINESS.md").write_text(readiness, encoding="utf-8")

    update_checksum_registry({
        CONFIG_RELATIVE, "docs/FINAL_DATA_QC_POLICY.md", "scripts/phase3b_data_readiness.py",
        "data/manifests/excluded_conflicting_duplicates_v1.csv",
        "data/manifests/final_clean_manifest_v1.csv", "data/splits/final_clean_split_v1.csv",
        "results/audit/final_data_qc_policy.json", "results/audit/final_dataset_flow.md",
        "results/audit/conflicting_duplicate_characterization.csv",
        "results/audit/final_clean_split_report.md", "results/audit/final_run_lock_v2.json",
        "results/audit/PHASE3B_DATA_READINESS.md", "tests/test_phase3b_data_readiness.py",
    })
    print(json.dumps({
        "status": final_status, "historical_records": len(samples), "conflict_groups": len(conflicting),
        "excluded_records": excluded_count, "final_clean_records": len(clean),
        "split_counts": dict(split_counts), "manifest_sha256": manifest_sha, "split_sha256": split_sha,
        "test_loader_used": False, "training_run": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
