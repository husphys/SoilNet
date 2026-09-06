#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load_json(relative):
    path = REPO / relative
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_csv(relative):
    path = REPO / relative
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    dataset = load_json("results/audit/dataset_summary.json")
    checkpoint = load_json("results/audit/checkpoint_summary.json")
    legacy = load_json("results/audit/legacy_summary.json")
    methods = load_csv("results/audit/methodology_audit.csv")
    claims = load_csv("results/audit/claims_evidence_matrix.csv")
    fallbacks = load_csv("data/manifests/historical_fallback_mapping.csv")
    checkpoint_rows = load_csv("results/audit/checkpoint_inventory.csv")
    statuses = Counter(row.get("evidence_status") for row in claims)
    training_only_notebooks = [row["relative_path"] for row in methods if row.get("evaluation_scope") == "TRAINING_SET_METRICS_ONLY"]
    all_fallbacks = bool(fallbacks) and all(row.get("exists", "").casefold() == "true" for row in fallbacks)
    verified_local = [row for row in checkpoint_rows if row.get("source") == "LOCAL" and row.get("status") == "VERIFIED_IDENTICAL"]
    verified_kind = "all verified overlaps are VICReg projector state_dicts, not evaluation-ready full fine-tuned models" if verified_local and all("projector" in row.get("filename", "").casefold() for row in verified_local) else "see checkpoint inventory for artifact types"
    named_candidates = [row for row in checkpoint_rows if row.get("source") == "LOCAL" and any(token in row.get("filename", "").casefold() for token in ("best", "final"))]
    large_file_report = f"""# Large file policy

- Local checkpoint files: {checkpoint.get('local_checkpoints', 'MISSING')}
- Total local bytes: {checkpoint.get('total_bytes_local', 'MISSING')}
- Files whose names contain `best` or `final`: {len(named_candidates)}
- Bytes in those name-only candidates: {sum(int(row['size_bytes']) for row in named_candidates)}
- Byte-identical local/GitHub content hashes: {checkpoint.get('verified_identical_unique_hashes', 'MISSING')}
- Approved checkpoints required for publication now: 0

Filename candidates are not selections. The five verified cross-source artifacts
are projector state dicts and do not by themselves reproduce final evaluation.
No local checkpoint has been copied, staged, or uploaded. A later selection
must document experiment ID, compatibility, metric lineage, and SHA256 before
Git LFS addition.
"""
    (REPO / "results" / "audit" / "LARGE_FILE_POLICY.md").write_text(large_file_report, encoding="utf-8")
    checksum_targets = [
        "data/manifests/dataset_inventory.csv", "data/manifests/soilnet_samples.csv",
        "data/manifests/historical_fallback_mapping.csv", "data/splits/final_split.csv",
        "data/splits/split_metadata.json", "results/audit/duplicate_report.csv",
    ]
    checksum_rows = []
    for relative in checksum_targets:
        path = REPO / relative
        if not path.exists():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        checksum_rows.append({"relative_path": relative, "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    checksum_path = REPO / "data" / "checksums" / "manifest_sha256.csv"
    with checksum_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(checksum_rows)
    claim_markdown = [
        "# Claims–evidence matrix",
        "",
        "Claimed manuscript values and exact table/figure mappings were not supplied; no value is inferred.",
        "",
        "| ID | Claim | Status | Required action |",
        "|---|---|---|---|",
    ]
    for row in claims:
        safe = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
        claim_markdown.append(f"| {safe(row['claim_id'])} | {safe(row['claim'])} | {safe(row['evidence_status'])} | {safe(row['required_action'])} |")
    claim_markdown.extend([
        "", "The full machine-readable lineage fields are in `results/audit/claims_evidence_matrix.csv`.", "",
    ])
    (REPO / "docs" / "CLAIMS_EVIDENCE_MATRIX.md").write_text("\n".join(claim_markdown), encoding="utf-8")

    decision_path = REPO / "results" / "audit" / "artifact_decision.csv"
    decisions = load_csv("results/audit/artifact_decision.csv")
    decisions = [row for row in decisions if not row.get("relative_path", "").startswith(("LOCAL:", "GITHUB:"))]
    decision_status = {
        "VERIFIED_IDENTICAL": ("KEEP_CHECKPOINT_REEVALUATE", "Byte-identical cross-source checkpoint; semantic lineage and held-out evaluation remain required."),
        "DUPLICATE": ("EXCLUDE_DUPLICATE", "Duplicate checkpoint content; retain one referenced copy only."),
        "INCOMPATIBLE": ("EXCLUDE_INVALID", "Safe structural inspection failed or format is unsupported."),
        "UNKNOWN_PROVENANCE": ("KEEP_AS_LEGACY", "Preserve locally but do not publish/use until provenance is established."),
        "LOCAL_ONLY": ("KEEP_CHECKPOINT_REEVALUATE", "Local-only checkpoint; verify experiment lineage before use."),
        "GITHUB_ONLY": ("KEEP_AS_LEGACY", "GitHub-only pinned legacy artifact."),
    }
    for row in checkpoint_rows:
        status, reason = decision_status.get(row["status"], ("KEEP_AS_LEGACY", "Unmapped checkpoint status."))
        decisions.append({"relative_path": f"{row['source']}:{row['relative_path']}", "status": status, "reason": reason})
    if decisions:
        with decision_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["relative_path", "status", "reason"])
            writer.writeheader()
            writer.writerows(decisions)
    report = f"""# FINAL AUDIT REPORT

Generated from machine-readable inventories. No long training was run.

## Direct answers

1. **Dataset size:** {dataset.get('total_images', 'MISSING')} image files among {dataset.get('total_files', 'MISSING')} total files; {dataset.get('readable_images', 'MISSING')} are readable.
2. **Why 2,057 historical samples:** `labels.csv` contains exactly {dataset.get('labeled_rows', 'MISSING')} data rows. This is the labeled subset, not all images under `DATA_ROOT`.
3. **Missing/corrupt:** {dataset.get('labeled_missing_or_corrupt', 'MISSING')} labeled rows remain unresolved after fallback mapping; {dataset.get('corrupt_images', 'MISSING')} inventoried images are corrupt/unreadable.
4. **Five fallbacks:** {'all five exist and are hash-mapped' if all_fallbacks else 'one or more are MISSING'}; affected legacy experiments remain `NOT_BITWISE_REPRODUCIBLE`.
5. **Duplicates:** {dataset.get('exact_duplicate_groups', 'MISSING')} exact groups, {dataset.get('exact_duplicate_files_beyond_first', 'MISSING')} excess exact-copy files, and {dataset.get('near_duplicate_components_with_nonidentical_sha256', 'MISSING')} candidate near-duplicate components containing nonidentical bytes at difference-hash Hamming threshold {dataset.get('near_duplicate_hamming_threshold', 'MISSING')}.
6. **Historical split:** no trustworthy held-out split was established by the static audit. The new split is prospective and cannot retroactively create unseen data for old checkpoints.
7. **Training-set evaluation notebooks/scripts:** {len(training_only_notebooks)} files; full list is in `methodology_audit.csv` (listed below).
8. **Verified checkpoints:** {checkpoint.get('verified_identical_unique_hashes', 'MISSING')} unique hashes are identical between local and GitHub sources; {verified_kind}.
9. **Local/GitHub overlap:** {checkpoint.get('verified_identical_unique_hashes', 'MISSING')} unique checkpoint hashes overlap. No full fine-tuned evaluation checkpoint is selected merely from this overlap.
10. **Experiments supporting manuscript:** source-level architecture/light claims are supported; experiment-level mappings remain in `experiment_graph.csv` because the manuscript/table mapping was not provided.
11. **Training behavior only:** historical fine-tuning regression/classification and mu-ablation downstream metrics detected on training loaders.
12. **Claims without artifacts:** irrigation, KNN, Decision Tree, and MLP are `MISSING`; GBM/SVM have result rows but no held-out lineage or implementation.
13. **Need reevaluation:** selected compatible checkpoints and classical models where frozen embeddings can be verified.
14. **Need rerun:** only a selected final fine-tune should be rerun if no truly unseen set can be proven. The full mu sweep is not justified by current evidence.
15. **Minimum GPU work:** first reuse verified checkpoints; recompute tables/CPU classical models; reevaluate on truly unseen data; otherwise one short selected group-split fine-tune. No full sweep.

## Training-set metric notebooks/scripts

""" + ("\n".join(f"- `{name}`" for name in training_only_notebooks) or "- MISSING until legacy audit runs") + f"""

## Priority table

| Priority | Work |
|---|---|
| NO_ACTION | Preserve raw data and immutable legacy notebooks. |
| CHEAP_RECOMPUTE | Rebuild tables from validated CSV; rerun classical models from verified frozen embeddings. |
| REEVALUATE_ONLY | Evaluate hash-selected compatible checkpoints on proven unseen groups. |
| SHORT_FINETUNE | One final selected configuration only if no unseen set exists. |
| FULL_RERUN | Not currently justified; use only if missing lineage cannot be recovered and the exact claim is essential. |

## Summary counts

### Dataset

- total files: {dataset.get('total_files', 'MISSING')}
- valid images: {dataset.get('readable_images', 'MISSING')}
- labeled rows: {dataset.get('labeled_rows', 'MISSING')}
- missing labeled images after mapping: {dataset.get('labeled_missing_or_corrupt', 'MISSING')}
- corrupt images: {dataset.get('corrupt_images', 'MISSING')}
- exact duplicate excess files: {dataset.get('exact_duplicate_files_beyond_first', 'MISSING')}
- candidate near-duplicate components with nonidentical bytes: {dataset.get('near_duplicate_components_with_nonidentical_sha256', 'MISSING')}

### Checkpoints

- local: {checkpoint.get('local_checkpoints', 'MISSING')}
- GitHub: {checkpoint.get('github_checkpoints', 'MISSING')}
- verified identical unique hashes: {checkpoint.get('verified_identical_unique_hashes', 'MISSING')}
- unique hashes: {checkpoint.get('unique_checkpoints_by_sha256', 'MISSING')}
- unknown provenance: {checkpoint.get('unknown_provenance', 'MISSING')}

### Claims

- supported: {statuses.get('SUPPORTED', 0)}
- training-only: {statuses.get('SUPPORTED_TRAINING_ONLY', 0)}
- partially supported: {statuses.get('PARTIALLY_SUPPORTED', 0)}
- need reevaluation: {statuses.get('NEED_REEVALUATION', 0)}
- need rerun: {statuses.get('NEED_RERUN', 0)}
- missing: {statuses.get('MISSING', 0)}
- conflicts: {statuses.get('CONFLICT_WITH_MANUSCRIPT', 0)} (not testable without manuscript values)

## Audit limitations

- GPU/NVML availability is recorded in `system_info.txt`; no GPU execution was assumed.
- No manuscript file was supplied, so exact claimed values/table/figure conflicts cannot be audited.
- Filename timestamps were not promoted to capture-session IDs without documentation.
- No historical consumed-sample manifest was found, so current extra images cannot be declared truly unseen by old checkpoints from filenames or mtimes alone.
- Checkpoint compatibility is a safe structural/key-pattern audit, not proof of semantic lineage.
"""
    (REPO / "results" / "audit" / "FINAL_AUDIT_REPORT.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
