#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def read_json(name):
    path = REPO / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main():
    dataset = read_json("results/audit/dataset_summary.json")
    checkpoints = read_json("results/audit/checkpoint_summary.json")
    claim_path = REPO / "results/audit/claims_evidence_matrix.csv"
    with claim_path.open(newline="", encoding="utf-8") as handle:
        statuses = Counter(row["evidence_status"] for row in csv.DictReader(handle))
    print("DATASET")
    print(f"total files: {dataset.get('total_files', 'MISSING')}")
    print(f"valid images: {dataset.get('readable_images', 'MISSING')}")
    print(f"labeled rows: {dataset.get('labeled_rows', 'MISSING')}")
    print(f"missing images: {dataset.get('labeled_missing_or_corrupt', 'MISSING')}")
    print(f"corrupt images: {dataset.get('corrupt_images', 'MISSING')}")
    print(f"exact duplicate excess files: {dataset.get('exact_duplicate_files_beyond_first', 'MISSING')}")
    print(f"candidate near-duplicate components: {dataset.get('near_duplicate_components_with_nonidentical_sha256', 'MISSING')}")
    print("\nCHECKPOINTS")
    print(f"local checkpoints: {checkpoints.get('local_checkpoints', 'MISSING')}")
    print(f"GitHub checkpoints: {checkpoints.get('github_checkpoints', 'MISSING')}")
    print(f"verified identical: {checkpoints.get('verified_identical_unique_hashes', 'MISSING')}")
    print(f"unique checkpoints: {checkpoints.get('unique_checkpoints_by_sha256', 'MISSING')}")
    print(f"unknown provenance: {checkpoints.get('unknown_provenance', 'MISSING')}")
    print("\nEXPERIMENTS")
    print(f"supported: {statuses['SUPPORTED']}")
    print(f"training-only: {statuses['SUPPORTED_TRAINING_ONLY']}")
    print(f"need reevaluation: {statuses['NEED_REEVALUATION']}")
    print(f"need rerun: {statuses['NEED_RERUN']}")
    print(f"missing: {statuses['MISSING']}")

    unsupported = sum(
        statuses[name]
        for name in (
            "SUPPORTED_TRAINING_ONLY",
            "NEED_REEVALUATION",
            "NEED_RERUN",
            "MISSING",
            "CONFLICT_WITH_MANUSCRIPT",
        )
    )
    print("\nMANUSCRIPT")
    print(f"claims supported: {statuses['SUPPORTED']}")
    print(f"partially supported: {statuses['PARTIALLY_SUPPORTED']}")
    print(f"unsupported or insufficient for held-out claims: {unsupported}")
    print(f"conflicting protocol/results: {statuses['CONFLICT_WITH_MANUSCRIPT']} (manuscript values not supplied)")

    print("\nRECOMPUTATION")
    print("no training: audit/table regeneration and recovery of verified artifacts")
    print("evaluation only: selected lineage-compatible checkpoints; none selected automatically")
    print("short fine-tune: 1 selected configuration, conditional on absence of truly unseen data")
    print("full rerun: 0 currently justified")

    ready = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    status = subprocess.run(
        ["git", "status", "--short", "--branch"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().replace("\n", "; ")
    repo_bytes = sum(
        path.stat().st_size
        for path in REPO.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(REPO).parts
    )
    print("\nGIT")
    print(f"files ready: {len(ready)} non-ignored files")
    print("files excluded: raw dataset, source checkpoints, local path config, caches, credentials")
    print(f"repository working-tree size: {repo_bytes} bytes")
    print("Git LFS payload size: 0 bytes (no checkpoint selected)")
    print(f"current git status: {status}")


if __name__ == "__main__":
    main()
