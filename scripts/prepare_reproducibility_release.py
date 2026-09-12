#!/usr/bin/env python3
"""Package frozen SoilNet artifacts without training or dataset image access.

The script only reads canonical run artifacts and writes public, path-normalized
copies. It intentionally excludes rolling/resume checkpoints, optimizer state,
raw images, and every P3 test artifact.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]

EXPERIMENTS = {
    "P0": {
        "run": "P0_FINAL_SOILNET_VICREG_MU27_LI_V4_BESTREG",
        "config": "P0_final_soilnet_v4_bestreg.yaml",
        "test_predictions": "P0_test_predictions.csv",
        "test_metrics": "P0_test_metrics.json",
    },
    "P1_noLI": {
        "run": "P1_SOILNET_VICREG_MU27_NO_LI_BESTREG",
        "config": "P1_no_li_bestreg.yaml",
        "test_predictions": "P1_noLI_test_predictions.csv",
        "test_metrics": "P1_noLI_test_metrics.json",
    },
    "P1_noSSL": {
        "run": "P1_SOILNET_IMAGENET_LI_NO_SSL_BESTREG",
        "config": "P1_no_ssl_bestreg.yaml",
        "test_predictions": "P1_noSSL_test_predictions.csv",
        "test_metrics": "P1_noSSL_test_metrics.json",
    },
    "P2": {
        "run": "P2_MOBILEVITV2_IMAGENET_LI_BESTREG",
        "config": "P2_mobilevitv2_imagenet_li_bestreg.yaml",
        "test_predictions": "P2_mobilevitv2_test_predictions.csv",
        "test_metrics": "P2_MobileViTv2_test_metrics.json",
    },
}


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _public_path(value: str, run_root: Path) -> str:
    mappings = (
        (str(REPO) + "/", "repo://"),
        (str(run_root) + "/", "release-artifact://"),
        (str(run_root.parent) + "/", "checkpoint-root://"),
        ("/mnt/e/soil_data_set/", "data-root://"),
        (str(run_root.parent / "soilnet_stale_p3") + "/", "withheld-stale-provenance://"),
    )
    for prefix, replacement in mappings:
        if value.startswith(prefix):
            return replacement + value[len(prefix):]
    return value


def _normalize_strings(value: Any, run_root: Path) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_strings(item, run_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_strings(item, run_root) for item in value]
    if isinstance(value, str):
        return _public_path(value, run_root)
    return value


def _copy_public_json(source: Path, destination: Path, run_root: Path) -> None:
    original = json.loads(source.read_text(encoding="utf-8"))
    normalized = _normalize_strings(original, run_root)
    if isinstance(normalized, dict):
        normalized["_public_release"] = {
            "normalization": "machine-local path strings replaced; numerical values unchanged",
            "source_artifact": _public_path(str(source), run_root),
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(normalized, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_split_manifests() -> None:
    source = REPO / "data/splits/final_clean_split_v1.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or ())
    expected = {"train": 1407, "validation": 289, "test": 231}
    for split, count in expected.items():
        selected = [row for row in rows if row["split"] == split]
        if len(selected) != count:
            raise RuntimeError(f"Locked {split} count changed: {len(selected)} != {count}")
        destination = REPO / f"data/splits/{split}_manifest.csv"
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected)
    _copy(REPO / "data/manifests/final_clean_manifest_v1.csv", REPO / "data/manifests/labeled_manifest.csv")


def package(run_root: Path) -> None:
    if not run_root.is_dir():
        raise FileNotFoundError(run_root)
    frozen_root = REPO / "results/frozen"
    for key, spec in EXPERIMENTS.items():
        source = run_root / spec["run"]
        destination = frozen_root / key
        for filename in ("training_history.csv", "validation_predictions.csv", "checkpoint_sha256.txt"):
            _copy(source / filename, destination / filename)
        _copy_public_json(source / "validation_metrics.json", destination / "validation_metrics.json", run_root)
        _copy_public_json(source / "run_metadata.json", destination / "run_metadata.json", run_root)
        _copy(
            REPO / "config/experiments" / spec["config"],
            destination / "resolved_config.yaml",
        )
        _copy(
            REPO / "results/final_test" / spec["test_predictions"],
            destination / "test_predictions.csv",
        )
        _copy_public_json(
            REPO / "results/final_test" / spec["test_metrics"],
            destination / "test_metrics.json",
            run_root,
        )

    p3_source = run_root / "P3_MOBILEVITV2_VICREG_LI_BESTREG"
    p3_destination = frozen_root / "P3"
    for filename in ("training_history.csv", "unlabeled_manifest.csv", "checkpoint_sha256.txt"):
        _copy(p3_source / "pretraining" / filename, p3_destination / "pretraining" / filename)
    _copy_public_json(
        p3_source / "pretraining/pretraining_metadata.json",
        p3_destination / "pretraining/pretraining_metadata.json",
        run_root,
    )
    for filename in ("training_history.csv", "validation_predictions.csv", "checkpoint_sha256.txt"):
        _copy(p3_source / "supervised" / filename, p3_destination / "supervised" / filename)
    for filename in ("validation_metrics.json", "run_metadata.json"):
        _copy_public_json(
            p3_source / "supervised" / filename,
            p3_destination / "supervised" / filename,
            run_root,
        )
    for filename in ("validation_comparison.csv", "validation_comparison.json"):
        source = p3_source / filename
        destination = p3_destination / filename
        if source.suffix == ".json":
            _copy_public_json(source, destination, run_root)
        else:
            _copy(source, destination)
    _copy(p3_source / "resolved_config.yaml", p3_destination / "resolved_config.yaml")
    _copy(
        p3_source / "pretraining/unlabeled_manifest.csv",
        REPO / "data/manifests/unlabeled_manifest.csv",
    )
    _write_split_manifests()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(os.environ["RUN_ARTIFACT_ROOT"]) if os.environ.get("RUN_ARTIFACT_ROOT") else None,
    )
    args = parser.parse_args()
    if args.run_root is None:
        raise RuntimeError("--run-root or RUN_ARTIFACT_ROOT is required")
    package(args.run_root.resolve())
    print("PACKAGING_COMPLETE_NO_TRAINING_NO_TEST_DATASET_ACCESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
