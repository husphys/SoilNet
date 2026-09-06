from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_paths(config_path: Path | None = None) -> dict[str, Path]:
    config_path = config_path or REPO_ROOT / "config" / "paths.local.yaml"
    config = load_yaml(config_path) if config_path.exists() else {}
    values = {
        "data_root": os.environ.get("DATA_ROOT") or config.get("dataset_root") or config.get("data_root"),
        "checkpoint_root": os.environ.get("CHECKPOINT_ROOT") or config.get("checkpoint_root"),
        "legacy_root": os.environ.get("LEGACY_ROOT") or config.get("legacy_root"),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ValueError(f"Missing path configuration: {', '.join(missing)}")
    return {name: Path(value).expanduser().resolve() for name, value in values.items()}


def resolve_run_artifact_root(config_path: Path | None = None) -> Path:
    config_path = config_path or REPO_ROOT / "config" / "paths.local.yaml"
    config = load_yaml(config_path) if config_path.exists() else {}
    value = os.environ.get("RUN_ARTIFACT_ROOT") or config.get("run_artifact_root")
    if not value:
        raise ValueError("Missing run_artifact_root or RUN_ARTIFACT_ROOT")
    return Path(value).expanduser().resolve()


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
