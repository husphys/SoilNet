#!/usr/bin/env python3
"""Write the deterministic SHA256 list for public release-critical files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "reproducibility/CHECKSUMS.sha256"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    exact = (
        "README.md",
        "LICENSE",
        ".gitattributes",
        ".gitignore",
        "pyproject.toml",
        "environment/soilnet.yml",
        "environment/requirements-lock.txt",
        "environment/README.md",
        "checkpoints/checkpoint_manifest.json",
        "data/DATA_PROVENANCE.md",
        "data/manifests/labeled_manifest.csv",
        "data/manifests/unlabeled_manifest.csv",
        "data/splits/final_clean_split_v1.csv",
        "data/splits/train_manifest.csv",
        "data/splits/validation_manifest.csv",
        "data/splits/test_manifest.csv",
    )
    patterns = (
        "config/experiments/*.yaml",
        "src/soilnet/**/*.py",
        "scripts/*.py",
        "tests/*.py",
        "checkpoints/**/*.md",
        "notebooks/final_experiments/*.ipynb",
        "reproducibility/executed_notebooks/*.ipynb",
        "results/frozen/**/*",
        "results/final_test/**/*",
        "results/model_registry/**/*",
        "results/p3_analysis/**/*",
        "results/paper/**/*",
        "results/edge/**/*",
    )
    paths = {REPO / value for value in exact}
    for pattern in patterns:
        paths.update(path for path in REPO.glob(pattern) if path.is_file())
    paths.update(
        path
        for path in (REPO / "reproducibility").glob("*")
        if path.is_file() and path.name != OUTPUT.name
    )
    rows = [(sha256(path), path.relative_to(REPO).as_posix()) for path in paths if path.is_file()]
    manifest = json.loads((REPO / "checkpoints/checkpoint_manifest.json").read_text(encoding="utf-8"))
    for record in manifest["checkpoints"]:
        if record["distribution"] == "github_release_asset":
            rows.append((record["sha256"], record["location"]))
    rows = sorted(set(rows), key=lambda row: row[1])
    OUTPUT.write_text("".join(f"{digest}  {path}\n" for digest, path in rows), encoding="utf-8")
    print(f"WROTE {len(rows)} CHECKSUMS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
