#!/usr/bin/env python3
"""Read-only audit of the frozen SoilNet reproducibility package."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RELEASE_ONLY_PREFIX = "checkpoints/release-assets/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_checksums(release_asset_dir: Path | None) -> tuple[int, int]:
    checked = missing_assets = 0
    for line in (REPO / "reproducibility/CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        expected, relative = line.split("  ", 1)
        path = REPO / relative
        if relative.startswith(RELEASE_ONLY_PREFIX):
            if release_asset_dir is None:
                missing_assets += 1
                continue
            path = release_asset_dir / Path(relative).name
        if not path.is_file():
            raise RuntimeError(f"Checksum target missing: {relative}")
        if sha256(path) != expected:
            raise RuntimeError(f"Checksum mismatch: {relative}")
        checked += 1
    return checked, missing_assets


def verify_baseline() -> int:
    count = 0
    inventory = REPO / "reproducibility/frozen_artifact_inventory_before_packaging.sha256"
    for line in inventory.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        expected, raw_path = line.split("  ", 1)
        path = Path(raw_path)
        if not path.is_absolute():
            path = REPO / path
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Frozen baseline changed or disappeared: {raw_path}")
        count += 1
    return count


def verify_manifests() -> dict[str, int]:
    expected = {"train": 1407, "validation": 289, "test": 231}
    combined = rows(REPO / "data/splits/final_clean_split_v1.csv")
    if len(combined) != 1927 or len({row["sample_id"] for row in combined}) != 1927:
        raise RuntimeError("Labeled manifest is not 1,927 unique sample IDs")
    for split, count in expected.items():
        selected = rows(REPO / f"data/splits/{split}_manifest.csv")
        if len(selected) != count or any(row["split"] != split for row in selected):
            raise RuntimeError(f"{split} manifest count/content mismatch")
        if any(not row.get("sample_id") or not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", "")) for row in selected):
            raise RuntimeError(f"{split} manifest lacks stable IDs/SHA256")
    unlabeled = rows(REPO / "data/manifests/unlabeled_manifest.csv")
    if len(unlabeled) != 11995:
        raise RuntimeError("Unlabeled manifest count mismatch")
    if any(Path(row["relative_path"]).is_absolute() or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) for row in unlabeled):
        raise RuntimeError("Unlabeled manifest path/SHA256 contract failed")
    return {**expected, "labeled": len(combined), "unlabeled": len(unlabeled)}


def verify_p3_boundary() -> None:
    p3 = REPO / "results/frozen/P3"
    if list(p3.rglob("*test*")):
        raise RuntimeError("P3 package contains a test-named artifact")
    metadata = json.loads((p3 / "supervised/run_metadata.json").read_text(encoding="utf-8"))
    factorial = json.loads((REPO / "results/p3_analysis/factorial_validation_summary.json").read_text(encoding="utf-8"))
    if metadata.get("test_evaluated") != "NO" or factorial.get("TEST_SET_OPENED") is not False:
        raise RuntimeError("P3 test firewall evidence failed")
    registry = (REPO / "reproducibility/EXPERIMENT_REGISTRY.md").read_text(encoding="utf-8")
    if "validation-only follow-up; test set not opened" not in registry:
        raise RuntimeError("P3 registry disclosure missing")


def verify_notebooks() -> tuple[int, int]:
    clean = executed = 0
    for path in sorted((REPO / "notebooks/final_experiments").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code" and (cell.get("execution_count") is not None or cell.get("outputs")):
                raise RuntimeError(f"Public source notebook is executed: {path.name}")
        clean += 1
    for path in sorted((REPO / "reproducibility/executed_notebooks").glob("*.ipynb")):
        json.loads(path.read_text(encoding="utf-8"))
        executed += 1
    return clean, executed


def scan_release_content() -> int:
    secret_patterns = (
        re.compile(r"ghp_[A-Za-z0-9]{20,}"),
        re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
        re.compile(r"(?i)(?:password|api[_-]?key|telegram[_-]?token)\s*[:=]\s*['\"][^'\"]{6,}"),
    )
    scanned = 0
    excluded_parts = {".git", ".pytest_cache", "__pycache__"}
    for path in REPO.rglob("*"):
        if not path.is_file() or any(part in excluded_parts for part in path.parts):
            continue
        if path.suffix.lower() in {".pth", ".png", ".pdf", ".pyc"} or path.stat().st_size > 5_000_000:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(pattern.search(text) for pattern in secret_patterns):
            raise RuntimeError(f"Potential credential/private key in {path.relative_to(REPO)}")
        scanned += 1
    public_roots = (
        REPO / "results/frozen",
        REPO / "notebooks/final_experiments",
        REPO / "reproducibility/executed_notebooks",
    )
    for root in public_roots:
        for path in root.rglob("*"):
            if path.is_file():
                text = path.read_text(encoding="utf-8", errors="ignore")
                if any(value in text for value in ("/home/diy-hus", "/mnt/d", "/mnt/e")):
                    raise RuntimeError(f"Source-machine path in public copy: {path.relative_to(REPO)}")
    if any(path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"} for path in (REPO / "data").rglob("*")):
        raise RuntimeError("Raw image found under data/")
    return scanned


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-asset-dir", type=Path)
    parser.add_argument("--require-release-assets", action="store_true")
    args = parser.parse_args()
    release_dir = args.release_asset_dir.resolve() if args.release_asset_dir else None
    checked, missing = verify_checksums(release_dir)
    if args.require_release_assets and missing:
        raise RuntimeError(f"{missing} GitHub Release assets were not provided")
    report = {
        "status": "PASS",
        "checksum_files_verified": checked,
        "release_assets_not_locally_required": missing,
        "frozen_baseline_files_verified": verify_baseline(),
        "split_counts": verify_manifests(),
        "notebooks": dict(zip(("clean", "executed"), verify_notebooks(), strict=True)),
        "text_files_secret_scanned": scan_release_content(),
        "P3_TEST_SET_OPENED": False,
        "training_run": False,
    }
    verify_p3_boundary()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
