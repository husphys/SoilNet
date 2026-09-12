#!/usr/bin/env python3
"""Normalize source-machine paths in public executed-notebook copies."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO / "reproducibility/executed_notebooks"
REPLACEMENTS = (
    (str(REPO) + "/", "repo://"),
    ("/mnt/d/check point/soilnet_final_runs/", "release-artifact://"),
    ("/mnt/d/check point/", "checkpoint-root://"),
    ("/mnt/e/soil_data_set/", "data-root://"),
    ("/home/diy-hus/", "user-home://"),
)


def normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if isinstance(value, str):
        for source, replacement in REPLACEMENTS:
            value = value.replace(source, replacement)
        return value
    return value


def main() -> int:
    for path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        notebook = normalize(json.loads(path.read_text(encoding="utf-8")))
        notebook.setdefault("metadata", {})["soilnet_public_release"] = {
            "executed_output_retained": True,
            "normalization": "machine-local path strings replaced; numerical outputs unchanged",
        }
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(path.relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
