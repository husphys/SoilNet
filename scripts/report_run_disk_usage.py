#!/usr/bin/env python3
"""Report external run checkpoint usage without deleting anything."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from soilnet.io import resolve_run_artifact_root


def main() -> int:
    root = resolve_run_artifact_root()
    rows = []
    if root.exists():
        for directory in sorted(path for path in root.iterdir() if path.is_dir()):
            checkpoints = sorted(path for path in directory.rglob("*.pth") if path.is_file())
            rows.append({
                "experiment": directory.name,
                "checkpoint_count": len(checkpoints),
                "total_size_bytes": sum(path.stat().st_size for path in checkpoints),
                "checkpoints": [str(path.relative_to(root)) for path in checkpoints],
                "retention_policy_pass": len(checkpoints) <= 2,
            })
    report = {
        "run_artifact_root": str(root),
        "experiment_count": len(rows),
        "checkpoint_count": sum(row["checkpoint_count"] for row in rows),
        "total_size_bytes": sum(row["total_size_bytes"] for row in rows),
        "experiments": rows,
        "files_deleted": 0,
    }
    print(json.dumps(report, indent=2))
    return 0 if all(row["retention_policy_pass"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
