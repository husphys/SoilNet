#!/usr/bin/env python3
"""Print only the compact Phase 2 handoff and canonical report path."""
from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    summary = json.loads((REPO / "results/audit/phase2_summary.json").read_text(encoding="utf-8"))
    print(
        "PHASE 2: "
        f"{summary['labeled_exact_unique']}/{summary['labeled_rows']} exact-unique/labeled; "
        f"checkpoints A/B/C={summary['checkpoint_root']}/{summary['legacy_github']}/{summary['data_root_embedded']}; "
        f"historical held-out={summary['valid_historical_heldout']}; "
        f"decision={summary['decision']}; training_run={str(summary['training_run']).lower()}"
    )
    print(REPO / "results/audit/PHASE2_DECISION.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
