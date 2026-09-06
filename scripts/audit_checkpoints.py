#!/usr/bin/env python3
"""Hash and safely inspect project-owned checkpoints without executing pickle code."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from soilnet.io import resolve_paths, sha256_file, write_csv, write_json


def scalar(value):
    return value if isinstance(value, (str, int, float, bool)) else ""


def infer_mu(path: Path, obj) -> str:
    if isinstance(obj, dict):
        for key in ("mu", "vicreg_mu"):
            if key in obj and scalar(obj[key]) != "":
                return str(obj[key])
        config = obj.get("config")
        if isinstance(config, dict) and scalar(config.get("mu")) != "":
            return str(config["mu"])
    match = re.search(r"mu[_ =-]?(\d+(?:\.0)?)", path.name, re.I)
    return match.group(1) if match else ""


def infer_model_name(path: Path, obj) -> str:
    if isinstance(obj, dict):
        for key in ("model_name", "architecture", "arch"):
            if scalar(obj.get(key)) != "":
                return str(obj[key])
        config = obj.get("config")
        if isinstance(config, dict):
            for key in ("model_name", "architecture", "arch"):
                if scalar(config.get(key)) != "":
                    return str(config[key])
    lower = path.name.casefold()
    if "soilnet" in lower or "vicreg" in lower:
        return "SoilNet_or_VICReg_inferred_from_filename"
    if "mobilevit" in lower:
        return "MobileViTV2_inferred_from_filename"
    if "mobilenet" in lower:
        return "MobileNetV2_inferred_from_filename"
    return ""


def inspect_checkpoint(path: Path) -> dict[str, object]:
    try:
        import torch
    except ImportError:
        return {"top_level_keys": "", "structure": "", "inspect_error": "TORCH_UNAVAILABLE"}
    try:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except (RuntimeError, ValueError):
            obj = torch.load(path, map_location="cpu", weights_only=True)
        result = {
            "top_level_keys": "",
            "structure": type(obj).__name__,
            "epoch_if_available": "",
            "model_name_if_available": infer_model_name(path, obj),
            "mu_if_available": infer_mu(path, obj),
            "metric_if_available": "",
            "optimizer_state_present": False,
            "compatible_model": "UNKNOWN",
            "inspect_error": "",
        }
        state = None
        if isinstance(obj, dict):
            keys = [str(key) for key in obj.keys()]
            result["top_level_keys"] = ";".join(keys[:100])
            result["epoch_if_available"] = scalar(obj.get("epoch"))
            result["optimizer_state_present"] = any(key in obj for key in ("optimizer", "optimizer_state_dict", "optim_state_dict"))
            metric_items = []
            for key, value in obj.items():
                if any(token in str(key).casefold() for token in ("loss", "acc", "f1", "rmse", "mae", "metric")) and scalar(value) != "":
                    metric_items.append(f"{key}={value}")
            result["metric_if_available"] = ";".join(metric_items[:20])
            for key in ("model_state_dict", "state_dict", "model", "encoder_state_dict"):
                if isinstance(obj.get(key), dict):
                    state = obj[key]
                    result["structure"] = f"dict containing {key}"
                    break
            if state is None and obj and all(hasattr(value, "shape") for value in obj.values()):
                state = obj
                result["structure"] = "raw_state_dict"
        if isinstance(state, dict):
            state_keys = [str(key).removeprefix("module.") for key in state]
            if any(key.startswith(("initial_conv.", "mnv2_block1.", "mobilevit_encoder.")) for key in state_keys):
                result["compatible_model"] = "SOILNET_KEY_PATTERN"
            elif any("projector" in key for key in state_keys):
                result["compatible_model"] = "VICREG_PROJECTOR_KEY_PATTERN"
            else:
                result["compatible_model"] = "STATE_DICT_UNMAPPED"
        del obj
        gc.collect()
        return result
    except Exception as exc:
        return {
            "top_level_keys": "", "structure": "LOAD_FAILED", "epoch_if_available": "",
            "model_name_if_available": infer_model_name(path, {}), "mu_if_available": infer_mu(path, {}),
            "metric_if_available": "", "optimizer_state_present": "", "compatible_model": "INCOMPATIBLE_OR_UNSUPPORTED",
            "inspect_error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path)
    parser.add_argument("--hash-only", action="store_true", help="Skip torch metadata inspection")
    parser.add_argument("--reuse-existing", action="store_true", help="Reclassify an existing inventory without reading source bytes")
    args = parser.parse_args()
    paths = resolve_paths(args.paths)
    sources = [("LOCAL", paths["checkpoint_root"]), ("GITHUB", paths["legacy_root"])]
    records = []
    inventory_path = REPO / "results" / "audit" / "checkpoint_inventory.csv"
    if args.reuse_existing:
        with inventory_path.open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
    else:
        for source, root in sources:
            files = sorted(root.rglob("*.pth"), key=lambda path: path.as_posix().casefold())
            for count, path in enumerate(files, 1):
                digest = sha256_file(path)
                metadata = {
                    "top_level_keys": "", "structure": "NOT_INSPECTED", "epoch_if_available": "",
                    "model_name_if_available": infer_model_name(path, {}), "mu_if_available": infer_mu(path, {}),
                    "metric_if_available": "", "optimizer_state_present": "", "compatible_model": "UNKNOWN",
                    "inspect_error": "HASH_ONLY",
                } if args.hash_only else inspect_checkpoint(path)
                records.append({
                    "checkpoint_id": f"sha256:{digest[:20]}",
                    "source": source,
                    "relative_path": path.relative_to(root).as_posix(),
                    "filename": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": digest,
                    **metadata,
                    "suspected_notebook": "TO_MAP_BY_EXPERIMENT_GRAPH",
                })
                if count % 25 == 0:
                    print(f"{source}: hashed/inspected {count}/{len(files)}", flush=True)
    by_hash = defaultdict(list)
    for record in records:
        if "projector" in record["filename"].casefold() and record["compatible_model"] == "STATE_DICT_UNMAPPED":
            record["compatible_model"] = "VICREG_PROJECTOR_KEY_PATTERN"
        by_hash[record["sha256"]].append(record)
    for record in records:
        same = by_hash[record["sha256"]]
        opposite = [item for item in same if item["source"] != record["source"]]
        record["matching_legacy_sha256"] = ";".join(item["relative_path"] for item in opposite if item["source"] == "GITHUB")
        if record["inspect_error"] not in {"", "HASH_ONLY", "TORCH_UNAVAILABLE"}:
            record["status"] = "INCOMPATIBLE"
        elif opposite:
            record["status"] = "VERIFIED_IDENTICAL"
        elif len(same) > 1:
            record["status"] = "DUPLICATE"
        elif record["source"] == "LOCAL" and (
            record["compatible_model"] in {"UNKNOWN", "STATE_DICT_UNMAPPED"}
            or not record["model_name_if_available"]
        ):
            record["status"] = "UNKNOWN_PROVENANCE"
        elif record["source"] == "LOCAL":
            record["status"] = "LOCAL_ONLY"
        else:
            record["status"] = "GITHUB_ONLY"
    fields = [
        "checkpoint_id", "source", "relative_path", "filename", "size_bytes", "sha256", "top_level_keys",
        "structure", "epoch_if_available", "model_name_if_available", "mu_if_available", "metric_if_available",
        "optimizer_state_present", "compatible_model", "matching_legacy_sha256", "suspected_notebook", "status", "inspect_error",
    ]
    write_csv(inventory_path, records, fields)
    hashes_by_source = {source: {row["sha256"] for row in records if row["source"] == source} for source, _ in sources}
    verified = hashes_by_source["LOCAL"] & hashes_by_source["GITHUB"]
    summary = {
        "local_checkpoints": sum(row["source"] == "LOCAL" for row in records),
        "github_checkpoints": sum(row["source"] == "GITHUB" for row in records),
        "verified_identical_unique_hashes": len(verified),
        "verified_identical_rows": sum(row["status"] == "VERIFIED_IDENTICAL" for row in records),
        "unique_checkpoints_by_sha256": len(by_hash),
        "unknown_provenance": sum(row["status"] == "UNKNOWN_PROVENANCE" for row in records),
        "local_only": sum(row["status"] == "LOCAL_ONLY" for row in records),
        "github_only": sum(row["status"] == "GITHUB_ONLY" for row in records),
        "duplicate_rows": sum(row["status"] == "DUPLICATE" for row in records),
        "incompatible_rows": sum(row["status"] == "INCOMPATIBLE" for row in records),
        "total_bytes_local": sum(int(row["size_bytes"]) for row in records if row["source"] == "LOCAL"),
        "policy": "No checkpoint is copied or selected automatically. Prefer verified final/best checkpoints after experiment mapping.",
    }
    write_json(REPO / "results" / "audit" / "checkpoint_summary.json", summary)
    selected_candidates = [row for row in records if row["source"] == "LOCAL" and re.search(r"(?:best|final)", row["filename"], re.I)]
    write_json(REPO / "results" / "audit" / "checkpoint_selection_candidates.json", {
        "candidate_count": len(selected_candidates),
        "candidate_total_bytes": sum(int(row["size_bytes"]) for row in selected_candidates),
        "warning": "Candidates are names only, not approved selections. Selection requires experiment/checkpoint/metric lineage.",
        "candidates": [{key: row[key] for key in ("checkpoint_id", "relative_path", "filename", "size_bytes", "sha256", "status")} for row in selected_candidates],
    })
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
