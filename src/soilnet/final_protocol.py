from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from soilnet.io import load_yaml, sha256_file, write_json


READY_STATUS = "READY_TO_RUN_FINAL_REVALIDATION"


def normalize_historical_soilnet_state(
    state: dict[str, Any], canonical_state: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, int]]:
    """Remove only verified legacy MobileViT registration aliases.

    Historical notebooks registered both ``mobilevit_full`` and its ``stages``
    child as ``mobilevit_encoder``. The forward path used the latter. Every
    canonical key must still exist with the exact expected shape; this is not a
    permissive/non-strict loader.
    """
    normalized = {str(key).removeprefix("module.").removeprefix("model."): value for key, value in state.items()}
    missing = set(canonical_state) - set(normalized)
    shape_mismatch = {
        key for key in canonical_state
        if key in normalized and tuple(normalized[key].shape) != tuple(canonical_state[key].shape)
    }
    extras = set(normalized) - set(canonical_state)
    if missing or shape_mismatch:
        raise RuntimeError(
            f"Historical state is not canonical-compatible: missing={len(missing)}, shape_mismatch={len(shape_mismatch)}"
        )
    if any(not key.startswith("mobilevit_full.") for key in extras):
        raise RuntimeError("Historical state has non-alias unexpected keys")
    stage_aliases = [key for key in extras if key.startswith("mobilevit_full.stages.")]
    for key in stage_aliases:
        canonical_key = "mobilevit_encoder." + key.removeprefix("mobilevit_full.stages.")
        if canonical_key not in normalized or not torch.equal(normalized[key], normalized[canonical_key]):
            raise RuntimeError(f"Legacy MobileViT alias mismatch: {key}")
    filtered = {key: normalized[key] for key in canonical_state}
    return filtered, {
        "canonical_keys": len(canonical_state), "legacy_keys": len(normalized),
        "verified_alias_keys_removed": len(extras), "verified_stage_alias_pairs": len(stage_aliases),
    }


def load_active_config(config_path: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    if config.get("protocol_status") != READY_STATUS or not config.get("training_allowed"):
        raise RuntimeError(f"Final training is blocked by protocol status: {config.get('protocol_status', 'MISSING')}")
    return config


def verify_locked_inputs(repo: Path, config_path: Path, lock_path: Path) -> dict[str, Any]:
    config = load_active_config(config_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    training_allowed = lock.get("training_allowed_by_protocol", lock.get("training_allowed"))
    if lock.get("lock_state") != "ACTIVE" or not training_allowed:
        raise RuntimeError(f"Final run lock is not active: {lock.get('lock_state', 'MISSING')}")
    current_commit = current_git_commit(repo)
    locked_commit = lock.get("git_commit")
    if locked_commit not in {None, "", "UNCOMMITTED"}:
        if current_commit != locked_commit:
            raise RuntimeError("Final run requires a Git state matching the run lock")
    else:
        if lock.get("git_state_policy") != "exact_content_hashes_when_no_commit_exists":
            raise RuntimeError("Uncommitted lock lacks an explicit exact-content policy")
        for relative, expected in lock.get("code_sha256", {}).items():
            path = repo / relative
            if not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(f"Locked code missing or changed: {relative}")
    if sha256_file(config_path) != lock.get("config_sha256"):
        raise RuntimeError("Final config SHA256 does not match the run lock")
    for config_key, lock_key in (
        ("final_clean_manifest", "final_clean_manifest_sha256"),
        ("final_split_manifest", "final_split_sha256"),
    ):
        relative = config["data"].get(config_key)
        if not relative:
            raise RuntimeError(f"Missing locked config field: data.{config_key}")
        path = repo / relative
        if not path.is_file() or sha256_file(path) != lock.get(lock_key):
            raise RuntimeError(f"Locked input missing or changed: {relative}")
    return {"config": config, "lock": lock}


def refuse_if_test_completed(marker_path: Path) -> None:
    if marker_path.exists():
        raise RuntimeError(f"Final test evaluation already completed: {marker_path}")


def current_git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNCOMMITTED"


def write_test_completion_marker(
    marker_path: Path,
    *,
    repo: Path,
    model_path: Path,
    split_path: Path,
    config_path: Path,
) -> None:
    refuse_if_test_completed(marker_path)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(repo),
        "model_sha256": sha256_file(model_path),
        "split_sha256": sha256_file(split_path),
        "config_sha256": sha256_file(config_path),
    }
    # Exclusive creation prevents an automatic second evaluation from replacing
    # evidence of the first one.
    descriptor = os.open(marker_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except Exception:
        marker_path.unlink(missing_ok=True)
        raise
