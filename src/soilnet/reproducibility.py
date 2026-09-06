from __future__ import annotations

import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def result_metadata(
    *, seed: int, config: str, dataset_manifest_sha256: str,
    split_manifest_sha256: str, checkpoint_sha256: str,
) -> dict[str, Any]:
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "UNCOMMITTED"
    return {
        "git_commit": git_commit,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "UNAVAILABLE",
        "seed": seed,
        "config": config,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "determinism_note": "Cross-platform/GPU bitwise identity is not guaranteed.",
    }
