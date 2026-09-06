from __future__ import annotations

import csv
import platform
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import torch

from soilnet.io import load_yaml, resolve_paths, resolve_run_artifact_root, sha256_file


REPO = Path(__file__).resolve().parents[3]
LOCKED_MANIFEST_SHA256 = "8ff45054d4b8e3df9758d0c112dc16719572b2267906fb3a5ed5b3262a6732bd"
LOCKED_SPLIT_SHA256 = "8927b8223b8c4c234d264ad9ea62ac2df6161124a79787a2e71eeb5cd23eae2f"
LOCKED_SSL_SHA256 = "42599ac025b8d8c8c5b8cca36624665f155e0ff588cd90de2fd47b08cd2d60d8"
P0_V3_EXPERIMENTS = {
    "P0_FINAL_SOILNET_VICREG_MU27_LI",
    "P1_ABLATION_NO_LI",
    "P1_ABLATION_NO_SSL",
}
P0_V4_EXPERIMENTS = {
    "P0_FINAL_SOILNET_VICREG_MU27_LI_V4_BESTREG",
    "P1_SOILNET_VICREG_MU27_NO_LI_BESTREG",
    "P1_SOILNET_IMAGENET_LI_NO_SSL_BESTREG",
    "P2_MOBILEVITV2_IMAGENET_LI_BESTREG",
}
RUNNABLE_STATUSES = {
    "READY", "PROTOCOL_REVISED_PRE_TEST", "OPTIONAL_NOT_YET_REQUIRED", "OPTIONAL",
}


@dataclass(frozen=True)
class ExperimentContext:
    config_path: Path
    config: dict[str, Any]
    config_sha256: str
    manifest_path: Path
    manifest_sha256: str
    split_path: Path
    split_sha256: str
    data_root: Path
    checkpoint_root: Path
    run_dir: Path


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def print_environment(context: ExperimentContext) -> dict[str, Any]:
    values = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": _version("torchvision"),
        "torchaudio": _version("torchaudio"),
        "timm": _version("timm"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU_BLOCKED",
        "torch_cuda_runtime": torch.version.cuda,
        "seed": context.config["seed"],
        "config_sha256": context.config_sha256,
        "manifest_sha256": context.manifest_sha256,
        "split_sha256": context.split_sha256,
    }
    for key, value in values.items():
        print(f"{key}: {value}")
    return values


def load_experiment_context(config_path: Path | str) -> ExperimentContext:
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO / path
    config = load_yaml(path)
    if config.get("protocol_status") not in RUNNABLE_STATUSES:
        raise RuntimeError(f"Experiment protocol is blocked: {config.get('protocol_status')}")
    if config.get("experiment_kind") == "classical_ml":
        locked = {"epochs": 0, "batch_size": 32, "seed": 20260905}
    elif config.get("experiment_id") in P0_V3_EXPERIMENTS | P0_V4_EXPERIMENTS:
        locked = {
            "epochs": 60, "batch_size": 32, "optimizer": "Adam",
            "learning_rate": 0.0001, "weight_decay": 0.0, "seed": 20260905,
        }
    else:
        locked = {
            "epochs": 15, "batch_size": 32, "optimizer": "Adam",
            "learning_rate": 0.0005, "weight_decay": 0.0, "seed": 20260905,
        }
    for key, expected in locked.items():
        if config.get(key) != expected:
            raise RuntimeError(f"Locked parameter changed: {key}={config.get(key)!r}, expected {expected!r}")
    paths = resolve_paths()
    run_root = resolve_run_artifact_root()
    manifest = REPO / config["dataset_manifest"]
    split = REPO / config["split"]
    manifest_sha = sha256_file(manifest)
    split_sha = sha256_file(split)
    if manifest_sha != config.get("dataset_manifest_sha256") or manifest_sha != LOCKED_MANIFEST_SHA256:
        raise RuntimeError("STOP: final clean manifest SHA256 mismatch")
    if split_sha != config.get("split_sha256") or split_sha != LOCKED_SPLIT_SHA256:
        raise RuntimeError("STOP: final clean split SHA256 mismatch")
    with split.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    counts = {name: sum(row["split"] == name for row in rows) for name in ("train", "validation", "test")}
    if counts != {"train": 1407, "validation": 289, "test": 231}:
        raise RuntimeError(f"STOP: locked split counts changed: {counts}")
    ssl = config.get("ssl_checkpoint")
    if ssl:
        if ssl.get("sha256") != LOCKED_SSL_SHA256:
            raise RuntimeError("STOP: configured SSL SHA256 changed")
        ssl_path = paths["checkpoint_root"] / ssl["relative_path"]
        if not ssl_path.is_file() or sha256_file(ssl_path) != LOCKED_SSL_SHA256:
            raise RuntimeError("STOP: selected SSL checkpoint missing or changed")
    initialization_checkpoint = config.get("initialization_checkpoint")
    if initialization_checkpoint:
        if initialization_checkpoint.get("scope") != "DATA_ROOT":
            raise RuntimeError("STOP: unsupported initialization-checkpoint scope")
        initialization_path = paths["data_root"] / initialization_checkpoint["relative_path"]
        expected_initialization_sha = initialization_checkpoint.get("sha256")
        if not initialization_path.is_file() or sha256_file(initialization_path) != expected_initialization_sha:
            raise RuntimeError("STOP: pre-VICReg initialization checkpoint missing or changed")
    run_dir = (run_root / config["output_dir"]).resolve()
    try:
        run_dir.relative_to(REPO)
    except ValueError:
        pass
    else:
        raise RuntimeError("Run artifacts must be stored outside the repository")
    return ExperimentContext(
        config_path=path, config=config, config_sha256=sha256_file(path),
        manifest_path=manifest, manifest_sha256=manifest_sha,
        split_path=split, split_sha256=split_sha,
        data_root=paths["data_root"], checkpoint_root=paths["checkpoint_root"], run_dir=run_dir,
    )
