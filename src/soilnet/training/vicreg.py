from __future__ import annotations

import csv
import copy
import json
import os
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from soilnet.io import sha256_file, sha256_text, write_json


class TwoViewUnlabeledDataset(Dataset):
    def __init__(self, data_root: Path, rows: list[dict[str, str]], transform):
        self.data_root = data_root
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        path = self.data_root / self.rows[index]["relative_path"]
        with Image.open(path) as image:
            image = image.convert("RGB")
            return self.transform(image), self.transform(image)


class VICRegProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    *,
    lambda_invariance: float,
    mu_variance: float,
    nu_covariance: float,
    epsilon: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    invariance = F.mse_loss(z1, z2)

    def variance_term(z: torch.Tensor) -> torch.Tensor:
        standard_deviation = torch.sqrt(z.var(dim=0) + epsilon)
        return torch.mean(F.relu(1 - standard_deviation))

    variance = variance_term(z1) + variance_term(z2)

    def covariance_term(z: torch.Tensor) -> torch.Tensor:
        centered = z - z.mean(dim=0)
        covariance = (centered.T @ centered) / (z.shape[0] - 1)
        off_diagonal = covariance - torch.diag(covariance.diag())
        return off_diagonal.pow(2).sum() / z.shape[1]

    covariance = covariance_term(z1) + covariance_term(z2)
    total = (
        lambda_invariance * invariance
        + mu_variance * variance
        + nu_covariance * covariance
    )
    return total, {"invariance": invariance, "variance": variance, "covariance": covariance}


def _augmentation(recipe: dict[str, Any]):
    augmentation = recipe["augmentation"]
    jitter = augmentation["color_jitter"]
    blur = augmentation["gaussian_blur"]
    return transforms.Compose([
        transforms.RandomResizedCrop(
            int(recipe["image_size"]), scale=tuple(augmentation["random_resized_crop_scale"])
        ),
        transforms.RandomHorizontalFlip(p=float(augmentation["horizontal_flip_probability"])),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=float(jitter["brightness"]),
                contrast=float(jitter["contrast"]),
                saturation=float(jitter["saturation"]),
                hue=float(jitter["hue"]),
            )
        ], p=float(jitter["probability"])),
        transforms.RandomGrayscale(p=float(augmentation["random_grayscale_probability"])),
        transforms.GaussianBlur(
            kernel_size=int(blur["kernel_size"]), sigma=tuple(blur["sigma"])
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=augmentation["normalization_mean"], std=augmentation["normalization_std"]
        ),
    ])


def build_verified_unlabeled_manifest(
    *, repo: Path, data_root: Path, recipe: dict[str, Any], destination: Path,
) -> tuple[list[dict[str, str]], str]:
    inventory_path = repo / recipe["inventory_manifest"]
    prefix = str(recipe["dataset_root_relative"]).rstrip("/") + "/"
    extensions = {str(value).casefold() for value in recipe["image_extensions"]}
    with inventory_path.open(newline="", encoding="utf-8") as handle:
        inventory = list(csv.DictReader(handle))
    rows = [
        {
            "relative_path": row["relative_path"],
            "size_bytes": row["size_bytes"],
            "sha256": row["sha256"],
        }
        for row in inventory
        if row["relative_path"].startswith(prefix)
        and Path(row["relative_path"]).suffix.casefold() in extensions
        and row["image_readable"] == "True"
    ]
    rows.sort(key=lambda row: row["relative_path"].casefold())
    expected_count = int(recipe["expected_image_count"])
    if len(rows) != expected_count:
        raise RuntimeError(f"STOP: expected {expected_count} unlabeled images, found {len(rows)}")
    for row in rows:
        path = data_root / row["relative_path"]
        if not path.is_file() or path.stat().st_size != int(row["size_bytes"]):
            raise RuntimeError(f"STOP: unlabeled image missing or size changed: {row['relative_path']}")
        if sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"STOP: unlabeled image SHA256 changed: {row['relative_path']}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    return rows, sha256_file(destination)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def validate_or_archive_stale_ssl_resume(
    *, config: dict[str, Any], config_sha256: str, unlabeled_manifest_sha256: str,
    experiment_root: Path, stale_archive_root: Path,
) -> dict[str, Any]:
    """Validate a P3 rolling checkpoint or safely archive one known stale shape.

    Automatic archival is deliberately narrow: the checkpoint must belong to
    this exact P3 experiment, architecture, and unlabeled manifest, and the
    sole identity mismatch must be the source-config hash. Final SSL-looking
    artifacts always block automatic archival.
    """
    output_dir = experiment_root / "pretraining"
    resume_path = output_dir / "last_resume_checkpoint.pth"
    final_paths = (
        output_dir / "pretraining_metadata.json",
        output_dir / "mobilevitv2_vicreg_encoder_final.pth",
        output_dir / "vicreg_projector_final.pth",
        output_dir / "training_history.csv",
        output_dir / "checkpoint_sha256.txt",
    )
    if not resume_path.is_file():
        return {
            "status": "NO_RESUME",
            "resume_path": str(resume_path),
            "start_epoch": 1,
            "stale_resume_detected": False,
            "test_loader_instantiated": False,
        }
    if any(path.exists() for path in final_paths):
        present = [str(path) for path in final_paths if path.exists()]
        raise RuntimeError(
            "STOP: rolling P3 SSL checkpoint coexists with final-looking SSL artifacts; "
            f"manual scientific review required: {present}"
        )
    payload = torch.load(resume_path, map_location="cpu", weights_only=True)
    expected = {
        "experiment_id": config["experiment_id"],
        "source_config_sha256": config_sha256,
        "unlabeled_manifest_sha256": unlabeled_manifest_sha256,
        "timm_model_name": config["timm_model_name"],
    }
    mismatches = {
        key: {
            "saved": payload.get(key),
            "expected": value,
        }
        for key, value in expected.items()
        if payload.get(key) != value
    }
    epoch = payload.get("epoch")
    maximum_resume_epoch = int(config["vicreg_pretraining"]["epochs"]) - 1
    if not isinstance(epoch, int) or not 1 <= epoch <= maximum_resume_epoch:
        raise RuntimeError(
            "STOP: VICReg resume checkpoint epoch mismatch: "
            f"saved={epoch!r}, expected_integer_range=1..{maximum_resume_epoch}"
        )
    required_states = (
        "backbone_state_dict", "projector_state_dict", "optimizer_state_dict",
        "scaler_state_dict", "torch_rng_state", "cuda_rng_state_all",
        "dataloader_generator_state", "history",
    )
    missing_states = [key for key in required_states if key not in payload]
    if missing_states:
        raise RuntimeError(f"STOP: VICReg resume checkpoint lacks required state: {missing_states}")
    if not mismatches:
        return {
            "status": "VALID_RESUME",
            "resume_path": str(resume_path),
            "resume_sha256": sha256_file(resume_path),
            "saved_metadata": {**{key: payload.get(key) for key in expected}, "epoch": epoch},
            "expected_metadata": {**expected, "epoch": f"1..{maximum_resume_epoch}"},
            "mismatches": {},
            "start_epoch": epoch + 1,
            "stale_resume_detected": False,
            "test_loader_instantiated": False,
        }
    if set(mismatches) != {"source_config_sha256"}:
        raise RuntimeError(f"STOP: VICReg resume checkpoint identity mismatch: {mismatches}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    archive_dir = stale_archive_root / timestamp
    archived_files = []
    related_paths = sorted(
        path for path in output_dir.glob("*resume*")
        if path.is_file()
    )
    if resume_path not in related_paths:
        related_paths.append(resume_path)
    for original in related_paths:
        relative = original.relative_to(experiment_root)
        archived = archive_dir / relative
        archived.parent.mkdir(parents=True, exist_ok=True)
        original_sha = sha256_file(original)
        shutil.move(str(original), str(archived))
        archived_files.append({
            "original_path": str(original),
            "archived_path": str(archived),
            "sha256": original_sha,
        })
    report = {
        "status": "STALE_RESUME_ARCHIVED",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "Interrupted P3 SSL resume matched experiment, architecture, and unlabeled manifest; only source_config_sha256 was stale.",
        "artifacts": archived_files,
        "saved_metadata": {**{key: payload.get(key) for key in expected}, "epoch": epoch},
        "expected_metadata": {**expected, "epoch": f"1..{maximum_resume_epoch}"},
        "mismatches": mismatches,
        "start_epoch": 1,
        "stale_resume_detected": True,
        "test_loader_instantiated": False,
    }
    report_path = archive_dir / "stale_artifact_report.json"
    write_json(report_path, report)
    report["report_path"] = str(report_path)
    report["archive_dir"] = str(archive_dir)
    return report


def build_resolved_p3_config(
    config: dict[str, Any], ssl_metadata: dict[str, Any],
) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    resolved["ssl_checkpoint"] = {
        "scope": "RUN_ARTIFACT_ROOT",
        "relative_path": str(
            Path(config["experiment_id"])
            / "pretraining"
            / Path(ssl_metadata["encoder_checkpoint_path"]).name
        ),
        "sha256": ssl_metadata["encoder_checkpoint_sha256"],
        "encoder_architecture": config["timm_model_name"],
        "unlabeled_manifest_sha256": ssl_metadata["unlabeled_manifest_sha256"],
    }
    return resolved


def _field_differences(existing: Any, expected: Any, prefix: str = "") -> list[dict[str, Any]]:
    if isinstance(existing, dict) and isinstance(expected, dict):
        rows = []
        for key in sorted(set(existing) | set(expected)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in existing:
                rows.append({"key": path, "existing_value": "<MISSING>", "current_expected_value": expected[key]})
            elif key not in expected:
                rows.append({"key": path, "existing_value": existing[key], "current_expected_value": "<MISSING>"})
            else:
                rows.extend(_field_differences(existing[key], expected[key], path))
        return rows
    if existing != expected:
        return [{"key": prefix, "existing_value": existing, "current_expected_value": expected}]
    return []


def reconcile_resolved_p3_config(
    *, config: dict[str, Any], config_sha256: str, experiment_root: Path,
    stale_archive_root: Path,
) -> dict[str, Any]:
    """Reconcile only a derived SSL-hash stale resolved config after SSL verifies."""
    ssl_metadata = verify_mobilevitv2_vicreg(
        config=config, config_sha256=config_sha256, experiment_root=experiment_root
    )
    expected = build_resolved_p3_config(config, ssl_metadata)
    expected_text = yaml.safe_dump(expected, sort_keys=False, allow_unicode=True)
    expected_text_sha = sha256_text(expected_text)
    resolved_path = experiment_root / "resolved_config.yaml"
    result = {
        "status": "CREATED",
        "resolved_config_path": str(resolved_path),
        "field_diff": [],
        "old_resolved_config_sha256": None,
        "new_resolved_config_sha256": expected_text_sha,
        "ssl_checkpoint_path": ssl_metadata["encoder_checkpoint_path"],
        "ssl_checkpoint_sha256": ssl_metadata["encoder_checkpoint_sha256"],
        "ssl_completed_epoch": ssl_metadata["epochs"],
        "ssl_identity_verification": "PASS",
        "test_loader_instantiated": False,
    }
    if resolved_path.is_file():
        old_sha = sha256_file(resolved_path)
        existing = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
        differences = _field_differences(existing, expected)
        result["old_resolved_config_sha256"] = old_sha
        result["field_diff"] = differences
        if not differences and old_sha == expected_text_sha:
            result["status"] = "MATCHED_CURRENT"
            result["new_resolved_config_sha256"] = old_sha
            return result
        allowed = {"ssl_checkpoint.sha256"}
        changed = {row["key"] for row in differences}
        if not changed.issubset(allowed):
            raise RuntimeError(
                "STOP: existing resolved P3 config changes actual experiment protocol: "
                f"{differences}"
            )
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        archive_dir = stale_archive_root / timestamp
        archive_dir.mkdir(parents=True, exist_ok=False)
        archived_path = archive_dir / "stale_resolved_config.yaml"
        shutil.move(str(resolved_path), str(archived_path))
        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "reason": "Current epoch-80 SSL identity verified; existing resolved config differed only by the derived SSL checkpoint SHA256.",
            "original_path": str(resolved_path),
            "archived_path": str(archived_path),
            "old_resolved_config_sha256": old_sha,
            "current_resolved_text_sha256": expected_text_sha,
            "field_diff": differences,
            "ssl_identity": {
                "experiment_id": ssl_metadata["experiment_id"],
                "source_config_sha256": ssl_metadata["source_config_sha256"],
                "unlabeled_manifest_sha256": ssl_metadata["unlabeled_manifest_sha256"],
                "timm_model_name": ssl_metadata["encoder_architecture"],
                "completed_epoch": ssl_metadata["epochs"],
                "checkpoint_sha256": ssl_metadata["encoder_checkpoint_sha256"],
            },
            "test_loader_instantiated": False,
        }
        report_path = archive_dir / "stale_resolved_config_report.json"
        write_json(report_path, report)
        result.update({
            "status": "STALE_RESOLVED_CONFIG_ARCHIVED",
            "archive_dir": str(archive_dir),
            "archived_path": str(archived_path),
            "report_path": str(report_path),
        })
    temporary = resolved_path.with_suffix(".yaml.tmp")
    temporary.write_text(expected_text, encoding="utf-8")
    os.replace(temporary, resolved_path)
    if sha256_file(resolved_path) != expected_text_sha:
        raise RuntimeError("STOP: newly written resolved P3 config SHA256 mismatch")
    return result


def reconcile_resolved_p3_before_ssl(
    *, config: dict[str, Any], config_sha256: str, experiment_root: Path,
    stale_archive_root: Path,
) -> dict[str, Any]:
    """Run before SSL: reconcile when a completed SSL exists, otherwise fail closed."""
    metadata_path = experiment_root / "pretraining/pretraining_metadata.json"
    resolved_path = experiment_root / "resolved_config.yaml"
    if metadata_path.is_file():
        return reconcile_resolved_p3_config(
            config=config,
            config_sha256=config_sha256,
            experiment_root=experiment_root,
            stale_archive_root=stale_archive_root,
        )
    if not resolved_path.exists():
        return {
            "status": "NO_RESOLVED_CONFIG_BEFORE_FIRST_SSL",
            "resolved_config_path": str(resolved_path),
            "test_loader_instantiated": False,
        }

    # A resolved config should normally only exist after final SSL.  The sole
    # safe partial-run exception is when an identity-checked rolling SSL
    # checkpoint and its manifest prove that the file belongs to this same
    # experiment/protocol.  Validate all protocol fields before moving either
    # artifact; never infer identity from the directory name alone.
    resume_path = experiment_root / "pretraining/last_resume_checkpoint.pth"
    manifest_path = experiment_root / "pretraining/unlabeled_manifest.csv"
    if not resume_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(
            "STOP: resolved_config.yaml exists without verified final SSL metadata or "
            "a complete partial-SSL identity; automatic reconciliation is not justified"
        )
    existing = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    existing_base = copy.deepcopy(existing)
    existing_ssl = existing_base.pop("ssl_checkpoint", None)
    current_base = copy.deepcopy(config)
    current_base.pop("ssl_checkpoint", None)
    protocol_differences = _field_differences(existing_base, current_base)
    manifest_sha = sha256_file(manifest_path)
    expected_relative_path = str(
        Path(config["experiment_id"])
        / "pretraining"
        / "mobilevitv2_vicreg_encoder_final.pth"
    )
    expected_ssl_identity = {
        "scope": "RUN_ARTIFACT_ROOT",
        "relative_path": expected_relative_path,
        "encoder_architecture": config["timm_model_name"],
        "unlabeled_manifest_sha256": manifest_sha,
    }
    ssl_identity_differences = []
    if not isinstance(existing_ssl, dict):
        ssl_identity_differences.append({
            "key": "ssl_checkpoint", "existing_value": existing_ssl,
            "current_expected_value": expected_ssl_identity,
        })
    else:
        ssl_identity_differences = _field_differences(
            {key: existing_ssl.get(key) for key in expected_ssl_identity},
            expected_ssl_identity,
            "ssl_checkpoint",
        )
    if protocol_differences or ssl_identity_differences:
        raise RuntimeError(
            "STOP: stale resolved config cannot be tied safely to the partial SSL run: "
            f"{protocol_differences + ssl_identity_differences}"
        )

    resume_resolution = validate_or_archive_stale_ssl_resume(
        config=config,
        config_sha256=config_sha256,
        unlabeled_manifest_sha256=manifest_sha,
        experiment_root=experiment_root,
        stale_archive_root=stale_archive_root,
    )
    if resume_resolution["status"] not in {"VALID_RESUME", "STALE_RESUME_ARCHIVED"}:
        raise RuntimeError(
            "STOP: partial SSL checkpoint did not provide a safe reconciliation identity"
        )
    old_sha = sha256_file(resolved_path)
    current_text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    field_diff = _field_differences(existing, config)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    archive_dir = stale_archive_root / timestamp
    archive_dir.mkdir(parents=True, exist_ok=False)
    archived_path = archive_dir / "stale_resolved_config.yaml"
    shutil.move(str(resolved_path), str(archived_path))
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reason": (
            "Resolved config predated final SSL; an identity-checked partial SSL "
            "checkpoint and manifest proved it belonged to this same protocol."
        ),
        "original_path": str(resolved_path),
        "archived_path": str(archived_path),
        "old_resolved_config_sha256": old_sha,
        "current_resolved_text_sha256": sha256_text(current_text),
        "field_diff": field_diff,
        "partial_ssl_resume_resolution": resume_resolution,
        "test_loader_instantiated": False,
    }
    report_path = archive_dir / "stale_resolved_config_report.json"
    write_json(report_path, report)
    return {
        "status": "STALE_RESOLVED_CONFIG_ARCHIVED_BEFORE_SSL",
        "resolved_config_path": str(resolved_path),
        "old_resolved_config_sha256": old_sha,
        "field_diff": field_diff,
        "archive_dir": str(archive_dir),
        "archived_path": str(archived_path),
        "report_path": str(report_path),
        "partial_ssl_resume_resolution": resume_resolution,
        "test_loader_instantiated": False,
    }


def validate_or_archive_stale_supervised_state(
    *, resolved_config: dict[str, Any], resolved_config_sha256: str,
    experiment_root: Path, stale_archive_root: Path,
) -> dict[str, Any]:
    """Prevent supervised resume across different SSL initializations."""
    supervised_dir = experiment_root / "supervised"
    resume_path = supervised_dir / "last_resume_checkpoint.pth"
    if not resume_path.is_file():
        return {"status": "NO_SUPERVISED_RESUME", "start_epoch": 1, "test_loader_instantiated": False}
    if (supervised_dir / "run_metadata.json").exists():
        raise RuntimeError("STOP: supervised resume coexists with completed run metadata")
    payload = torch.load(resume_path, map_location="cpu", weights_only=True)
    ssl_sha = resolved_config["ssl_checkpoint"]["sha256"]
    expected = {
        "experiment_id": resolved_config["experiment_id"],
        "config_sha256": resolved_config_sha256,
        "manifest_sha256": resolved_config["dataset_manifest_sha256"],
        "split_sha256": resolved_config["split_sha256"],
        "ssl_initialization_sha256": ssl_sha,
        "selected_ssl_checkpoint_sha256": ssl_sha,
        "model_architecture": resolved_config["architecture"],
    }
    mismatches = {
        key: {"saved": payload.get(key), "expected": value}
        for key, value in expected.items() if payload.get(key) != value
    }
    epoch = payload.get("epoch")
    if not isinstance(epoch, int) or not 1 <= epoch < int(resolved_config["epochs"]):
        raise RuntimeError(f"STOP: invalid supervised resume epoch: {epoch!r}")
    if not mismatches:
        return {
            "status": "VALID_SUPERVISED_RESUME", "start_epoch": epoch + 1,
            "resume_sha256": sha256_file(resume_path), "mismatches": {},
            "test_loader_instantiated": False,
        }
    identity_keys = {"experiment_id", "manifest_sha256", "split_sha256", "model_architecture"}
    if any(key in mismatches for key in identity_keys):
        raise RuntimeError(f"STOP: unsafe supervised resume identity mismatch: {mismatches}")
    allowed = {"config_sha256", "ssl_initialization_sha256", "selected_ssl_checkpoint_sha256"}
    if not set(mismatches).issubset(allowed):
        raise RuntimeError(f"STOP: unsupported supervised resume mismatch: {mismatches}")
    saved_ssl_values = {
        payload.get("ssl_initialization_sha256"), payload.get("selected_ssl_checkpoint_sha256")
    }
    if len(saved_ssl_values) != 1 or None in saved_ssl_values or ssl_sha in saved_ssl_values:
        raise RuntimeError(f"STOP: ambiguous supervised SSL lineage mismatch: {mismatches}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    archive_dir = stale_archive_root / timestamp
    archived = []
    for path in sorted(supervised_dir.iterdir()):
        if not path.is_file():
            continue
        destination = archive_dir / "supervised" / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = sha256_file(path)
        shutil.move(str(path), str(destination))
        archived.append({"original_path": str(path), "archived_path": str(destination), "sha256": digest})
    report = {
        "status": "STALE_SUPERVISED_STATE_ARCHIVED",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "Partial supervised run used a different verified SSL initialization and cannot be resumed with the current epoch-80 SSL checkpoint.",
        "saved_epoch": epoch,
        "saved_metadata": {key: payload.get(key) for key in expected},
        "expected_metadata": expected,
        "mismatches": mismatches,
        "artifacts": archived,
        "start_epoch": 1,
        "test_loader_instantiated": False,
    }
    report_path = archive_dir / "stale_supervised_state_report.json"
    write_json(report_path, report)
    report["archive_dir"] = str(archive_dir)
    report["report_path"] = str(report_path)
    return report


def run_vicreg_one_batch_preflight(
    *, config: dict[str, Any], repo: Path, data_root: Path, device: str = "cuda",
) -> dict[str, Any]:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU_BLOCKED: VICReg engineering preflight requires CUDA")
    recipe = config["vicreg_pretraining"]
    inventory_path = repo / recipe["inventory_manifest"]
    prefix = str(recipe["dataset_root_relative"]).rstrip("/") + "/"
    extensions = {str(value).casefold() for value in recipe["image_extensions"]}
    with inventory_path.open(newline="", encoding="utf-8") as handle:
        rows = [
            {"relative_path": row["relative_path"]}
            for row in csv.DictReader(handle)
            if row["relative_path"].startswith(prefix)
            and Path(row["relative_path"]).suffix.casefold() in extensions
            and row["image_readable"] == "True"
        ]
    rows.sort(key=lambda row: row["relative_path"].casefold())
    if len(rows) != int(recipe["expected_image_count"]):
        raise RuntimeError("STOP: VICReg preflight unlabeled count mismatch")
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    selected = torch.device(device)
    backbone = timm.create_model(
        config["timm_model_name"], pretrained=True, num_classes=0, global_pool="avg"
    ).to(selected)
    feature_count = int(backbone.num_features)
    projector = VICRegProjector(
        feature_count,
        int(recipe["projector_hidden_dim"]),
        int(recipe["projector_output_dim"]),
    ).to(selected)
    dataset = TwoViewUnlabeledDataset(data_root, rows, _augmentation(recipe))
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    view1, view2 = next(iter(loader))
    feature1 = backbone(view1.to(selected))
    feature2 = backbone(view2.to(selected))
    projection1 = projector(feature1)
    projection2 = projector(feature2)
    loss, components = vicreg_loss(
        projection1,
        projection2,
        lambda_invariance=float(recipe["lambda_invariance"]),
        mu_variance=float(recipe["mu_variance"]),
        nu_covariance=float(recipe["nu_covariance"]),
        epsilon=float(recipe["epsilon"]),
    )
    loss.backward()
    report = {
        "status": "PASS",
        "device": str(selected),
        "image_count_in_locked_pool": len(rows),
        "batch_size": 2,
        "feature_shape": list(feature1.shape),
        "projection_shape": list(projection1.shape),
        "vicreg_total_loss": float(loss.detach()),
        "vicreg_components": {name: float(value.detach()) for name, value in components.items()},
        "loss_finite": bool(torch.isfinite(loss.detach()).item()),
        "backward_batches": 1,
        "optimizer_step_performed": False,
        "test_loader_instantiated": False,
        "test_images_loaded": 0,
    }
    del loader, dataset, projector, backbone, view1, view2, feature1, feature2, projection1, projection2, loss
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def pretrain_mobilevitv2_vicreg(
    *, config: dict[str, Any], config_sha256: str, repo: Path, data_root: Path,
    experiment_root: Path, stale_archive_root: Path | None = None,
    run_ssl_pretraining: bool = True,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU_BLOCKED: VICReg pretraining requires CUDA")
    recipe = config["vicreg_pretraining"]
    output_dir = experiment_root / "pretraining"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "pretraining_metadata.json"
    encoder_path = output_dir / "mobilevitv2_vicreg_encoder_final.pth"
    projector_path = output_dir / "vicreg_projector_final.pth"
    resume_path = output_dir / "last_resume_checkpoint.pth"
    if metadata_path.is_file():
        completed = verify_mobilevitv2_vicreg(
            config=config, config_sha256=config_sha256, experiment_root=experiment_root
        )
        print(json.dumps({
            "EXPERIMENT_ID": config["experiment_id"],
            "CONFIG_SHA256": config_sha256,
            "UNLABELED_MANIFEST_SHA256": completed["unlabeled_manifest_sha256"],
            "SPLIT_SHA256": config["split_sha256"],
            "TIMM_MODEL_NAME": config["timm_model_name"],
            "RUN_SSL_PRETRAINING": run_ssl_pretraining,
            "SSL_RESUME_STATUS": "COMPLETED_SSL_VERIFIED_REUSE",
            "SSL_START_EPOCH": None,
            "TRAIN_COUNT": config["split_counts"]["train"],
            "VALIDATION_COUNT": config["split_counts"]["validation"],
            "TEST_LOADER_INSTANTIATED": False,
        }, indent=2), flush=True)
        completed["resume_resolution"] = {
            "status": "COMPLETED_SSL_VERIFIED_REUSE", "start_epoch": None,
            "stale_resume_detected": False,
        }
        return completed
    if not run_ssl_pretraining:
        raise RuntimeError(
            "STOP: RUN_SSL_PRETRAINING=False but no verified completed P3 SSL "
            "checkpoint exists; refusing to start VICReg"
        )
    allowed = {"unlabeled_manifest.csv", resume_path.name}
    unexpected = {path.name for path in output_dir.iterdir() if path.is_file()} - allowed
    if unexpected:
        raise RuntimeError(f"STOP: VICReg output directory contains unexpected files: {sorted(unexpected)}")

    started_utc = datetime.now(timezone.utc).isoformat()
    started_monotonic = time.monotonic()
    rows, unlabeled_manifest_sha = build_verified_unlabeled_manifest(
        repo=repo,
        data_root=data_root,
        recipe=recipe,
        destination=output_dir / "unlabeled_manifest.csv",
    )
    archive_root = stale_archive_root or experiment_root.parent.parent / "soilnet_stale_p3"
    resume_resolution = validate_or_archive_stale_ssl_resume(
        config=config,
        config_sha256=config_sha256,
        unlabeled_manifest_sha256=unlabeled_manifest_sha,
        experiment_root=experiment_root,
        stale_archive_root=archive_root,
    )
    pre_ssl_log = {
        "EXPERIMENT_ID": config["experiment_id"],
        "CONFIG_SHA256": config_sha256,
        "UNLABELED_MANIFEST_SHA256": unlabeled_manifest_sha,
        "SPLIT_SHA256": config["split_sha256"],
        "TIMM_MODEL_NAME": config["timm_model_name"],
        "RUN_SSL_PRETRAINING": run_ssl_pretraining,
        "SSL_RESUME_STATUS": resume_resolution["status"],
        "SSL_START_EPOCH": resume_resolution["start_epoch"],
        "TRAIN_COUNT": config["split_counts"]["train"],
        "VALIDATION_COUNT": config["split_counts"]["validation"],
        "TEST_LOADER_INSTANTIATED": False,
    }
    if resume_resolution["stale_resume_detected"]:
        pre_ssl_log.update({
            "STALE_SSL_RESUME_DETECTED": True,
            "STALE_SSL_RESUME_ARCHIVED": resume_resolution["archive_dir"],
            "STALE_ARTIFACT_REPORT": resume_resolution["report_path"],
            "SSL_RESTART_FROM_EPOCH": 1,
        })
    print(json.dumps(pre_ssl_log, indent=2), flush=True)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    device = torch.device("cuda")
    backbone = timm.create_model(
        config["timm_model_name"], pretrained=True, num_classes=0, global_pool="avg"
    ).to(device)
    feature_count = int(backbone.num_features)
    if feature_count != int(recipe["projector_input_dim"]):
        raise RuntimeError(
            f"STOP: MobileViTv2 feature dimension {feature_count} != locked projector input "
            f"{recipe['projector_input_dim']}"
        )
    projector = VICRegProjector(
        feature_count,
        int(recipe["projector_hidden_dim"]),
        int(recipe["projector_output_dim"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        list(backbone.parameters()) + list(projector.parameters()),
        lr=float(recipe["learning_rate"]),
        weight_decay=float(recipe["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(recipe["use_amp"]))
    generator = torch.Generator().manual_seed(seed)
    dataset = TwoViewUnlabeledDataset(data_root, rows, _augmentation(recipe))
    loader = DataLoader(
        dataset,
        batch_size=int(recipe["batch_size"]),
        shuffle=True,
        drop_last=bool(recipe["drop_last"]),
        num_workers=int(recipe["num_workers"]),
        pin_memory=True,
        persistent_workers=False,
        generator=generator,
    )
    history: list[dict[str, float | int]] = []
    start_epoch = 1
    if resume_path.is_file():
        payload = torch.load(resume_path, map_location="cpu", weights_only=True)
        expected = {
            "experiment_id": config["experiment_id"],
            "source_config_sha256": config_sha256,
            "unlabeled_manifest_sha256": unlabeled_manifest_sha,
            "timm_model_name": config["timm_model_name"],
        }
        mismatches = {
            key: {
                "saved": payload.get(key),
                "expected": value,
            }
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"STOP: VICReg resume checkpoint identity mismatch: {mismatches}")
        backbone.load_state_dict(payload["backbone_state_dict"], strict=True)
        projector.load_state_dict(payload["projector_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scaler.load_state_dict(payload["scaler_state_dict"])
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
        generator.set_state(payload["dataloader_generator_state"].cpu())
        history = list(payload["history"])
        start_epoch = int(payload["epoch"]) + 1
        print(json.dumps({"vicreg_resume": "ACCEPTED", "start_epoch": start_epoch}))

    for epoch in range(start_epoch, int(recipe["epochs"]) + 1):
        epoch_started = time.monotonic()
        backbone.train()
        projector.train()
        totals = {"loss": 0.0, "invariance": 0.0, "variance": 0.0, "covariance": 0.0}
        batches = 0
        for view1, view2 in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(recipe["use_amp"])):
                projection1 = projector(backbone(view1.to(device, non_blocking=True)))
                projection2 = projector(backbone(view2.to(device, non_blocking=True)))
                loss, components = vicreg_loss(
                    projection1,
                    projection2,
                    lambda_invariance=float(recipe["lambda_invariance"]),
                    mu_variance=float(recipe["mu_variance"]),
                    nu_covariance=float(recipe["nu_covariance"]),
                    epsilon=float(recipe["epsilon"]),
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            totals["loss"] += float(loss.detach())
            for name, value in components.items():
                totals[name] += float(value.detach())
            batches += 1
        row = {
            "epoch": epoch,
            "batches": batches,
            "vicreg_total_loss": totals["loss"] / batches,
            "invariance_loss": totals["invariance"] / batches,
            "variance_loss": totals["variance"] / batches,
            "covariance_loss": totals["covariance"] / batches,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.monotonic() - epoch_started,
        }
        history.append(row)
        _atomic_torch_save({
            "experiment_id": config["experiment_id"],
            "source_config_sha256": config_sha256,
            "epoch": epoch,
            "timm_model_name": config["timm_model_name"],
            "unlabeled_manifest_sha256": unlabeled_manifest_sha,
            "backbone_state_dict": backbone.state_dict(),
            "projector_state_dict": projector.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
            "dataloader_generator_state": generator.get_state(),
            "history": history,
        }, resume_path)
        print(json.dumps({"stage": "vicreg", **row}), flush=True)

    _atomic_torch_save({
        "experiment_id": config["experiment_id"],
        "source_config_sha256": config_sha256,
        "encoder_architecture": config["timm_model_name"],
        "initialization": "ImageNet then VICReg",
        "epoch": int(recipe["epochs"]),
        "unlabeled_manifest_sha256": unlabeled_manifest_sha,
        "backbone_state_dict": backbone.state_dict(),
        "vicreg_recipe": recipe,
        "final_vicreg_loss": history[-1]["vicreg_total_loss"],
    }, encoder_path)
    _atomic_torch_save({
        "experiment_id": config["experiment_id"],
        "source_config_sha256": config_sha256,
        "epoch": int(recipe["epochs"]),
        "unlabeled_manifest_sha256": unlabeled_manifest_sha,
        "projector_state_dict": projector.state_dict(),
        "vicreg_recipe": recipe,
    }, projector_path)
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    encoder_sha = sha256_file(encoder_path)
    projector_sha = sha256_file(projector_path)
    (output_dir / "checkpoint_sha256.txt").write_text(
        f"{encoder_path.name}  {encoder_sha}\n{projector_path.name}  {projector_sha}\n",
        encoding="utf-8",
    )
    resume_path.unlink(missing_ok=True)
    metadata = {
        "experiment_id": config["experiment_id"],
        "source_config_sha256": config_sha256,
        "training_completed": True,
        "training_started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "training_duration_seconds": time.monotonic() - started_monotonic,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "seed": seed,
        "encoder_architecture": config["timm_model_name"],
        "encoder_parameters": sum(parameter.numel() for parameter in backbone.parameters()),
        "projector_parameters": sum(parameter.numel() for parameter in projector.parameters()),
        "unlabeled_image_count": len(rows),
        "unlabeled_manifest_path": str(output_dir / "unlabeled_manifest.csv"),
        "unlabeled_manifest_sha256": unlabeled_manifest_sha,
        "inventory_manifest_path": str(repo / recipe["inventory_manifest"]),
        "inventory_manifest_sha256": sha256_file(repo / recipe["inventory_manifest"]),
        "epochs": int(recipe["epochs"]),
        "batches_per_epoch": len(loader),
        "final_vicreg_loss": history[-1]["vicreg_total_loss"],
        "encoder_checkpoint_path": str(encoder_path),
        "encoder_checkpoint_sha256": encoder_sha,
        "projector_checkpoint_path": str(projector_path),
        "projector_checkpoint_sha256": projector_sha,
        "vicreg_recipe": recipe,
        "test_images_loaded": 0,
        "test_loader_instantiated": False,
        "resume_resolution": resume_resolution,
    }
    write_json(metadata_path, metadata)
    return metadata


def verify_mobilevitv2_vicreg(
    *, config: dict[str, Any], config_sha256: str, experiment_root: Path,
) -> dict[str, Any]:
    output_dir = experiment_root / "pretraining"
    metadata_path = output_dir / "pretraining_metadata.json"
    encoder_path = output_dir / "mobilevitv2_vicreg_encoder_final.pth"
    projector_path = output_dir / "vicreg_projector_final.pth"
    history_path = output_dir / "training_history.csv"
    manifest_path = output_dir / "unlabeled_manifest.csv"
    required = (metadata_path, encoder_path, projector_path, history_path, manifest_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"STOP: verified P3 VICReg reuse artifacts are missing: {missing}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "experiment_id": config["experiment_id"],
        "source_config_sha256": config_sha256,
        "encoder_architecture": config["timm_model_name"],
        "epochs": int(config["vicreg_pretraining"]["epochs"]),
        "unlabeled_image_count": int(config["vicreg_pretraining"]["expected_image_count"]),
        "training_completed": True,
    }
    mismatches = {
        key: {"observed": metadata.get(key), "expected": value}
        for key, value in expected.items() if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"STOP: P3 VICReg metadata mismatch: {mismatches}")
    observed_hashes = {
        "encoder_checkpoint_sha256": sha256_file(encoder_path),
        "projector_checkpoint_sha256": sha256_file(projector_path),
        "unlabeled_manifest_sha256": sha256_file(manifest_path),
    }
    hash_mismatches = {
        key: {"observed": value, "expected": metadata.get(key)}
        for key, value in observed_hashes.items() if value != metadata.get(key)
    }
    if hash_mismatches:
        raise RuntimeError(f"STOP: P3 VICReg artifact hash mismatch: {hash_mismatches}")
    checkpoint = torch.load(encoder_path, map_location="cpu", weights_only=True)
    checkpoint_expected = {
        "experiment_id": config["experiment_id"],
        "source_config_sha256": config_sha256,
        "epoch": int(config["vicreg_pretraining"]["epochs"]),
        "unlabeled_manifest_sha256": metadata["unlabeled_manifest_sha256"],
    }
    checkpoint_mismatches = {
        key: {"saved": checkpoint.get(key), "expected": value}
        for key, value in checkpoint_expected.items() if checkpoint.get(key) != value
    }
    saved_model_name = checkpoint.get("timm_model_name", checkpoint.get("encoder_architecture"))
    if saved_model_name != config["timm_model_name"]:
        checkpoint_mismatches["timm_model_name"] = {
            "saved": saved_model_name, "expected": config["timm_model_name"]
        }
    if checkpoint_mismatches:
        raise RuntimeError(f"STOP: P3 VICReg encoder payload metadata mismatch: {checkpoint_mismatches}")
    if not isinstance(checkpoint.get("backbone_state_dict"), dict):
        raise RuntimeError("STOP: P3 VICReg encoder payload has no backbone_state_dict")
    metadata["reuse_verification"] = "PASS"
    metadata["timm_model_name_verified"] = saved_model_name
    return metadata
