from __future__ import annotations

import csv
import json
import os
import platform
import time
from datetime import datetime, timezone
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from soilnet.data import SoilNetDataset
from soilnet.evaluation.predictions import prediction_rows_and_metrics, write_prediction_artifacts
from soilnet.final_protocol import current_git_commit
from soilnet.io import sha256_file, write_json
from soilnet.models import build_model, model_complexity
from soilnet.reproducibility import set_determinism
from soilnet.utils import ExperimentContext


def deterministic_transform(config: dict[str, Any]):
    """Historical training evidence contains no random online augmentation."""
    return transforms.Compose([
        transforms.Resize(tuple(config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=config["normalization_mean"], std=config["normalization_std"]),
    ])


def build_train_loader(context: ExperimentContext) -> DataLoader:
    dataset = SoilNetDataset(
        context.split_path, context.data_root,
        transform=deterministic_transform(context.config), split="train",
    )
    generator = torch.Generator().manual_seed(int(context.config["seed"]))
    return DataLoader(
        dataset, batch_size=int(context.config["batch_size"]), shuffle=True,
        num_workers=int(context.config["num_workers"]),
        pin_memory=torch.cuda.is_available(), generator=generator,
    )


def build_val_loader(context: ExperimentContext) -> DataLoader:
    dataset = SoilNetDataset(
        context.split_path, context.data_root,
        transform=deterministic_transform(context.config), split="validation",
    )
    return DataLoader(
        dataset, batch_size=int(context.config["batch_size"]), shuffle=False,
        num_workers=int(context.config["num_workers"]), pin_memory=torch.cuda.is_available(),
    )


def compute_losses(
    predicted_regression: torch.Tensor,
    predicted_classification: torch.Tensor,
    regression_target: torch.Tensor,
    classification_target: torch.Tensor,
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    regression = nn.MSELoss()(predicted_regression, regression_target)
    classification = nn.CrossEntropyLoss()(predicted_classification, classification_target)
    regression_weight = float(config["loss"]["regression_weight"])
    classification_weight = float(config["loss"]["classification_weight"])
    total = regression_weight * regression + classification_weight * classification
    return {"total_loss": total, "regression_loss": regression, "classification_loss": classification}


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    if config["optimizer"] != "Adam":
        raise RuntimeError("Locked optimizer must be Adam")
    return torch.optim.Adam(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )


def run_one_batch_preflight(
    context: ExperimentContext,
    *,
    device: str | torch.device | None = None,
    train_loader: DataLoader | None = None,
    val_loader: DataLoader | None = None,
) -> dict[str, Any]:
    """Check one train/validation batch on a temporary model without stepping."""
    set_determinism(int(context.config["seed"]))
    selected = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, load_report = build_model(
        context.config, context.checkpoint_root, data_root=context.data_root,
        run_artifact_root=context.run_artifact_root,
    )
    model.to(selected)
    optimizer = build_optimizer(model, context.config)
    train_loader = train_loader or build_train_loader(context)
    val_loader = val_loader or build_val_loader(context)
    model.train()
    image, light, regression, classification = next(iter(train_loader))
    train_shapes = {
        "train_image_shape": list(image.shape),
        "train_li_shape": list(light.shape),
        "train_regression_target_shape": list(regression.shape),
        "train_class_target_shape": list(classification.shape),
    }
    optimizer.zero_grad(set_to_none=True)
    outputs = model(image.to(selected), light.to(selected) if context.config["use_li"] else None)
    losses = compute_losses(outputs[0], outputs[1], regression.to(selected), classification.to(selected), context.config)
    losses["total_loss"].backward()
    model.eval()
    with torch.inference_mode():
        image, light, regression, classification = next(iter(val_loader))
        outputs = model(image.to(selected), light.to(selected) if context.config["use_li"] else None)
        validation_shapes = [list(outputs[0].shape), list(outputs[1].shape)]
    return {
        "status": "PASS", "device": str(selected), "train_batches": 1,
        "validation_batches": 1, "backward_batches": 1,
        "train_samples": len(train_loader.dataset), "validation_samples": len(val_loader.dataset),
        **train_shapes,
        "output_shapes": validation_shapes, "load_report": load_report,
        "loss_components_finite": {
            key: bool(torch.isfinite(value.detach()).item()) for key, value in losses.items()
        },
        "optimizer": optimizer.__class__.__name__, "optimizer_step_performed": False,
        "temporary_model": True,
        "research_metrics_created": False, "test_loader_instantiated": False,
        "li_consumed_by_model": bool(context.config["use_li"]),
        "vicreg_checkpoint_loaded": bool(
            load_report and load_report.get("vicreg_checkpoint_loaded")
        ),
        "initialization_provenance": context.config.get("initialization_provenance"),
    }


def run_one_batch_smoke(context: ExperimentContext, *, device: str | torch.device | None = None) -> dict[str, Any]:
    """Backward-compatible alias for the non-mutating engineering preflight."""
    return run_one_batch_preflight(context, device=device)


def _save_checkpoint(
    path: Path, model: nn.Module, context: ExperimentContext, epoch: int,
    *, optimizer: torch.optim.Optimizer | None = None, resume: bool = False,
    history: list[dict[str, Any]] | None = None,
    checkpoint_metrics: dict[str, float] | None = None,
) -> str:
    payload = {
        "experiment_id": context.config["experiment_id"],
        "ablation_type": context.config.get("ablation_type"),
        "protocol_version": context.config.get("protocol_version"),
        "checkpoint_selection": context.config.get("checkpoint_selection", "fixed_endpoint"),
        "selection_formula": context.config.get("selection_formula"),
        "epoch": epoch,
        "model_architecture": context.config["architecture"],
        "li_consumed_by_model": bool(context.config["use_li"]),
        "li_ablation_strategy": context.config.get("li_ablation_strategy"),
        "model_state_dict": model.state_dict(),
        "config_sha256": context.config_sha256,
        "manifest_sha256": context.manifest_sha256,
        "split_sha256": context.split_sha256,
        "ssl_initialization_sha256": (
            context.config.get("ssl_checkpoint") or {}
        ).get("sha256"),
        "selected_ssl_checkpoint_sha256": (
            context.config.get("ssl_checkpoint") or {}
        ).get("sha256"),
        "initialization_checkpoint_sha256": (
            context.config.get("initialization_checkpoint") or {}
        ).get("sha256"),
        "initialization_provenance": context.config.get("initialization_provenance"),
        "vicreg_checkpoint_loaded": bool(context.config.get("ssl_checkpoint")),
        "initialization_source_kind": (
            "vicreg_ssl" if context.config.get("ssl_checkpoint")
            else "historical_pre_vicreg_imagenet_snapshot"
            if context.config.get("initialization_checkpoint") else None
        ),
        "seed": int(context.config["seed"]),
        "training_protocol_parameters": {
            "epochs": int(context.config["epochs"]),
            "batch_size": int(context.config["batch_size"]),
            "optimizer": context.config["optimizer"],
            "learning_rate": float(context.config["learning_rate"]),
            "weight_decay": float(context.config["weight_decay"]),
            "use_amp": bool(context.config["use_amp"]),
        },
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if resume:
        payload["history"] = history or []
    if checkpoint_metrics:
        payload.update(checkpoint_metrics)
    torch.save(payload, path)
    return sha256_file(path)


def _validation_loss_summary(model, loader, device, config) -> dict[str, float]:
    model.eval()
    totals = {"total_loss": 0.0, "regression_loss": 0.0, "classification_loss": 0.0}
    with torch.inference_mode():
        for image, light, regression, classification in loader:
            prediction = model(image.to(device), light.to(device) if config["use_li"] else None)
            losses = compute_losses(
                prediction[0], prediction[1], regression.to(device), classification.to(device), config
            )
            for key in totals:
                totals[key] += float(losses[key].detach())
    return {key: value / len(loader) for key, value in totals.items()}


def _validate_resume_payload(payload: dict[str, Any], context: ExperimentContext) -> None:
    expected = {
        "experiment_id": context.config["experiment_id"],
        "model_architecture": context.config["architecture"],
        "config_sha256": context.config_sha256,
        "manifest_sha256": context.manifest_sha256,
        "split_sha256": context.split_sha256,
        "ssl_initialization_sha256": (context.config.get("ssl_checkpoint") or {}).get("sha256"),
        "initialization_checkpoint_sha256": (
            context.config.get("initialization_checkpoint") or {}
        ).get("sha256"),
        "seed": int(context.config["seed"]),
    }
    mismatches = {
        key: {"saved": payload.get(key), "expected": value}
        for key, value in expected.items() if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"STOP: rolling resume checkpoint does not match this run: {mismatches}")


def _package_version(name: str) -> str:
    try:
        return package_metadata.version(name)
    except package_metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def _uses_best_regression_selection(config: dict[str, Any]) -> bool:
    return (
        config.get("save_validation_best") is True
        and config.get("primary_task") == "regression"
        and config.get("checkpoint_selection") == "minimum_mean_validation_RMSE"
        and config.get("selection_formula") == "(SM0_RMSE + SM20_RMSE) / 2"
        and config.get("primary_checkpoint") == "validation_best_regression.pth"
        and config.get("epoch_60_checkpoint") == "epoch_60_final.pth"
    )


def _regression_selection_metrics(row: dict[str, Any]) -> dict[str, float]:
    sm0_rmse = float(row["validation_SM_0_rmse"])
    sm20_rmse = float(row["validation_SM_20_rmse"])
    return {
        "SM0_RMSE": sm0_rmse,
        "SM20_RMSE": sm20_rmse,
        "mean_validation_regression_RMSE": (sm0_rmse + sm20_rmse) / 2.0,
    }


def train_experiment(
    context: ExperimentContext, *, resume_if_available: bool = True,
) -> dict[str, Any]:
    """Execute the config-locked train/validation protocol; never loads test."""
    started_monotonic = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU_BLOCKED: full training requires CUDA")
    if context.config.get("use_amp") is not False:
        raise RuntimeError("AMP is not authorized by the locked historical protocol")
    set_determinism(int(context.config["seed"]))
    context.run_dir.mkdir(parents=True, exist_ok=True)
    best_regression_selection = _uses_best_regression_selection(context.config)
    if context.config.get("save_validation_best") and not best_regression_selection:
        raise RuntimeError("STOP: unsupported or incomplete validation checkpoint-selection policy")
    resume_path = context.run_dir / "last_resume_checkpoint.pth"
    best_path = context.run_dir / "validation_best_regression.pth"
    completed_metadata = context.run_dir / "run_metadata.json"
    if completed_metadata.is_file():
        completed = json.loads(completed_metadata.read_text(encoding="utf-8"))
        if completed.get("training_completed") is True:
            raise RuntimeError("STOP: this experiment is already completed; do not overwrite or rerun it")
    if best_regression_selection:
        existing_files = {path.name for path in context.run_dir.iterdir() if path.is_file()}
        allowed_resume_files = {resume_path.name, best_path.name}
        if resume_path.is_file():
            unexpected = existing_files - allowed_resume_files
            if unexpected:
                raise RuntimeError(f"STOP: BESTREG resume directory contains unexpected artifacts: {sorted(unexpected)}")
        elif existing_files:
            raise RuntimeError("STOP: BESTREG output directory is not empty and has no valid rolling resume checkpoint")
    model, load_report = build_model(
        context.config, context.checkpoint_root, data_root=context.data_root,
        run_artifact_root=context.run_artifact_root,
    )
    device = torch.device("cuda")
    model.to(device)
    optimizer = build_optimizer(model, context.config)
    train_loader, val_loader = build_train_loader(context), build_val_loader(context)
    best_mean_val_rmse = float("inf")
    best_epoch: int | None = None
    best_metrics: dict[str, float] | None = None
    history: list[dict[str, Any]] = []
    start_epoch = 1
    if resume_path.is_file() and resume_if_available:
        resume_payload = torch.load(resume_path, map_location=device, weights_only=True)
        _validate_resume_payload(resume_payload, context)
        saved_epoch = int(resume_payload.get("epoch", 0))
        if not 1 <= saved_epoch < int(context.config["epochs"]):
            raise RuntimeError(f"STOP: invalid saved resume epoch: {saved_epoch}")
        model.load_state_dict(resume_payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        history = list(resume_payload.get("history", []))
        if len(history) != saved_epoch:
            raise RuntimeError("STOP: rolling resume history does not match saved epoch")
        if best_regression_selection:
            prior_best_rows = [row for row in history if row.get("is_best_regression") is True]
            if not prior_best_rows:
                raise RuntimeError("STOP: v4 resume history has no validation-best checkpoint record")
            prior_best = prior_best_rows[-1]
            best_epoch = int(prior_best["epoch"])
            best_metrics = _regression_selection_metrics(prior_best)
            best_mean_val_rmse = best_metrics["mean_validation_regression_RMSE"]
            if not best_path.is_file():
                raise RuntimeError("STOP: v4 validation-best checkpoint is missing during resume")
            best_payload = torch.load(best_path, map_location="cpu", weights_only=True)
            _validate_resume_payload(best_payload, context)
            if int(best_payload.get("epoch", 0)) != best_epoch:
                raise RuntimeError("STOP: v4 validation-best checkpoint epoch does not match resume history")
        start_epoch = saved_epoch + 1
        print(json.dumps({
            "resume": "ACCEPTED", "saved_epoch": saved_epoch,
            "config_sha256": context.config_sha256,
            "manifest_sha256": context.manifest_sha256,
            "split_sha256": context.split_sha256,
            "ssl_initialization_sha256": (context.config.get("ssl_checkpoint") or {}).get("sha256"),
        }))
    elif resume_path.is_file():
        raise RuntimeError("STOP: rolling checkpoint exists but RESUME_IF_AVAILABLE is false")
    for epoch in range(start_epoch, int(context.config["epochs"]) + 1):
        model.train()
        totals = {"total_loss": 0.0, "regression_loss": 0.0, "classification_loss": 0.0}
        for image, light, regression, classification in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(image.to(device), light.to(device) if context.config["use_li"] else None)
            losses = compute_losses(
                prediction[0], prediction[1], regression.to(device), classification.to(device), context.config
            )
            losses["total_loss"].backward()
            optimizer.step()
            for key in totals:
                totals[key] += float(losses[key].detach())
        validation_losses = _validation_loss_summary(model, val_loader, device, context.config)
        _, validation_metrics = prediction_rows_and_metrics(model, val_loader, device, context.config)
        row = {"epoch": epoch, **{key: value / len(train_loader) for key, value in totals.items()}}
        row.update({f"validation_{key}": value for key, value in validation_losses.items()})
        for target in ("SM_0", "SM_20"):
            for metric, value in validation_metrics["regression"][target].items():
                row[f"validation_{target}_{metric}"] = value
        for metric, value in validation_metrics["classification"].items():
            row[f"validation_{metric}"] = value
        if best_regression_selection:
            selection_metrics = _regression_selection_metrics(row)
            mean_val_rmse = selection_metrics["mean_validation_regression_RMSE"]
            is_best_regression = mean_val_rmse < best_mean_val_rmse
            row["mean_regression_RMSE"] = mean_val_rmse
            row["is_best_regression"] = is_best_regression
            if is_best_regression:
                best_mean_val_rmse = mean_val_rmse
                best_epoch = epoch
                best_metrics = selection_metrics
                _save_checkpoint(
                    best_path, model, context, epoch, optimizer=optimizer,
                    checkpoint_metrics=selection_metrics,
                )
                print(
                    "BEST REGRESSION UPDATED | "
                    f"epoch={epoch} | mean_RMSE={mean_val_rmse:.4f} | "
                    f"SM0={selection_metrics['SM0_RMSE']:.4f} | "
                    f"SM20={selection_metrics['SM20_RMSE']:.4f}"
                )
        history.append(row)
        _save_checkpoint(
            resume_path, model, context, epoch, optimizer=optimizer, resume=True, history=history
        )
        print(json.dumps({
            "epoch": row["epoch"],
            "train_total_loss": row["total_loss"],
            "train_regression_loss": row["regression_loss"],
            "train_classification_loss": row["classification_loss"],
            "validation_total_loss": row["validation_total_loss"],
            "SM0_RMSE": row["validation_SM_0_rmse"],
            "SM0_MAE": row["validation_SM_0_mae"],
            "SM0_R2": row["validation_SM_0_r2"],
            "SM20_RMSE": row["validation_SM_20_rmse"],
            "SM20_MAE": row["validation_SM_20_mae"],
            "SM20_R2": row["validation_SM_20_r2"],
            "mean_validation_RMSE": row.get("mean_regression_RMSE"),
            "current_best_epoch": best_epoch if best_regression_selection else None,
            "current_best_mean_validation_RMSE": (
                best_mean_val_rmse if best_regression_selection else None
            ),
            "best_checkpoint_updated": row.get("is_best_regression", False),
            "classification_accuracy": row["validation_accuracy"],
            "Macro-F1": row["validation_macro_f1"],
            "regression_metric_scale": "original_0_to_100_percentage_points",
        }))
    final_epoch = int(context.config["epochs"])
    endpoint_path = context.run_dir / str(context.config.get("epoch_60_checkpoint", context.config["primary_checkpoint"]))
    final_row_metrics = _regression_selection_metrics(history[-1])
    endpoint_sha = _save_checkpoint(
        endpoint_path, model, context, final_epoch, optimizer=optimizer,
        checkpoint_metrics=final_row_metrics,
    )
    resume_path.unlink(missing_ok=True)
    if best_regression_selection:
        if best_epoch is None or best_metrics is None or not best_path.is_file():
            raise RuntimeError("STOP: v4 completed without a validation-best regression checkpoint")
        primary = best_path
        primary_sha = sha256_file(primary)
        best_payload = torch.load(primary, map_location=device, weights_only=True)
        _validate_resume_payload(best_payload, context)
        model.load_state_dict(best_payload["model_state_dict"], strict=True)
        primary_epoch = best_epoch
    else:
        primary = endpoint_path
        primary_sha = endpoint_sha
        primary_epoch = final_epoch
    with (context.run_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    validation_rows, validation_metrics = prediction_rows_and_metrics(model, val_loader, device, context.config)
    validation_metrics.update({
        "generated_from_checkpoint": str(primary),
        "generated_from_checkpoint_sha256": primary_sha,
        "generated_from_checkpoint_epoch": primary_epoch,
        "checkpoint_selection": context.config.get("checkpoint_selection", "fixed_endpoint"),
    })
    write_prediction_artifacts(
        context.run_dir / "validation_predictions.csv",
        context.run_dir / "validation_metrics.json",
        validation_rows,
        validation_metrics,
    )
    _, final_training_metrics = prediction_rows_and_metrics(model, train_loader, device, context.config)
    checksum_lines = [f"{primary.name}  {primary_sha}"]
    if endpoint_path != primary:
        checksum_lines.append(f"{endpoint_path.name}  {endpoint_sha}")
    (context.run_dir / "checkpoint_sha256.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    checkpoint_paths = [path for path in context.run_dir.glob("*.pth") if path.is_file()]
    if len(checkpoint_paths) > 2:
        raise RuntimeError("Checkpoint retention policy violated")
    metadata = {
        "experiment_id": context.config["experiment_id"],
        "ablation_type": context.config.get("ablation_type"),
        "protocol_version": context.config.get("protocol_version"),
        "primary_task": context.config.get("primary_task"),
        "checkpoint_selection": context.config.get("checkpoint_selection", "fixed_endpoint"),
        "selection_formula": context.config.get("selection_formula"),
        "training_completed": True,
        "training_started_utc": started_utc,
        "training_duration_seconds": time.monotonic() - started_monotonic,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(Path(__file__).resolve().parents[3]),
        "checkpoint_path": str(primary), "checkpoint_sha256": primary_sha,
        "primary_checkpoint_path": str(primary), "primary_checkpoint_sha256": primary_sha,
        "epoch_60_checkpoint_path": str(endpoint_path), "epoch_60_checkpoint_sha256": endpoint_sha,
        "config_path": str(context.config_path), "config_sha256": context.config_sha256,
        "manifest_sha256": context.manifest_sha256, "split_sha256": context.split_sha256,
        "dataset_manifest_sha256": context.manifest_sha256,
        "train_count": int(context.config.get("split_counts", {}).get("train", len(train_loader.dataset))),
        "validation_count": int(context.config.get("split_counts", {}).get("validation", len(val_loader.dataset))),
        "test_count": int(context.config.get("split_counts", {}).get("test", 231)),
        "split_counts": context.config.get("split_counts", {
            "train": len(train_loader.dataset), "validation": len(val_loader.dataset), "test": 231,
        }),
        "model_architecture": context.config["architecture"],
        "li_consumed_by_model": bool(context.config["use_li"]),
        "li_ablation_strategy": context.config.get("li_ablation_strategy"),
        "python": platform.python_version(), "pytorch": torch.__version__,
        "torchvision": _package_version("torchvision"), "timm": _package_version("timm"),
        "cuda_build": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "device": str(device),
        "seed": int(context.config["seed"]), "epochs": final_epoch,
        "batch_size": int(context.config["batch_size"]),
        "optimizer": context.config["optimizer"], "learning_rate": float(context.config["learning_rate"]),
        "lr": float(context.config["learning_rate"]),
        "weight_decay": float(context.config["weight_decay"]),
        "ssl_checkpoint_sha256": (context.config.get("ssl_checkpoint") or {}).get("sha256"),
        "selected_ssl_checkpoint_sha256": (context.config.get("ssl_checkpoint") or {}).get("sha256"),
        "initialization_checkpoint_sha256": (
            context.config.get("initialization_checkpoint") or {}
        ).get("sha256"),
        "initialization_provenance": context.config.get("initialization_provenance"),
        "vicreg_checkpoint_loaded": bool(
            load_report and load_report.get("vicreg_checkpoint_loaded")
        ),
        "initialization_source_kind": load_report.get("source_kind") if load_report else None,
        "final_checkpoint_sha256": endpoint_sha,
        "epochs_completed": final_epoch, "primary_checkpoint_epoch": primary_epoch,
        "best_epoch": best_epoch,
        "best_SM0_RMSE": best_metrics["SM0_RMSE"] if best_metrics else None,
        "best_SM20_RMSE": best_metrics["SM20_RMSE"] if best_metrics else None,
        "best_mean_validation_RMSE": (
            best_metrics["mean_validation_regression_RMSE"] if best_metrics else None
        ),
        "final_training_metrics": final_training_metrics,
        "validation_metrics": validation_metrics,
        "test_evaluated": "NO" if best_regression_selection else False,
        "checkpoint_count": len(checkpoint_paths),
        "checkpoint_total_bytes": sum(path.stat().st_size for path in checkpoint_paths),
        "complexity": model_complexity(model, primary),
        "ssl_load_report": load_report if context.config.get("ssl_checkpoint") else None,
        "initialization_load_report": load_report,
        "bitwise_reproducibility_across_hardware_claimed": False,
    }
    write_json(context.run_dir / "run_metadata.json", metadata)
    return metadata
