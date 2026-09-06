from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from soilnet.data import SoilNetDataset
from soilnet.evaluation.metrics import classification_metrics, regression_metrics
from soilnet.evaluation.predictions import prediction_rows_and_metrics, write_prediction_artifacts
from soilnet.io import load_yaml, sha256_file, write_csv, write_json
from soilnet.models import build_frozen_model, model_complexity
from soilnet.reproducibility import set_determinism
from soilnet.training.engine import deterministic_transform


REPO = Path(__file__).resolve().parents[2]
SPLIT_SHA256 = "8927b8223b8c4c234d264ad9ea62ac2df6161124a79787a2e71eeb5cd23eae2f"
MANIFEST_SHA256 = "8ff45054d4b8e3df9758d0c112dc16719572b2267906fb3a5ed5b3262a6732bd"
SPLIT_COUNTS = {"train": 1407, "validation": 289, "test": 231}
TRAINING_SEED = 20260905
BOOTSTRAP_SEED = 20260906
N_BOOTSTRAP = 10000
PUBLICATION_URL = "https://github.com/husphys-gif/SoilNet.git"
LEGACY_URLS = {
    "https://github.com/diy-hus/SoilNet",
    "https://github.com/diy-hus/SoilNet.git",
}
FINAL_TEST_DIR = REPO / "results/final_test"
FINAL_TEST_LOCK = FINAL_TEST_DIR / "FINAL_TEST_LOCK.json"
LEGACY_TEST_MARKER = REPO / "results/final/test_evaluation_completed.json"
DEPLOYMENT_CHECKPOINT = REPO / "checkpoints/deployment/P0_FINAL_SOILNET_VICREG_MU27_LI_V4_BESTREG.pth"


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "P0": {
        "experiment_id": "P0_FINAL_SOILNET_VICREG_MU27_LI_V4_BESTREG",
        "model_role": "primary_full_model",
        "config": "config/experiments/P0_final_soilnet_v4_bestreg.yaml",
        "checkpoint_path": "/mnt/d/check point/soilnet_final_runs/P0_FINAL_SOILNET_VICREG_MU27_LI_V4_BESTREG/validation_best_regression.pth",
        "checkpoint_sha256": "eba009dfd45ec21174a8e40b16148e0455e902286c7db55933487221da761379",
        "best_epoch": 45,
        "architecture": "SoilNet",
        "initialization": "ImageNet then VICReg mu=27",
        "LI": True,
        "VICReg": True,
        "validation": {"SM0_RMSE": 14.420743601668894, "SM20_RMSE": 14.636912724403645},
        "prediction_file": "P0_test_predictions.csv",
    },
    "P1_noLI": {
        "experiment_id": "P1_SOILNET_VICREG_MU27_NO_LI_BESTREG",
        "model_role": "LI_ablation",
        "config": "config/experiments/P1_no_li_bestreg.yaml",
        "checkpoint_path": "/mnt/d/check point/soilnet_final_runs/P1_SOILNET_VICREG_MU27_NO_LI_BESTREG/validation_best_regression.pth",
        "checkpoint_sha256": "a9d8de995b0673e9ec39bfc4afac00d6a2a773ad9ffbea0027ff2d0c4fab820c",
        "best_epoch": 53,
        "architecture": "SoilNet",
        "initialization": "ImageNet then VICReg mu=27",
        "LI": False,
        "VICReg": True,
        "validation_mean_RMSE": 15.125815592570847,
        "prediction_file": "P1_noLI_test_predictions.csv",
    },
    "P1_noSSL": {
        "experiment_id": "P1_SOILNET_IMAGENET_LI_NO_SSL_BESTREG",
        "model_role": "VICReg_ablation_and_architecture_control",
        "config": "config/experiments/P1_no_ssl_bestreg.yaml",
        "checkpoint_path": "/mnt/d/check point/soilnet_final_runs/P1_SOILNET_IMAGENET_LI_NO_SSL_BESTREG/validation_best_regression.pth",
        "checkpoint_sha256": "50c0f15567ab71a87b04569790b9cc7afd016989e3899d56e1898d8a6fcfb044",
        "best_epoch": 52,
        "architecture": "SoilNet",
        "initialization": "historical pre-VICReg ImageNet snapshot",
        "LI": True,
        "VICReg": False,
        "validation_mean_RMSE": 17.029435352353772,
        "prediction_file": "P1_noSSL_test_predictions.csv",
    },
    "P2_MobileViTv2": {
        "experiment_id": "P2_MOBILEVITV2_IMAGENET_LI_BESTREG",
        "model_role": "architecture_control_against_P1_noSSL",
        "config": "config/experiments/P2_mobilevitv2_imagenet_li_bestreg.yaml",
        "checkpoint_path": "/mnt/d/check point/soilnet_final_runs/P2_MOBILEVITV2_IMAGENET_LI_BESTREG/validation_best_regression.pth",
        "checkpoint_sha256": None,
        "best_epoch": None,
        "architecture": "MobileViTv2-0.5 (timm mobilevitv2_050.cvnets_in1k)",
        "initialization": "ImageNet CVNets weights",
        "LI": True,
        "VICReg": False,
        "prediction_file": "P2_mobilevitv2_test_predictions.csv",
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Required artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _package_version(name: str) -> str:
    try:
        return package_metadata.version(name)
    except package_metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def software_environment() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": _package_version("torchvision"),
        "timm": _package_version("timm"),
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }


def _validate_protocol_config(path: Path) -> dict[str, Any]:
    config = load_yaml(path)
    expected = {
        "seed": TRAINING_SEED,
        "epochs": 60,
        "batch_size": 32,
        "optimizer": "Adam",
        "learning_rate": 0.0001,
        "weight_decay": 0.0,
        "split_sha256": SPLIT_SHA256,
        "checkpoint_selection": "minimum_mean_validation_RMSE",
        "selection_formula": "(SM0_RMSE + SM20_RMSE) / 2",
        "primary_checkpoint": "validation_best_regression.pth",
        "epoch_60_checkpoint": "epoch_60_final.pth",
        "test_evaluation": "prohibited_in_training_notebook",
    }
    drift = {key: {"expected": value, "observed": config.get(key)} for key, value in expected.items() if config.get(key) != value}
    if drift:
        raise RuntimeError(f"Frozen protocol drift in {path.name}: {drift}")
    if config.get("split_counts") is not None and config.get("split_counts") != SPLIT_COUNTS:
        raise RuntimeError(f"Frozen split-count metadata drift in {path.name}")
    return config


def verify_p2_fairness() -> dict[str, Any]:
    control = _validate_protocol_config(REPO / MODEL_SPECS["P1_noSSL"]["config"])
    baseline = _validate_protocol_config(REPO / MODEL_SPECS["P2_MobileViTv2"]["config"])
    matched = (
        "dataset_manifest", "dataset_manifest_sha256", "split", "split_sha256", "split_counts",
        "seed", "epochs", "batch_size", "optimizer", "learning_rate", "weight_decay", "loss",
        "metrics", "image_size", "normalization_mean", "normalization_std", "train_augmentation",
        "validation_transform", "use_amp", "num_workers", "use_li", "num_classes", "class_mapping",
        "primary_task", "save_validation_best", "checkpoint_selection", "selection_formula",
        "primary_checkpoint", "epoch_60_checkpoint", "test_evaluation",
    )
    mismatch = {key: {"P1_noSSL": control.get(key), "P2": baseline.get(key)} for key in matched if control.get(key) != baseline.get(key)}
    if mismatch:
        raise RuntimeError(f"P2 is not a fair architecture control: {mismatch}")
    if baseline.get("timm_model_name") != "mobilevitv2_050.cvnets_in1k":
        raise RuntimeError("P2 exact TIMM variant drifted")
    if baseline.get("initialization") != "imagenet" or baseline.get("ssl_checkpoint") is not None:
        raise RuntimeError("P2 must use ImageNet initialization with no VICReg")
    return {"status": "PASS", "matched_fields": list(matched), "intended_delta": "architecture"}


def _verify_completed_model(key: str) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = MODEL_SPECS[key]
    config = _validate_protocol_config(REPO / spec["config"])
    checkpoint = Path(spec["checkpoint_path"])
    run_dir = checkpoint.parent
    metadata = _read_json(run_dir / "run_metadata.json")
    required_files = (
        checkpoint, run_dir / "epoch_60_final.pth", run_dir / "training_history.csv",
        run_dir / "validation_predictions.csv", run_dir / "validation_metrics.json",
        run_dir / "checkpoint_sha256.txt",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise RuntimeError(f"Frozen artifacts missing for {key}: {missing}")
    if metadata.get("experiment_id") != spec["experiment_id"] or metadata.get("training_completed") is not True:
        raise RuntimeError(f"{key} is not a completed expected experiment")
    if metadata.get("test_evaluated") not in (False, "NO"):
        raise RuntimeError(f"{key} is not test-clean")
    if metadata.get("checkpoint_path") != spec["checkpoint_path"]:
        raise RuntimeError(f"{key} checkpoint path changed")
    observed_sha = sha256_file(checkpoint)
    expected_sha = spec.get("checkpoint_sha256") or metadata.get("checkpoint_sha256")
    if observed_sha != expected_sha or metadata.get("checkpoint_sha256") != expected_sha:
        raise RuntimeError(f"{key} checkpoint SHA256 mismatch: {observed_sha}")
    if metadata.get("split_sha256") != SPLIT_SHA256 or metadata.get("seed") != TRAINING_SEED:
        raise RuntimeError(f"{key} split or seed drifted")
    if key != "P2_MobileViTv2" and int(metadata.get("best_epoch")) != int(spec["best_epoch"]):
        raise RuntimeError(f"{key} frozen best epoch changed")
    validation = _read_json(run_dir / "validation_metrics.json")
    if validation.get("generated_from_checkpoint_sha256") != expected_sha:
        raise RuntimeError(f"{key} validation artifacts were not generated from the frozen best checkpoint")
    return metadata, config


def write_frozen_model_registry() -> dict[str, Any]:
    """Create the four-model registry only after P2 has completed successfully."""
    verify_p2_fairness()
    models = []
    for key, spec in MODEL_SPECS.items():
        metadata, config = _verify_completed_model(key)
        complexity = metadata.get("complexity") or model_complexity(
            build_frozen_model(config), Path(spec["checkpoint_path"])
        )
        model = {
            "key": key,
            "experiment_id": spec["experiment_id"],
            "model_role": spec["model_role"],
            "checkpoint_path": spec["checkpoint_path"],
            "checkpoint_sha256": metadata["checkpoint_sha256"],
            "best_epoch": int(metadata["best_epoch"]),
            "architecture": spec["architecture"],
            "initialization": spec["initialization"],
            "LI": bool(spec["LI"]),
            "VICReg": bool(spec["VICReg"]),
            "split_sha256": SPLIT_SHA256,
            "seed": TRAINING_SEED,
            "validation_metrics": metadata["validation_metrics"],
            "config_path": spec["config"],
            "config_sha256": metadata["config_sha256"],
            "validation_predictions_path": str(Path(spec["checkpoint_path"]).parent / "validation_predictions.csv"),
            "validation_metrics_path": str(Path(spec["checkpoint_path"]).parent / "validation_metrics.json"),
            "complexity": complexity,
            "frozen": True,
            "mutable": False,
        }
        models.append(model)
    registry = {
        "registry_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "TEST_OPENED": "NO",
        "TEST_EVALUATED": "NO",
        "MODEL_SET_FROZEN": "YES",
        "dataset_manifest_sha256": MANIFEST_SHA256,
        "split_sha256": SPLIT_SHA256,
        "split_counts": SPLIT_COUNTS,
        "seed": TRAINING_SEED,
        "model_count": 4,
        "models": models,
    }
    write_json(REPO / "results/model_registry/frozen_model_registry.json", registry)
    return registry


def preflight_frozen_registry(*, allow_completed_test: bool = False) -> dict[str, Any]:
    """Complete every identity check before the sole test Dataset is created."""
    if (FINAL_TEST_LOCK.exists() or LEGACY_TEST_MARKER.exists()) and not allow_completed_test:
        raise RuntimeError("FINAL_TEST_ALREADY_COMPLETED")
    registry_path = REPO / "results/model_registry/frozen_model_registry.json"
    registry = _read_json(registry_path)
    if registry.get("TEST_OPENED") != "NO" or registry.get("TEST_EVALUATED") != "NO":
        raise RuntimeError("Registry does not preserve the pre-test firewall")
    if registry.get("MODEL_SET_FROZEN") != "YES" or registry.get("model_count") != 4:
        raise RuntimeError("Exactly four frozen models are required")
    if registry.get("split_sha256") != SPLIT_SHA256 or registry.get("split_counts") != SPLIT_COUNTS:
        raise RuntimeError("Registry split identity/counts drifted")
    split_path = REPO / "data/splits/final_clean_split_v1.csv"
    if sha256_file(split_path) != SPLIT_SHA256:
        raise RuntimeError("Locked split file SHA256 changed")
    observed = {entry.get("key"): entry for entry in registry.get("models", [])}
    if set(observed) != set(MODEL_SPECS):
        raise RuntimeError("Frozen registry model set changed")
    for key, spec in MODEL_SPECS.items():
        entry = observed[key]
        if entry.get("experiment_id") != spec["experiment_id"] or entry.get("checkpoint_path") != spec["checkpoint_path"]:
            raise RuntimeError(f"{key} identity/path drifted")
        if entry.get("frozen") is not True or entry.get("mutable") is not False:
            raise RuntimeError(f"{key} is not immutable")
        checkpoint = Path(entry["checkpoint_path"])
        if not checkpoint.is_file() or sha256_file(checkpoint) != entry.get("checkpoint_sha256"):
            raise RuntimeError(f"{key} checkpoint missing or changed")
        if spec.get("checkpoint_sha256") and entry["checkpoint_sha256"] != spec["checkpoint_sha256"]:
            raise RuntimeError(f"{key} locked checkpoint SHA256 changed")
        if entry.get("split_sha256") != SPLIT_SHA256 or entry.get("seed") != TRAINING_SEED:
            raise RuntimeError(f"{key} split/seed drifted")
        for field in ("validation_predictions_path", "validation_metrics_path"):
            if not Path(entry[field]).is_file():
                raise RuntimeError(f"{key} validation artifact missing: {field}")
        config = _validate_protocol_config(REPO / entry["config_path"])
        if sha256_file(REPO / entry["config_path"]) != entry.get("config_sha256"):
            raise RuntimeError(f"{key} config changed after training")
        if config["experiment_id"] != entry["experiment_id"]:
            raise RuntimeError(f"{key} config identity mismatch")
    return registry


def _metric_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    sm0, sm20 = metrics["regression"]["SM_0"], metrics["regression"]["SM_20"]
    return {
        "n": metrics["n"],
        "SM0_RMSE": sm0["rmse"], "SM0_MAE": sm0["mae"], "SM0_R2": sm0["r2"], "SM0_ME": sm0["me"],
        "SM20_RMSE": sm20["rmse"], "SM20_MAE": sm20["mae"], "SM20_R2": sm20["r2"], "SM20_ME": sm20["me"],
        "mean_RMSE": (sm0["rmse"] + sm20["rmse"]) / 2.0,
        "mean_MAE": (sm0["mae"] + sm20["mae"]) / 2.0,
        "Accuracy": metrics["classification"]["accuracy"],
        "Macro_F1": metrics["classification"]["macro_f1"],
        "Macro_Precision": metrics["classification"]["macro_precision"],
        "Macro_Recall": metrics["classification"]["macro_recall"],
        "regression_scale": "original_0_to_100_percentage_points",
    }


def _read_prediction_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != SPLIT_COUNTS["test"]:
        raise RuntimeError(f"Prediction row count changed: {path}")
    return {
        "sample_id": np.asarray([row["sample_id"] for row in rows]),
        "SM0_true": np.asarray([float(row["SM_0_true"]) for row in rows]),
        "SM0_pred": np.asarray([float(row["SM_0_pred"]) for row in rows]),
        "SM20_true": np.asarray([float(row["SM_20_true"]) for row in rows]),
        "SM20_pred": np.asarray([float(row["SM_20_pred"]) for row in rows]),
    }


def paired_bootstrap(output_dir: Path) -> list[dict[str, Any]]:
    data = {key: _read_prediction_csv(output_dir / spec["prediction_file"]) for key, spec in MODEL_SPECS.items()}
    reference = data["P0"]
    for key, values in data.items():
        if not np.array_equal(reference["sample_id"], values["sample_id"]):
            raise RuntimeError(f"Paired sample index/order mismatch for {key}")
        for target in ("SM0", "SM20"):
            if not np.allclose(reference[f"{target}_true"], values[f"{target}_true"], rtol=0, atol=1e-7):
                raise RuntimeError(f"Paired truth mismatch for {key}/{target}")
    comparisons = [
        ("P0_vs_P1_noLI", "P0", "P1_noLI", "LI contribution"),
        ("P0_vs_P1_noSSL", "P0", "P1_noSSL", "VICReg contribution"),
        ("P1_noSSL_vs_P2_MobileViTv2", "P1_noSSL", "P2_MobileViTv2", "architecture contribution under ImageNet+LI/noSSL control"),
        ("P0_vs_P2_MobileViTv2_context_only", "P0", "P2_MobileViTv2", "overall-system context; not an isolated architecture effect"),
    ]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sample_indices = rng.integers(0, SPLIT_COUNTS["test"], size=(N_BOOTSTRAP, SPLIT_COUNTS["test"]))
    rows: list[dict[str, Any]] = []
    for comparison, model_a, model_b, interpretation in comparisons:
        distributions: dict[str, np.ndarray] = {}
        points: dict[str, float] = {}
        for target in ("SM0", "SM20"):
            truth = reference[f"{target}_true"]
            error_a = data[model_a][f"{target}_pred"] - truth
            error_b = data[model_b][f"{target}_pred"] - truth
            rmse_a = np.sqrt(np.mean(np.square(error_a[sample_indices]), axis=1))
            rmse_b = np.sqrt(np.mean(np.square(error_b[sample_indices]), axis=1))
            mae_a = np.mean(np.abs(error_a[sample_indices]), axis=1)
            mae_b = np.mean(np.abs(error_b[sample_indices]), axis=1)
            distributions[f"{target}_RMSE"] = rmse_a - rmse_b
            distributions[f"{target}_MAE"] = mae_a - mae_b
            points[f"{target}_RMSE"] = float(np.sqrt(np.mean(error_a ** 2)) - np.sqrt(np.mean(error_b ** 2)))
            points[f"{target}_MAE"] = float(np.mean(np.abs(error_a)) - np.mean(np.abs(error_b)))
        distributions["mean_RMSE"] = (distributions["SM0_RMSE"] + distributions["SM20_RMSE"]) / 2.0
        distributions["mean_MAE"] = (distributions["SM0_MAE"] + distributions["SM20_MAE"]) / 2.0
        points["mean_RMSE"] = (points["SM0_RMSE"] + points["SM20_RMSE"]) / 2.0
        points["mean_MAE"] = (points["SM0_MAE"] + points["SM20_MAE"]) / 2.0
        for metric, distribution in distributions.items():
            rows.append({
                "comparison": comparison, "model_a": model_a, "model_b": model_b,
                "delta_definition": "metric(model_a) - metric(model_b)",
                "interpretation": interpretation, "metric": metric,
                "point_estimate": points[metric], "bootstrap_mean": float(np.mean(distribution)),
                "ci_2_5": float(np.percentile(distribution, 2.5)),
                "ci_97_5": float(np.percentile(distribution, 97.5)),
                "bootstrap_seed": BOOTSTRAP_SEED, "n_bootstrap": N_BOOTSTRAP,
                "n_paired_test_samples": SPLIT_COUNTS["test"], "ci_method": "paired_test_set_percentile",
            })
    write_csv(output_dir / "paired_bootstrap_results.csv", rows, list(rows[0]))
    write_json(output_dir / "paired_bootstrap_results.json", {
        "method": "paired bootstrap over shared test sample indices with replacement",
        "uncertainty_scope": "test-set sampling only; not multiple-training-seed uncertainty",
        "bootstrap_seed": BOOTSTRAP_SEED, "n_bootstrap": N_BOOTSTRAP,
        "n_paired_test_samples": SPLIT_COUNTS["test"], "confidence_interval": "95% percentile",
        "results": rows,
    })
    (output_dir / "PAIRED_BOOTSTRAP_METHOD.md").write_text(
        "# Paired bootstrap method\n\nAll frozen models predict the same 231 held-out samples in the same order. "
        "Using seed 20260906, 10,000 bootstrap samples draw shared sample indices with replacement. "
        "Each delta is metric(model_a) - metric(model_b); negative error deltas favor model_a. "
        "Intervals are the 2.5th and 97.5th percentiles. These are paired test-set bootstrap "
        "intervals and do not represent multiple-training-seed uncertainty.\n",
        encoding="utf-8",
    )
    return rows


def generate_final_test_figures(output_dir: Path) -> None:
    """Render figures from persisted prediction CSVs only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=False)
    predictions = {key: _read_prediction_csv(output_dir / spec["prediction_file"]) for key, spec in MODEL_SPECS.items()}
    p0 = predictions["P0"]
    plotting_rows = []
    for index, sample_id in enumerate(p0["sample_id"]):
        plotting_rows.append({
            "sample_id": sample_id, "SM0_true": p0["SM0_true"][index], "SM0_pred": p0["SM0_pred"][index],
            "SM0_residual": p0["SM0_pred"][index] - p0["SM0_true"][index],
            "SM20_true": p0["SM20_true"][index], "SM20_pred": p0["SM20_pred"][index],
            "SM20_residual": p0["SM20_pred"][index] - p0["SM20_true"][index],
        })
    write_csv(figure_dir / "P0_scatter_residual_data.csv", plotting_rows, list(plotting_rows[0]))
    for target in ("SM0", "SM20"):
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        ax.scatter(p0[f"{target}_true"], p0[f"{target}_pred"], s=18, alpha=0.7)
        bounds = [min(p0[f"{target}_true"].min(), p0[f"{target}_pred"].min()), max(p0[f"{target}_true"].max(), p0[f"{target}_pred"].max())]
        ax.plot(bounds, bounds, color="black", linewidth=1)
        ax.set(xlabel=f"Actual {target} (percentage points)", ylabel=f"Predicted {target} (percentage points)")
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(figure_dir / f"P0_actual_vs_predicted_{target}.{extension}", dpi=300)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        ax.hist(p0[f"{target}_pred"] - p0[f"{target}_true"], bins=20, edgecolor="black")
        ax.axvline(0, color="black", linewidth=1)
        ax.set(xlabel=f"{target} residual: predicted - actual (percentage points)", ylabel="Test samples")
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(figure_dir / f"P0_residual_distribution_{target}.{extension}", dpi=300)
        plt.close(fig)
    metric_rows = []
    for key, spec in MODEL_SPECS.items():
        values = predictions[key]
        mean_rmse = sum(float(np.sqrt(np.mean((values[f"{target}_pred"] - values[f"{target}_true"]) ** 2))) for target in ("SM0", "SM20")) / 2.0
        metric_rows.append({"model": key, "mean_RMSE": mean_rmse})
    write_csv(figure_dir / "model_mean_RMSE_plot_data.csv", metric_rows, ["model", "mean_RMSE"])
    for filename, keys, title in (
        ("ablation_mean_RMSE", ("P0", "P1_noLI", "P1_noSSL"), "Frozen ablation comparison"),
        ("architecture_comparison", ("P1_noSSL", "P2_MobileViTv2"), "Architecture control: ImageNet + LI + no VICReg"),
    ):
        selected = [row for row in metric_rows if row["model"] in keys]
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.bar([row["model"] for row in selected], [row["mean_RMSE"] for row in selected])
        ax.set(ylabel="Mean test RMSE (percentage points)", title=title)
        ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(figure_dir / f"{filename}.{extension}", dpi=300)
        plt.close(fig)


def run_final_test_once(data_root: Path) -> dict[str, Any]:
    """Evaluate exactly four preflighted models on one shared test Dataset."""
    registry = preflight_frozen_registry()  # MUST remain before Dataset/DataLoader construction.
    if FINAL_TEST_DIR.exists():
        raise RuntimeError("FINAL_TEST_OUTPUT_EXISTS_WITHOUT_LOCK")
    staging = REPO / f"results/.final_test_in_progress_{os.getpid()}"
    if staging.exists():
        raise RuntimeError(f"Unexpected staging directory: {staging}")
    staging.mkdir(parents=True)
    try:
        write_json(staging / "frozen_model_registry_snapshot.json", registry)
        common_config = load_yaml(REPO / MODEL_SPECS["P0"]["config"])
        set_determinism(TRAINING_SEED)
        test_dataset = SoilNetDataset(
            REPO / common_config["split"], Path(data_root),
            transform=deterministic_transform(common_config), split="test",
        )
        if len(test_dataset) != SPLIT_COUNTS["test"]:
            raise RuntimeError(f"Locked test count changed: {len(test_dataset)}")
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        metrics_by_model: dict[str, Any] = {}
        registry_by_key = {entry["key"]: entry for entry in registry["models"]}
        for key, spec in MODEL_SPECS.items():
            entry = registry_by_key[key]
            config = load_yaml(REPO / entry["config_path"])
            model = build_frozen_model(config)
            payload = torch.load(Path(entry["checkpoint_path"]), map_location="cpu", weights_only=True)
            state = payload.get("model_state_dict", payload)
            model.load_state_dict(state, strict=True)
            model.to(device).eval()
            with torch.inference_mode():
                prediction_rows, raw_metrics = prediction_rows_and_metrics(model, test_loader, device, config)
            if raw_metrics["n"] != SPLIT_COUNTS["test"]:
                raise RuntimeError(f"Test prediction count changed for {key}")
            summary = _metric_summary(raw_metrics)
            metrics_by_model[key] = {"experiment_id": spec["experiment_id"], **summary, "raw": raw_metrics}
            write_prediction_artifacts(staging / spec["prediction_file"], staging / f"{key}_test_metrics.json", prediction_rows, raw_metrics)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        rows = [{"model": key, "experiment_id": value["experiment_id"], **{k: v for k, v in value.items() if k not in {"experiment_id", "raw"}}} for key, value in metrics_by_model.items()]
        write_json(staging / "final_test_metrics.json", {
            "regression_scale": "original_0_to_100_percentage_points", "test_n": SPLIT_COUNTS["test"],
            "models": metrics_by_model,
        })
        write_csv(staging / "final_test_metrics.csv", rows, list(rows[0]))
        write_csv(staging / "final_ablation_table.csv", [row for row in rows if row["model"] in {"P0", "P1_noLI", "P1_noSSL"}], list(rows[0]))
        write_csv(staging / "final_architecture_comparison.csv", [row for row in rows if row["model"] in {"P1_noSSL", "P2_MobileViTv2"}], list(rows[0]))
        paired_bootstrap(staging)
        generate_final_test_figures(staging)
        (staging / "FINAL_TEST_REPORT.md").write_text(
            "# Final held-out test report\n\nThe four frozen validation-selected models were evaluated once on the same "
            "231-sample held-out test set. Regression values are on the original 0-100 percentage-point scale. "
            "Model selection used validation only; classification metrics are secondary. See `final_test_metrics.csv` "
            "and `paired_bootstrap_results.csv`.\n",
            encoding="utf-8",
        )
        lock = {
            "TEST_OPENED": "YES", "TEST_EVALUATED": "YES", "MODEL_SET_FROZEN": "YES",
            "timestamp": datetime.now(timezone.utc).isoformat(), "split_sha256": SPLIT_SHA256,
            "test_n": SPLIT_COUNTS["test"],
            "model_sha256": {entry["key"]: entry["checkpoint_sha256"] for entry in registry["models"]},
            "software_environment": software_environment(),
            "bootstrap": {"seed": BOOTSTRAP_SEED, "n_bootstrap": N_BOOTSTRAP, "ci": "95% percentile", "paired_by": "sample index"},
            "policy": "No training notebook may use these outcomes for model changes.",
        }
        write_json(staging / "FINAL_TEST_LOCK.json", lock)
        staging.rename(FINAL_TEST_DIR)
        return lock
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
