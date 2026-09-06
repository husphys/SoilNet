from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch
import joblib
import numpy as np

from soilnet.data import SoilNetDataset
from soilnet.evaluation.predictions import prediction_rows_and_metrics, write_prediction_artifacts
from soilnet.evaluation.metrics import classification_metrics, regression_metrics
from soilnet.final_protocol import current_git_commit
from soilnet.io import sha256_file, write_json
from soilnet.models import build_model
from soilnet.reproducibility import set_determinism
from soilnet.training.engine import deterministic_transform
from soilnet.utils import load_experiment_context


REPO = Path(__file__).resolve().parents[3]


def build_test_loader(context):
    dataset = SoilNetDataset(
        context.split_path, context.data_root,
        transform=deterministic_transform(context.config), split="test",
    )
    if len(dataset) != 231:
        raise RuntimeError(f"STOP: locked test count changed: {len(dataset)}")
    return torch.utils.data.DataLoader(
        dataset, batch_size=int(context.config["batch_size"]), shuffle=False, num_workers=0
    )


def _guard_marker(allow_manual_rerun: bool) -> tuple[Path, str]:
    marker = REPO / "results/final/test_evaluation_completed.json"
    rerun_suffix = ""
    if marker.exists():
        if not allow_manual_rerun:
            raise RuntimeError("Final test evaluation already completed; explicit manual override is required")
        rerun_suffix = datetime.now(timezone.utc).strftime("_MANUAL_RERUN_%Y%m%dT%H%M%SZ")
        print("WARNING: explicit manual test rerun; original global marker will not be overwritten")
    return marker, rerun_suffix


def _preflight_deep(config_paths: Iterable[str | Path]):
    prepared = []
    for config_path in config_paths:
        context = load_experiment_context(config_path)
        metadata_path = context.run_dir / "run_metadata.json"
        if not metadata_path.is_file():
            raise RuntimeError(f"Training-completion metadata missing: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("training_completed") is not True or metadata.get("test_evaluated") is not False:
            raise RuntimeError(f"Model is not frozen and test-clean: {context.config['experiment_id']}")
        if metadata.get("config_sha256") != context.config_sha256:
            raise RuntimeError("Experiment config changed after training")
        checkpoint_path = Path(metadata["checkpoint_path"])
        if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != metadata["checkpoint_sha256"]:
            raise RuntimeError("Frozen model checkpoint missing or changed")
        prepared.append((context, metadata, checkpoint_path))
    return prepared


def _evaluate_preflighted_deep(prepared, rerun_suffix: str):
    records, marker_models = {}, {}
    for context, metadata, checkpoint_path in prepared:
        set_determinism(int(context.config["seed"]))
        model, _ = build_model(context.config, context.checkpoint_root, pretrained_override=False)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        device = torch.device("cuda")
        model.to(device)
        prediction_rows, metrics = prediction_rows_and_metrics(
            model, build_test_loader(context), device, context.config
        )
        output = context.run_dir / f"final_test{rerun_suffix}"
        output.mkdir(parents=True, exist_ok=False)
        write_prediction_artifacts(
            output / "test_predictions.csv", output / "test_metrics.json", prediction_rows, metrics
        )
        records[context.config["experiment_id"]] = metrics
        marker_models[context.config["experiment_id"]] = {
            "checkpoint_sha256": metadata["checkpoint_sha256"],
            "config_sha256": context.config_sha256,
        }
    return records, marker_models


@torch.inference_mode()
def _test_embeddings(context, device):
    model, _ = build_model(context.config, context.checkpoint_root)
    model.to(device).eval()
    loader = build_test_loader(context)
    features, regression, classification, source_rows = [], [], [], []
    cursor = 0
    for image, light, target_regression, target_classification in loader:
        features.append(model.extract_features(image.to(device), light.to(device)).cpu().numpy())
        regression.append(target_regression.numpy() * 100.0)
        classification.append(target_classification.numpy())
        source_rows.extend(loader.dataset.rows[cursor:cursor + len(image)])
        cursor += len(image)
    return np.concatenate(features), np.concatenate(regression), np.concatenate(classification), source_rows


def _preflight_classical(config_path: str | Path):
    context = load_experiment_context(config_path)
    metadata_path = context.run_dir / "run_metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"Classical training-completion metadata missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("training_completed") is not True or metadata.get("test_evaluated") is not False:
        raise RuntimeError("Classical estimators are not frozen and test-clean")
    if metadata.get("config_sha256") != context.config_sha256:
        raise RuntimeError("Classical config changed after estimator fitting")
    estimators = {}
    for name in ("KNN", "SVM", "GBM", "DT", "MLP"):
        paths = {
            "classifier": context.run_dir / f"{name}_classifier.joblib",
            "SM_0": context.run_dir / f"{name}_SM_0_regressor.joblib",
            "SM_20": context.run_dir / f"{name}_SM_20_regressor.joblib",
        }
        if not all(path.is_file() for path in paths.values()):
            raise RuntimeError(f"Frozen classical estimator missing: {name}")
        estimators[name] = paths
    return context, metadata, estimators


def _evaluate_preflighted_classical(prepared, rerun_suffix: str):
    context, _, estimator_paths = prepared
    device = torch.device("cuda")
    features, truth_regression, truth_class, source_rows = _test_embeddings(context, device)
    records, marker_models = {}, {}
    for name, paths in estimator_paths.items():
        classifier = joblib.load(paths["classifier"])
        regressors = [joblib.load(paths[target]) for target in ("SM_0", "SM_20")]
        predicted_class = classifier.predict(features)
        probabilities = classifier.predict_proba(features) if hasattr(classifier, "predict_proba") else None
        predicted_regression = np.column_stack([
            regressors[index].predict(features) for index in range(2)
        ])
        rows = []
        for index, source in enumerate(source_rows):
            rows.append({
                "sample_id": source["sample_id"], "relative_path": source["relative_path"],
                "SM_0_true": truth_regression[index, 0], "SM_0_pred": predicted_regression[index, 0],
                "SM_20_true": truth_regression[index, 1], "SM_20_pred": predicted_regression[index, 1],
                "class_true": int(truth_class[index]), "class_pred": int(predicted_class[index]),
                "class_probabilities": "" if probabilities is None else ";".join(
                    f"{value:.10g}" for value in probabilities[index]
                ),
            })
        metrics = {
            "classification": classification_metrics(truth_class, predicted_class),
            "regression": {
                target: regression_metrics(truth_regression[:, index], predicted_regression[:, index])
                for index, target in enumerate(("SM_0", "SM_20"))
            },
            "n": len(rows), "regression_scale": "original_0_to_100_percentage_points",
        }
        output = context.run_dir / f"final_test{rerun_suffix}" / name
        output.mkdir(parents=True, exist_ok=False)
        write_prediction_artifacts(output / "test_predictions.csv", output / "test_metrics.json", rows, metrics)
        with (output / "classification_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["model", *metrics["classification"]])
            writer.writeheader(); writer.writerow({"model": name, **metrics["classification"]})
        regression_rows = [
            {"model": name, "target": target, **values}
            for target, values in metrics["regression"].items()
        ]
        with (output / "regression_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(regression_rows[0]))
            writer.writeheader(); writer.writerows(regression_rows)
        records[name] = metrics
        marker_models[f"P2_CLASSICAL_ML:{name}"] = {
            "config_sha256": context.config_sha256,
            "estimator_sha256": {kind: sha256_file(path) for kind, path in paths.items()},
        }
    return records, marker_models


def evaluate_all_frozen_models_once(
    deep_config_paths: Iterable[str | Path],
    classical_config_path: str | Path | None = None,
    *,
    allow_manual_rerun: bool = False,
) -> dict:
    marker, rerun_suffix = _guard_marker(allow_manual_rerun)
    selected_deep = list(deep_config_paths)
    if not selected_deep and classical_config_path is None:
        raise RuntimeError("STOP: explicitly select at least one frozen experiment before test access")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU_BLOCKED: final evaluation requires CUDA")
    # Complete all config/artifact preflight checks before constructing the
    # first test loader or computing the first prospective metric.
    deep_prepared = _preflight_deep(selected_deep)
    classical_prepared = _preflight_classical(classical_config_path) if classical_config_path else None
    deep_records, deep_markers = _evaluate_preflighted_deep(deep_prepared, rerun_suffix)
    if classical_prepared is None:
        classical_records, classical_markers = {}, {}
    else:
        classical_records, classical_markers = _evaluate_preflighted_classical(classical_prepared, rerun_suffix)
    context = deep_prepared[0][0] if deep_prepared else classical_prepared[0]
    records = {"deep": deep_records, "classical": classical_records}
    marker_models = {**deep_markers, **classical_markers}
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(REPO),
        "manifest_sha256": context.manifest_sha256,
        "split_sha256": context.split_sha256,
        "model_checkpoint_hashes_and_config_hashes": marker_models,
        "manual_rerun": bool(rerun_suffix),
    }
    if not rerun_suffix:
        marker.parent.mkdir(parents=True, exist_ok=True)
        descriptor = marker.open("x", encoding="utf-8")
        with descriptor:
            json.dump(payload, descriptor, indent=2, sort_keys=True)
            descriptor.write("\n")
    return records


def evaluate_frozen_deep_models_once(
    config_paths: Iterable[str | Path], *, allow_manual_rerun: bool = False,
) -> dict:
    """Compatibility entrypoint for deep-only evaluation with the same marker."""
    marker, rerun_suffix = _guard_marker(allow_manual_rerun)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU_BLOCKED: final deep-model evaluation requires CUDA")
    prepared = _preflight_deep(config_paths)
    records, marker_models = _evaluate_preflighted_deep(prepared, rerun_suffix)
    context = prepared[0][0]
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(), "git_commit": current_git_commit(REPO),
        "manifest_sha256": context.manifest_sha256, "split_sha256": context.split_sha256,
        "model_checkpoint_hashes_and_config_hashes": marker_models, "manual_rerun": bool(rerun_suffix),
    }
    if not rerun_suffix:
        marker.parent.mkdir(parents=True, exist_ok=True)
        with marker.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True); handle.write("\n")
    return records
