from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from torch.utils.data import DataLoader

from soilnet.data import SoilNetDataset
from soilnet.evaluation import classification_metrics, regression_metrics
from soilnet.final_protocol import current_git_commit
from soilnet.io import write_json
from soilnet.models import build_model
from soilnet.reproducibility import set_determinism
from soilnet.training.engine import deterministic_transform
from soilnet.utils import ExperimentContext


def _feature_loader(context: ExperimentContext, split: str) -> DataLoader:
    if split not in {"train", "validation"}:
        raise RuntimeError("Classical model-selection notebook may load only train/validation")
    dataset = SoilNetDataset(
        context.split_path, context.data_root,
        transform=deterministic_transform(context.config), split=split,
    )
    return DataLoader(dataset, batch_size=int(context.config["batch_size"]), shuffle=False, num_workers=0)


@torch.inference_mode()
def extract_embeddings(context: ExperimentContext, split: str, device: torch.device):
    model, load_report = build_model(context.config, context.checkpoint_root)
    model.to(device).eval()
    loader = _feature_loader(context, split)
    features, regression, classification, sample_ids = [], [], [], []
    cursor = 0
    for image, light, target_regression, target_classification in loader:
        batch_features = model.extract_features(image.to(device), light.to(device)).cpu().numpy()
        features.append(batch_features)
        regression.append(target_regression.numpy() * 100.0)
        classification.append(target_classification.numpy())
        sample_ids.extend(row["sample_id"] for row in loader.dataset.rows[cursor:cursor + len(image)])
        cursor += len(image)
    return {
        "features": np.concatenate(features),
        "regression": np.concatenate(regression),
        "classification": np.concatenate(classification),
        "sample_ids": np.asarray(sample_ids),
        "load_report": load_report,
    }


def build_classical_estimators(seed: int):
    classifiers = {
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="uniform")),
        "SVM": make_pipeline(StandardScaler(), SVC(
            kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
            probability=True, random_state=seed,
        )),
        "GBM": make_pipeline(StandardScaler(), GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=seed,
        )),
        "DT": make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=seed)),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(100,), alpha=0.0001, max_iter=500, random_state=seed,
        )),
    }
    regressors = {
        "KNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5, weights="uniform")),
        "SVM": make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1.0, gamma="scale", epsilon=0.1)),
        "GBM": make_pipeline(StandardScaler(), GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=seed,
        )),
        "DT": make_pipeline(StandardScaler(), DecisionTreeRegressor(random_state=seed)),
        "MLP": make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(100,), alpha=0.0001, max_iter=500, random_state=seed,
        )),
    }
    return classifiers, regressors


def run_classical_experiment(context: ExperimentContext) -> dict[str, Any]:
    """Fit fixed classical models on train embeddings; validate only."""
    set_determinism(int(context.config["seed"]))
    context.run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = extract_embeddings(context, "train", device)
    validation = extract_embeddings(context, "validation", device)
    for name, values in (("train", train), ("validation", validation)):
        np.savez_compressed(
            context.run_dir / f"embeddings_{name}.npz",
            features=values["features"], regression=values["regression"],
            classification=values["classification"], sample_ids=values["sample_ids"],
        )
    classifiers, regressors = build_classical_estimators(int(context.config["seed"]))
    classification_rows, regression_rows = [], []
    for model_name, estimator in classifiers.items():
        estimator.fit(train["features"], train["classification"])
        prediction = estimator.predict(validation["features"])
        metrics = classification_metrics(validation["classification"], prediction)
        classification_rows.append({"model": model_name, **metrics})
        joblib.dump(estimator, context.run_dir / f"{model_name}_classifier.joblib")
    for model_name, estimator in regressors.items():
        per_target = {}
        for index, target in enumerate(("SM_0", "SM_20")):
            fitted = estimator if index == 0 else build_classical_estimators(int(context.config["seed"]))[1][model_name]
            fitted.fit(train["features"], train["regression"][:, index])
            prediction = fitted.predict(validation["features"])
            metrics = regression_metrics(validation["regression"][:, index], prediction)
            regression_rows.append({"model": model_name, "target": target, **metrics})
            joblib.dump(fitted, context.run_dir / f"{model_name}_{target}_regressor.joblib")
            per_target[target] = metrics
    for filename, rows in (
        ("validation_classification_metrics.csv", classification_rows),
        ("validation_regression_metrics.csv", regression_rows),
    ):
        with (context.run_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    metadata = {
        "experiment_id": context.config["experiment_id"], "training_completed": True,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(Path(__file__).resolve().parents[3]),
        "config_sha256": context.config_sha256, "manifest_sha256": context.manifest_sha256,
        "split_sha256": context.split_sha256, "feature_extractor": "verified_vicreg_mu27_soilnet",
        "feature_checkpoint_sha256": context.config["ssl_checkpoint"]["sha256"],
        "train_samples": len(train["features"]), "validation_samples": len(validation["features"]),
        "classification_and_regression_reported_separately": True,
        "large_grid_search_used": False, "test_evaluated": False,
        "ssl_load_report": train["load_report"],
    }
    write_json(context.run_dir / "run_metadata.json", metadata)
    return metadata
