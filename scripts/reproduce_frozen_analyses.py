#!/usr/bin/env python3
"""Reproduce metrics and bootstrap analyses from frozen prediction CSVs.

No model, image, dataset object, training function, validation loader, or test
loader is created. The held-out evidence is re-analysed only from already
frozen P0/P1/P2 predictions; P3 remains validation-only.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np

from soilnet.evaluation.metrics import classification_metrics, regression_metrics


VALIDATION_KEYS = ("P0", "P1_noLI", "P1_noSSL", "P2", "P3")
TEST_KEYS = ("P0", "P1_noLI", "P1_noSSL", "P2")
FINAL_BOOTSTRAP_SEED = 20260906
FACTORIAL_BOOTSTRAP_SEED = 20260912
N_BOOTSTRAP = 10_000


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def arrays(path: Path) -> dict[str, np.ndarray]:
    rows = read_rows(path)
    return {
        "sample_id": np.asarray([row["sample_id"] for row in rows]),
        "SM0_true": np.asarray([float(row["SM_0_true"]) for row in rows]),
        "SM0_pred": np.asarray([float(row["SM_0_pred"]) for row in rows]),
        "SM20_true": np.asarray([float(row["SM_20_true"]) for row in rows]),
        "SM20_pred": np.asarray([float(row["SM_20_pred"]) for row in rows]),
        "class_true": np.asarray([int(row["class_true"]) for row in rows]),
        "class_pred": np.asarray([int(row["class_pred"]) for row in rows]),
    }


def calculate(data: dict[str, np.ndarray]) -> dict:
    return {
        "regression": {
            "SM_0": regression_metrics(data["SM0_true"], data["SM0_pred"]),
            "SM_20": regression_metrics(data["SM20_true"], data["SM20_pred"]),
        },
        "classification": classification_metrics(data["class_true"], data["class_pred"]),
        "n": len(data["sample_id"]),
        "regression_scale": "original_0_to_100_percentage_points",
    }


def compare_metrics(observed: dict, expected: dict, label: str) -> None:
    if observed["n"] != expected["n"]:
        raise RuntimeError(f"{label}: sample count mismatch")
    for target in ("SM_0", "SM_20"):
        for metric in ("rmse", "mae", "me", "r2"):
            delta = abs(observed["regression"][target][metric] - expected["regression"][target][metric])
            if delta > 5e-6:
                raise RuntimeError(f"{label}: {target}/{metric} mismatch {delta}")
    for metric in ("accuracy", "macro_f1", "macro_precision", "macro_recall"):
        delta = abs(observed["classification"][metric] - expected["classification"][metric])
        if delta > 5e-12:
            raise RuntimeError(f"{label}: classification/{metric} mismatch {delta}")


def final_paired_bootstrap(data: dict[str, dict[str, np.ndarray]]) -> list[dict]:
    reference = data["P0"]
    for key, values in data.items():
        if not np.array_equal(reference["sample_id"], values["sample_id"]):
            raise RuntimeError(f"Held-out paired sample order mismatch: {key}")
        for target in ("SM0", "SM20"):
            if not np.allclose(reference[f"{target}_true"], values[f"{target}_true"], rtol=0, atol=1e-7):
                raise RuntimeError(f"Held-out truth mismatch: {key}/{target}")
    comparisons = (
        ("P0_vs_P1_noLI", "P0", "P1_noLI", "LI contribution"),
        ("P0_vs_P1_noSSL", "P0", "P1_noSSL", "VICReg contribution"),
        ("P1_noSSL_vs_P2_MobileViTv2", "P1_noSSL", "P2", "architecture contribution under ImageNet+LI/noSSL control"),
        ("P0_vs_P2_MobileViTv2_context_only", "P0", "P2", "overall-system context; not an isolated architecture effect"),
    )
    rng = np.random.default_rng(FINAL_BOOTSTRAP_SEED)
    indices = rng.integers(0, 231, size=(N_BOOTSTRAP, 231))
    rows = []
    for comparison, model_a, model_b, interpretation in comparisons:
        distributions, points = {}, {}
        for target in ("SM0", "SM20"):
            truth = reference[f"{target}_true"]
            error_a = data[model_a][f"{target}_pred"] - truth
            error_b = data[model_b][f"{target}_pred"] - truth
            distributions[f"{target}_RMSE"] = (
                np.sqrt(np.mean(np.square(error_a[indices]), axis=1))
                - np.sqrt(np.mean(np.square(error_b[indices]), axis=1))
            )
            distributions[f"{target}_MAE"] = (
                np.mean(np.abs(error_a[indices]), axis=1) - np.mean(np.abs(error_b[indices]), axis=1)
            )
            points[f"{target}_RMSE"] = float(
                np.sqrt(np.mean(error_a ** 2)) - np.sqrt(np.mean(error_b ** 2))
            )
            points[f"{target}_MAE"] = float(np.mean(np.abs(error_a)) - np.mean(np.abs(error_b)))
        for metric in ("RMSE", "MAE"):
            distributions[f"mean_{metric}"] = (
                distributions[f"SM0_{metric}"] + distributions[f"SM20_{metric}"]
            ) / 2.0
            points[f"mean_{metric}"] = (points[f"SM0_{metric}"] + points[f"SM20_{metric}"]) / 2.0
        for metric, distribution in distributions.items():
            rows.append({
                "comparison": comparison,
                "model_a": model_a,
                "model_b": "P2_MobileViTv2" if model_b == "P2" else model_b,
                "delta_definition": "metric(model_a) - metric(model_b)",
                "interpretation": interpretation,
                "metric": metric,
                "point_estimate": points[metric],
                "bootstrap_mean": float(np.mean(distribution)),
                "ci_2_5": float(np.percentile(distribution, 2.5)),
                "ci_97_5": float(np.percentile(distribution, 97.5)),
                "bootstrap_seed": FINAL_BOOTSTRAP_SEED,
                "n_bootstrap": N_BOOTSTRAP,
                "n_paired_test_samples": 231,
                "ci_method": "paired_test_set_percentile",
            })
    return rows


def factorial_bootstrap(data: dict[str, dict[str, np.ndarray]]) -> list[dict]:
    keys = ("P0", "P1_noSSL", "P2", "P3")
    reference = data["P0"]
    for key in keys:
        if not np.array_equal(reference["sample_id"], data[key]["sample_id"]):
            raise RuntimeError(f"Validation paired sample order mismatch: {key}")
    rng = np.random.default_rng(FACTORIAL_BOOTSTRAP_SEED)
    indices = rng.integers(0, 289, size=(N_BOOTSTRAP, 289))
    metrics, points = {}, {}
    for key in keys:
        per_model, point = {}, {}
        for target in ("SM0", "SM20"):
            error = data[key][f"{target}_pred"] - data[key][f"{target}_true"]
            per_model[f"{target} RMSE"] = np.sqrt(np.mean(np.square(error[indices]), axis=1))
            per_model[f"{target} MAE"] = np.mean(np.abs(error[indices]), axis=1)
            point[f"{target} RMSE"] = float(np.sqrt(np.mean(np.square(error))))
            point[f"{target} MAE"] = float(np.mean(np.abs(error)))
        for metric in ("RMSE", "MAE"):
            per_model[f"mean {metric}"] = (per_model[f"SM0 {metric}"] + per_model[f"SM20 {metric}"]) / 2
            point[f"mean {metric}"] = (point[f"SM0 {metric}"] + point[f"SM20 {metric}"]) / 2
        metrics[key], points[key] = per_model, point
    effects = {
        "SoilNet SSL effect (P0 - P1_noSSL)": ("P0", "P1_noSSL"),
        "MobileViTv2 SSL effect (P3 - P2)": ("P3", "P2"),
    }
    metric_order = ("SM0 RMSE", "SM20 RMSE", "mean RMSE", "SM0 MAE", "SM20 MAE", "mean MAE")
    replicates = {
        effect: {metric: metrics[left][metric] - metrics[right][metric] for metric in metric_order}
        for effect, (left, right) in effects.items()
    }
    interaction = "Architecture × SSL interaction: (P3 - P2) - (P0 - P1_noSSL)"
    replicates[interaction] = {
        metric: replicates["MobileViTv2 SSL effect (P3 - P2)"][metric]
        - replicates["SoilNet SSL effect (P0 - P1_noSSL)"][metric]
        for metric in metric_order
    }
    rows = []
    for effect in (*effects, interaction):
        for metric in metric_order:
            values = replicates[effect][metric]
            if effect == interaction:
                point = points["P3"][metric] - points["P2"][metric] - (points["P0"][metric] - points["P1_noSSL"][metric])
            else:
                left, right = effects[effect]
                point = points[left][metric] - points[right][metric]
            rows.append({
                "effect": effect,
                "metric": metric,
                "point estimate": point,
                "bootstrap mean": float(values.mean()),
                "95% CI lower": float(np.percentile(values, 2.5)),
                "95% CI upper": float(np.percentile(values, 97.5)),
                "proportion > 0": float(np.mean(values > 0)),
                "proportion < 0": float(np.mean(values < 0)),
                "replicates": N_BOOTSTRAP,
                "seed": FACTORIAL_BOOTSTRAP_SEED,
            })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_rows(observed: list[dict], expected_path: Path, keys: tuple[str, ...]) -> None:
    expected = read_rows(expected_path)
    if len(observed) != len(expected):
        raise RuntimeError(f"Row count mismatch against {expected_path}")
    for left, right in zip(observed, expected, strict=True):
        if any(str(left[key]) != right[key] for key in keys):
            raise RuntimeError(f"Identity mismatch against {expected_path}")
        for key in set(left) - set(keys):
            # Prediction CSVs intentionally store decimal text, so recomputed
            # points may differ from in-memory values by a few micro-units.
            if isinstance(left[key], (int, float)) and abs(float(left[key]) - float(right[key])) > 5e-6:
                raise RuntimeError(f"Numeric mismatch for {key} against {expected_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    validation, test, metric_report = {}, {}, {"validation": {}, "test": {}}
    for key in VALIDATION_KEYS:
        base = REPO / "results/frozen" / key
        prediction = base / ("supervised/validation_predictions.csv" if key == "P3" else "validation_predictions.csv")
        metric_path = base / ("supervised/validation_metrics.json" if key == "P3" else "validation_metrics.json")
        validation[key] = arrays(prediction)
        observed = calculate(validation[key])
        expected = json.loads(metric_path.read_text(encoding="utf-8"))
        compare_metrics(observed, expected, f"{key}/validation")
        metric_report["validation"][key] = observed
    for key in TEST_KEYS:
        base = REPO / "results/frozen" / key
        test[key] = arrays(base / "test_predictions.csv")
        observed = calculate(test[key])
        expected = json.loads((base / "test_metrics.json").read_text(encoding="utf-8"))
        compare_metrics(observed, expected, f"{key}/test")
        metric_report["test"][key] = observed

    final_rows = final_paired_bootstrap(test)
    compare_rows(final_rows, REPO / "results/final_test/paired_bootstrap_results.csv", ("comparison", "metric"))
    factorial_rows = factorial_bootstrap(validation)
    compare_rows(factorial_rows, REPO / "results/p3_analysis/vicreg_paired_bootstrap.csv", ("effect", "metric"))

    complexity_rows = []
    for key in VALIDATION_KEYS:
        base = REPO / "results/frozen" / key
        metadata_path = base / ("supervised/run_metadata.json" if key == "P3" else "run_metadata.json")
        complexity = json.loads(metadata_path.read_text(encoding="utf-8"))["complexity"]
        complexity_rows.append({
            "model": "P2_MobileViTv2" if key == "P2" else key,
            "checkpoint_size_bytes": complexity["checkpoint_size_bytes"],
            "input_shape": complexity["input_shape"],
            "li_input_dimension": complexity["li_input_dimension"],
            "parameters": complexity["parameters"],
            "trainable_parameters": complexity["trainable_parameters"],
        })
    published_complexity = read_rows(REPO / "results/paper/08_model_complexity.csv")
    if complexity_rows[:4] != [
        {key: int(value) if key in {"checkpoint_size_bytes", "li_input_dimension", "parameters", "trainable_parameters"} else value
         for key, value in row.items()}
        for row in published_complexity
    ]:
        raise RuntimeError("P0/P1/P2 complexity table does not reproduce")

    (output / "recomputed_metrics.json").write_text(json.dumps(metric_report, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "paired_test_bootstrap.csv", final_rows)
    write_csv(output / "factorial_validation_bootstrap.csv", factorial_rows)
    write_csv(output / "model_complexity.csv", complexity_rows)
    (output / "status.json").write_text(json.dumps({
        "status": "PASS",
        "source": "frozen prediction CSVs",
        "training_run": False,
        "image_files_opened": 0,
        "P3_TEST_SET_OPENED": False,
        "test_scope": list(TEST_KEYS),
        "validation_scope": list(VALIDATION_KEYS),
        "paired_bootstrap_replicates": N_BOOTSTRAP,
    }, indent=2) + "\n", encoding="utf-8")
    print("FROZEN_ANALYSIS_REPRODUCTION_PASS; P3_TEST_SET_OPENED=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
