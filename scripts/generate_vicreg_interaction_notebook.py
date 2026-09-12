from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "notebooks/final_experiments/14_vicreg_architecture_interaction_analysis.ipynb"


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


def markdown(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


book = nbf.v4.new_notebook()
book.metadata.update({
    "kernelspec": {"display_name": "Python (soilnet)", "language": "python", "name": "soilnet"},
    "language_info": {"name": "python", "version": "3.11"},
})
book.cells = [
    markdown("""
    # VICReg architecture-interaction analysis — frozen validation only

    This notebook reads only the four frozen validation prediction, metric, metadata,
    and selected-checkpoint artifacts. It does not construct a dataset/model/loader,
    does not train, and does not access held-out test samples.

    **Scientific question:** whether the validation-scale VICReg effect differs
    by architecture. **Configuration:** 10,000 paired bootstrap replicates at
    seed 20260912. **Dataset:** frozen prediction CSVs only. **Split:** validation
    only (289 aligned samples). **Checkpoint:** identities are inherited from
    frozen metadata. **Expected outputs:** four-model metrics, paired effects,
    interaction effects, and factorial summary.
    """),
    code("""
    from __future__ import annotations

    import hashlib
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from IPython.display import display

    def find_repo(start: Path) -> Path:
        for candidate in (start.resolve(), *start.resolve().parents):
            if (candidate / "pyproject.toml").is_file() and (candidate / "src/soilnet").is_dir():
                return candidate
        raise RuntimeError("STOP: execute from inside the SoilNet checkout")

    REPO = find_repo(Path.cwd())

    FROZEN_ROOT = REPO / "results/frozen"
    OUTPUT_DIR = REPO / "results/p3_analysis"
    NOTEBOOK_PATH = REPO / "notebooks/final_experiments/14_vicreg_architecture_interaction_analysis.ipynb"
    GENERATOR_PATH = REPO / "scripts/generate_vicreg_interaction_notebook.py"
    N_VALIDATION = 289
    BOOTSTRAP_REPLICATES = 10_000
    BOOTSTRAP_SEED = 20260912
    LOCKED_MANIFEST_SHA256 = "8ff45054d4b8e3df9758d0c112dc16719572b2267906fb3a5ed5b3262a6732bd"
    LOCKED_SPLIT_SHA256 = "8927b8223b8c4c234d264ad9ea62ac2df6161124a79787a2e71eeb5cd23eae2f"
    TEST_SET_OPENED = False

    EXPERIMENTS = {
        "P0": {
            "label": "SoilNet + VICReg + LI",
            "architecture": "SoilNet",
            "initialization": "+VICReg",
            "experiment_id": "P0_FINAL_SOILNET_VICREG_MU27_LI_V4_BESTREG",
            "artifact_dir": FROZEN_ROOT / "P0",
        },
        "P1_noSSL": {
            "label": "SoilNet + ImageNet only + LI",
            "architecture": "SoilNet",
            "initialization": "ImageNet-only",
            "experiment_id": "P1_SOILNET_IMAGENET_LI_NO_SSL_BESTREG",
            "artifact_dir": FROZEN_ROOT / "P1_noSSL",
        },
        "P2": {
            "label": "MobileViTv2 + ImageNet only + LI",
            "architecture": "MobileViTv2",
            "initialization": "ImageNet-only",
            "experiment_id": "P2_MOBILEVITV2_IMAGENET_LI_BESTREG",
            "artifact_dir": FROZEN_ROOT / "P2",
        },
        "P3": {
            "label": "MobileViTv2 + VICReg + LI",
            "architecture": "MobileViTv2",
            "initialization": "+VICReg",
            "experiment_id": "P3_MOBILEVITV2_VICREG_LI_BESTREG",
            "artifact_dir": FROZEN_ROOT / "P3/supervised",
        },
    }

    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    """),
    markdown("## 1. Frozen-artifact identity verification"),
    code("""
    frames, metrics_by_model, metadata_by_model = {}, {}, {}
    source_hashes_before = {}
    identity_rows = []

    for key, spec in EXPERIMENTS.items():
        directory = spec["artifact_dir"]
        paths = {
            "predictions": directory / "validation_predictions.csv",
            "metrics": directory / "validation_metrics.json",
            "metadata": directory / "run_metadata.json",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise RuntimeError(f"STOP: missing frozen {key} validation artifacts: {missing}")
        frame = pd.read_csv(paths["predictions"], dtype={"sample_id": str, "relative_path": str})
        metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        expected_sha = metadata["primary_checkpoint_sha256"]
        failures = []
        if metadata.get("experiment_id") != spec["experiment_id"]:
            failures.append("experiment_id")
        if metadata.get("training_completed") is not True or metadata.get("epochs_completed") != 60:
            failures.append("frozen_training_completion")
        if metadata.get("manifest_sha256") != LOCKED_MANIFEST_SHA256:
            failures.append("manifest_sha256")
        if metadata.get("split_sha256") != LOCKED_SPLIT_SHA256:
            failures.append("split_sha256")
        if len(frame) != N_VALIDATION or metrics.get("n") != N_VALIDATION:
            failures.append("validation_count")
        if frame["sample_id"].isna().any() or not frame["sample_id"].is_unique:
            failures.append("sample_id_uniqueness")
        if metrics.get("generated_from_checkpoint_sha256") != expected_sha:
            failures.append("metrics_checkpoint_sha256")
        if metrics.get("generated_from_checkpoint_epoch") != metadata.get("best_epoch"):
            failures.append("best_epoch")
        if failures:
            raise RuntimeError(f"STOP: frozen identity verification failed for {key}: {failures}")
        frames[key], metrics_by_model[key], metadata_by_model[key] = frame, metrics, metadata
        source_hashes_before[key] = {
            **{name: sha256_file(path) for name, path in paths.items()},
            "checkpoint_declared_sha256": expected_sha,
        }
        identity_rows.append({
            "model": key,
            "experiment_id": metadata["experiment_id"],
            "n_validation": len(frame),
            "best_epoch": metadata["best_epoch"],
            "checkpoint_sha256": expected_sha,
            "identity_verification": "PASS",
        })

    identity_table = pd.DataFrame(identity_rows)
    display(identity_table)
    print({"TEST_SET_OPENED": TEST_SET_OPENED})
    """),
    markdown("## 2. Exact paired alignment and ground-truth verification"),
    code("""
    reference = frames["P0"]
    alignment_columns = ["sample_id", "relative_path"]
    truth_columns = ["SM_0_true", "SM_20_true"]
    alignment_report = {}
    for key, frame in frames.items():
        ids_order_equal = frame["sample_id"].tolist() == reference["sample_id"].tolist()
        paths_order_equal = frame["relative_path"].tolist() == reference["relative_path"].tolist()
        sm0_truth_equal = np.array_equal(frame["SM_0_true"].to_numpy(), reference["SM_0_true"].to_numpy())
        sm20_truth_equal = np.array_equal(frame["SM_20_true"].to_numpy(), reference["SM_20_true"].to_numpy())
        alignment_report[key] = {
            "n": len(frame),
            "sample_ids_same_order": ids_order_equal,
            "relative_paths_same_order": paths_order_equal,
            "SM0_ground_truth_exact": sm0_truth_equal,
            "SM20_ground_truth_exact": sm20_truth_equal,
        }
        if not all((ids_order_equal, paths_order_equal, sm0_truth_equal, sm20_truth_equal)):
            raise RuntimeError(f"STOP: paired validation alignment failed for {key}")

    display(pd.DataFrame(alignment_report).T)
    print({
        "VALIDATION_ALIGNMENT": "PASS",
        "N_VALIDATION": N_VALIDATION,
        "SPLIT_SHA256": LOCKED_SPLIT_SHA256,
        "TEST_SET_OPENED": TEST_SET_OPENED,
    })
    """),
    markdown("## 3. Exact frozen validation metrics"),
    code("""
    def metric_values(frame: pd.DataFrame):
        values = {}
        for target in ("SM_0", "SM_20"):
            truth = frame[f"{target}_true"].to_numpy(dtype=np.float64)
            prediction = frame[f"{target}_pred"].to_numpy(dtype=np.float64)
            residual = prediction - truth
            values[target] = {
                "rmse": float(np.sqrt(np.mean(residual ** 2))),
                "mae": float(np.mean(np.abs(residual))),
                "r2": float(1.0 - np.sum(residual ** 2) / np.sum((truth - truth.mean()) ** 2)),
            }
        return values

    exact_rows = []
    point_metrics = {}
    for key in ("P0", "P1_noSSL", "P2", "P3"):
        recomputed = metric_values(frames[key])
        frozen = metrics_by_model[key]["regression"]
        for target in ("SM_0", "SM_20"):
            for metric in ("rmse", "mae", "r2"):
                if not np.isclose(recomputed[target][metric], frozen[target][metric], rtol=0, atol=5e-6):
                    raise RuntimeError(f"STOP: {key} {target} {metric} does not reproduce frozen metrics")
        point_metrics[key] = {
            "SM0 RMSE": frozen["SM_0"]["rmse"],
            "SM0 MAE": frozen["SM_0"]["mae"],
            "SM0 R²": frozen["SM_0"]["r2"],
            "SM20 RMSE": frozen["SM_20"]["rmse"],
            "SM20 MAE": frozen["SM_20"]["mae"],
            "SM20 R²": frozen["SM_20"]["r2"],
        }
        point_metrics[key]["mean RMSE"] = (point_metrics[key]["SM0 RMSE"] + point_metrics[key]["SM20 RMSE"]) / 2
        point_metrics[key]["mean MAE"] = (point_metrics[key]["SM0 MAE"] + point_metrics[key]["SM20 MAE"]) / 2
        exact_rows.append({
            "experiment": key,
            "configuration": EXPERIMENTS[key]["label"],
            **point_metrics[key],
            "best epoch": int(metadata_by_model[key]["best_epoch"]),
            "parameters": int(metadata_by_model[key]["complexity"]["parameters"]),
        })

    exact_metrics = pd.DataFrame(exact_rows)
    display(exact_metrics)
    """),
    markdown("""
    ## 4. Paired bootstrap with 10,000 common-index replicates

    Every replicate resamples the same 289 row indices for all four models.
    For all error differences below, **difference < 0 means the model on the left is better**.
    """),
    code("""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    common_indices = rng.integers(0, N_VALIDATION, size=(BOOTSTRAP_REPLICATES, N_VALIDATION))
    bootstrap_metrics = {}
    for key, frame in frames.items():
        per_model = {}
        for target, short in (("SM_0", "SM0"), ("SM_20", "SM20")):
            truth = frame[f"{target}_true"].to_numpy(dtype=np.float64)
            prediction = frame[f"{target}_pred"].to_numpy(dtype=np.float64)
            absolute_error = np.abs(prediction - truth)
            squared_error = (prediction - truth) ** 2
            per_model[f"{short} RMSE"] = np.sqrt(np.mean(squared_error[common_indices], axis=1))
            per_model[f"{short} MAE"] = np.mean(absolute_error[common_indices], axis=1)
        per_model["mean RMSE"] = (per_model["SM0 RMSE"] + per_model["SM20 RMSE"]) / 2
        per_model["mean MAE"] = (per_model["SM0 MAE"] + per_model["SM20 MAE"]) / 2
        bootstrap_metrics[key] = per_model

    EFFECTS = {
        "SoilNet SSL effect (P0 - P1_noSSL)": ("P0", "P1_noSSL"),
        "MobileViTv2 SSL effect (P3 - P2)": ("P3", "P2"),
    }
    METRIC_ORDER = ["SM0 RMSE", "SM20 RMSE", "mean RMSE", "SM0 MAE", "SM20 MAE", "mean MAE"]
    effect_replicates = {}
    for effect, (left, right) in EFFECTS.items():
        effect_replicates[effect] = {
            metric: bootstrap_metrics[left][metric] - bootstrap_metrics[right][metric]
            for metric in METRIC_ORDER
        }
    interaction_name = "Architecture × SSL interaction: (P3 - P2) - (P0 - P1_noSSL)"
    effect_replicates[interaction_name] = {
        metric: (
            effect_replicates["MobileViTv2 SSL effect (P3 - P2)"][metric]
            - effect_replicates["SoilNet SSL effect (P0 - P1_noSSL)"][metric]
        )
        for metric in METRIC_ORDER
    }

    def full_sample_effect(effect: str, metric: str) -> float:
        if effect == interaction_name:
            return (
                point_metrics["P3"][metric] - point_metrics["P2"][metric]
                - (point_metrics["P0"][metric] - point_metrics["P1_noSSL"][metric])
            )
        left, right = EFFECTS[effect]
        return point_metrics[left][metric] - point_metrics[right][metric]

    bootstrap_rows = []
    for effect in (*EFFECTS, interaction_name):
        for metric in METRIC_ORDER:
            values = effect_replicates[effect][metric]
            low, high = np.percentile(values, [2.5, 97.5])
            bootstrap_rows.append({
                "effect": effect,
                "metric": metric,
                "point estimate": full_sample_effect(effect, metric),
                "bootstrap mean": float(values.mean()),
                "95% CI lower": float(low),
                "95% CI upper": float(high),
                "proportion > 0": float(np.mean(values > 0)),
                "proportion < 0": float(np.mean(values < 0)),
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
            })
    bootstrap_summary = pd.DataFrame(bootstrap_rows)
    display(bootstrap_summary)
    """),
    markdown("## 5. Factorial mean-RMSE table and sign-aware validation conclusion"),
    code("""
    def summary_row(effect: str, metric: str = "mean RMSE"):
        return bootstrap_summary.loc[
            (bootstrap_summary["effect"] == effect) & (bootstrap_summary["metric"] == metric)
        ].iloc[0]

    soil_effect = summary_row("SoilNet SSL effect (P0 - P1_noSSL)")
    mobile_effect = summary_row("MobileViTv2 SSL effect (P3 - P2)")
    interaction_effect = summary_row(interaction_name)
    factorial_table = pd.DataFrame([
        {
            "Architecture": "SoilNet",
            "ImageNet-only": point_metrics["P1_noSSL"]["mean RMSE"],
            "+VICReg": point_metrics["P0"]["mean RMSE"],
            "VICReg effect": soil_effect["point estimate"],
            "95% CI": f"[{soil_effect['95% CI lower']:.6f}, {soil_effect['95% CI upper']:.6f}]",
        },
        {
            "Architecture": "MobileViTv2",
            "ImageNet-only": point_metrics["P2"]["mean RMSE"],
            "+VICReg": point_metrics["P3"]["mean RMSE"],
            "VICReg effect": mobile_effect["point estimate"],
            "95% CI": f"[{mobile_effect['95% CI lower']:.6f}, {mobile_effect['95% CI upper']:.6f}]",
        },
    ])
    interaction_low = float(interaction_effect["95% CI lower"])
    interaction_high = float(interaction_effect["95% CI upper"])
    if interaction_low <= 0 <= interaction_high:
        conclusion = (
            "The 95% paired-bootstrap CI for the architecture-by-SSL interaction includes 0; "
            "these validation data do not establish an architecture-dependent VICReg effect."
        )
    elif float(interaction_effect["point estimate"]) > 0:
        conclusion = (
            "Validation evidence supports an architecture-dependent VICReg effect: the positive "
            "interaction means the VICReg error change is less favorable for MobileViTv2 than for SoilNet."
        )
    else:
        conclusion = (
            "Validation evidence supports an architecture-dependent VICReg effect: the negative "
            "interaction means the VICReg error change is more favorable for MobileViTv2 than for SoilNet."
        )
    display(factorial_table)
    print(conclusion)
    """),
    markdown("## 6. Write isolated analysis artifacts and verify sources remained unchanged"),
    code("""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_DIR / "four_model_validation_metrics.csv"
    bootstrap_path = OUTPUT_DIR / "vicreg_paired_bootstrap.csv"
    interaction_path = OUTPUT_DIR / "vicreg_interaction_bootstrap.csv"
    summary_path = OUTPUT_DIR / "factorial_validation_summary.json"

    exact_metrics.to_csv(metrics_path, index=False, float_format="%.12g")
    bootstrap_summary.to_csv(bootstrap_path, index=False, float_format="%.12g")
    bootstrap_summary.loc[bootstrap_summary["effect"] == interaction_name].to_csv(
        interaction_path, index=False, float_format="%.12g"
    )

    source_hashes_after = {}
    for key, spec in EXPERIMENTS.items():
        directory = spec["artifact_dir"]
        metadata = metadata_by_model[key]
        source_hashes_after[key] = {
            "predictions": sha256_file(directory / "validation_predictions.csv"),
            "metrics": sha256_file(directory / "validation_metrics.json"),
            "metadata": sha256_file(directory / "run_metadata.json"),
            "checkpoint_declared_sha256": metadata["primary_checkpoint_sha256"],
        }
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("STOP: a frozen source artifact changed during analysis")

    notebook_sha256 = sha256_file(NOTEBOOK_PATH)
    artifact_rows = [
        {"path": str(path.relative_to(REPO)), "sha256": sha256_file(path)}
        for path in (metrics_path, bootstrap_path, interaction_path)
    ]
    artifact_rows.append({"path": str(NOTEBOOK_PATH.relative_to(REPO)), "sha256": notebook_sha256})
    artifact_rows.append({
        "path": str(GENERATOR_PATH.relative_to(REPO)), "sha256": sha256_file(GENERATOR_PATH)
    })
    final_summary = {
        "title": "FINAL FACTORIAL VALIDATION SUMMARY",
        "scope": "VALIDATION_ONLY",
        "validation_sample_count": N_VALIDATION,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "difference_convention": "difference < 0 means the model on the left has lower error",
        "interaction_formula": "(P3 - P2) - (P0 - P1_noSSL)",
        "interaction_sign": {
            "negative": "VICReg error change is more favorable for MobileViTv2 than SoilNet",
            "positive": "VICReg error change is less favorable for MobileViTv2 than SoilNet",
        },
        "exact_mean_validation_RMSE": {
            key: point_metrics[key]["mean RMSE"] for key in ("P0", "P1_noSSL", "P2", "P3")
        },
        "mean_RMSE_effects": {
            "soilnet_vicreg": soil_effect.to_dict(),
            "mobilevitv2_vicreg": mobile_effect.to_dict(),
            "architecture_x_ssl_interaction": interaction_effect.to_dict(),
        },
        "conclusion": conclusion,
        "validation_alignment": alignment_report,
        "identity_verification": identity_rows,
        "source_artifact_sha256": source_hashes_before,
        "source_artifacts_unchanged": True,
        "TEST_SET_OPENED": TEST_SET_OPENED,
        "notebook_sha256": notebook_sha256,
        "new_artifacts": artifact_rows + [
            {
                "path": str(summary_path.relative_to(REPO)),
                "sha256": None,
                "note": "Self-hash is intentionally omitted from the JSON itself; verify externally.",
            }
        ],
    }
    summary_path.write_text(json.dumps(final_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print({"OUTPUT_DIR": str(OUTPUT_DIR), "SOURCE_ARTIFACTS_UNCHANGED": True})
    """),
    markdown("## 7. Final factorial validation summary"),
    code("""
    print("FINAL FACTORIAL VALIDATION SUMMARY")
    print(json.dumps({
        "exact_mean_validation_RMSE": final_summary["exact_mean_validation_RMSE"],
        "SoilNet_VICReg_effect_mean_RMSE": final_summary["mean_RMSE_effects"]["soilnet_vicreg"],
        "MobileViTv2_VICReg_effect_mean_RMSE": final_summary["mean_RMSE_effects"]["mobilevitv2_vicreg"],
        "architecture_x_SSL_interaction_mean_RMSE": final_summary["mean_RMSE_effects"]["architecture_x_ssl_interaction"],
        "conclusion": conclusion,
        "TEST_SET_OPENED": TEST_SET_OPENED,
        "new_artifacts": final_summary["new_artifacts"],
        "notebook_sha256": notebook_sha256,
    }, indent=2, ensure_ascii=False))
    """),
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(book, OUTPUT)
print(OUTPUT)
