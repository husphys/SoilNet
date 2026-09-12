#!/usr/bin/env python3
"""Static notebook/config firewall checks; never executes notebook cells."""
from __future__ import annotations

import ast
import argparse
import json
import re
import sys
from pathlib import Path

import nbformat
import timm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from soilnet.io import load_yaml, sha256_file, write_json


EXPECTED = [
    "00_gpu_environment_check.ipynb",
    "01_P0_final_soilnet_v4_bestreg.ipynb",
    "02_P1_no_li_bestreg.ipynb",
    "03_P1_no_ssl_bestreg.ipynb",
    "09_P2_mobilevitv2_imagenet_li_bestreg.ipynb",
    "10_final_frozen_test_evaluation.ipynb",
    "11_raspberry_pi_benchmark.ipynb",
    "12_finalize_paper_artifacts_and_publish.ipynb",
    "13_P3_mobilevitv2_vicreg_bestreg.ipynb",
    "14_vicreg_architecture_interaction_analysis.ipynb",
]
CONFIGS = [
    "P0_final_soilnet.yaml", "P0_final_soilnet_v4_bestreg.yaml",
    "P1_no_li.yaml", "P1_no_li_bestreg.yaml",
    "P1_no_ssl.yaml", "P1_no_ssl_bestreg.yaml",
    "P1_mobilenetv2.yaml", "P1_mobilevitv2.yaml", "P2_mobilenetv3.yaml",
    "P2_efficientnet_b0.yaml", "P2_classical_ml.yaml",
    "P2_mobilevitv2_imagenet_li_bestreg.yaml",
    "P3_mobilevitv2_vicreg_li_bestreg.yaml",
]
REQUIRED_CONFIG = {
    "experiment_id", "architecture", "initialization", "ssl_checkpoint", "use_li",
    "dataset_manifest", "split", "seed", "epochs", "batch_size", "optimizer", "learning_rate",
    "weight_decay", "loss", "metrics", "output_dir",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--read-only", action="store_true", help="print checks without rewriting the audit JSON")
    args = parser.parse_args()
    errors = []
    executed_cells = 0
    training_notebook_cells_executed = 0
    gpu_notebook_execution_status = "NOT_RUN"
    directory = REPO / "notebooks/final_experiments"
    observed = sorted(path.name for path in directory.glob("*.ipynb"))
    if observed != sorted(EXPECTED):
        errors.append(f"notebook set mismatch: {observed}")
    for filename in EXPECTED:
        path = directory / filename
        if not path.is_file():
            continue
        book = nbformat.read(path, as_version=4)
        try:
            nbformat.validate(book)
        except Exception as exc:
            errors.append(f"{filename}: nbformat invalid: {exc}")
        if not book.cells or book.cells[0].cell_type != "markdown":
            errors.append(f"{filename}: first cell is not Markdown")
        else:
            header = book.cells[0].source.casefold()
            for term in ("scientific question", "configuration", "dataset", "split", "expected outputs"):
                if term not in header:
                    errors.append(f"{filename}: first Markdown missing {term}")
            if "checkpoint" not in header and "initialization" not in header:
                errors.append(f"{filename}: first Markdown missing checkpoint/initialization")
        source = "\n".join(cell.source for cell in book.cells if cell.cell_type == "code")
        if re.search(r"[A-Za-z]:\\\\", source):
            errors.append(f"{filename}: hard-coded Windows path")
        for index, cell in enumerate(book.cells):
            if cell.cell_type == "code":
                try:
                    ast.parse(cell.source)
                except SyntaxError as exc:
                    errors.append(f"{filename}: code cell {index} syntax error: {exc}")
                if cell.get("outputs") or cell.get("execution_count") is not None:
                    executed_cells += 1
                    if filename != "00_gpu_environment_check.ipynb":
                        training_notebook_cells_executed += 1
                    if any(output.get("output_type") == "error" for output in cell.get("outputs", [])):
                        errors.append(f"{filename}: saved execution error found")
                    # Completed P0/P1 notebooks may retain successful execution
                    # evidence. Error outputs remain prohibited above.
        if filename == "00_gpu_environment_check.ipynb" and any(
            cell.get("execution_count") is not None for cell in book.cells if cell.cell_type == "code"
        ):
            output_text = "\n".join(
                "".join(output.get("text", []))
                for cell in book.cells if cell.cell_type == "code"
                for output in cell.get("outputs", []) if output.get("output_type") == "stream"
            )
            output_errors = [
                output for cell in book.cells if cell.cell_type == "code"
                for output in cell.get("outputs", []) if output.get("output_type") == "error"
            ]
            if output_errors:
                errors.append("GPU notebook saved an execution error")
            required_gpu_evidence = (
                "cuda_available: True", "gpu: NVIDIA GeForce RTX 3050",
                "'GPU_SMOKE': 'PASS'", "'research_metrics': False", "'test_evaluated': 'NO'",
            )
            if all(marker in output_text for marker in required_gpu_evidence):
                gpu_notebook_execution_status = "PASS"
            else:
                errors.append("GPU notebook saved output is incomplete or not a safe PASS")
        if filename[0:2].isdigit() and filename[0:2] in {f"{value:02d}" for value in range(1, 9)}:
            automatic_p1 = filename in {"02_P1_no_li_bestreg.ipynb", "03_P1_no_ssl_bestreg.ipynb"}
            if automatic_p1 and "RUN_TRAINING = True" not in source:
                errors.append(f"{filename}: missing automatic Run-All training flag")
            if not automatic_p1 and not filename.startswith("01_P0_") and "RUN_TRAINING = False" not in source:
                errors.append(f"{filename}: missing disabled training flag")
            if 'split="test"' in source or "build_test_loader" in source:
                errors.append(f"{filename}: training/model-selection notebook references a test loader")
            if '"test_evaluated": "NO"' not in source:
                errors.append(f"{filename}: Experiment Summary does not state test=NO")
        if filename == "01_P0_final_soilnet.ipynb":
            markdown_source = "\n".join(cell.source for cell in book.cells if cell.cell_type == "markdown")
            for section in range(1, 18):
                if f"## {section}." not in markdown_source:
                    errors.append(f"P0 notebook missing manual-run section {section}")
            required_p0 = [
                "RUN_PREFLIGHT = True", "RESUME_IF_AVAILABLE = True",
                "epoch_60_final.pth", "optimizer_step_performed", "test_loader\": \"NOT_CREATED",
            ]
            for marker in required_p0:
                if marker not in source:
                    errors.append(f"P0 notebook missing safety marker: {marker}")
            if "optimizer.step(" in source:
                errors.append("P0 notebook preflight contains an optimizer step")
        if filename == "09_final_test_evaluation.ipynb":
            if "RUN_FINAL_TEST = False" not in source:
                errors.append("final test notebook lacks disabled final-test flag")
            if "SELECTED_DEEP_CONFIGS = []" not in source:
                errors.append("final test notebook does not require explicit model selection")
        if filename == "09_P2_mobilevitv2_imagenet_li_bestreg.ipynb":
            required = (
                "mobilevitv2_050.cvnets_in1k", "run_one_batch_preflight", "train_experiment",
                "write_frozen_model_registry", "TEST_OPENED", "resume_if_available=True",
            )
            if any(marker not in source for marker in required):
                errors.append("final P2 notebook is missing an automatic protocol gate")
            if 'split="test"' in source or "build_test_loader" in source:
                errors.append("final P2 notebook references test construction")
        if filename == "10_final_frozen_test_evaluation.ipynb":
            if source.index("registry = preflight_frozen_registry()") > source.index("final_lock = run_final_test_once"):
                errors.append("final test preflight is not visibly ordered before evaluation")
            for forbidden in ("train_experiment", "build_optimizer", ".backward(", ".train()"):
                if forbidden in source:
                    errors.append(f"final test notebook contains prohibited training token: {forbidden}")
        if filename == "11_raspberry_pi_benchmark.ipynb":
            if "run_raspberry_pi_benchmark" not in source or "RPI_HARDWARE_REQUIRED" not in path.read_text(encoding="utf-8"):
                errors.append("Raspberry Pi notebook lacks hardware gate")
        if filename == "12_finalize_paper_artifacts_and_publish.ipynb":
            for forbidden in ("train_experiment", "run_final_test_once", "DataLoader", "model("):
                if forbidden in source:
                    errors.append(f"publication notebook contains training/evaluation token: {forbidden}")
            if "publish_reproducibility_repository" not in source:
                errors.append("publication notebook lacks guarded publication entrypoint")
        if filename == "13_P3_mobilevitv2_vicreg_bestreg.ipynb":
            if 'split="test"' in source or "build_test_loader" in source or "evaluate_final_test" in source:
                errors.append("P3 notebook references prohibited test construction/evaluation")
            if "TEST_LOADER_INSTANTIATED" not in source or "train_experiment" not in source:
                errors.append("P3 notebook lacks frozen test firewall or supervised workflow")
        if filename == "14_vicreg_architecture_interaction_analysis.ipynb":
            for forbidden in ("SoilNetDataset", "DataLoader", "train_experiment", 'split="test"'):
                if forbidden in source:
                    errors.append(f"factorial notebook contains prohibited token: {forbidden}")
            if "TEST_SET_OPENED = False" not in source or 'REPO / "results/frozen"' not in source:
                errors.append("factorial notebook lacks public frozen-input/test-firewall contract")
    available = set(timm.list_models())
    for filename in CONFIGS:
        config = load_yaml(REPO / "config/experiments" / filename)
        missing = sorted(REQUIRED_CONFIG - set(config))
        if missing:
            errors.append(f"{filename}: missing config keys {missing}")
        if config.get("dataset_manifest_sha256") != "8ff45054d4b8e3df9758d0c112dc16719572b2267906fb3a5ed5b3262a6732bd":
            errors.append(f"{filename}: manifest hash mismatch")
        if config.get("split_sha256") != "8927b8223b8c4c234d264ad9ea62ac2df6161124a79787a2e71eeb5cd23eae2f":
            errors.append(f"{filename}: split hash mismatch")
        model_name = config.get("timm_model_name")
        if model_name and model_name not in available and model_name.split(".", 1)[0] not in available:
            errors.append(f"{filename}: timm model unavailable: {model_name}")
        if config.get("experiment_kind") != "classical_ml":
            if config.get("experiment_id") in {
                "P0_FINAL_SOILNET_VICREG_MU27_LI", "P0_FINAL_SOILNET_VICREG_MU27_LI_V4_BESTREG",
                "P1_ABLATION_NO_LI", "P1_ABLATION_NO_SSL",
                "P1_SOILNET_VICREG_MU27_NO_LI_BESTREG",
                "P1_SOILNET_IMAGENET_LI_NO_SSL_BESTREG",
                "P2_MOBILEVITV2_IMAGENET_LI_BESTREG",
                "P3_MOBILEVITV2_VICREG_LI_BESTREG",
            }:
                expected = {"epochs": 60, "batch_size": 32, "optimizer": "Adam", "learning_rate": 0.0001, "weight_decay": 0.0}
            else:
                expected = {"epochs": 15, "batch_size": 32, "optimizer": "Adam", "learning_rate": 0.0005, "weight_decay": 0.0}
            for key, value in expected.items():
                if config.get(key) != value:
                    errors.append(f"{filename}: locked {key} mismatch")
    p0 = load_yaml(REPO / "config/experiments/P0_final_soilnet.yaml")
    if p0.get("protocol_status") != "PROTOCOL_REVISED_PRE_TEST" or p0.get("primary_checkpoint") != "epoch_60_final.pth":
        errors.append("P0 v3 status or primary checkpoint mismatch")
    p0_v4 = load_yaml(REPO / "config/experiments/P0_final_soilnet_v4_bestreg.yaml")
    if (
        p0_v4.get("protocol_version") != "P0-v4-bestreg"
        or p0_v4.get("primary_checkpoint") != "validation_best_regression.pth"
        or p0_v4.get("checkpoint_selection") != "minimum_mean_validation_RMSE"
    ):
        errors.append("P0 v4 best-regression checkpoint policy mismatch")
    for name in ("P1_no_li.yaml", "P1_no_ssl.yaml"):
        if load_yaml(REPO / "config/experiments" / name).get("protocol_status") != "OPTIONAL_NOT_YET_REQUIRED":
            errors.append(f"{name}: optional P1 status mismatch")
    if sha256_file(REPO / "data/manifests/final_clean_manifest_v1.csv") != "8ff45054d4b8e3df9758d0c112dc16719572b2267906fb3a5ed5b3262a6732bd":
        errors.append("live manifest hash mismatch")
    if sha256_file(REPO / "data/splits/final_clean_split_v1.csv") != "8927b8223b8c4c234d264ad9ea62ac2df6161124a79787a2e71eeb5cd23eae2f":
        errors.append("live split hash mismatch")
    if (REPO / "results/final/test_evaluation_completed.json").exists():
        errors.append("final-test marker exists before notebook training")
    report = {
        "status": "PASS" if not errors else "FAIL", "errors": errors,
        "notebooks_checked": len(EXPECTED), "configs_checked": len(CONFIGS),
        "notebook_cells_executed": executed_cells,
        "gpu_diagnostic_cells_executed": executed_cells - training_notebook_cells_executed,
        "training_notebook_cells_executed": training_notebook_cells_executed,
        "gpu_notebook_execution_status": gpu_notebook_execution_status,
        "test_loader_instantiated": False,
    }
    if not args.read_only:
        write_json(REPO / "results/audit/notebook_static_check.json", report)
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
