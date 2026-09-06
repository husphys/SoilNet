#!/usr/bin/env python3
"""Generate only the four final sequential notebooks; never execute them."""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "notebooks/final_experiments"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


def book(cells):
    value = nbf.v4.new_notebook(cells=cells)
    value.metadata["kernelspec"] = {"display_name": "Python (soilnet)", "language": "python", "name": "soilnet"}
    value.metadata["language_info"] = {"name": "python", "version": "3.11"}
    return value


def boot() -> str:
    return dedent("""
    from pathlib import Path
    import sys

    REPO = Path.cwd().resolve()
    while REPO != REPO.parent and not (REPO / "pyproject.toml").is_file():
        REPO = REPO.parent
    if not (REPO / "pyproject.toml").is_file():
        raise RuntimeError("Open this notebook from inside the SoilNet repository")
    sys.path.insert(0, str(REPO / "src"))
    """).strip() + "\n"


def notebook_09():
    return book([
        md("""
        # P2 MobileViTv2 ImageNet + LI architecture control

        **Scientific question:** What is the architecture contribution relative to P1-noSSL when initialization family, LI fusion, targets, loss, split, seed, and supervised protocol are fixed?
        **Configuration:** `config/experiments/P2_mobilevitv2_imagenet_li_bestreg.yaml`
        **Dataset:** 1,927 clean unique labeled images.
        **Split:** 1,407 train / 289 validation / 231 sealed test; SHA256 `8927b822...3eae2f`.
        **Checkpoint/initialization:** `timm/mobilevitv2_050.cvnets_in1k` ImageNet weights; no VICReg.
        **Expected outputs:** external P2 best/epoch-60 checkpoints, history, validation artifacts, metadata, then the four-model frozen registry.

        Run after notebooks 01–03 have completed. This notebook trains for all 60 epochs and never creates a test dataset or loader.
        """),
        md("## 1. Deterministic environment setup\n\nSet the CUDA deterministic workspace before importing PyTorch, then load only train/validation workflow functions."),
        code(dedent("""
        import os
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        """) + boot() + dedent("""
        import json
        import torch
        from soilnet.final_sequence import verify_p2_fairness, write_frozen_model_registry
        from soilnet.training import run_one_batch_preflight, train_experiment
        from soilnet.utils import load_experiment_context, print_environment

        CONFIG_PATH = REPO / "config/experiments/P2_mobilevitv2_imagenet_li_bestreg.yaml"
        """)),
        md("## 2. Locked hashes and fair-control protocol\n\nValidate the exact split, seed, optimizer, transformations, LI semantics, BESTREG rule, and the intended architecture-only delta against P1-noSSL."),
        code("""
        fairness = verify_p2_fairness()
        context = load_experiment_context(CONFIG_PATH)
        assert context.split_sha256 == "8927b8223b8c4c234d264ad9ea62ac2df6161124a79787a2e71eeb5cd23eae2f"
        assert context.config["timm_model_name"] == "mobilevitv2_050.cvnets_in1k"
        assert context.config["initialization"] == "imagenet" and context.config["ssl_checkpoint"] is None
        assert context.config["use_li"] is True
        assert context.config["selection_formula"] == "(SM0_RMSE + SM20_RMSE) / 2"
        print_environment(context)
        print(fairness)
        """),
        md("## 3. CUDA gate and new-output safety\n\nThe external P2 destination must be absent or contain only a valid rolling resume. A completed run is never overwritten."),
        code("""
        if not torch.cuda.is_available():
            raise RuntimeError("GPU_BLOCKED: P2 training requires CUDA")
        metadata_path = context.run_dir / "run_metadata.json"
        if metadata_path.is_file() and json.loads(metadata_path.read_text(encoding="utf-8")).get("training_completed") is True:
            raise RuntimeError("STOP: P2 output directory already contains a completed run")
        print({"device": torch.cuda.get_device_name(0), "output_directory": str(context.run_dir), "test": "NOT_OPENED"})
        """),
        md("## 4. Temporary-model preflight\n\nBuild a disposable ImageNet-initialized baseline and check one train/validation batch. No optimizer step, checkpoint, or research metric is produced."),
        code("""
        preflight = run_one_batch_preflight(context, device=torch.device("cuda"))
        if preflight.get("status") != "PASS" or preflight.get("optimizer_step_performed") is not False or preflight.get("test_loader_instantiated") is not False:
            raise RuntimeError("P2_PREFLIGHT_FAILED")
        print(preflight)
        """),
        md("## 5. Automatic 60-epoch supervised run\n\nTrain all 60 epochs with Adam 1e-4, batch 32, weight decay 0, and no early stopping. Strictly lower mean validation regression RMSE replaces the best checkpoint."),
        code("""
        run_metadata = train_experiment(context, resume_if_available=True)
        assert run_metadata["training_completed"] is True
        assert run_metadata["epochs_completed"] == 60
        assert run_metadata["test_evaluated"] == "NO"
        """),
        md("## 6. Freeze all four models\n\nVerify P0, both P1 ablations, and P2 checkpoint identities plus validation artifacts, then write the registry while the test firewall remains closed."),
        code("""
        registry = write_frozen_model_registry()
        assert registry["model_count"] == 4
        assert registry["TEST_OPENED"] == "NO" and registry["MODEL_SET_FROZEN"] == "YES"
        print(json.dumps({"registry": "results/model_registry/frozen_model_registry.json", "models": [m["experiment_id"] for m in registry["models"]], "TEST_OPENED": "NO"}, indent=2))
        """),
    ])


def notebook_10():
    return book([
        md("""
        # One-time final frozen test evaluation

        **Scientific question:** What are the final held-out metrics and paired test-set bootstrap intervals for the four frozen models?
        **Configuration:** the immutable four-model registry created by Notebook 09.
        **Dataset:** locked 1,927-image clean dataset.
        **Split:** the sole 231-sample held-out test, opened only after full preflight.
        **Checkpoint:** four validation-selected checkpoints with registry-locked SHA256 values.
        **Expected outputs:** per-sample predictions, final metrics/tables, paired bootstrap files, paper figures, and `FINAL_TEST_LOCK.json` under `results/final_test/`.

        This notebook performs no training, tuning, checkpoint selection, optimizer creation, backward pass, or `model.train()` call.
        """),
        md("## 1. Imports and machine-local data root\n\nResolve the configured raw-data root without creating any dataset or loader."),
        code(boot() + dedent("""
        import json
        from soilnet.final_sequence import preflight_frozen_registry, run_final_test_once
        from soilnet.io import resolve_paths
        """)),
        md("## 2. Complete frozen-registry preflight\n\nBefore test construction, verify exactly four immutable models, exact paths/hashes, split identity/count metadata, validation artifacts, and absence of a completed test lock."),
        code("""
        registry = preflight_frozen_registry()
        assert registry["model_count"] == 4 and registry["TEST_OPENED"] == "NO"
        print(json.dumps({"preflight": "PASS", "models": [m["experiment_id"] for m in registry["models"]], "test_loader": "NOT_CREATED_YET"}, indent=2))
        """),
        md("## 3. Evaluate once and lock\n\nOnly after the preceding preflight passes, create one shared test dataset/loader, evaluate all four frozen models under inference mode, persist predictions, bootstrap by paired sample index, plot from saved prediction CSVs, and write the final lock last."),
        code("""
        paths = resolve_paths()
        final_lock = run_final_test_once(paths["data_root"])
        assert final_lock["TEST_EVALUATED"] == "YES" and final_lock["test_n"] == 231
        print(json.dumps(final_lock, indent=2))
        """),
    ])


def notebook_11():
    return book([
        md("""
        # Raspberry Pi frozen-P0 deployment benchmark

        **Scientific question:** What is frozen P0 CPU inference latency on actual Raspberry Pi hardware?
        **Configuration:** primary all-logical-core setting plus a one-thread secondary setting.
        **Dataset/split:** no dataset and no test access; representative tensors have shapes `1x3x224x224` and `1x1`.
        **Checkpoint:** bundled frozen P0 deployment checkpoint, SHA256 `eba009df...761379`.
        **Expected outputs:** environment, JSON/CSV measurements, report, and paper-ready latency CSV under `results/edge/`.

        Run this notebook on the actual Raspberry Pi after Notebook 10. It refuses desktop/proxy execution with `RPI_HARDWARE_REQUIRED`.
        """),
        md("## 1. Hardware, benchmark, and automatic result publication\n\nVerify the exact publication remote, clean/safe Git state, identity and write authentication; synchronize `main`; require Raspberry Pi hardware; verify the one allowed checkpoint; benchmark CPU inference with 50 warmups and 200 timed iterations; then commit and push only the five allowlisted `results/edge` files."),
        code(boot() + dedent("""
        import json
        from soilnet.publication import prepare_raspberry_pi_publication, publish_raspberry_pi_results
        from soilnet.rpi_benchmark import run_raspberry_pi_benchmark

        publication_preflight = prepare_raspberry_pi_publication()
        benchmark = run_raspberry_pi_benchmark()
        assert benchmark["hardware_verified"] == "Raspberry Pi"
        assert benchmark["settings"][0]["warmup_iterations"] == 50
        assert benchmark["settings"][0]["timed_iterations"] == 200
        edge_publication = publish_raspberry_pi_results()
        assert edge_publication["push"] == "PASS"
        print(json.dumps({"benchmark": benchmark, "publication": edge_publication}, indent=2))
        """)),
    ])


def notebook_12():
    return book([
        md("""
        # Final paper artifacts and publication

        **Scientific question:** Which locked artifacts support each manuscript claim, and is the complete reproducibility snapshot safe to publish?
        **Configuration:** fixed publication target `https://github.com/husphys-gif/SoilNet.git`.
        **Dataset/split:** metadata/manifests only; no raw images, dataset loading, or model evaluation.
        **Checkpoint:** only the single frozen P0 deployment binary is allowlisted; all training collections remain external.
        **Expected outputs:** ten numbered paper CSVs, consolidated summary/docs, an audited Git commit, dry-run, and push to `publication/main`.

        Run last, after Notebook 11 outputs have been copied into this same checkout. This notebook performs no training and no model inference/evaluation.
        """),
        md("## 1. Safely retrieve Raspberry Pi evidence and validate prerequisites\n\nAllow only recognized local Notebook 09/10/12 and experiment-result changes, stash them, fetch and rebase from exact `publication/main`, restore them without overwriting, then require four completed frozen runs, the final-test lock/predictions/metrics/bootstrap/figures, and verified Raspberry Pi outputs."),
        code(boot() + dedent("""
        import json
        from soilnet.publication import (
            aggregate_paper_artifacts, publish_reproducibility_repository,
            retrieve_raspberry_pi_results, validate_publication_inputs,
        )

        edge_retrieval = retrieve_raspberry_pi_results()
        validated = validate_publication_inputs()
        print({"edge_retrieval": edge_retrieval["status"], "prerequisites": "PASS", "models": len(validated["registry"]["models"]), "test_locked": True, "rpi_verified": True})
        """)),
        md("## 2. Aggregate paper-ready evidence\n\nCreate deterministic tables and documentation while keeping validation, held-out test, historical evidence, deployment evidence, and irrigation proof-of-concept claims separate."),
        code("""
        paper_files = aggregate_paper_artifacts()
        print({"paper_artifacts": [str(path.relative_to(REPO)) for path in paper_files]})
        """),
        md("## 3. Audit, commit, dry-run, and publish\n\nUse an explicit file allowlist, verify size/raw-data/checkpoint/secret policies and Git identity, configure only the exact `publication` remote, test authentication without printing credentials, commit idempotently, dry-run, then push `main`. The legacy repository is never a target."),
        code("""
        publication = publish_reproducibility_repository()
        print(json.dumps(publication, indent=2))
        """),
    ])


def main() -> None:
    outputs = {
        "09_P2_mobilevitv2_imagenet_li_bestreg.ipynb": notebook_09(),
        "10_final_frozen_test_evaluation.ipynb": notebook_10(),
        "11_raspberry_pi_benchmark.ipynb": notebook_11(),
        "12_finalize_paper_artifacts_and_publish.ipynb": notebook_12(),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, notebook in outputs.items():
        nbf.write(notebook, OUTPUT / name)
        print(OUTPUT / name)


if __name__ == "__main__":
    main()
