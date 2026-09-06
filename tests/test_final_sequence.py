import ast
import json
from pathlib import Path

import nbformat
import numpy as np
import pytest

from soilnet.final_sequence import (
    BOOTSTRAP_SEED,
    MODEL_SPECS,
    N_BOOTSTRAP,
    PUBLICATION_URL,
    SPLIT_COUNTS,
    SPLIT_SHA256,
    preflight_frozen_registry,
    verify_p2_fairness,
)
from soilnet.io import load_yaml, sha256_file
from soilnet.rpi_benchmark import P0_SHA256, require_raspberry_pi_hardware


REPO = Path(__file__).resolve().parents[1]
BOOKS = REPO / "notebooks/final_experiments"


def _source(name: str) -> str:
    book = nbformat.read(BOOKS / name, as_version=4)
    nbformat.validate(book)
    for cell in book.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            assert cell.execution_count is None and not cell.outputs
    return "\n".join(cell.source for cell in book.cells if cell.cell_type == "code")


def test_final_sequence_notebooks_are_unexecuted_and_ordered():
    names = (
        "09_P2_mobilevitv2_imagenet_li_bestreg.ipynb",
        "10_final_frozen_test_evaluation.ipynb",
        "11_raspberry_pi_benchmark.ipynb",
        "12_finalize_paper_artifacts_and_publish.ipynb",
    )
    for name in names:
        assert (BOOKS / name).is_file()
        _source(name)
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert all(readme.index(name) < readme.index(names[index + 1]) for index, name in enumerate(names[:-1]))


def test_p2_is_exact_fair_bestreg_control_and_has_no_test_access():
    result = verify_p2_fairness()
    assert result["status"] == "PASS" and result["intended_delta"] == "architecture"
    config = load_yaml(REPO / "config/experiments/P2_mobilevitv2_imagenet_li_bestreg.yaml")
    assert config["timm_model_name"] == "mobilevitv2_050.cvnets_in1k"
    assert config["initialization"] == "imagenet" and config["ssl_checkpoint"] is None
    assert config["use_li"] is True
    assert config["epochs"] == 60 and config["batch_size"] == 32
    assert config["optimizer"] == "Adam" and config["learning_rate"] == 1e-4 and config["weight_decay"] == 0
    assert config["seed"] == 20260905 and config["split_sha256"] == SPLIT_SHA256
    assert config["checkpoint_selection"] == "minimum_mean_validation_RMSE"
    assert config["selection_formula"] == "(SM0_RMSE + SM20_RMSE) / 2"
    source = _source("09_P2_mobilevitv2_imagenet_li_bestreg.ipynb")
    assert "train_experiment(context, resume_if_available=True)" in source
    assert "run_one_batch_preflight" in source
    assert 'split="test"' not in source and "build_test_loader" not in source


def test_locked_frozen_checkpoint_identities_are_exact():
    assert MODEL_SPECS["P0"]["checkpoint_sha256"] == "eba009dfd45ec21174a8e40b16148e0455e902286c7db55933487221da761379"
    assert MODEL_SPECS["P1_noLI"]["checkpoint_sha256"] == "a9d8de995b0673e9ec39bfc4afac00d6a2a773ad9ffbea0027ff2d0c4fab820c"
    assert MODEL_SPECS["P1_noSSL"]["checkpoint_sha256"] == "50c0f15567ab71a87b04569790b9cc7afd016989e3899d56e1898d8a6fcfb044"


def test_final_test_preflight_precedes_single_dataset_and_no_training():
    notebook_source = _source("10_final_frozen_test_evaluation.ipynb")
    assert notebook_source.index("registry = preflight_frozen_registry()") < notebook_source.index("final_lock = run_final_test_once")
    for token in ("train_experiment", "build_optimizer", ".backward(", ".train()"):
        assert token not in notebook_source
    module = (REPO / "src/soilnet/final_sequence.py").read_text(encoding="utf-8")
    function_start = module.index("def run_final_test_once")
    function_source = module[function_start:]
    assert function_source.index("preflight_frozen_registry()") < function_source.index("SoilNetDataset(")
    assert function_source.count("SoilNetDataset(") == 1
    assert function_source.count("DataLoader(") == 1
    assert "torch.inference_mode()" in function_source
    assert "optimizer" not in function_source.casefold()


def test_bootstrap_is_fixed_paired_by_shared_indices_and_has_10000_iterations():
    assert BOOTSTRAP_SEED == 20260906 and N_BOOTSTRAP == 10000
    source = (REPO / "src/soilnet/final_sequence.py").read_text(encoding="utf-8")
    assert "sample_indices = rng.integers" in source
    assert "error_a[sample_indices]" in source and "error_b[sample_indices]" in source
    assert "np.array_equal(reference[\"sample_id\"]" in source


def test_figures_are_built_from_saved_prediction_csvs():
    source = (REPO / "src/soilnet/final_sequence.py").read_text(encoding="utf-8")
    figure_start = source.index("def generate_final_test_figures")
    figure_end = source.index("def run_final_test_once")
    figure_source = source[figure_start:figure_end]
    assert "_read_prediction_csv" in figure_source
    assert "prediction_file" in figure_source


def test_rpi_gate_and_single_bundled_checkpoint():
    with pytest.raises(RuntimeError, match="RPI_HARDWARE_REQUIRED"):
        require_raspberry_pi_hardware({"device_tree_model": "desktop", "cpu_model": "x86"})
    checkpoint = REPO / "checkpoints/deployment/P0_FINAL_SOILNET_VICREG_MU27_LI_V4_BESTREG.pth"
    assert checkpoint.is_file() and sha256_file(checkpoint) == P0_SHA256
    assert [path for path in (REPO / "checkpoints").rglob("*.pth")] == [checkpoint]
    source = _source("11_raspberry_pi_benchmark.ipynb")
    assert "run_raspberry_pi_benchmark" in source
    assert source.index("prepare_raspberry_pi_publication()") < source.index("run_raspberry_pi_benchmark()")
    assert source.index("run_raspberry_pi_benchmark()") < source.index("publish_raspberry_pi_results()")


def test_publication_target_and_legacy_firewall_are_literal_and_no_blanket_add():
    source = (REPO / "src/soilnet/publication.py").read_text(encoding="utf-8")
    assert PUBLICATION_URL == "https://github.com/husphys-gif/SoilNet.git"
    assert "https://github.com/diy-hus/SoilNet" in (REPO / "src/soilnet/final_sequence.py").read_text(encoding="utf-8")
    assert '_git("add", "--", *relative_files' in source
    assert 'git add .' not in source and 'git add -A' not in source
    notebook_source = _source("12_finalize_paper_artifacts_and_publish.ipynb")
    for token in ("train_experiment", "run_final_test_once", "DataLoader"):
        assert token not in notebook_source
    assert notebook_source.index("retrieve_raspberry_pi_results()") < notebook_source.index("validate_publication_inputs()")
    publication = (REPO / "src/soilnet/publication.py").read_text(encoding="utf-8")
    assert "EDGE_RESULT_FILES" in publication
    assert '_git("add", "--", *EDGE_RESULT_FILES)' in publication
    assert "_sync_preserving_allowed_changes" in publication


def test_registry_preflight_checks_before_any_test_construction():
    source = (REPO / "src/soilnet/final_sequence.py").read_text(encoding="utf-8")
    start = source.index("def preflight_frozen_registry")
    end = source.index("def _metric_summary")
    preflight_source = source[start:end]
    assert "SoilNetDataset" not in preflight_source and "DataLoader" not in preflight_source
    assert "FINAL_TEST_ALREADY_COMPLETED" in preflight_source
