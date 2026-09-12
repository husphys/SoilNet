import ast
import json
from pathlib import Path

import nbformat
import pytest
import torch

from soilnet.io import load_yaml, sha256_file
from soilnet.models import ExperimentSoilNet, build_model
from soilnet.evaluation.final_test import evaluate_all_frozen_models_once
from soilnet.training.engine import (
    _regression_selection_metrics,
    _save_checkpoint,
    _uses_best_regression_selection,
    _validate_resume_payload,
)
from soilnet.utils import load_experiment_context


REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO / "notebooks/final_experiments"


def test_final_notebooks_are_valid_and_saved_outputs_have_no_errors():
    paths = sorted(NOTEBOOKS.glob("*.ipynb"))
    assert len(paths) == 10
    for path in paths:
        book = nbformat.read(path, as_version=4)
        nbformat.validate(book)
        assert book.cells[0].cell_type == "markdown"
        for cell in book.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
                if cell.outputs:
                    assert all(output.output_type != "error" for output in cell.outputs)


def test_training_notebooks_cannot_construct_test_loader():
    paths = [
        path for path in sorted(NOTEBOOKS.glob("*.ipynb"))
        if path.name[:2] in {f"{index:02d}" for index in range(1, 9)}
    ]
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert "build_test_loader" not in source
        assert 'split=\\"test\\"' not in source
        if path.name in {"02_P1_no_li_bestreg.ipynb", "03_P1_no_ssl_bestreg.ipynb"}:
            assert "RUN_TRAINING = True" in source
        elif not path.name.startswith("01_P0_"):
            assert "RUN_TRAINING = False" in source
    final_source = (NOTEBOOKS / "10_final_frozen_test_evaluation.ipynb").read_text(encoding="utf-8")
    assert "preflight_frozen_registry" in final_source
    assert "run_final_test_once" in final_source


def test_deep_configs_keep_their_predeclared_protocol_and_hashes():
    for path in sorted((REPO / "config/experiments").glob("P[012]_*.yaml")):
        config = load_yaml(path)
        if config["experiment_kind"] == "classical_ml" or config["protocol_status"] == "SKIPPED_VARIANT_UNRESOLVED":
            continue
        if config["experiment_id"] in {
            "P0_FINAL_SOILNET_VICREG_MU27_LI", "P0_FINAL_SOILNET_VICREG_MU27_LI_V4_BESTREG",
            "P1_ABLATION_NO_LI", "P1_ABLATION_NO_SSL",
            "P1_SOILNET_VICREG_MU27_NO_LI_BESTREG",
            "P1_SOILNET_IMAGENET_LI_NO_SSL_BESTREG",
            "P2_MOBILEVITV2_IMAGENET_LI_BESTREG",
        }:
            assert config["epochs"] == 60
            assert config["learning_rate"] == 0.0001
            expected_primary = (
                "validation_best_regression.pth"
                if config.get("save_validation_best") is True
                else "epoch_60_final.pth"
            )
            assert config["primary_checkpoint"] == expected_primary
        else:
            assert config["epochs"] == 15
            assert config["learning_rate"] == 0.0005
        assert config["batch_size"] == 32
        assert config["optimizer"] == "Adam"
        assert config["weight_decay"] == 0.0
        assert config["use_amp"] is False
        assert config["test_evaluation"] == "prohibited_in_training_notebook"
        assert config["dataset_manifest_sha256"] == sha256_file(REPO / config["dataset_manifest"])
        assert config["split_sha256"] == sha256_file(REPO / config["split"])


def test_mobilenetv3_is_explicitly_blocked_not_silently_selected():
    config = load_yaml(REPO / "config/experiments/P2_mobilenetv3.yaml")
    assert config["protocol_status"] == "SKIPPED_VARIANT_UNRESOLVED"
    assert config["timm_model_name"] is None
    with pytest.raises(RuntimeError, match="blocked"):
        load_experiment_context(REPO / "config/experiments/P2_mobilenetv3.yaml")


def test_no_li_model_preserves_dual_heads_and_checkpoint_shapes():
    p0 = ExperimentSoilNet(num_classes=10, use_li_signal=True, backbone_pretrained=False)
    no_li = ExperimentSoilNet(num_classes=10, use_li_signal=False, backbone_pretrained=False)
    assert p0.state_dict().keys() == no_li.state_dict().keys()
    assert all(p0.state_dict()[key].shape == no_li.state_dict()[key].shape for key in p0.state_dict())
    assert no_li.reg_head[-1].out_features == 2
    assert no_li.cls_head[-1].out_features == 10


def test_checkpoint_policy_is_bounded_and_test_marker_absent():
    source = (REPO / "src/soilnet/training/engine.py").read_text(encoding="utf-8")
    assert "last_resume_checkpoint.pth" in source
    assert 'context.config["primary_checkpoint"]' in source
    assert "ssl_initialization_sha256" in source
    assert "resume_if_available" in source
    assert "len(checkpoint_paths) > 2" in source
    assert not (REPO / "results/final/test_evaluation_completed.json").exists()
    bundled = sorted(path.relative_to(REPO).as_posix() for path in REPO.rglob("*.pth") if path.is_file())
    assert bundled == sorted([
        "checkpoints/deployment/P0_FINAL_SOILNET_VICREG_MU27_LI_V4_BESTREG.pth",
        "checkpoints/deployment/P1_SOILNET_VICREG_MU27_NO_LI_BESTREG.pth",
    ])
    assert sha256_file(REPO / bundled[0]) in {
        "eba009dfd45ec21174a8e40b16148e0455e902286c7db55933487221da761379",
        "a9d8de995b0673e9ec39bfc4afac00d6a2a773ad9ffbea0027ff2d0c4fab820c",
    }
    assert sha256_file(REPO / bundled[1]) in {
        "eba009dfd45ec21174a8e40b16148e0455e902286c7db55933487221da761379",
        "a9d8de995b0673e9ec39bfc4afac00d6a2a773ad9ffbea0027ff2d0c4fab820c",
    }


def test_p0_v4_notebook_preflight_does_not_step_final_weights():
    book = nbformat.read(NOTEBOOKS / "01_P0_final_soilnet_v4_bestreg.ipynb", as_version=4)
    source = "\n".join(cell.source for cell in book.cells if cell.cell_type == "code")
    assert "RESUME_IF_AVAILABLE = True" in source
    assert "optimizer.step(" not in source
    assert "epoch_60_final.pth" in source


def test_p0_v4_bestreg_notebook_declares_selection_and_keeps_training_disabled():
    path = NOTEBOOKS / "01_P0_final_soilnet_v4_bestreg.ipynb"
    book = nbformat.read(path, as_version=4)
    source = "\n".join(cell.source for cell in book.cells if cell.cell_type == "code")
    assert "P0_final_soilnet_v4_bestreg.yaml" in source
    assert "RESUME_IF_AVAILABLE = True" in source
    assert "validation_best_regression.pth" in source
    assert "epoch_60_final.pth" in source
    assert 'os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")' in source
    assert "build_test_loader" not in source
    assert 'split="test"' not in source


@pytest.mark.parametrize(
    ("filename", "config_name", "experiment_id"),
    [
        (
            "02_P1_no_li_bestreg.ipynb", "P1_no_li_bestreg.yaml",
            "P1_SOILNET_VICREG_MU27_NO_LI_BESTREG",
        ),
        (
            "03_P1_no_ssl_bestreg.ipynb", "P1_no_ssl_bestreg.yaml",
            "P1_SOILNET_IMAGENET_LI_NO_SSL_BESTREG",
        ),
    ],
)
def test_p1_bestreg_notebooks_are_sequential_automatic_run_all(
    filename, config_name, experiment_id,
):
    book = nbformat.read(NOTEBOOKS / filename, as_version=4)
    code_cells = [cell.source for cell in book.cells if cell.cell_type == "code"]
    source = "\n".join(code_cells)
    first = code_cells[0]
    assert first.index('os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"') < first.index("import torch")
    assert f'config/experiments/{config_name}' in source
    assert experiment_id in source
    assert "RUN_TRAINING = True" in source
    assert source.index("run_one_batch_preflight(context, device=device)") < source.index(
        "train_experiment(context, resume_if_available=True)"
    )
    assert 'if PREFLIGHT_PASSED is not True:' in source
    assert 'if not torch.cuda.is_available():' in source
    assert "validation_best_regression.pth" in source
    assert "epoch_60_final.pth" in source
    assert "build_test_loader" not in source
    assert 'split="test"' not in source
    assert "09_final_test_evaluation" not in source


def test_p1_bestreg_configs_match_p0_v4_supervised_protocol():
    p0 = load_yaml(REPO / "config/experiments/P0_final_soilnet_v4_bestreg.yaml")
    locked = (
        "dataset_manifest", "dataset_manifest_sha256", "split", "split_sha256",
        "seed", "epochs", "batch_size", "optimizer", "learning_rate", "weight_decay",
        "loss", "metrics", "image_size", "normalization_mean", "normalization_std",
        "train_augmentation", "validation_transform", "use_amp", "num_workers",
        "primary_task", "checkpoint_selection", "selection_formula",
        "primary_checkpoint", "epoch_60_checkpoint", "test_evaluation",
    )
    for filename in ("P1_no_li_bestreg.yaml", "P1_no_ssl_bestreg.yaml"):
        config = load_yaml(REPO / "config/experiments" / filename)
        assert all(config[key] == p0[key] for key in locked)


def test_p1_no_li_uses_locked_vicreg_and_does_not_consume_li():
    config = load_yaml(REPO / "config/experiments/P1_no_li_bestreg.yaml")
    assert config["use_li"] is False
    assert config["li_ablation_strategy"] == "zero_vector_preserve_head_capacity_and_ssl_compatibility"
    assert config["ssl_checkpoint"]["sha256"] == "42599ac025b8d8c8c5b8cca36624665f155e0ff588cd90de2fd47b08cd2d60d8"
    source = (REPO / "src/soilnet/models/experiment_soilnet.py").read_text(encoding="utf-8")
    assert "if self.use_li_signal:" in source
    assert "light_features = torch.zeros" in source


def test_p1_no_ssl_loads_exact_pre_vicreg_snapshot_without_vicreg():
    context = load_experiment_context(REPO / "config/experiments/P1_no_ssl_bestreg.yaml")
    assert context.config["ssl_checkpoint"] is None
    assert context.config["initialization"] == "historical_pre_vicreg_imagenet_snapshot"
    model, report = build_model(
        context.config, context.checkpoint_root, data_root=context.data_root
    )
    assert isinstance(model, ExperimentSoilNet)
    assert report["source_kind"] == "historical_pre_vicreg_imagenet_snapshot"
    assert report["checkpoint_sha256"] == "7fbfac2b637613c6f48b3fef6f2c56be07080eedfba73da1d2b553e0b41f3631"
    assert report["vicreg_checkpoint_loaded"] is False
    assert report["matched_keys"] == report["canonical_keys"] == 577
    components = context.config["initialization_provenance"]["components"]
    assert components["mnv2_block1"] == "timm/mobilenetv2_100.ra_in1k ImageNet pretrained"
    assert components["mobilevit_encoder"] == "timm/mobilevitv2_050.cvnets_in1k ImageNet pretrained"
    assert "newly initialized" in components["regression_head"]


def test_p1_bestreg_engine_has_no_early_stop_and_reloads_best_before_validation_exports():
    source = (REPO / "src/soilnet/training/engine.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    train = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "train_experiment"
    )
    epoch_loop = next(
        node for node in ast.walk(train)
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "epoch"
    )
    assert not any(isinstance(node, (ast.Break, ast.Return)) for node in ast.walk(epoch_loop))
    reload_index = source.index('model.load_state_dict(best_payload["model_state_dict"], strict=True)')
    export_index = source.index("validation_rows, validation_metrics = prediction_rows_and_metrics")
    assert reload_index < export_index
    assert "mean_val_rmse < best_mean_val_rmse" in source
    assert 'context.run_dir / "validation_best_regression.pth"' in source
    assert '"mean_regression_RMSE"' in source
    assert '"is_best_regression"' in source


def test_p1_run_metadata_schema_is_explicit_and_test_remains_no():
    source = (REPO / "src/soilnet/training/engine.py").read_text(encoding="utf-8")
    required = {
        "ablation_type", "primary_task", "checkpoint_selection", "selection_formula",
        "best_epoch", "best_SM0_RMSE", "best_SM20_RMSE",
        "best_mean_validation_RMSE", "primary_checkpoint_path",
        "primary_checkpoint_sha256", "epoch_60_checkpoint_path",
        "epoch_60_checkpoint_sha256", "train_count", "validation_count", "test_count",
        "dataset_manifest_sha256", "split_sha256", "optimizer", "lr", "batch_size",
        "epochs", "weight_decay", "device", "gpu_name", "training_completed",
        "test_evaluated", "initialization_provenance",
        "li_consumed_by_model", "li_ablation_strategy", "vicreg_checkpoint_loaded",
        "initialization_source_kind",
    }
    assert all(f'"{key}"' in source for key in required)
    assert '"test_evaluated": "NO" if best_regression_selection else False' in source


def test_p0_v3_statuses_do_not_make_optional_experiments_block_p0():
    p0 = load_yaml(REPO / "config/experiments/P0_final_soilnet.yaml")
    assert p0["protocol_status"] == "PROTOCOL_REVISED_PRE_TEST"
    for filename in ("P1_no_li.yaml", "P1_no_ssl.yaml"):
        assert load_yaml(REPO / "config/experiments" / filename)["protocol_status"] == "OPTIONAL_NOT_YET_REQUIRED"
    for filename in ("P1_mobilenetv2.yaml", "P1_mobilevitv2.yaml", "P2_efficientnet_b0.yaml", "P2_classical_ml.yaml"):
        assert load_yaml(REPO / "config/experiments" / filename)["protocol_status"] == "OPTIONAL"


def test_p0_v4_config_changes_only_checkpoint_selection_and_identity():
    v3 = load_yaml(REPO / "config/experiments/P0_final_soilnet.yaml")
    v4 = load_yaml(REPO / "config/experiments/P0_final_soilnet_v4_bestreg.yaml")
    locked = (
        "architecture", "ssl", "mu", "initialization", "ssl_checkpoint", "use_li",
        "num_classes", "class_mapping", "dataset_manifest", "dataset_manifest_sha256",
        "split", "split_sha256", "seed", "epochs", "batch_size", "optimizer",
        "learning_rate", "weight_decay", "loss", "metrics", "image_size",
        "normalization_mean", "normalization_std", "train_augmentation",
        "validation_transform", "use_amp", "num_workers", "test_evaluation",
    )
    assert all(v4[key] == v3[key] for key in locked)
    assert _uses_best_regression_selection(v4)
    assert v4["experiment_id"] == "P0_FINAL_SOILNET_VICREG_MU27_LI_V4_BESTREG"
    assert v4["output_dir"] == "P0_FINAL_SOILNET_VICREG_MU27_LI_V4_BESTREG"


def test_bestreg_formula_uses_only_the_two_validation_regression_rmse_values():
    row = {
        "validation_SM_0_rmse": 14.0,
        "validation_SM_20_rmse": 16.0,
        "validation_accuracy": 0.0,
        "validation_macro_f1": 1.0,
    }
    assert _regression_selection_metrics(row) == {
        "SM0_RMSE": 14.0,
        "SM20_RMSE": 16.0,
        "mean_validation_regression_RMSE": 15.0,
    }


def test_bestreg_checkpoint_contains_required_provenance(tmp_path):
    config = load_yaml(REPO / "config/experiments/P0_final_soilnet_v4_bestreg.yaml")
    context = type("Context", (), {
        "config": config,
        "config_sha256": "config-hash",
        "manifest_sha256": config["dataset_manifest_sha256"],
        "split_sha256": config["split_sha256"],
    })()
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    path = tmp_path / "validation_best_regression.pth"
    metrics = {"SM0_RMSE": 14.0, "SM20_RMSE": 16.0, "mean_validation_regression_RMSE": 15.0}
    _save_checkpoint(path, model, context, 38, optimizer=optimizer, checkpoint_metrics=metrics)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "epoch", "model_state_dict", "optimizer_state_dict", "SM0_RMSE", "SM20_RMSE",
        "mean_validation_regression_RMSE", "experiment_id", "ssl_initialization_sha256",
        "selected_ssl_checkpoint_sha256",
        "manifest_sha256", "split_sha256", "seed", "training_protocol_parameters",
    }
    assert required <= set(payload)
    assert payload["epoch"] == 38


def test_p0_resume_identity_gate_accepts_exact_payload_and_rejects_drift():
    context = load_experiment_context(REPO / "config/experiments/P0_final_soilnet.yaml")
    payload = {
        "experiment_id": context.config["experiment_id"],
        "model_architecture": context.config["architecture"],
        "config_sha256": context.config_sha256,
        "manifest_sha256": context.manifest_sha256,
        "split_sha256": context.split_sha256,
        "ssl_initialization_sha256": context.config["ssl_checkpoint"]["sha256"],
        "initialization_checkpoint_sha256": None,
        "seed": context.config["seed"],
    }
    _validate_resume_payload(payload, context)
    payload["split_sha256"] = "changed"
    with pytest.raises(RuntimeError, match="rolling resume checkpoint"):
        _validate_resume_payload(payload, context)


def test_final_test_requires_explicit_nonempty_model_selection_before_loader_access():
    with pytest.raises(RuntimeError, match="explicitly select"):
        evaluate_all_frozen_models_once([], None)
