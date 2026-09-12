import ast
import json
from pathlib import Path

import nbformat
import pytest
import torch
import yaml

from soilnet.io import load_yaml, sha256_file, write_json
from soilnet.training.vicreg import (
    build_resolved_p3_config,
    reconcile_resolved_p3_before_ssl,
    reconcile_resolved_p3_config,
    validate_or_archive_stale_ssl_resume,
    validate_or_archive_stale_supervised_state,
)


REPO = Path(__file__).resolve().parents[1]


def test_p3_matches_p2_supervised_protocol_except_vicreg_initialization():
    p2 = load_yaml(REPO / "config/experiments/P2_mobilevitv2_imagenet_li_bestreg.yaml")
    p3 = load_yaml(REPO / "config/experiments/P3_mobilevitv2_vicreg_li_bestreg.yaml")
    fields = (
        "architecture", "timm_model_name", "use_li", "li_fusion", "num_classes",
        "class_mapping", "dataset_manifest", "dataset_manifest_sha256", "split",
        "split_sha256", "split_counts", "seed", "epochs", "batch_size", "optimizer",
        "learning_rate", "weight_decay", "loss", "metrics", "image_size",
        "normalization_mean", "normalization_std", "train_augmentation",
        "validation_transform", "use_amp", "num_workers", "primary_task",
        "save_validation_best", "checkpoint_selection", "selection_formula",
        "primary_checkpoint", "epoch_60_checkpoint", "test_evaluation",
    )
    assert all(p2[field] == p3[field] for field in fields)
    assert p2["initialization"] == "imagenet"
    assert p3["initialization"] == "imagenet_then_vicreg"
    assert p3["initialization_provenance"]["soilnet_vicreg_checkpoint_loaded"] is False


def test_p3_recipe_matches_final_soilnet_vicreg_notebook():
    p3 = load_yaml(REPO / "config/experiments/P3_mobilevitv2_vicreg_li_bestreg.yaml")
    recipe = p3["vicreg_pretraining"]
    assert recipe["expected_image_count"] == 11995
    assert recipe["epochs"] == 80
    assert recipe["batch_size"] == 16
    assert recipe["learning_rate"] == 0.0001
    assert recipe["projector_output_dim"] == 128
    assert recipe["lambda_invariance"] == 25.0
    assert recipe["mu_variance"] == 27.0
    assert recipe["nu_covariance"] == 1.0
    source = (REPO / "src/soilnet/training/vicreg.py").read_text(encoding="utf-8")
    assert 'torch.amp.GradScaler("cuda"' in source
    assert 'torch.amp.autocast("cuda"' in source
    assert "torch.cuda.amp" not in source


def test_p3_runner_has_no_test_dataset_or_test_evaluation_call():
    path = REPO / "scripts/run_p3_mobilevitv2_vicreg.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"build_test_loader", "evaluate_all_frozen_models_once", "evaluate_final_test"}
    called = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not (called & forbidden)
    assert 'split="test"' not in source
    assert "SoilNet VICReg checkpoint" in source
    assert "reconcile_resolved_p3_before_ssl" in source
    assert "--run-ssl-pretraining" in source


def test_p3_notebook_is_fresh_kernel_run_all_and_test_firewalled():
    path = REPO / "notebooks/final_experiments/13_P3_mobilevitv2_vicreg_bestreg.ipynb"
    book = nbformat.read(path, as_version=4)
    nbformat.validate(book)
    source = "\n".join(cell.source for cell in book.cells if cell.cell_type == "code")
    for cell in book.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
    assert "RUN_SSL_PRETRAINING = False" in source
    assert "importlib.reload" in source
    assert "verify_mobilevitv2_vicreg" in source
    assert "pretrain_mobilevitv2_vicreg" in source
    assert "reconcile_resolved_p3_before_ssl" in source
    assert "reconcile_resolved_p3_config" in source
    assert "validate_or_archive_stale_supervised_state" in source
    assert "train_experiment(context" in source
    assert "validation_best_regression.pth" not in source or "primary_checkpoint_path" in source
    assert "TEST_LOADER_INSTANTIATED" in source
    assert "build_test_loader" not in source
    assert 'split="test"' not in source
    assert "evaluate_final_test" not in source


def _resume_payload(*, config_hash="old-config", manifest_hash="manifest", model_name="mobilevitv2_050.cvnets_in1k"):
    return {
        "experiment_id": "P3_MOBILEVITV2_VICREG_LI_BESTREG",
        "source_config_sha256": config_hash,
        "unlabeled_manifest_sha256": manifest_hash,
        "timm_model_name": model_name,
        "epoch": 3,
        "backbone_state_dict": {},
        "projector_state_dict": {},
        "optimizer_state_dict": {},
        "scaler_state_dict": {},
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": [],
        "dataloader_generator_state": torch.Generator().get_state(),
        "history": [{"epoch": 1}, {"epoch": 2}, {"epoch": 3}],
    }


def _resume_config():
    return {
        "experiment_id": "P3_MOBILEVITV2_VICREG_LI_BESTREG",
        "timm_model_name": "mobilevitv2_050.cvnets_in1k",
        "vicreg_pretraining": {"epochs": 80},
    }


def test_stale_p3_resume_archives_only_config_hash_mismatch_idempotently(tmp_path):
    experiment_root = tmp_path / "runs/P3_MOBILEVITV2_VICREG_LI_BESTREG"
    resume = experiment_root / "pretraining/last_resume_checkpoint.pth"
    resume.parent.mkdir(parents=True)
    torch.save(_resume_payload(), resume)
    result = validate_or_archive_stale_ssl_resume(
        config=_resume_config(),
        config_sha256="current-config",
        unlabeled_manifest_sha256="manifest",
        experiment_root=experiment_root,
        stale_archive_root=tmp_path / "soilnet_stale_p3",
    )
    assert result["status"] == "STALE_RESUME_ARCHIVED"
    assert result["start_epoch"] == 1
    assert not resume.exists()
    archived = Path(result["artifacts"][0]["archived_path"])
    assert archived.is_file()
    report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
    assert set(report["mismatches"]) == {"source_config_sha256"}
    second = validate_or_archive_stale_ssl_resume(
        config=_resume_config(), config_sha256="current-config",
        unlabeled_manifest_sha256="manifest", experiment_root=experiment_root,
        stale_archive_root=tmp_path / "soilnet_stale_p3",
    )
    assert second["status"] == "NO_RESUME"


def test_valid_p3_resume_is_reused_and_architecture_mismatch_stops(tmp_path):
    experiment_root = tmp_path / "runs/P3_MOBILEVITV2_VICREG_LI_BESTREG"
    resume = experiment_root / "pretraining/last_resume_checkpoint.pth"
    resume.parent.mkdir(parents=True)
    torch.save(_resume_payload(config_hash="current-config"), resume)
    valid = validate_or_archive_stale_ssl_resume(
        config=_resume_config(), config_sha256="current-config",
        unlabeled_manifest_sha256="manifest", experiment_root=experiment_root,
        stale_archive_root=tmp_path / "soilnet_stale_p3",
    )
    assert valid["status"] == "VALID_RESUME"
    assert valid["start_epoch"] == 4
    assert resume.is_file()

    torch.save(_resume_payload(model_name="wrong-model"), resume)
    with pytest.raises(RuntimeError, match="timm_model_name"):
        validate_or_archive_stale_ssl_resume(
            config=_resume_config(), config_sha256="current-config",
            unlabeled_manifest_sha256="manifest", experiment_root=experiment_root,
            stale_archive_root=tmp_path / "soilnet_stale_p3",
        )
    assert resume.is_file()


def _completed_ssl(tmp_path, config, config_sha="current-config"):
    experiment_root = tmp_path / "runs" / config["experiment_id"]
    pretraining = experiment_root / "pretraining"
    pretraining.mkdir(parents=True)
    manifest = pretraining / "unlabeled_manifest.csv"
    manifest.write_text("relative_path,size_bytes,sha256\na.jpg,1,x\n", encoding="utf-8")
    manifest_sha = sha256_file(manifest)
    encoder = pretraining / "mobilevitv2_vicreg_encoder_final.pth"
    torch.save({
        "experiment_id": config["experiment_id"],
        "source_config_sha256": config_sha,
        "encoder_architecture": config["timm_model_name"],
        "epoch": 80,
        "unlabeled_manifest_sha256": manifest_sha,
        "backbone_state_dict": {},
    }, encoder)
    projector = pretraining / "vicreg_projector_final.pth"
    torch.save({"projector_state_dict": {}}, projector)
    (pretraining / "training_history.csv").write_text("epoch,loss\n80,1\n", encoding="utf-8")
    metadata = {
        "experiment_id": config["experiment_id"],
        "source_config_sha256": config_sha,
        "encoder_architecture": config["timm_model_name"],
        "epochs": 80,
        "unlabeled_image_count": 11995,
        "training_completed": True,
        "encoder_checkpoint_path": str(encoder),
        "encoder_checkpoint_sha256": sha256_file(encoder),
        "projector_checkpoint_sha256": sha256_file(projector),
        "unlabeled_manifest_sha256": manifest_sha,
    }
    write_json(pretraining / "pretraining_metadata.json", metadata)
    return experiment_root, metadata


def test_stale_resolved_config_archives_only_derived_ssl_hash(tmp_path):
    config = load_yaml(REPO / "config/experiments/P3_mobilevitv2_vicreg_li_bestreg.yaml")
    experiment_root, metadata = _completed_ssl(tmp_path, config)
    stale = build_resolved_p3_config(config, metadata)
    stale["ssl_checkpoint"]["sha256"] = "old-ssl"
    resolved_path = experiment_root / "resolved_config.yaml"
    resolved_path.write_text(yaml.safe_dump(stale, sort_keys=False), encoding="utf-8")
    result = reconcile_resolved_p3_config(
        config=config, config_sha256="current-config", experiment_root=experiment_root,
        stale_archive_root=tmp_path / "stale",
    )
    assert result["status"] == "STALE_RESOLVED_CONFIG_ARCHIVED"
    assert [row["key"] for row in result["field_diff"]] == ["ssl_checkpoint.sha256"]
    assert Path(result["archived_path"]).is_file()
    current = load_yaml(resolved_path)
    assert current["ssl_checkpoint"]["sha256"] == metadata["encoder_checkpoint_sha256"]

    current["seed"] += 1
    resolved_path.write_text(yaml.safe_dump(current, sort_keys=False), encoding="utf-8")
    with pytest.raises(RuntimeError, match="actual experiment protocol"):
        reconcile_resolved_p3_config(
            config=config, config_sha256="current-config", experiment_root=experiment_root,
            stale_archive_root=tmp_path / "stale",
        )


def test_partial_supervised_state_from_old_ssl_is_archived(tmp_path):
    config = load_yaml(REPO / "config/experiments/P3_mobilevitv2_vicreg_li_bestreg.yaml")
    experiment_root, metadata = _completed_ssl(tmp_path, config)
    resolved = build_resolved_p3_config(config, metadata)
    resolved_path = experiment_root / "resolved_config.yaml"
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    supervised = experiment_root / "supervised"
    supervised.mkdir()
    payload = {
        "experiment_id": config["experiment_id"],
        "config_sha256": "old-resolved",
        "manifest_sha256": config["dataset_manifest_sha256"],
        "split_sha256": config["split_sha256"],
        "ssl_initialization_sha256": "old-ssl",
        "selected_ssl_checkpoint_sha256": "old-ssl",
        "model_architecture": config["architecture"],
        "epoch": 22,
    }
    torch.save(payload, supervised / "last_resume_checkpoint.pth")
    torch.save({**payload, "epoch": 13}, supervised / "validation_best_regression.pth")
    result = validate_or_archive_stale_supervised_state(
        resolved_config=resolved,
        resolved_config_sha256=sha256_file(resolved_path),
        experiment_root=experiment_root,
        stale_archive_root=tmp_path / "stale",
    )
    assert result["status"] == "STALE_SUPERVISED_STATE_ARCHIVED"
    assert result["start_epoch"] == 1
    assert not list(supervised.glob("*.pth"))
    assert len(result["artifacts"]) == 2


def test_pre_ssl_reconcile_archives_only_identity_proven_partial_run(tmp_path):
    config = load_yaml(REPO / "config/experiments/P3_mobilevitv2_vicreg_li_bestreg.yaml")
    experiment_root = tmp_path / "runs" / config["experiment_id"]
    pretraining = experiment_root / "pretraining"
    pretraining.mkdir(parents=True)
    manifest = pretraining / "unlabeled_manifest.csv"
    manifest.write_text("relative_path,size_bytes,sha256\na.jpg,1,x\n", encoding="utf-8")
    manifest_sha = sha256_file(manifest)
    torch.save(
        _resume_payload(config_hash="current-config", manifest_hash=manifest_sha),
        pretraining / "last_resume_checkpoint.pth",
    )
    stale = dict(config)
    stale["ssl_checkpoint"] = {
        "scope": "RUN_ARTIFACT_ROOT",
        "relative_path": (
            f"{config['experiment_id']}/pretraining/"
            "mobilevitv2_vicreg_encoder_final.pth"
        ),
        "sha256": "not-final-yet",
        "encoder_architecture": config["timm_model_name"],
        "unlabeled_manifest_sha256": manifest_sha,
    }
    resolved_path = experiment_root / "resolved_config.yaml"
    resolved_path.write_text(yaml.safe_dump(stale, sort_keys=False), encoding="utf-8")

    result = reconcile_resolved_p3_before_ssl(
        config=config,
        config_sha256="current-config",
        experiment_root=experiment_root,
        stale_archive_root=tmp_path / "stale",
    )
    assert result["status"] == "STALE_RESOLVED_CONFIG_ARCHIVED_BEFORE_SSL"
    assert not resolved_path.exists()
    assert Path(result["archived_path"]).is_file()
    assert (pretraining / "last_resume_checkpoint.pth").is_file()
    assert result["partial_ssl_resume_resolution"]["status"] == "VALID_RESUME"


def test_pre_ssl_reconcile_stops_when_partial_identity_is_not_exact(tmp_path):
    config = load_yaml(REPO / "config/experiments/P3_mobilevitv2_vicreg_li_bestreg.yaml")
    experiment_root = tmp_path / "runs" / config["experiment_id"]
    pretraining = experiment_root / "pretraining"
    pretraining.mkdir(parents=True)
    manifest = pretraining / "unlabeled_manifest.csv"
    manifest.write_text("relative_path,size_bytes,sha256\na.jpg,1,x\n", encoding="utf-8")
    manifest_sha = sha256_file(manifest)
    torch.save(
        _resume_payload(config_hash="current-config", manifest_hash=manifest_sha),
        pretraining / "last_resume_checkpoint.pth",
    )
    stale = dict(config)
    stale["seed"] += 1
    stale["ssl_checkpoint"] = {
        "scope": "RUN_ARTIFACT_ROOT",
        "relative_path": (
            f"{config['experiment_id']}/pretraining/"
            "mobilevitv2_vicreg_encoder_final.pth"
        ),
        "sha256": "not-final-yet",
        "encoder_architecture": config["timm_model_name"],
        "unlabeled_manifest_sha256": manifest_sha,
    }
    resolved_path = experiment_root / "resolved_config.yaml"
    resolved_path.write_text(yaml.safe_dump(stale, sort_keys=False), encoding="utf-8")

    with pytest.raises(RuntimeError, match="cannot be tied safely"):
        reconcile_resolved_p3_before_ssl(
            config=config,
            config_sha256="current-config",
            experiment_root=experiment_root,
            stale_archive_root=tmp_path / "stale",
        )
    assert resolved_path.is_file()
    assert (pretraining / "last_resume_checkpoint.pth").is_file()
