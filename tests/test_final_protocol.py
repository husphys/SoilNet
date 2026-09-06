import ast
from pathlib import Path

import pytest

from soilnet.final_protocol import load_active_config, refuse_if_test_completed


REPO = Path(__file__).resolve().parents[1]


def test_blocked_phase3a_config_cannot_start_training():
    with pytest.raises(RuntimeError, match="BLOCKED_DATA_CONFLICT"):
        load_active_config(REPO / "config/experiments/final_revalidation.yaml")


def test_existing_test_marker_prevents_rerun(tmp_path):
    marker = tmp_path / "test_evaluation_completed.json"
    marker.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="already completed"):
        refuse_if_test_completed(marker)


def test_no_final_manifest_or_split_was_generated_while_labels_conflict():
    assert not (REPO / "data/manifests/final_unique_manifest_v1.csv").exists()
    assert not (REPO / "data/splits/final_unique_split_v1.csv").exists()
    assert not (REPO / "results/final/test_evaluation_completed.json").exists()


def test_training_entrypoint_never_constructs_test_dataset():
    tree = ast.parse((REPO / "scripts/train_final_revalidation.py").read_text(encoding="utf-8"))
    split_literals = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "SoilNetDataset":
            for keyword in node.keywords:
                if keyword.arg == "split" and isinstance(keyword.value, ast.Constant):
                    split_literals.append(keyword.value.value)
    assert split_literals == ["train", "validation"]
