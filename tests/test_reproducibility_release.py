import ast
import csv
import json
import subprocess
import sys
from pathlib import Path

import nbformat

from soilnet.io import sha256_file


REPO = Path(__file__).resolve().parents[1]


def test_public_manifests_have_locked_counts_and_hashes():
    expected = {
        "train": (1407, "631dc280f9593b050b0f7ab91dc49d7048d0ae03ade51da096b4d384b486aa97"),
        "validation": (289, "f6b71b441872502e5f6231b27a762cbd913015e31ef7592dcb5c62da6c62bb90"),
        "test": (231, "3be3096b86a8b599a32384d71abcaaee5b6162e84a80c1bf8662003a2db7bc19"),
    }
    for split, (count, digest) in expected.items():
        path = REPO / f"data/splits/{split}_manifest.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == count
        assert all(row["split"] == split and row["sample_id"] and row["sha256"] for row in rows)
        assert sha256_file(path) == digest
    with (REPO / "data/manifests/unlabeled_manifest.csv").open(newline="", encoding="utf-8") as handle:
        unlabeled = list(csv.DictReader(handle))
    assert len(unlabeled) == 11995
    assert all(row["relative_path"] and row["sha256"] for row in unlabeled)


def test_p3_release_is_validation_only_and_test_guard_precedes_checkpoint_access():
    p3 = REPO / "results/frozen/P3"
    assert not list(p3.rglob("*test*"))
    metadata = json.loads((p3 / "supervised/run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["test_evaluated"] == "NO"
    summary = json.loads((REPO / "results/p3_analysis/factorial_validation_summary.json").read_text(encoding="utf-8"))
    assert summary["TEST_SET_OPENED"] is False
    process = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/reproduce_checkpoint_evaluation.py"),
            "--experiment",
            "P3",
            "--split",
            "test",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
    )
    assert process.returncode != 0
    assert "P3_TEST_PROHIBITED" in process.stderr


def test_frozen_analysis_script_has_no_dataset_model_or_training_calls():
    source = (REPO / "scripts/reproduce_frozen_analyses.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "SoilNetDataset" not in source
    assert "DataLoader" not in source
    assert "train_experiment" not in source
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not (called & {"build_model", "build_frozen_model", "train_experiment"})


def test_public_notebook_sources_are_clean_and_executed_copies_are_separate():
    for path in sorted((REPO / "notebooks/final_experiments").glob("*.ipynb")):
        book = nbformat.read(path, as_version=4)
        nbformat.validate(book)
        for cell in book.cells:
            if cell.cell_type == "code":
                assert cell.execution_count is None
                assert not cell.outputs
                ast.parse(cell.source)
    executed = sorted((REPO / "reproducibility/executed_notebooks").glob("*.ipynb"))
    assert len(executed) == 7
    assert any(
        cell.get("outputs")
        for path in executed
        for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]
        if cell["cell_type"] == "code"
    )


def test_public_release_copies_have_no_source_machine_absolute_paths():
    roots = (
        REPO / "results/frozen",
        REPO / "notebooks/final_experiments",
        REPO / "reproducibility/executed_notebooks",
    )
    forbidden = ("/home/diy-hus", "/mnt/d", "/mnt/e")
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file():
                text = path.read_text(encoding="utf-8")
                assert not any(token in text for token in forbidden), path
