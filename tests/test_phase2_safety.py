from pathlib import Path
import csv


REPO = Path(__file__).resolve().parents[1]


def test_image_manifest_never_contains_checkpoint_files():
    with (REPO / "data/manifests/dataset_inventory.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert not any(row["relative_path"].casefold().endswith(".pth") for row in rows)


def test_data_root_checkpoint_scope_is_not_merged_into_phase1_inventory():
    with (REPO / "results/audit/checkpoint_inventory.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert sum(row["source"] == "LOCAL" for row in rows) == 969
    assert all("DATA_ROOT_EMBEDDED_ARTIFACTS" not in row["source"] for row in rows)


def test_data_root_checkpoint_inventory_is_separate_and_complete():
    with (REPO / "results/audit/data_root_checkpoint_inventory.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 108
    assert len({row["relative_path_from_data_root"] for row in rows}) == 108
    assert all(row["relative_path_from_data_root"].casefold().endswith(".pth") for row in rows)


def test_phase2_does_not_invent_historical_heldout_or_unseen_data():
    with (REPO / "results/audit/historical_split_inventory.csv").open(newline="", encoding="utf-8") as handle:
        splits = list(csv.DictReader(handle))
    with (REPO / "results/audit/unseen_labeled_candidates.csv").open(newline="", encoding="utf-8") as handle:
        unseen = list(csv.DictReader(handle))
    assert splits and not any(row["genuine_heldout"].casefold() == "true" for row in splits)
    assert len(unseen) == 2057
    assert not any(row["status"] == "CONFIRMED_UNSEEN" for row in unseen)
