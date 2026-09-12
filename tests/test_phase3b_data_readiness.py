import csv
import json
from collections import defaultdict
from pathlib import Path

from soilnet.io import sha256_file


REPO = Path(__file__).resolve().parents[1]
EXCLUSION = REPO / "data/manifests/excluded_conflicting_duplicates_v1.csv"
MANIFEST = REPO / "data/manifests/final_clean_manifest_v1.csv"
SPLIT = REPO / "data/splits/final_clean_split_v1.csv"
LOCK = REPO / "results/audit/final_run_lock_v2.json"


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_excluded_hashes_are_absent_and_final_hashes_are_unique():
    excluded = rows(EXCLUSION)
    final = rows(MANIFEST)
    excluded_hashes = {row["sha256"] for row in excluded}
    final_hashes = [row["sha256"] for row in final]
    assert len(excluded) == 130
    assert len(excluded_hashes) == 46
    assert {row["reason"] for row in excluded} == {"EXACT_DUPLICATE_CONFLICT"}
    assert len(final) == 1927
    assert len(final_hashes) == len(set(final_hashes))
    assert excluded_hashes.isdisjoint(final_hashes)


def test_no_exact_image_target_conflict_remains():
    grouped = defaultdict(list)
    for row in rows(MANIFEST):
        grouped[row["sha256"]].append(row)
    assert all(len(group) == 1 for group in grouped.values())
    assert all(len({row["SM_0"] for row in group}) == 1 for group in grouped.values())
    assert all(len({row["SM_20"] for row in group}) == 1 for group in grouped.values())


def test_no_cross_split_group_or_sample_leakage():
    group_splits = defaultdict(set)
    near_splits = defaultdict(set)
    sample_splits = defaultdict(set)
    sha_splits = defaultdict(set)
    for row in rows(SPLIT):
        group_splits[row["group_id"]].add(row["split"])
        near_splits[row["near_duplicate_group_id"]].add(row["split"])
        sample_splits[row["sample_id"]].add(row["split"])
        sha_splits[row["sha256"]].add(row["split"])
    assert all(len(value) == 1 for value in group_splits.values())
    assert all(len(value) == 1 for value in near_splits.values())
    assert all(len(value) == 1 for value in sample_splits.values())
    assert all(len(value) == 1 for value in sha_splits.values())


def test_only_explicit_deployment_checkpoints_are_in_repository():
    checkpoints = sorted(path for path in REPO.rglob("*.pth") if path.is_file())
    assert sorted(path.relative_to(REPO).as_posix() for path in checkpoints) == sorted([
        "checkpoints/deployment/P0_FINAL_SOILNET_VICREG_MU27_LI_V4_BESTREG.pth",
        "checkpoints/deployment/P1_SOILNET_VICREG_MU27_NO_LI_BESTREG.pth",
    ])
    assert {sha256_file(path) for path in checkpoints} == {
        "eba009dfd45ec21174a8e40b16148e0455e902286c7db55933487221da761379",
        "a9d8de995b0673e9ec39bfc4afac00d6a2a773ad9ffbea0027ff2d0c4fab820c",
    }


def test_final_manifest_and_split_hashes_match_v2_lock():
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    assert sha256_file(MANIFEST) == lock["final_clean_manifest_sha256"]
    assert sha256_file(SPLIT) == lock["final_split_sha256"]


def test_phase3b_did_not_open_the_prospective_test_firewall():
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    assert lock["test_loader_used_in_phase3b"] is False
    assert lock["test_metrics_read_or_computed_in_phase3b"] is False
    assert not (REPO / "results/final/test_evaluation_completed.json").exists()
