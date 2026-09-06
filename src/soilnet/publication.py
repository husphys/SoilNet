from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from soilnet.final_sequence import (
    DEPLOYMENT_CHECKPOINT,
    FINAL_TEST_DIR,
    FINAL_TEST_LOCK,
    LEGACY_URLS,
    MANIFEST_SHA256,
    MODEL_SPECS,
    PUBLICATION_URL,
    REPO,
    SPLIT_COUNTS,
    SPLIT_SHA256,
    preflight_frozen_registry,
)
from soilnet.io import sha256_file, write_csv


P0_DEPLOYMENT_SHA256 = "eba009dfd45ec21174a8e40b16148e0455e902286c7db55933487221da761379"
EDGE_RESULT_FILES = (
    "results/edge/rpi_benchmark.json",
    "results/edge/rpi_benchmark.csv",
    "results/edge/RPI_BENCHMARK_REPORT.md",
    "results/edge/rpi_environment.txt",
    "results/edge/paper_ready_latency_summary.csv",
)
EDGE_COMMIT_MESSAGE = "Add Raspberry Pi deployment benchmark"
FINAL_COMMIT_MESSAGE = "Final SoilNet reproducibility results and Q1 evaluation"
FINAL_NOTEBOOKS = {
    "00_gpu_environment_check.ipynb",
    "01_P0_final_soilnet_v4_bestreg.ipynb",
    "02_P1_no_li_bestreg.ipynb",
    "03_P1_no_ssl_bestreg.ipynb",
    "09_P2_mobilevitv2_imagenet_li_bestreg.ipynb",
    "10_final_frozen_test_evaluation.ipynb",
    "11_raspberry_pi_benchmark.ipynb",
    "12_finalize_paper_artifacts_and_publish.ipynb",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Required publication artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"Required publication artifact is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_publication_inputs() -> dict[str, Any]:
    registry = preflight_frozen_registry(allow_completed_test=True)
    lock = _read_json(FINAL_TEST_LOCK)
    if lock.get("TEST_OPENED") != "YES" or lock.get("TEST_EVALUATED") != "YES" or lock.get("MODEL_SET_FROZEN") != "YES":
        raise RuntimeError("Final-test lock is invalid")
    if lock.get("split_sha256") != SPLIT_SHA256 or lock.get("test_n") != SPLIT_COUNTS["test"]:
        raise RuntimeError("Final-test split identity changed")
    required_test = [
        "frozen_model_registry_snapshot.json", "P0_test_predictions.csv", "P1_noLI_test_predictions.csv",
        "P1_noSSL_test_predictions.csv", "P2_mobilevitv2_test_predictions.csv", "final_test_metrics.json",
        "final_test_metrics.csv", "final_ablation_table.csv", "final_architecture_comparison.csv",
        "FINAL_TEST_REPORT.md", "paired_bootstrap_results.json", "paired_bootstrap_results.csv",
        "PAIRED_BOOTSTRAP_METHOD.md", "FINAL_TEST_LOCK.json",
    ]
    missing = [name for name in required_test if not (FINAL_TEST_DIR / name).is_file()]
    required_figures = [
        f"P0_actual_vs_predicted_{target}.{extension}" for target in ("SM0", "SM20") for extension in ("png", "pdf")
    ] + [
        f"P0_residual_distribution_{target}.{extension}" for target in ("SM0", "SM20") for extension in ("png", "pdf")
    ] + [
        f"{stem}.{extension}" for stem in ("ablation_mean_RMSE", "architecture_comparison") for extension in ("png", "pdf")
    ] + ["P0_scatter_residual_data.csv", "model_mean_RMSE_plot_data.csv"]
    missing.extend(f"figures/{name}" for name in required_figures if not (FINAL_TEST_DIR / "figures" / name).is_file())
    if missing:
        raise RuntimeError(f"Final-test publication artifacts missing: {missing}")
    edge = REPO / "results/edge"
    required_edge = (
        "rpi_benchmark.json", "rpi_benchmark.csv", "RPI_BENCHMARK_REPORT.md",
        "rpi_environment.txt", "paper_ready_latency_summary.csv",
    )
    if any(not (edge / name).is_file() for name in required_edge):
        raise RuntimeError("RPI_BENCHMARK_MISSING")
    rpi = _read_json(edge / "rpi_benchmark.json")
    if rpi.get("hardware_verified") != "Raspberry Pi" or rpi.get("checkpoint_sha256") != P0_DEPLOYMENT_SHA256:
        raise RuntimeError("Raspberry Pi benchmark identity is invalid")
    for key, expected in lock.get("model_sha256", {}).items():
        entry = next((model for model in registry["models"] if model["key"] == key), None)
        if entry is None or entry["checkpoint_sha256"] != expected:
            raise RuntimeError(f"Final-test lock/registry mismatch for {key}")
    return {"registry": registry, "lock": lock, "rpi": rpi}


def _validation_row(entry: dict[str, Any]) -> dict[str, Any]:
    metrics = entry["validation_metrics"]
    sm0, sm20 = metrics["regression"]["SM_0"], metrics["regression"]["SM_20"]
    return {
        "model": entry["key"], "experiment_id": entry["experiment_id"], "best_epoch": entry["best_epoch"],
        "SM0_RMSE": sm0["rmse"], "SM0_MAE": sm0["mae"], "SM0_R2": sm0["r2"],
        "SM20_RMSE": sm20["rmse"], "SM20_MAE": sm20["mae"], "SM20_R2": sm20["r2"],
        "mean_RMSE": (sm0["rmse"] + sm20["rmse"]) / 2.0,
        "mean_MAE": (sm0["mae"] + sm20["mae"]) / 2.0,
        "Accuracy": metrics["classification"]["accuracy"], "Macro_F1": metrics["classification"]["macro_f1"],
        "scale": "original_0_to_100_percentage_points",
    }


def aggregate_paper_artifacts() -> list[Path]:
    validated = validate_publication_inputs()
    registry = validated["registry"]
    output = REPO / "results/paper"
    output.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    def save(name: str, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
        path = output / name
        write_csv(path, rows, fields or list(rows[0]))
        created.append(path)

    save("01_dataset_summary.csv", [{
        "dataset": "final_clean_unique_labeled", "total": 1927, **SPLIT_COUNTS,
        "manifest_sha256": MANIFEST_SHA256, "split_sha256": SPLIT_SHA256,
        "raw_data_in_repository": "NO",
    }])
    registry_rows = [{
        "model": entry["key"], "experiment_id": entry["experiment_id"], "role": entry["model_role"],
        "architecture": entry["architecture"], "initialization": entry["initialization"],
        "LI": entry["LI"], "VICReg": entry["VICReg"], "checkpoint_sha256": entry["checkpoint_sha256"],
        "best_epoch": entry["best_epoch"], "split_sha256": entry["split_sha256"], "seed": entry["seed"],
    } for entry in registry["models"]]
    save("02_model_registry.csv", registry_rows)
    validation_rows = [_validation_row(entry) for entry in registry["models"]]
    save("03_validation_ablation_table.csv", validation_rows)
    final_rows = _read_csv(FINAL_TEST_DIR / "final_test_metrics.csv")
    save("04_final_test_performance.csv", final_rows)
    save("05_final_test_ablation.csv", [row for row in final_rows if row["model"] in {"P0", "P1_noLI", "P1_noSSL"}])
    save("06_architecture_comparison.csv", [row for row in final_rows if row["model"] in {"P1_noSSL", "P2_MobileViTv2"}])
    bootstrap_rows = _read_csv(FINAL_TEST_DIR / "paired_bootstrap_results.csv")
    save("07_paired_bootstrap_ci.csv", bootstrap_rows)
    complexity_rows = []
    for entry in registry["models"]:
        complexity_rows.append({"model": entry["key"], **entry["complexity"]})
    save("08_model_complexity.csv", complexity_rows)
    save("09_raspberry_pi_latency.csv", _read_csv(REPO / "results/edge/paper_ready_latency_summary.csv"))
    claims = [
        {"claim": "Final P0 regression and classification performance", "evidence_class": "SUPPORTED_BY_FINAL_TEST", "evidence": "04_final_test_performance.csv"},
        {"claim": "LI contribution: P0 versus P1-noLI", "evidence_class": "SUPPORTED_BY_VALIDATION_ABLATION", "evidence": "03_validation_ablation_table.csv and paired final-test context"},
        {"claim": "VICReg contribution: P0 versus P1-noSSL", "evidence_class": "SUPPORTED_BY_VALIDATION_ABLATION", "evidence": "03_validation_ablation_table.csv and paired final-test context"},
        {"claim": "Architecture contribution: P1-noSSL versus P2", "evidence_class": "SUPPORTED_BY_FINAL_TEST", "evidence": "06_architecture_comparison.csv and 07_paired_bootstrap_ci.csv"},
        {"claim": "Raspberry Pi CPU latency", "evidence_class": "SUPPORTED_BY_DEPLOYMENT_BENCHMARK", "evidence": "09_raspberry_pi_latency.csv"},
        {"claim": "Irrigation demonstration", "evidence_class": "PROOF_OF_CONCEPT_ONLY", "evidence": "historical provenance; no generalized causal estimate"},
        {"claim": "Old SoilNet 0.086/0.066 metrics", "evidence_class": "REMOVE_OR_REWRITE", "evidence": "incompatible or unrecovered provenance"},
        {"claim": "Old SoilNet 0.098/0.072 metrics", "evidence_class": "REMOVE_OR_REWRITE", "evidence": "incompatible or unrecovered provenance"},
        {"claim": "2,250 labeled images", "evidence_class": "REMOVE_OR_REWRITE", "evidence": "final QC count is 1,927"},
        {"claim": "Direct Barlow/MoCo/SimCLR superiority", "evidence_class": "SUPPORTED_BY_HISTORICAL_PROVENANCE_ONLY", "evidence": "do not claim superiority unless provenance is recovered"},
        {"claim": "Classical ML Table 7 as direct final benchmark", "evidence_class": "REMOVE_OR_REWRITE", "evidence": "not evaluated in final frozen test"},
        {"claim": "Direct SOTA superiority across incomparable datasets", "evidence_class": "REMOVE_OR_REWRITE", "evidence": "datasets/protocols are not directly comparable"},
        {"claim": "Independent 200-image validation", "evidence_class": "SUPPORTED_BY_HISTORICAL_PROVENANCE_ONLY", "evidence": "retain only if provenance is recovered"},
        {"claim": "Generalized 50% irrigation saving", "evidence_class": "PROOF_OF_CONCEPT_ONLY", "evidence": "rewrite as demonstration, not generalized causal claim"},
    ]
    save("10_claim_evidence_matrix.csv", claims)
    (output / "FINAL_RESULTS_SUMMARY.md").write_text(
        "# Final results summary\n\n"
        "## VALIDATION\n\nFrozen-model selection and ablations are in `03_validation_ablation_table.csv`.\n\n"
        "## HELD-OUT TEST\n\nOne-time results on 231 samples are in `04_final_test_performance.csv`; regression uses original 0-100 percentage points.\n\n"
        "## HISTORICAL DEVELOPMENT EVIDENCE\n\nHistorical values are provenance context only and are not mixed with final validation/test evidence.\n\n"
        "## DEPLOYMENT BENCHMARK\n\nActual Raspberry Pi CPU measurements are in `09_raspberry_pi_latency.csv`.\n\n"
        "## PROOF-OF-CONCEPT IRRIGATION\n\nIrrigation evidence remains proof-of-concept only; no generalized saving or causal effect is claimed.\n",
        encoding="utf-8",
    )
    created.append(output / "FINAL_RESULTS_SUMMARY.md")
    _write_final_docs(claims)
    return created


def _write_final_docs(claims: list[dict[str, str]]) -> None:
    (REPO / "docs/FINAL_EXPERIMENTAL_PROTOCOL.md").write_text(
        "# Final experimental protocol\n\nThe locked split contains 1,407 train, 289 validation, and 231 held-out test samples "
        f"(SHA256 `{SPLIT_SHA256}`, seed 20260905). All supervised runs use 60 epochs, batch 32, Adam, "
        "learning rate 1e-4, weight decay 0, no early stopping, and strict minimum mean validation regression RMSE. "
        "Classification is auxiliary and never selects checkpoints. Notebook 10 alone opens test after all four models are frozen. "
        "After `FINAL_TEST_LOCK.json` exists, test outcomes must not drive model changes.\n",
        encoding="utf-8",
    )
    (REPO / "docs/FINAL_REPRODUCIBILITY_STATEMENT.md").write_text(
        "# Final reproducibility statement\n\nRaw images and training checkpoint collections are not distributed. Content-addressed manifests, "
        "configs, code, frozen checkpoint identities, per-sample final predictions, metrics, bootstrap outputs, and figures are included. "
        "The one bundled binary is the frozen P0 deployment checkpoint, included solely for dataset-free Raspberry Pi benchmarking. "
        "Its SHA256 is verified before loading. Historical evidence is explicitly separated from prospective validation and held-out test evidence.\n",
        encoding="utf-8",
    )
    lines = ["# Claim evidence matrix", "", "| Claim | Classification | Evidence |", "|---|---|---|"]
    lines.extend(f"| {row['claim']} | {row['evidence_class']} | {row['evidence']} |" for row in claims)
    (REPO / "docs/CLAIM_EVIDENCE_MATRIX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REPO / "docs/LEGACY_RESULT_STATUS.md").write_text(
        "# Legacy result status\n\nThe legacy `diy-hus/SoilNet` repository is immutable provenance only. Old 0.086/0.066 and "
        "0.098/0.072 values, the 2,250-image count, direct SSL-method superiority, classical Table 7 comparisons, "
        "cross-dataset SOTA claims, an independent 200-image validation claim, and generalized 50% irrigation saving "
        "must be removed or rewritten unless their exact provenance is recovered. Final claims use only the evidence classes in "
        "`CLAIM_EVIDENCE_MATRIX.md`.\n",
        encoding="utf-8",
    )


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=REPO, text=True, capture_output=True, check=check)


def _publication_remote() -> str:
    """Return/create an exact publication remote; never target the legacy URL."""
    remotes = {
        line.split()[0]: line.split()[1]
        for line in _git("remote", "-v").stdout.splitlines()
        if line.endswith("(push)")
    }
    if any(
        url in LEGACY_URLS and name in {"origin", "publication"}
        for name, url in remotes.items()
    ):
        raise RuntimeError("Legacy repository overwrite is prohibited")
    if "publication" in remotes:
        if remotes["publication"] != PUBLICATION_URL:
            raise RuntimeError("publication remote points to an unexpected repository")
        return "publication"
    if remotes.get("origin") == PUBLICATION_URL:
        _git("remote", "add", "publication", PUBLICATION_URL)
        return "publication"
    raise RuntimeError("No exact husphys-gif/SoilNet publication remote")


def _require_main_identity_and_auth(remote: str) -> None:
    if _git("branch", "--show-current").stdout.strip() != "main":
        raise RuntimeError("Publication must run on the main branch")
    author_name = _git("config", "--get", "user.name", check=False).stdout.strip()
    author_email = _git("config", "--get", "user.email", check=False).stdout.strip()
    if not author_name or not author_email:
        raise RuntimeError("GIT_IDENTITY_REQUIRED")
    # A push dry-run verifies write authentication without printing credentials
    # or changing the remote repository.
    probe = _git("push", "--dry-run", remote, "HEAD:refs/heads/main", check=False)
    if probe.returncode != 0:
        raise RuntimeError("GITHUB_AUTH_REQUIRED")


def _worktree_paths() -> list[str]:
    paths = []
    for line in _git("status", "--porcelain=v1", "--untracked-files=all").stdout.splitlines():
        value = line[3:]
        if " -> " in value:
            value = value.split(" -> ", 1)[1]
        paths.append(value)
    return paths


def _is_allowed(path: str, allowed: tuple[str, ...]) -> bool:
    return any(path == value or (value.endswith("/") and path.startswith(value)) for value in allowed)


def _sync_preserving_allowed_changes(remote: str, allowed: tuple[str, ...], label: str) -> None:
    """Fetch/rebase while round-tripping only explicitly recognized local work."""
    dirty = _worktree_paths()
    unexpected = [path for path in dirty if not _is_allowed(path, allowed)]
    if unexpected:
        raise RuntimeError(f"UNSAFE_GIT_STATE: unrelated files would block rebase: {unexpected}")
    stashed = False
    if dirty:
        before = _git("rev-parse", "--verify", "refs/stash", check=False).stdout.strip()
        result = _git("stash", "push", "--include-untracked", "-m", f"soilnet-{label}", check=False)
        after = _git("rev-parse", "--verify", "refs/stash", check=False).stdout.strip()
        if result.returncode != 0 or not after or after == before:
            raise RuntimeError("SAFE_STASH_FAILED")
        stashed = True
    fetch = _git("fetch", remote, "main", check=False)
    if fetch.returncode != 0:
        if stashed:
            _git("stash", "pop", check=False)
        raise RuntimeError("GITHUB_AUTH_REQUIRED")
    rebase = _git("rebase", f"{remote}/main", check=False)
    if rebase.returncode != 0:
        _git("rebase", "--abort", check=False)
        if stashed:
            _git("stash", "pop", check=False)
        raise RuntimeError("SAFE_REBASE_CONFLICT")
    if stashed:
        restored = _git("stash", "pop", check=False)
        if restored.returncode != 0:
            raise RuntimeError("SAFE_STASH_RESTORE_CONFLICT")


def prepare_raspberry_pi_publication() -> dict[str, Any]:
    """Run before benchmarking, while Notebook 11 is still clean on disk."""
    remote = _publication_remote()
    _sync_preserving_allowed_changes(
        remote,
        ("notebooks/final_experiments/11_raspberry_pi_benchmark.ipynb",),
        "rpi-preflight",
    )
    _require_main_identity_and_auth(remote)
    return {"status": "PASS", "remote": PUBLICATION_URL, "branch": "main"}


def publish_raspberry_pi_results() -> dict[str, Any]:
    """Commit/push only the mandatory Raspberry Pi outputs."""
    remote = _publication_remote()
    paths = [REPO / value for value in EDGE_RESULT_FILES]
    _audit_files(paths)
    rpi = _read_json(REPO / "results/edge/rpi_benchmark.json")
    if rpi.get("hardware_verified") != "Raspberry Pi" or rpi.get("checkpoint_sha256") != P0_DEPLOYMENT_SHA256:
        raise RuntimeError("Raspberry Pi result identity is invalid")
    _git("add", "--", *EDGE_RESULT_FILES)
    staged = _git("diff", "--cached", "--name-only", "--diff-filter=ACMR").stdout.splitlines()
    unexpected = sorted(set(staged) - set(EDGE_RESULT_FILES))
    if unexpected:
        raise RuntimeError(f"Edge publication staged unrelated files: {unexpected}")
    if staged:
        if _git("diff", "--cached", "--check", check=False).returncode != 0:
            raise RuntimeError("Edge result git diff check failed")
        _git("commit", "-m", EDGE_COMMIT_MESSAGE)
        status = "COMMITTED"
    else:
        status = "IDENTICAL_RESULTS_ALREADY_COMMITTED"
    _sync_preserving_allowed_changes(
        remote,
        ("notebooks/final_experiments/11_raspberry_pi_benchmark.ipynb",),
        "rpi-push",
    )
    _require_main_identity_and_auth(remote)
    pushed = _git("push", remote, "main", check=False)
    if pushed.returncode != 0:
        raise RuntimeError("GITHUB_AUTH_REQUIRED")
    return {"status": status, "push": "PASS", "remote": PUBLICATION_URL, "staged_files": list(EDGE_RESULT_FILES)}


def retrieve_raspberry_pi_results() -> dict[str, Any]:
    """Safely fetch/rebase Pi results before workstation aggregation."""
    remote = _publication_remote()
    allowed = (
        "notebooks/final_experiments/09_P2_mobilevitv2_imagenet_li_bestreg.ipynb",
        "notebooks/final_experiments/10_final_frozen_test_evaluation.ipynb",
        "notebooks/final_experiments/12_finalize_paper_artifacts_and_publish.ipynb",
        "results/model_registry/", "results/final_test/", "results/paper/",
    )
    _sync_preserving_allowed_changes(remote, allowed, "workstation-edge-pull")
    missing = [value for value in EDGE_RESULT_FILES if not (REPO / value).is_file()]
    if missing:
        raise RuntimeError(f"RPI_BENCHMARK_MISSING: {missing}")
    rpi = _read_json(REPO / "results/edge/rpi_benchmark.json")
    if rpi.get("hardware_verified") != "Raspberry Pi" or rpi.get("checkpoint_sha256") != P0_DEPLOYMENT_SHA256:
        raise RuntimeError("Retrieved Raspberry Pi result identity is invalid")
    return {"status": "PASS", "remote": PUBLICATION_URL, "files": list(EDGE_RESULT_FILES)}


def _publication_files() -> list[Path]:
    roots = ["src", "config", "tests", "docs", "legacy", "data/manifests", "data/splits", "data/checksums", "results/audit", "results/metrics", "results/tables", "results/figures", "results/model_registry", "results/final_test", "results/edge", "results/paper"]
    paths = [REPO / name for name in (".gitattributes", ".gitignore", "README.md", "CITATION.cff", "LICENSE", "environment.yml", "requirements-lock.txt", "pyproject.toml", "notebooks/README.md")]
    paths.extend((REPO / "checkpoints/README.md", REPO / "checkpoints/deployment/README.md"))
    paths.extend(REPO / "notebooks/final_experiments" / name for name in sorted(FINAL_NOTEBOOKS))
    for root in roots:
        base = REPO / root
        if base.exists():
            paths.extend(path for path in base.rglob("*") if path.is_file())
    paths.extend(path for path in (REPO / "scripts").glob("*.py") if path.name != "generate_final_notebooks.py")
    excluded = {
        REPO / "results/audit/environment.json", REPO / "results/audit/system_info.txt",
        REPO / "results/audit/legacy_path_references.csv", REPO / "config/paths.local.yaml",
    }
    paths = [path for path in paths if path not in excluded and "__pycache__" not in path.parts and ".ipynb_checkpoints" not in path.parts]
    paths.append(DEPLOYMENT_CHECKPOINT)
    return sorted(set(paths))


def _audit_files(paths: Iterable[Path]) -> None:
    secret_patterns = [re.compile(pattern) for pattern in (
        r"ghp_[A-Za-z0-9]{20,}", r"github_pat_[A-Za-z0-9_]{20,}", r"AKIA[0-9A-Z]{16}",
        r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----", r"(?i)(?:api[_-]?key|access[_-]?token)\s*[:=]\s*['\"][^'\"]{12,}",
    )]
    for path in paths:
        relative = path.relative_to(REPO)
        if not path.is_file():
            raise RuntimeError(f"Allowlisted publication file is missing: {relative}")
        if path.stat().st_size > 100 * 1024 * 1024:
            raise RuntimeError(f"Publication file exceeds 100 MiB: {relative}")
        suffix = path.suffix.casefold()
        if suffix in {".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
            raise RuntimeError(f"Raw/image file is not allowlisted for publication: {relative}")
        if suffix == ".pth" and (relative.as_posix() != "checkpoints/deployment/P0_FINAL_SOILNET_VICREG_MU27_LI_V4_BESTREG.pth" or sha256_file(path) != P0_DEPLOYMENT_SHA256):
            raise RuntimeError(f"Unapproved checkpoint binary: {relative}")
        if suffix in {".png", ".pdf"} and not relative.as_posix().startswith("results/final_test/figures/"):
            raise RuntimeError(f"Non-paper binary image/document found: {relative}")
        if suffix not in {".pth", ".png", ".pdf"} and path.stat().st_size <= 5 * 1024 * 1024:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(pattern.search(text) for pattern in secret_patterns):
                raise RuntimeError(f"Potential credential/token in publication file: {relative}")


def publish_reproducibility_repository() -> dict[str, Any]:
    """Audit, explicitly stage, commit, dry-run, and push only publication files."""
    validate_publication_inputs()
    static_check = subprocess.run(
        [sys.executable, "scripts/check_final_notebooks.py", "--read-only"],
        cwd=REPO, text=True, capture_output=True,
    )
    unit_tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], cwd=REPO, text=True, capture_output=True,
    )
    if static_check.returncode != 0 or unit_tests.returncode != 0:
        raise RuntimeError("REPOSITORY_VERIFICATION_FAILED")
    files = _publication_files()
    _audit_files(files)
    status_before = _git("status", "--short").stdout.splitlines()
    remote = _publication_remote()
    _require_main_identity_and_auth(remote)
    relative_files = [str(path.relative_to(REPO)) for path in files]
    for offset in range(0, len(relative_files), 100):
        _git("add", "--", *relative_files[offset:offset + 100])
    staged = [REPO / line for line in _git("diff", "--cached", "--name-only", "--diff-filter=ACMR").stdout.splitlines()]
    _audit_files(staged)
    if _git("diff", "--cached", "--check", check=False).returncode != 0:
        raise RuntimeError("git diff --cached --check failed")
    if not staged:
        head = _git("rev-parse", "HEAD", check=False).stdout.strip()
        remote_head = _git("ls-remote", remote, "refs/heads/main", check=False).stdout.split()
        if head and remote_head and remote_head[0] == head:
            return {"status": "ALREADY_PUBLISHED", "commit": head, "remote": PUBLICATION_URL, "repository_tests": "PASS"}
        subject = _git("log", "-1", "--pretty=%s", check=False).stdout.strip()
        if head and subject == FINAL_COMMIT_MESSAGE:
            _sync_preserving_allowed_changes(
                remote,
                ("notebooks/final_experiments/12_finalize_paper_artifacts_and_publish.ipynb",),
                "final-retry",
            )
            _require_main_identity_and_auth(remote)
            pushed = _git("push", remote, "main", check=False)
            if pushed.returncode != 0:
                raise RuntimeError("GITHUB_AUTH_REQUIRED")
            return {"status": "PUBLISHED_EXISTING_FINAL_COMMIT", "commit": head, "remote": PUBLICATION_URL, "repository_tests": "PASS"}
        raise RuntimeError("No staged changes, but HEAD is not the final publication commit")
    _git("commit", "-m", FINAL_COMMIT_MESSAGE)
    commit = _git("rev-parse", "HEAD").stdout.strip()
    _sync_preserving_allowed_changes(
        remote,
        ("notebooks/final_experiments/12_finalize_paper_artifacts_and_publish.ipynb",),
        "final-push",
    )
    commit = _git("rev-parse", "HEAD").stdout.strip()
    _require_main_identity_and_auth(remote)
    pushed = _git("push", remote, "main", check=False)
    if pushed.returncode != 0:
        raise RuntimeError("GITHUB_AUTH_REQUIRED")
    return {
        "status": "PUBLISHED", "commit": commit, "remote": PUBLICATION_URL,
        "legacy_remote_touched": False, "repository_tests": "PASS",
        "status_entries_before_staging": len(status_before),
    }
