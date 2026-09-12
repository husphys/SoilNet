# SoilNet reproducibility pre-release audit

Audit date: 2026-09-12 UTC

Overall status: **PASS**

No training, model forward pass, raw-image evaluation, or new held-out-test
access occurred during packaging. P3 remained validation-only.

| Gate | Status | Evidence |
|---|---|---|
| Initial Git state captured before copy/edit | PASS | `pre_release_git_state.txt`; detached HEAD and pre-existing P3 worktree changes recorded |
| Remote freshness | PASS | fetched `origin/main`; local base and `origin/main` both `0a9438928aeac62ec0d8255b2b402c03dba6cf67`; no merge/rebase required |
| Pre-packaging frozen-artifact inventory | PASS | `frozen_artifact_inventory_before_packaging.sha256` |
| Frozen P0/P1/P2/P3 artifacts unchanged | PASS | 107 baseline files re-hashed successfully |
| Final experiment registry | PASS | five entries P0, P1_noLI, P1_noSSL, P2, P3 with required protocol/identity fields |
| Required source/config/notebook structure | PASS | `src/soilnet/{models,training,utils}`, `config/experiments`, and ten final notebooks present |
| Notebook static validation | PASS | 10 notebooks / 13 configs; 0 executed public-source cells; 0 test loaders instantiated |
| Clean/executed notebook separation | PASS | 10 clean sources and 7 path-normalized executed provenance copies; both hashes recorded |
| Final artifact coverage | PASS | histories, validation artifacts, final-tested P0/P1/P2 test artifacts, bootstrap, complexity, Pi benchmark, and evidence matrix indexed |
| Stale/rolling/interrupted artifact exclusion | PASS | no resume/rolling checkpoint or stale P3 archive copied into `results/frozen` or checkpoint distribution |
| Checkpoint byte identity | PASS | 2 Git checkpoints plus 6 staged GitHub Release assets verified against the manifest |
| Strict checkpoint loading | PASS | P0, P1_noLI, P1_noSSL, P2, and P3 instantiated and strict-loaded; no dataset/forward pass |
| Public checksum inventory | PASS | all repository critical artifacts and six staged Release assets verified |
| Labeled dataset identity/count | PASS | 1,927 unique sample IDs; 1,407 / 289 / 231 split |
| Unlabeled manifest | PASS | 11,995 relative paths with SHA256 |
| Raw-data license/provenance gate | PASS-WITH-WITHHOLDING | no raw image published; unverified labeled/unlabeled redistribution rights disclosed |
| P3 test firewall | PASS | `TEST_SET_OPENED=False`; no P3 test artifact; CLI rejects P3 test before checkpoint/dataset access |
| Prediction-level metric reproduction | PASS | validation for P0/P1/P2/P3 and existing held-out evidence for P0/P1/P2 reproduced from frozen CSVs |
| Paired bootstrap reproduction | PASS | 10,000 held-out paired replicates and 10,000 factorial validation replicates matched frozen outputs |
| P0/P1/P2 complexity reproduction | PASS | published four-row table verified; P3 complexity appended only in reproduced output |
| Secret/private-key scan | PASS | repository text scan found no GitHub token, Telegram token value, API key, password value, or private SSH key |
| New public-copy machine paths | PASS | no `/home/diy-hus`, `/mnt/d`, or `/mnt/e` in public notebooks, executed copies, or `results/frozen` |
| Full automated test suite | PASS | `64 passed` |
| Fresh-clone smoke | PASS | temporary Git snapshot cloned to a new directory; checksum audit and frozen-analysis reproduction both passed |

## Known limitations

- Raw labeled and unlabeled images are not redistributed. Reviewer checkpoint
  inference therefore requires lawful local reconstruction against manifests.
- Some pre-existing historical files under `results/audit/` retain local/legacy
  path evidence. They predate this package, are intentionally preserved as audit
  provenance, and are excluded from normalized public run copies.
- Six large canonical files are GitHub Release assets rather than Git blobs.
  Their byte-identical local staging passed before publication.
- Notebook 14 had no executed notebook output to preserve; its frozen CSV/JSON
  outputs are available under `results/p3_analysis/`.
- The environment is pinned and tested, but bitwise equality across GPU models
  and CUDA platforms is not claimed.
- Raspberry Pi radio/hardware utilities may require platform-specific packages
  outside the manuscript analysis environment; they are not required for metric
  reproduction.

## Publication gate

All local reproducibility gates are PASS. Commit/push/tag/release publication is
authorized by the request, provided the branch remains fast-forward and GitHub
authentication succeeds. Force-push and history rewriting are prohibited.
