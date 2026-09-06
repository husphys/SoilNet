# SoilNet reproducibility repository

This repository contains the evidence-first, leakage-controlled workflow for the
final SoilNet manuscript experiments. It preserves historical provenance while
keeping prospective validation, the one-time held-out test, and deployment
measurements explicitly separate.

## Final contribution

The frozen experiment set isolates three questions:

- P0: full SoilNet with light intensity (LI) and VICReg mu=27.
- P1-noLI: the same system without a sample-dependent LI signal.
- P1-noSSL: the same SoilNet architecture with the exact historical pre-VICReg
  ImageNet snapshot, LI, and no VICReg.
- P2-MobileViTv2: `mobilevitv2_050.cvnets_in1k` with ImageNet initialization,
  the same 32-dimensional LI fusion and dual heads, and no VICReg. Its fair
  architecture comparison is P1-noSSL versus P2.

All supervised runs use 60 epochs, batch size 32, Adam at `1e-4`, weight decay
0, seed 20260905, and no early stopping. Strictly minimum validation
`(SM0_RMSE + SM20_RMSE) / 2` selects `validation_best_regression.pth`;
classification never selects a checkpoint.

## Dataset QC and split

The final dataset contains 1,927 clean unique labeled images after excluding
conflicting exact-image groups. The fixed split is 1,407 train / 289 validation
/ 231 held-out test, with SHA256
`8927b8223b8c4c234d264ad9ea62ac2df6161124a79787a2e71eeb5cd23eae2f`.
Regression metrics use the original 0-100 percentage-point scale.

Raw images are not included. Configure an ignored `config/paths.local.yaml`
from `config/paths.example.yaml`, or set `DATA_ROOT`, `CHECKPOINT_ROOT`, and
`RUN_ARTIFACT_ROOT`.

## Final notebook order

1. `01_P0_final_soilnet_v4_bestreg.ipynb` — completed/frozen on the CUDA workstation.
2. `02_P1_no_li_bestreg.ipynb` — completed/frozen on the CUDA workstation.
3. `03_P1_no_ssl_bestreg.ipynb` — completed/frozen on the CUDA workstation.
4. `09_P2_mobilevitv2_imagenet_li_bestreg.ipynb` — train/validate P2 on the CUDA workstation; never accesses test.
5. `10_final_frozen_test_evaluation.ipynb` — evaluate the four frozen models once on the CUDA workstation; the only test-opening notebook.
6. `11_raspberry_pi_benchmark.ipynb` — benchmark frozen P0 on actual Raspberry Pi hardware and automatically commit/push only `results/edge/`; no dataset access.
7. `12_finalize_paper_artifacts_and_publish.ipynb` — safely fetch/rebase the Pi commit, restore local experiment outputs, aggregate, and publish; no training/evaluation.

Restart the `soilnet` kernel and use **Run All** for each remaining notebook in
order. Clone/pull the bootstrap repository on the Pi once and configure its Git
identity/authentication before Notebook 11. No benchmark-result files are copied
manually: Notebook 11 pushes them and Notebook 12 retrieves them automatically.

## Test-lock policy

Notebook 10 validates the exact four-model registry, checkpoint hashes, split,
counts, validation artifacts, and immutability before constructing one shared
test dataset/loader. It writes `results/final_test/FINAL_TEST_LOCK.json` last.
After that lock exists, no training or model change may use test outcomes.

## Result map

- `results/model_registry/`: four frozen validation-selected models.
- `results/final_test/`: per-sample predictions, metrics, paired bootstrap,
  figures, and final test lock.
- `results/edge/`: actual Raspberry Pi environment and latency measurements.
- `results/paper/`: numbered manuscript-ready tables and consolidated summary.
- `results/audit/`: dataset, checkpoint, and provenance audits.

## Raspberry Pi benchmark

The single frozen P0 deployment checkpoint is deliberately bundled at
`checkpoints/deployment/P0_FINAL_SOILNET_VICREG_MU27_LI_V4_BESTREG.pth`
(SHA256 `eba009dfd45ec21174a8e40b16148e0455e902286c7db55933487221da761379`).
No `/mnt/d` mount or separate checkpoint transfer is required. Notebook 11
refuses to run unless Raspberry Pi hardware is detected. It stages exactly five
allowlisted edge-result files, safely rebases, and pushes them to `main`.

## Environment and verification

Create the pinned environment with `environment.yml`/`requirements-lock.txt`,
then run:

```bash
conda run -n soilnet python scripts/check_final_notebooks.py --read-only
conda run -n soilnet pytest -q
```

Publication targets only `https://github.com/husphys-gif/SoilNet.git`. The
legacy `https://github.com/diy-hus/SoilNet` repository is historical provenance
and is never a push target. Notebook 12 uses an explicit staging allowlist and
will stop for missing Git identity/authentication without printing credentials.

## Citation and license

See `CITATION.cff` for software citation metadata and `LICENSE` for the MIT
license. Cite the associated SoilNet paper when bibliographic details become
available; this repository does not invent missing paper metadata.
