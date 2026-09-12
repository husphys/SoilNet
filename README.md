# SoilNet manuscript reproducibility release

This repository freezes the code, configurations, data manifests, checkpoints,
predictions, and analyses used by the final SoilNet experiment set. The task is
joint prediction of soil moisture at the surface (SM-0) and at 20 cm depth
(SM-20), with an auxiliary moisture-class output.

The registry contains five frozen entries:

- **P0:** SoilNet + VICReg + light-intensity input (LI).
- **P1_noLI:** SoilNet + VICReg without sample-dependent LI.
- **P1_noSSL:** SoilNet + ImageNet initialization + LI, without VICReg.
- **P2:** MobileViTv2 + ImageNet initialization + LI.
- **P3:** MobileViTv2 + VICReg + LI; validation-only follow-up.

P0, P1_noLI, P1_noSSL, and P2 have frozen held-out-test evidence. **P3 is
validation-only; its test set was not opened.** The repository reports the
factorial validation analysis without extending P3 into held-out evaluation.
It does not make a general superiority or lightweightness claim.

## What is frozen

- [Experiment registry](reproducibility/EXPERIMENT_REGISTRY.md)
- [Artifact index](reproducibility/ARTIFACT_INDEX.md)
- [Reproduction guide](reproducibility/REPRODUCING_RESULTS.md)
- [Checksums](reproducibility/CHECKSUMS.sha256)
- [Data provenance and redistribution decision](data/DATA_PROVENANCE.md)
- [Pre-release audit](reproducibility/PRE_RELEASE_AUDIT.md)

Clean notebook sources are under `notebooks/final_experiments/`. Path-normalized
executed copies retained for provenance are under
`reproducibility/executed_notebooks/`. Final run artifacts are indexed under
`results/frozen/`; statistical outputs remain under `results/final_test/` and
`results/p3_analysis/`.

## Dataset boundary

The labeled manifest contains 1,927 unique image-measurement pairs split into
1,407 train, 289 validation, and 231 held-out test samples. The VICReg unlabeled
pool manifest contains 11,995 images. Both manifests use stable relative paths,
sample identifiers where applicable, and per-file SHA256 values.

Raw images are **not** distributed because redistribution rights have not been
verified for the complete labeled and unlabeled collections. In particular,
third-party/web-derived provenance within the unlabeled pool is not licensed
here. See `data/DATA_PROVENANCE.md`; do not describe this release as making all
data publicly available.

## Checkpoints

The P0 and P1_noLI inference checkpoints are stored directly in the repository.
The other canonical checkpoints and retraining initializations are distributed
as assets of GitHub Release `soilnet-manuscript-repro-v1` to keep large binaries
out of normal Git history. No checkpoint was converted or re-exported. See
`checkpoints/checkpoint_manifest.json` and `checkpoints/README.md`.

## Quick verification without retraining

Create the environment, then recompute all frozen prediction-level metrics and
bootstrap analyses:

```bash
conda env create -f environment/soilnet.yml
conda run -n soilnet pip install -r environment/requirements-lock.txt
conda run -n soilnet python scripts/verify_reproducibility_release.py
conda run -n soilnet python scripts/reproduce_frozen_analyses.py \
  --output-dir reproduced-results
```

This analysis command does not instantiate a dataset or open image files. It
uses the already frozen prediction CSVs; the P3 test boundary remains closed.
Checkpoint-based inference and optional retraining instructions are explicitly
separated in `reproducibility/REPRODUCING_RESULTS.md`.

## Final notebook sequence

The clean source notebooks retain the original workflow order:

1. `01_P0_final_soilnet_v4_bestreg.ipynb`
2. `02_P1_no_li_bestreg.ipynb`
3. `03_P1_no_ssl_bestreg.ipynb`
4. `09_P2_mobilevitv2_imagenet_li_bestreg.ipynb`
5. `10_final_frozen_test_evaluation.ipynb` — existing P0/P1/P2 test evidence only
6. `11_raspberry_pi_benchmark.ipynb`
7. `12_finalize_paper_artifacts_and_publish.ipynb`
8. `13_P3_mobilevitv2_vicreg_bestreg.ipynb` — validation only
9. `14_vicreg_architecture_interaction_analysis.ipynb` — frozen validation analysis

## Embedded prototype

The repository includes an embedded Raspberry Pi prototype and frozen benchmark
evidence under `deployment/`, `firmware/`, and `results/edge/`. These artifacts
document a prototype implementation; they do not establish a generalized causal
irrigation-saving claim.

## License and citation

Software is released under the [MIT License](LICENSE). This software license
does not grant redistribution rights for withheld source images. See
`CITATION.cff` for citation metadata.
