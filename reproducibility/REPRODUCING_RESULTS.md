# Reproducing the frozen SoilNet results

The fast path verifies reported analyses from frozen predictions. Checkpoint
inference is separate and requires lawful local access to the raw images.
Retraining is optional, expensive, and is not required to verify reported
metrics.

## 1. Create the environment

```bash
conda env create -f environment/soilnet.yml
conda run -n soilnet pip install -r environment/requirements-lock.txt
```

The pinned versions and historical CUDA environment are described in
`environment/README.md`. Cross-device bitwise identity is not claimed.

## 2. Verify dataset manifests

```bash
conda run -n soilnet python scripts/verify_reproducibility_release.py
sha256sum data/manifests/labeled_manifest.csv \
  data/splits/train_manifest.csv \
  data/splits/validation_manifest.csv \
  data/splits/test_manifest.csv \
  data/manifests/unlabeled_manifest.csv
```

Expected counts are 1,927 labeled, 1,407 train, 289 validation, 231 test, and
11,995 unlabeled. Raw images are withheld; see `data/DATA_PROVENANCE.md`.

## 3. Download and verify canonical checkpoints

P0 and P1_noLI are already in the checkout. Download the remaining original
checkpoint files from the GitHub Release:

```bash
mkdir -p checkpoints/release-assets
gh release download soilnet-manuscript-repro-v1 \
  --repo husphys-gif/SoilNet \
  --dir checkpoints/release-assets
sha256sum -c reproducibility/CHECKSUMS.sha256
```

`checkpoints/checkpoint_manifest.json` records each checkpoint role, byte size,
SHA256, and public location. These are canonical original checkpoint files, not
deployment exports.

## 4. Instantiate and strict-load P0/P1/P2/P3

These commands do not create a dataset or run inference:

```bash
conda run -n soilnet python scripts/reproduce_checkpoint_evaluation.py \
  --experiment P0 --instantiate-only
conda run -n soilnet python scripts/reproduce_checkpoint_evaluation.py \
  --experiment P1_noLI --instantiate-only
conda run -n soilnet python scripts/reproduce_checkpoint_evaluation.py \
  --experiment P1_noSSL --instantiate-only
conda run -n soilnet python scripts/reproduce_checkpoint_evaluation.py \
  --experiment P2 --instantiate-only
conda run -n soilnet python scripts/reproduce_checkpoint_evaluation.py \
  --experiment P3 --instantiate-only
```

Each command verifies SHA256 and uses strict state-dict loading.

## 5. Reproduce evaluation from frozen checkpoints

Set DATA_ROOT to a lawful local reconstruction matching the public manifest.
Write outputs to a new directory; never overwrite `results/frozen/`.

```bash
export DATA_ROOT=/path/to/verified/soil_data_set

# Validation is available for every model.
for model in P0 P1_noLI P1_noSSL P2 P3; do
  conda run -n soilnet python scripts/reproduce_checkpoint_evaluation.py \
    --experiment "$model" --split validation --data-root "$DATA_ROOT" \
    --output-dir "reproduced-results/checkpoint-evaluation/$model"
done

# Held-out reproduction is restricted to models already final-tested.
for model in P0 P1_noLI P1_noSSL P2; do
  conda run -n soilnet python scripts/reproduce_checkpoint_evaluation.py \
    --experiment "$model" --split test --data-root "$DATA_ROOT" \
    --output-dir "reproduced-results/checkpoint-evaluation/$model"
done
```

The CLI refuses `--experiment P3 --split test` before it resolves a checkpoint
or constructs a dataset. **P3: validation-only follow-up; test set not opened.**

## 6. Reproduce metrics and paired bootstrap without images

```bash
conda run -n soilnet python scripts/reproduce_frozen_analyses.py \
  --output-dir reproduced-results/frozen-analysis
```

This recomputes validation metrics for P0/P1_noLI/P1_noSSL/P2/P3, existing test
metrics for P0/P1_noLI/P1_noSSL/P2, and the 10,000-replicate paired held-out
bootstrap. It reads frozen prediction CSVs only.

## 7. Reproduce the model-complexity table

The same command writes
`reproduced-results/frozen-analysis/model_complexity.csv` and verifies the
P0/P1/P2 rows against `results/paper/08_model_complexity.csv`. P3 complexity
comes from its frozen metadata and is appended without changing the manuscript
table.

## 8. Reproduce factorial validation analysis

`scripts/reproduce_frozen_analyses.py` also writes the 10,000-replicate
factorial validation bootstrap and verifies it against
`results/p3_analysis/vicreg_paired_bootstrap.csv`. For a notebook presentation,
open and run:

```bash
conda run -n soilnet jupyter notebook \
  notebooks/final_experiments/14_vicreg_architecture_interaction_analysis.ipynb
```

Notebook 14 uses only `results/frozen/` validation artifacts and writes only
factorial analysis outputs. It does not load a test dataset.

## Retraining from source is a different operation

Retraining is not part of verification above. It requires raw images, the
downloaded initialization checkpoints, CUDA, and new empty output directories.
The clean P0/P1/P2 notebooks and P3 CLI preserve the frozen recipes:

- P0: `01_P0_final_soilnet_v4_bestreg.ipynb`
- P1_noLI: `02_P1_no_li_bestreg.ipynb`
- P1_noSSL: `03_P1_no_ssl_bestreg.ipynb`
- P2: `09_P2_mobilevitv2_imagenet_li_bestreg.ipynb`
- P3: `python scripts/run_p3_mobilevitv2_vicreg.py --run-ssl-pretraining`

Use a new RUN_ARTIFACT_ROOT. Never point retraining at the frozen release
directories. Retraining may reproduce the protocol but is not expected to be
byte-identical across GPU/software platforms.
