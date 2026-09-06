# Reproducibility protocol

## Preconditions

Work from a WSL-native checkout. Configure local roots in the ignored
`config/paths.local.yaml` or environment variables. Do not modify the dataset,
checkpoint root, or pinned legacy checkout.

## Exact audit commands

```bash
conda activate soilnet
python scripts/capture_environment.py
python scripts/smoke_models.py
python scripts/audit_legacy.py
python scripts/audit_dataset.py --refresh
python scripts/audit_result_artifacts.py
python scripts/audit_checkpoints.py
python scripts/build_final_report.py
pytest -q
python scripts/verify_artifacts.py
git diff --check
```

Expected outputs are the CSV/JSON inventories under `data/` and
`results/audit/`, ending with `verification.json` status `PASS`. The commands
perform no training. `audit_dataset.py` and `audit_checkpoints.py` may take
minutes because they hash every source byte.

## Evaluation metadata

Every new metric CSV must have adjacent JSON recording Git commit, UTC
timestamp, Python/PyTorch/CUDA/cuDNN/GPU, seed, config, dataset manifest hash,
split manifest hash, and checkpoint SHA256. Use
`soilnet.reproducibility.result_metadata`.

For regression metrics reported in percentage points, use absolute tolerance
`1e-4` when rerunning on the same stored predictions. For classification ratios
use absolute tolerance `1e-6` on the same predictions. A full GPU forward pass
may require wider, explicitly reported tolerances. Bitwise identity across GPU,
driver, CUDA, cuDNN, or platform versions is not promised.

## Hardware notes

The audit was completed in WSL2 on CPU because the operating system blocked
NVML/GPU access. The tested PyTorch wheels include a CUDA 11.8 runtime, but a
system CUDA Toolkit is neither required nor installed by this repository. See
`results/audit/system_info.txt` and `results/audit/environment.json` for the
captured kernel, driver/GPU availability, package versions, and disk state.
CPU-only audit and smoke commands are supported; any later GPU evaluation must
record its actual GPU, driver, CUDA, and cuDNN metadata next to its result CSV.

## Minimal recomputation policy

1. Reuse only hash-verified SSL checkpoints and feature extractors.
2. Regenerate tables from stored CSV only after evaluation scope is proven.
3. If frozen embeddings exist, rerun KNN/GBM/SVM/Decision Tree/MLP without
   retraining the backbone.
4. Reevaluate a checkpoint only on data proven unseen by manifest/hash/group.
5. If no unseen set exists, rerun one final selected fine-tuning experiment on
   the prospective group split. Do not rerun the full mu sweep by default.
