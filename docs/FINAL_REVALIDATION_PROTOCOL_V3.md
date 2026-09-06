# Final revalidation protocol v3

## Status

`PROTOCOL_REVISED_PRE_TEST`

This revision was frozen before any prospective test loader, prediction, or
metric was created. It does not overwrite or reinterpret
`results/audit/final_run_lock_v2.json`.

## Evidence and revision decision

| Item | Epochs | Batch | Optimizer | Learning rate | Weight decay |
|---|---:|---:|---|---:|---:|
| `OLD_MANUSCRIPT_PROTOCOL` | 15 | 32 | Adam | 5e-4 | 0 |
| `HISTORICAL_CODE_EVIDENCE` | 60 | 32 | Adam | 1e-4 | not specified |
| `FINAL_REVALIDATION_V3` | 60 | 32 | Adam | 1e-4 | 0 |

Reason: the final protocol was revised before held-out test evaluation to match
the strongest surviving audited GitHub fine-tuning implementation evidence.
Weight decay remains zero from the already locked manuscript protocol because
no historical code evidence authorizes a different value.

The data-root notebook named `Finetune_VicReg_mu=27.ipynb` is a distinct
surviving branch (100 epochs, batch 16, Adam 1e-4). It is not silently treated
as the 60/32 GitHub source. The GitHub repository is now empty, so its original
notebook cannot currently be re-read; the 60/32 values are preserved from the
completed audit and the explicit pre-test revision decision.

## Immutable scientific identities

- Experiment: `P0_FINAL_SOILNET_VICREG_MU27_LI`
- Architecture: SoilNet, image plus LI, two regression outputs and ten classes
- Seed: `20260905`, retained from the pre-test Phase 3B v2 lock
- Manifest SHA256:
  `8ff45054d4b8e3df9758d0c112dc16719572b2267906fb3a5ed5b3262a6732bd`
- Split SHA256:
  `8927b8223b8c4c234d264ad9ea62ac2df6161124a79787a2e71eeb5cd23eae2f`
- Split counts: train 1,407; validation 289; sealed test 231
- SSL checkpoint: `checkpoints_VicReg_original_NEW/vicreg_model_final_mu_27.0.pth`
- SSL SHA256:
  `42599ac025b8d8c8c5b8cca36624665f155e0ff588cd90de2fd47b08cd2d60d8`
- SSL lineage: `VERIFIED_LINEAGE`; no SSL rerun or alternate checkpoint

Targets use `SM_0 / 100`, `SM_20 / 100`, and `LI / 100` internally. Loss is
`MSELoss + CrossEntropyLoss` with unit weights. Validation regression metrics
are restored to the original 0–100 percentage-point scale.

## Training and artifact policy

The primary manuscript checkpoint is `epoch_60_final.pth`; validation-best is
optional and secondary. A single `last_resume_checkpoint.pth` is overwritten
after each epoch and removed after successful epoch 60. Resume is rejected
unless experiment ID, architecture, config, manifest, split, and SSL hashes all
match. A completed run is not automatically repeated for a better validation
number.

Full training requires CUDA and never falls back to CPU or a smaller batch.
The P0 notebook creates train and validation loaders only. Prospective test
access remains exclusively behind the disabled notebook 09 gate.

## Environment interpretation

Historical surviving GitHub notebook evidence records Python 3.10.16, PyTorch
2.6.0+cu118, and CUDA available. The current revalidation environment is Python
3.11.15, PyTorch 2.6.0+cu118, torchvision 0.21.0+cu118, and timm 1.0.29.
The current timm version is not claimed to be the original historical version.
It is a tested compatible reproduction environment because model construction,
strict 577/577 checkpoint loading, forward/backward smoke, and tests pass.
Bitwise reproducibility across hardware is not claimed.
