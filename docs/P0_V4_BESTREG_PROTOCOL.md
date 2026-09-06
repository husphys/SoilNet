# P0-v4 best-regression protocol

## Status and reason for this revision

`P0-v4-bestreg` is a prospective checkpoint-selection revision prepared after
P0-v3 completed its locked 60-epoch run. P0-v3 completed normally, but its
epoch-60 validation results showed overfitting relative to the historical
validation trajectory. Because P0-v3 retained only `epoch_60_final.pth`, its
historical intermediate epochs cannot be selected retrospectively.

P0-v4 therefore performs one new full 60-epoch run and changes only checkpoint
selection and artifact provenance. It does not change the data manifest, split,
seed, architecture, VICReg μ=27 initialization, optimizer, learning rate,
weight decay, batch size, losses, transforms, AMP setting, or epoch budget. It
does not use early stopping.

No claim is made that P0-v4 is better than P0-v3 before P0-v4 is run.

## Prospective primary endpoint

After validation at every epoch, compute regression RMSE on the original 0–100
percentage-point scale and define:

```text
mean_validation_regression_RMSE = (SM0_RMSE + SM20_RMSE) / 2
```

The primary checkpoint is the epoch with the minimum
`mean_validation_regression_RMSE`. Classification metrics are recorded as
secondary metrics and are never used for checkpoint selection. Equal values do
not replace the earlier checkpoint because updates require a strict decrease.

Training always continues through epoch 60. The run retains both:

- `validation_best_regression.pth` — primary model-selection artifact
- `epoch_60_final.pth` — endpoint/protocol artifact

After epoch 60, the primary checkpoint is reloaded before
`validation_predictions.csv` and `validation_metrics.json` are generated. The
run metadata records the selected epoch, both checkpoint paths and hashes, the
selection formula, and the checkpoint used for final validation artifacts.

## Identity and resume firewall

- Experiment ID: `P0_FINAL_SOILNET_VICREG_MU27_LI_V4_BESTREG`
- Output directory:
  `/mnt/d/check point/soilnet_final_runs/P0_FINAL_SOILNET_VICREG_MU27_LI_V4_BESTREG/`
- SSL checkpoint SHA256:
  `42599ac025b8d8c8c5b8cca36624665f155e0ff588cd90de2fd47b08cd2d60d8`
- Manifest SHA256:
  `8ff45054d4b8e3df9758d0c112dc16719572b2267906fb3a5ed5b3262a6732bd`
- Split SHA256:
  `8927b8223b8c4c234d264ad9ea62ac2df6161124a79787a2e71eeb5cd23eae2f`

Resume accepts only the rolling checkpoint in this v4 output directory and
requires exact experiment, architecture, config, manifest, split, and SSL
checkpoint identity. It cannot resume from the P0-v3 directory. If no v4
rolling checkpoint exists, initialization begins from the locked VICReg μ=27
checkpoint.

## Test firewall

This revision was defined without constructing a test dataset or loader,
calculating test metrics, or inspecting test predictions. The training path
constructs only train and validation loaders, and `test_evaluated` remains
`NO`. Notebook 09 remains disabled and is outside this protocol-preparation
step.
