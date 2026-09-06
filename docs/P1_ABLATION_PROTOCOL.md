# P1 ablation protocol

## Purpose and design

The P1 experiments isolate one factor at a time from the completed P0-v4
BESTREG reference. Both use the same locked supervised protocol and prospective
validation-only checkpoint rule. They do not reuse P0-v4's best epoch or
validation values for selection.

| Experiment | Image | LI | ImageNet initialization | VICReg μ=27 |
|---|:---:|:---:|:---:|:---:|
| P0-v4 | ✓ | ✓ | ✓ | ✓ |
| P1-noLI | ✓ | ✗ | ✓ | ✓ |
| P1-noSSL | ✓ | ✓ | ✓ | ✗ |

Held constant: the clean manifest and split, seed `20260905`, deterministic
resize/normalization, regression and classification targets, loss weights,
SoilNet image branch, two regression outputs, ten-class output, 60 epochs,
batch size 32, Adam, learning rate `1e-4`, weight decay 0, no AMP, metric
implementation, original 0–100 regression metric scale, artifact schema, and
provenance requirements.

## P1-noLI semantics

`P1_SOILNET_VICREG_MU27_NO_LI_BESTREG` loads the exact selected VICReg μ=27
checkpoint with SHA256
`42599ac025b8d8c8c5b8cca36624665f155e0ff588cd90de2fd47b08cd2d60d8`.
The image branch and dual-head checkpoint shapes remain identical to P0-v4.
The model forward path does not consume the LI tensor: it replaces the 32-D
sample-dependent LI embedding with a 32-D zero control before the unchanged
heads. Thus no feature substitutes for LI, no hidden dimension is increased,
and checkpoint coverage/head capacity are not confounded with LI availability.

## P1-noSSL initialization provenance

`P1_SOILNET_IMAGENET_LI_NO_SSL_BESTREG` never loads a VICReg checkpoint. It
loads the exact state immediately before the selected original μ27 VICReg run:

`DATA_ROOT:Soil_Labeled_Data/soilNet_di chuyen tu o C/Model/SoilNet_orginal.pth`

SHA256:
`7fbfac2b637613c6f48b3fef6f2c56be07080eedfba73da1d2b553e0b41f3631`.

Historical `Model/Create_SoilNet.py` created this snapshot immediately after
constructing SoilNet and before any training. The selected μ27 notebook then
loaded this same file before VICReg. Its two MobileNetV2 segments originate
from `timm/mobilenetv2_100.ra_in1k`, and its MobileViTv2 segment originates
from `timm/mobilevitv2_050.cvnets_in1k`, all with ImageNet pretrained weights.
The initial convolution, adapters, final convolution, LI projection, and both
task heads were newly initialized when the snapshot was created. Loading the
hash-locked snapshot preserves their exact historical realization and avoids a
version- or RNG-dependent reconstruction. LI remains enabled exactly as in
P0-v4.

This controls removal of the VICReg stage, not removal of ImageNet
initialization. It does not claim that ImageNet and VICReg effects are
independent of the historical architecture or initialization snapshot.

## Checkpoint and execution policy

After every epoch, both runs compute:

```text
mean_validation_regression_RMSE = (SM0_RMSE + SM20_RMSE) / 2
```

A strict decrease replaces `validation_best_regression.pth`. Classification
metrics are secondary and never select checkpoints. Training never early-stops
and always proceeds through epoch 60, when `epoch_60_final.pth` is also saved.
The best checkpoint is then reloaded before final validation predictions and
metrics are exported. Both checkpoint SHA256 values are recorded explicitly.

Each notebook is a sequential clean-kernel Run-All workflow. It raises before
training if config, hashes, initialization, CUDA, temporary-model preflight, or
no-step assertions fail. A completed output directory is never overwritten;
an incomplete run can resume only when experiment, architecture, config,
manifest, split, SSL/initialization checkpoint, and seed identities match.

## Test firewall and limitation

The P1 notebooks and shared training path construct only train and validation
loaders. They do not read test images, compute test metrics or predictions, or
invoke notebook 09. `test_evaluated` remains `NO`.

These are single-seed ablations. Any eventual difference is conditional on the
locked seed, split, architecture, and optimization protocol; it does not by
itself quantify between-seed uncertainty or prove a general causal effect.
