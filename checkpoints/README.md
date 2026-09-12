# Frozen checkpoints

`checkpoint_manifest.json` is the authoritative map from experiment/stage to
canonical checkpoint SHA256, byte size, and distribution location.

Two existing deployment checkpoints are stored in Git:

- P0: `checkpoints/deployment/P0_FINAL_SOILNET_VICREG_MU27_LI_V4_BESTREG.pth`
- P1_noLI: `checkpoints/deployment/P1_SOILNET_VICREG_MU27_NO_LI_BESTREG.pth`

The GitHub Release tagged `soilnet-manuscript-repro-v1` carries the other
original frozen files:

- P1_noSSL validation-best checkpoint
- P2 validation-best checkpoint
- P3 validation-best checkpoint
- P3 epoch-80 MobileViTv2 VICReg encoder
- SoilNet VICReg μ=27 initialization used by P0/P1_noLI
- historical pre-VICReg ImageNet snapshot used by P1_noSSL

Download assets into `checkpoints/release-assets/`, then verify them against
`reproducibility/CHECKSUMS.sha256`. No state-dict-only export, quantization, or
weight conversion is used; these are byte-identical canonical checkpoints.
Rolling/resume and epoch-60 supervised snapshots are intentionally excluded.
