# Frozen P0 deployment checkpoint

This directory contains exactly one model binary: the frozen, validation-selected
P0 checkpoint used by the Raspberry Pi latency notebook. It is included so that
edge benchmarking does not require the training dataset or a Windows `/mnt/d`
mount.

- Experiment: `P0_FINAL_SOILNET_VICREG_MU27_LI_V4_BESTREG`
- Selection: minimum validation `(SM0_RMSE + SM20_RMSE) / 2`
- Selected epoch: 45
- SHA256: `eba009dfd45ec21174a8e40b16148e0455e902286c7db55933487221da761379`
- Size: below GitHub's 100 MiB per-file limit

Notebook 11 recomputes the SHA256 and refuses to benchmark a different file.
No P1/P2, epoch-60, resume, SSL, or historical checkpoint is published here.
