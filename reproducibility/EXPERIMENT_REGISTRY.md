# Frozen experiment registry

All supervised experiments use the fixed labeled split of 1,407 train / 289
validation / 231 test records, seed `20260905`, 60 epochs, batch size 32, Adam,
learning rate `1e-4`, and weight decay 0. The validation-selected checkpoint is
the strict minimum of `(SM0_RMSE + SM20_RMSE) / 2`; checkpoint selection does
not use classification or held-out-test metrics.

| experiment_id | architecture | initialization | VICReg | LI | train / validation / test | epochs | batch | optimizer | learning rate | seed | best epoch | checkpoint SHA256 | parameters | evaluation status |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| P0 | SoilNet | ImageNet then frozen VICReg μ=27 | yes | yes | 1407 / 289 / 231 | 60 | 32 | Adam | 0.0001 | 20260905 | 45 | `eba009dfd45ec21174a8e40b16148e0455e902286c7db55933487221da761379` | 3,682,133 | frozen validation and final held-out test |
| P1_noLI | SoilNet | ImageNet then frozen VICReg μ=27 | yes | no sample-dependent LI | 1407 / 289 / 231 | 60 | 32 | Adam | 0.0001 | 20260905 | 53 | `a9d8de995b0673e9ec39bfc4afac00d6a2a773ad9ffbea0027ff2d0c4fab820c` | 3,682,133 | frozen validation and final held-out test |
| P1_noSSL | SoilNet | frozen historical pre-VICReg ImageNet snapshot | no | yes | 1407 / 289 / 231 | 60 | 32 | Adam | 0.0001 | 20260905 | 52 | `50c0f15567ab71a87b04569790b9cc7afd016989e3899d56e1898d8a6fcfb044` | 3,682,133 | frozen validation and final held-out test |
| P2 | MobileViTv2-0.5 (`mobilevitv2_050.cvnets_in1k`) | ImageNet CVNets | no | yes | 1407 / 289 / 231 | 60 | 32 | Adam | 0.0001 | 20260905 | 60 | `0a080d5129a2e57f1c2baf420b5af9aadf913fc0aab0d0050b6d662f24c28a8d` | 1,189,189 | frozen validation and final held-out test |
| P3 | MobileViTv2-0.5 (`mobilevitv2_050.cvnets_in1k`) | ImageNet then experiment-specific VICReg | yes | yes | 1407 / 289 / 231 metadata only | 60 | 32 | Adam | 0.0001 | 20260905 | 58 | `e2119a5b7a289925584fba8d3cd57040b7835461fc39d686bda464227de50020` | 1,189,189 | **validation-only follow-up; test set not opened** |

P3 VICReg pretraining used 11,995 unlabeled images, 80 epochs, batch size 16,
Adam at `1e-4`, and the frozen encoder SHA256
`32a1a62233d39bf6d5de5f803e7781be81e6a96851efc1531f87ec6f80ec5580`.

## Identity map

| ID | Config | Frozen run artifacts | Checkpoint distribution |
|---|---|---|---|
| P0 | `config/experiments/P0_final_soilnet_v4_bestreg.yaml` | `results/frozen/P0/` | Git checkout |
| P1_noLI | `config/experiments/P1_no_li_bestreg.yaml` | `results/frozen/P1_noLI/` | Git checkout |
| P1_noSSL | `config/experiments/P1_no_ssl_bestreg.yaml` | `results/frozen/P1_noSSL/` | GitHub Release asset |
| P2 | `config/experiments/P2_mobilevitv2_imagenet_li_bestreg.yaml` | `results/frozen/P2/` | GitHub Release asset |
| P3 | source config plus frozen `results/frozen/P3/resolved_config.yaml` | `results/frozen/P3/` | GitHub Release assets |

`results/frozen/*/run_metadata.json` files are public path-normalized views of
the frozen metadata. Only machine-local path strings changed; numerical values
did not. Original source hashes are preserved in
`frozen_artifact_inventory_before_packaging.sha256`.
