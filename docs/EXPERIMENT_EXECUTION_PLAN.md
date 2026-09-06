# Final P0-P2 experiment execution plan

P0 and both P1 ablations are completed and frozen. Their external artifacts
must not be changed or rerun.

| Order | Notebook | Machine | Purpose | Training | Test access | Output |
|---:|---|---|---|---|---|---|
| 1 | `09_P2_mobilevitv2_imagenet_li_bestreg.ipynb` | CUDA workstation | Fair MobileViTv2 architecture control and four-model freeze | 60 epochs | No | External P2 directory; `results/model_registry/` |
| 2 | `10_final_frozen_test_evaluation.ipynb` | Workstation with raw data and frozen checkpoints | One-time four-model held-out inference, paired bootstrap, figures, lock | No | Yes, exactly once | `results/final_test/` |
| 3 | `11_raspberry_pi_benchmark.ipynb` | Actual Raspberry Pi | Frozen P0 CPU latency, then allowlisted edge commit/push | No | No | `results/edge/` and publication `main` |
| 4 | `12_finalize_paper_artifacts_and_publish.ipynb` | Workstation | Safely fetch/rebase Pi results, restore local experiment outputs, aggregate, audit, and publish | No | No | `results/paper/`, documentation, publication `main` |

The bootstrap repository is pushed before experiments and includes the P0
deployment binary. The Pi starts from that repository, Notebook 11 pushes its
edge-only commit, and Notebook 12 retrieves it automatically. There is no
manual checkpoint or benchmark-result copy step.

Notebook 09 accesses train and validation only. Notebook 10 refuses to create
the sole test dataset/loader until all four immutable checkpoint identities and
validation artifacts pass. Notebook 11 refuses non-Raspberry-Pi hardware.
Notebook 12 refuses missing test or Pi evidence and performs no model inference.
