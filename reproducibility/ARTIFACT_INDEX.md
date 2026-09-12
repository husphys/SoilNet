# Final artifact index

| Evidence | Public location | Scope |
|---|---|---|
| Labeled manifest | `data/manifests/labeled_manifest.csv` | 1,927 unique labeled pairs |
| Train / validation / test manifests | `data/splits/{train,validation,test}_manifest.csv` | 1,407 / 289 / 231 |
| Unlabeled manifest | `data/manifests/unlabeled_manifest.csv` | 11,995 entries; raw files withheld |
| Histories, validation predictions/metrics, metadata, resolved configs | `results/frozen/<ID>/` | Canonical non-temporary run artifacts |
| P0/P1/P2 test predictions and metrics | `results/frozen/{P0,P1_noLI,P1_noSSL,P2}/test_*` | Existing held-out evidence only |
| P3 pretraining artifacts | `results/frozen/P3/pretraining/` | History, metadata, manifest, checkpoint hashes |
| P3 supervised artifacts | `results/frozen/P3/supervised/` | History, validation predictions/metrics, metadata, hashes |
| P3 P2 comparison | `results/frozen/P3/validation_comparison.{csv,json}` | Validation only |
| Final paired bootstrap | `results/final_test/paired_bootstrap_results.{csv,json}` | P0/P1/P2 held-out predictions |
| Factorial validation analysis | `results/p3_analysis/` | P0, P1_noSSL, P2, P3 validation predictions |
| Complexity table | `results/paper/08_model_complexity.csv` | P0/P1/P2; P3 complexity is in registry/metadata |
| Raspberry Pi benchmark | `results/edge/` and `results/paper/09_raspberry_pi_latency.csv` | Frozen prototype benchmark |
| Claims/evidence matrix | `results/paper/10_claim_evidence_matrix.csv` | Manuscript evidence boundaries |
| Clean notebooks | `notebooks/final_experiments/` | Public source, no executed outputs |
| Executed notebook provenance | `reproducibility/executed_notebooks/` | Path-normalized output copies |
| Checkpoint identities/locations | `checkpoints/checkpoint_manifest.json` | Original weights; Git or Release assets |

No rolling checkpoint, resume checkpoint, optimizer snapshot, stale P3 archive,
temporary notebook, or raw image is included.
