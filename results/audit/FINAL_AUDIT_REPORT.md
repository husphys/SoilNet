# FINAL AUDIT REPORT

Generated from machine-readable inventories. No long training was run.

## Direct answers

1. **Dataset size:** 14053 image files among 14270 total files; 14053 are readable.
2. **Why 2,057 historical samples:** `labels.csv` contains exactly 2057 data rows. This is the labeled subset, not all images under `DATA_ROOT`.
3. **Missing/corrupt:** 0 labeled rows remain unresolved after fallback mapping; 0 inventoried images are corrupt/unreadable.
4. **Five fallbacks:** all five exist and are hash-mapped; affected legacy experiments remain `NOT_BITWISE_REPRODUCIBLE`.
5. **Duplicates:** 2905 exact groups, 8875 excess exact-copy files, and 834 candidate near-duplicate components containing nonidentical bytes at difference-hash Hamming threshold 5.
6. **Historical split:** no trustworthy held-out split was established by the static audit. The new split is prospective and cannot retroactively create unseen data for old checkpoints.
7. **Training-set evaluation notebooks/scripts:** 42 files; full list is in `methodology_audit.csv` (listed below).
8. **Verified checkpoints:** 5 unique hashes are identical between local and GitHub sources; all verified overlaps are VICReg projector state_dicts, not evaluation-ready full fine-tuned models.
9. **Local/GitHub overlap:** 5 unique checkpoint hashes overlap. No full fine-tuned evaluation checkpoint is selected merely from this overlap.
10. **Experiments supporting manuscript:** source-level architecture/light claims are supported; experiment-level mappings remain in `experiment_graph.csv` because the manuscript/table mapping was not provided.
11. **Training behavior only:** historical fine-tuning regression/classification and mu-ablation downstream metrics detected on training loaders.
12. **Claims without artifacts:** irrigation, KNN, Decision Tree, and MLP are `MISSING`; GBM/SVM have result rows but no held-out lineage or implementation.
13. **Need reevaluation:** selected compatible checkpoints and classical models where frozen embeddings can be verified.
14. **Need rerun:** only a selected final fine-tune should be rerun if no truly unseen set can be proven. The full mu sweep is not justified by current evidence.
15. **Minimum GPU work:** first reuse verified checkpoints; recompute tables/CPU classical models; reevaluate on truly unseen data; otherwise one short selected group-split fine-tune. No full sweep.

## Training-set metric notebooks/scripts

- `Base_line_models_pretrain_ImageNet.py`
- `finetune_mu=20.ipynb`
- `finetune_mu=23.ipynb`
- `finetune_mu=25.ipynb`
- `finetune_mu=27.ipynb`
- `finetune_mu=30.ipynb`
- `finetune_mu=33.ipynb`
- `finetune_mu=40.ipynb`
- `finetune_new/Finetune_VicReg_mu=20.ipynb`
- `finetune_new/Finetune_VicReg_mu=23.ipynb`
- `finetune_new/Finetune_VicReg_mu=25.ipynb`
- `finetune_new/Finetune_VicReg_mu=27.ipynb`
- `finetune_new/Finetune_VicReg_mu=30.ipynb`
- `finetune_new/Finetune_VicReg_mu=33.ipynb`
- `finetune_new/Finetune_VicReg_TinyImageNet_mu=20.ipynb`
- `finetune_new/Finetune_VicReg_TinyImageNet_mu=23.ipynb`
- `finetune_new/Finetune_VicReg_TinyImageNet_mu=25.ipynb`
- `finetune_new/Finetune_VicReg_TinyImageNet_mu=27.ipynb`
- `finetune_new/Finetune_VicReg_TinyImageNet_mu=30.ipynb`
- `finetune_new/Finetune_VicReg_TinyImageNet_mu=33.ipynb`
- `finetune_tinyImage_mu=20.ipynb`
- `finetune_tinyImage_mu=23.ipynb`
- `finetune_tinyImage_mu=25.ipynb`
- `finetune_tinyImage_mu=27.ipynb`
- `finetune_tinyImage_mu=30.ipynb`
- `finetune_tinyImage_mu=33.ipynb`
- `pp_SoiNet_orginal_vicREG-mu=2.ipynb`
- `pp_SoiNet_orginal_vicREG-mu=23.ipynb`
- `pp_SoiNet_orginal_vicREG-mu=25.ipynb`
- `pp_SoiNet_orginal_vicREG-mu=27.ipynb`
- `pp_SoiNet_orginal_vicREG-mu=30.ipynb`
- `pp_SoiNet_orginal_vicREG-mu=33.ipynb`
- `pp_SoiNet_orginal_vicREG-mu=40.ipynb`
- `pp_SoiNet_orginal_vicREG-mu=80.ipynb`
- `pp_SoiNet_orginal_vicREG_mu=20.ipynb`
- `SoilNet_pretrain_ImageNet.py`
- `Vicreg_soiNet_tinyImagenet_mu=20.ipynb`
- `Vicreg_soiNet_tinyImagenet_mu=23.ipynb`
- `Vicreg_soiNet_tinyImagenet_mu=25.ipynb`
- `Vicreg_soiNet_tinyImagenet_mu=27.ipynb`
- `Vicreg_soiNet_tinyImagenet_mu=30.ipynb`
- `Vicreg_soiNet_tinyImagenet_mu=33.ipynb`

## Priority table

| Priority | Work |
|---|---|
| NO_ACTION | Preserve raw data and immutable legacy notebooks. |
| CHEAP_RECOMPUTE | Rebuild tables from validated CSV; rerun classical models from verified frozen embeddings. |
| REEVALUATE_ONLY | Evaluate hash-selected compatible checkpoints on proven unseen groups. |
| SHORT_FINETUNE | One final selected configuration only if no unseen set exists. |
| FULL_RERUN | Not currently justified; use only if missing lineage cannot be recovered and the exact claim is essential. |

## Summary counts

### Dataset

- total files: 14270
- valid images: 14053
- labeled rows: 2057
- missing labeled images after mapping: 0
- corrupt images: 0
- exact duplicate excess files: 8875
- candidate near-duplicate components with nonidentical bytes: 834

### Checkpoints

- local: 969
- GitHub: 10
- verified identical unique hashes: 5
- unique hashes: 974
- unknown provenance: 303

### Claims

- supported: 2
- training-only: 4
- partially supported: 4
- need reevaluation: 2
- need rerun: 0
- missing: 4
- conflicts: 0 (not testable without manuscript values)

## Audit limitations

- GPU/NVML availability is recorded in `system_info.txt`; no GPU execution was assumed.
- No manuscript file was supplied, so exact claimed values/table/figure conflicts cannot be audited.
- Filename timestamps were not promoted to capture-session IDs without documentation.
- No historical consumed-sample manifest was found, so current extra images cannot be declared truly unseen by old checkpoints from filenames or mtimes alone.
- Checkpoint compatibility is a safe structural/key-pattern audit, not proof of semantic lineage.
