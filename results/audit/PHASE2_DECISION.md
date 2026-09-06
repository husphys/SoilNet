# PHASE 2 DECISION

| Item | Decision | Evidence | Required action |
|---|---|---|---|
| Labeled dataset | 1973 exact-unique images from 2,057 rows | SHA256 labeled audit | Keep row count and duplicate count separate |
| 2,250 versus 2,057 | UNRESOLVED | No count evidence for 2,250; `labels.csv` proves 2,057 | Obtain draft/capture manifest with sample IDs |
| Fine-tuned checkpoint | 30 in DATA_ROOT; 0 in the 969-file CHECKPOINT_ROOT scope | Safe tensor/key classification | Do not select by filename |
| Historical held-out evidence | 0 valid checkpoints | Fine-tune and SOTA sources use all-row training/evaluation | Do not report as test performance |
| Confirmed unseen labeled data | 0 | All 2,057 row identities are exposed by high-confidence fine-tune generations | Prospective split is for a new run only |
| Preselection | PRESELECTION_UNRESOLVED | Manuscript/final config mapping is absent | Freeze architecture, SSL source and mu before training |
| Final recovery | READY_AFTER_ONE_SHORT_FINETUNE | Old checkpoints cannot recover clean held-out metrics | Run only FINAL_LEAKAGE_CONTROLLED_REVALIDATION after preselection |
| Public release | REVIEW_REQUIRED | No binaries copied; machine paths remain in some evidence files | Sanitize/review before a later commit |

## Direct answers

1. **Unique labeled samples:** 1973 exact SHA256 images across 2,057 labeled rows; 84 rows are redundant exact copies in 46 exact groups.
2. **2,250 vs 2,057:** UNRESOLVED. No audited artifact establishes 2,250 as an image count. 2,057 has strongest provenance.
3. **Fine-tuned SoilNet checkpoints:** 0 in CHECKPOINT_ROOT; 30 safely classified candidates in DATA_ROOT.
4. **Strongest lineage:** the `Best_finetuned_*` and direct `finetune_mu_*` families have HIGH_CONFIDENCE filename/save-target/architecture lineage, but none is VERIFIED because save-time SHA evidence is absent. No single mu is uniquely selected.
5. **Valid held-out evidence:** none. No linked fine-tuned checkpoint has a reproducible held-out split with sample identifiers.
6. **Confirmed unseen labeled data:** none. A newly assigned prospective test row is not historically unseen.
7. **Historical Accuracy/F1:** retain only as `TRAINING_SET_METRICS_ONLY`; downstream classical rows require reevaluation.
8. **Historical RMSE/MAE:** retain only as training behavior; not held-out performance.
9. **SSL artifacts:** existing hash-inventoried backbone/projector artifacts do not require a new SSL sweep, but soil-pool SSL is transductive and no pure-external verified final encoder is selected.
10. **Cheap classical regeneration:** KNN, SVM, GBM, Decision Tree and MLP after a frozen encoder/config decision; no CNN training in that step.
11. **New final fine-tune:** yes, only if held-out generalization metrics remain required by the manuscript.
12. **Exact training:** `FINAL_LEAKAGE_CONTROLLED_REVALIDATION`: preselected SoilNet only; 1,496 train, 317 validation, validation-only selection, 244 test once.
13. **Mu sweep:** no. Mu/config must be frozen from historical/manuscript evidence before the new test is touched.
14. **SSL rerun:** no evidence currently justifies rerunning SSL.
15. **Full rerun:** none justified.

## Checkpoints discovered inside DATA_ROOT

1. `.pth` files: 108.
2. Unique SHA256: 107.
2a. Redundant files within DATA_ROOT scope: 1.
3. Files byte-identical to CHECKPOINT_ROOT: 0.
4. Files byte-identical to legacy GitHub: 4.
5. FINETUNED_SOILNET_DUALHEAD: 30.
6. Checkpoints with regression and classification heads: 51.
7. Checkpoints with a light branch: 51.
8. Fine-tuned lineage: VERIFIED=0; HIGH_CONFIDENCE=24. Save-time SHA evidence is required for VERIFIED.
9. Fine-tuned candidates linked to training on all 2,057 rows: 25.
10. Genuine historical held-out checkpoints: 0.
11. These artifacts do not change the one-short-fine-tune decision: they recover model bytes and lineage, but not a clean historical held-out protocol.

## Scope totals

- CHECKPOINT_ROOT: 969 files
- LEGACY_GITHUB: 10 files
- DATA_ROOT_EMBEDDED_ARTIFACTS: 108 files
- CROSS-SCOPE UNIQUE SHA256: 1077
- VERIFIED FINE-TUNED LINEAGE: 0
- HIGH-CONFIDENCE FINE-TUNED LINEAGE: 24
- SOTA/baseline artifacts separately audited: 8

## Final decision

**READY_AFTER_ONE_SHORT_FINETUNE**

Execution gate: **PRESELECTION_UNRESOLVED**. No training may begin until final architecture, SSL source and mu are fixed without using prospective validation/test outcomes.
