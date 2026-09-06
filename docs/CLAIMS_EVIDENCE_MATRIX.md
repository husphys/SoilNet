# Claims–evidence matrix

Claimed manuscript values were not supplied; PHASE 2 does not alter claims or invent values.

Dataset counts have distinct meanings and are not interchangeable: 2,057 is
the strongest-provenance historical labeled table; 1,927 is the PHASE 3B final
conflict-free set after excluding all 130 rows in conflicting exact-image
groups; 2,250 is an unresolved historical manuscript count. The QC flow does
not retroactively rewrite the manuscript claim.

| ID | Claim | Evidence status | Historical metric | Exposure | Recovery |
|---|---|---|---|---|---|
| ARCH | SoilNet architecture | SUPPORTED | UNKNOWN | UNKNOWN | KEEP |
| LIGHT | Light-intensity input | SUPPORTED | UNKNOWN | UNKNOWN | KEEP |
| SM0 | SM_0 regression performance | SUPPORTED_TRAINING_ONLY | TRAIN | CONFIRMED_LEAKAGE | SHORT_FINETUNE |
| SM20 | SM_20 regression performance | SUPPORTED_TRAINING_ONLY | TRAIN | CONFIRMED_LEAKAGE | SHORT_FINETUNE |
| CLASS | Moisture classification performance | SUPPORTED_TRAINING_ONLY | TRAIN | CONFIRMED_LEAKAGE | SHORT_FINETUNE |
| SSL | SSL-VICReg | PARTIALLY_SUPPORTED | TRAIN | POTENTIAL_LEAKAGE | KEEP |
| MU | mu ablation | SUPPORTED_TRAINING_ONLY | TRAIN | CONFIRMED_LEAKAGE | MISSING |
| TINY | TinyImageNet pretraining | PARTIALLY_SUPPORTED | TRAIN | POTENTIAL_LEAKAGE | KEEP |
| BASE | Baseline models | PARTIALLY_SUPPORTED | TRAIN | CONFIRMED_LEAKAGE | MISSING |
| KNN | KNN | MISSING | UNKNOWN | UNKNOWN | CHEAP_RECOMPUTE |
| GBM | GBM | NEED_REEVALUATION | TEST | CONFIRMED_LEAKAGE | CHEAP_RECOMPUTE |
| SVM | SVM | NEED_REEVALUATION | TEST | CONFIRMED_LEAKAGE | CHEAP_RECOMPUTE |
| DT | Decision Tree | MISSING | UNKNOWN | UNKNOWN | CHEAP_RECOMPUTE |
| MLP | MLP | MISSING | UNKNOWN | UNKNOWN | CHEAP_RECOMPUTE |
| EDGE | Edge/deployment | PARTIALLY_SUPPORTED | UNKNOWN | UNKNOWN | MISSING |
| IRR | Irrigation experiment | MISSING | UNKNOWN | UNKNOWN | MISSING |

Machine-readable lineage and notes are in `results/audit/claims_evidence_matrix.csv`.
