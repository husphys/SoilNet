# Claim evidence matrix

| Claim | Classification | Required evidence/action |
|---|---|---|
| Final frozen-model performance | SUPPORTED_BY_FINAL_TEST | Notebook 10 metrics and predictions |
| LI and VICReg ablations | SUPPORTED_BY_VALIDATION_ABLATION | Frozen validation-selected P0/P1 registry; paired test context is secondary |
| P1-noSSL versus P2 architecture contribution | SUPPORTED_BY_FINAL_TEST | Controlled protocol plus paired final-test comparison |
| Raspberry Pi CPU latency | SUPPORTED_BY_DEPLOYMENT_BENCHMARK | Notebook 11 on detected Raspberry Pi hardware |
| Historical SSL/development observations | SUPPORTED_BY_HISTORICAL_PROVENANCE_ONLY | Do not promote without recovered comparable provenance |
| Irrigation demonstration | PROOF_OF_CONCEPT_ONLY | Do not generalize or claim causal savings |
| Old 0.086/0.066 SoilNet metrics | REMOVE_OR_REWRITE | Incompatible/unrecovered provenance |
| Old 0.098/0.072 SoilNet metrics | REMOVE_OR_REWRITE | Incompatible/unrecovered provenance |
| 2,250 labeled-image count | REMOVE_OR_REWRITE | Final QC count is 1,927 |
| Direct Barlow/MoCo/SimCLR superiority | REMOVE_OR_REWRITE | Recover exact comparable provenance first |
| Classical ML Table 7 as direct final benchmark | REMOVE_OR_REWRITE | Not in the frozen final test set |
| Direct SOTA superiority across incomparable datasets | REMOVE_OR_REWRITE | Protocols/datasets are not directly comparable |
| Independent 200-image validation | REMOVE_OR_REWRITE | Recover provenance first |
| Generalized 50% irrigation saving | REMOVE_OR_REWRITE | Retain only as proof-of-concept language |
