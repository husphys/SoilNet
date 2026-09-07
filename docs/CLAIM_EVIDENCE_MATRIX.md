# Claim evidence matrix

| Claim | Classification | Evidence |
|---|---|---|
| Final P0 regression and classification performance | SUPPORTED_BY_FINAL_TEST | 04_final_test_performance.csv |
| LI contribution: P0 versus P1-noLI | SUPPORTED_BY_VALIDATION_ABLATION | 03_validation_ablation_table.csv and paired final-test context |
| VICReg contribution: P0 versus P1-noSSL | SUPPORTED_BY_VALIDATION_ABLATION | 03_validation_ablation_table.csv and paired final-test context |
| Architecture contribution: P1-noSSL versus P2 | SUPPORTED_BY_FINAL_TEST | 06_architecture_comparison.csv and 07_paired_bootstrap_ci.csv |
| Raspberry Pi CPU latency | SUPPORTED_BY_DEPLOYMENT_BENCHMARK | 09_raspberry_pi_latency.csv |
| Irrigation demonstration | PROOF_OF_CONCEPT_ONLY | historical provenance; no generalized causal estimate |
| Old SoilNet 0.086/0.066 metrics | REMOVE_OR_REWRITE | incompatible or unrecovered provenance |
| Old SoilNet 0.098/0.072 metrics | REMOVE_OR_REWRITE | incompatible or unrecovered provenance |
| 2,250 labeled images | REMOVE_OR_REWRITE | final QC count is 1,927 |
| Direct Barlow/MoCo/SimCLR superiority | SUPPORTED_BY_HISTORICAL_PROVENANCE_ONLY | do not claim superiority unless provenance is recovered |
| Classical ML Table 7 as direct final benchmark | REMOVE_OR_REWRITE | not evaluated in final frozen test |
| Direct SOTA superiority across incomparable datasets | REMOVE_OR_REWRITE | datasets/protocols are not directly comparable |
| Independent 200-image validation | SUPPORTED_BY_HISTORICAL_PROVENANCE_ONLY | retain only if provenance is recovered |
| Generalized 50% irrigation saving | PROOF_OF_CONCEPT_ONLY | rewrite as demonstration, not generalized causal claim |
