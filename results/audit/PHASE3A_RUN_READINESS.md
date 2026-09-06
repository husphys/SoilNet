# PHASE 3A RUN READINESS

| Item | Answer | Evidence / action |
|---|---|---|
| 1. μ preselection resolved? | YES — `PRESELECTION_RESOLVED_FROM_HISTORICAL_MANUSCRIPT` | Locked before prospective evaluation |
| 2. Evidence source? | Historical manuscript assertion supplied for PHASE 3A | Manuscript file itself is not in audited roots |
| 3. Selected SSL μ=27 checkpoint? | YES — `CHECKPOINT_ROOT:checkpoints_VicReg_original_NEW/vicreg_model_final_mu_27.0.pth` | One selected among 96 μ=27 VICReg-related candidates |
| 4. SHA256? | `42599ac025b8d8c8c5b8cca36624665f155e0ff588cd90de2fd47b08cd2d60d8` | Safe-load plus independent lineage checks |
| 5. 2,057 duplicate labels consistent? | NO — 46/46 exact groups conflict | Every group conflicts on SM_0 and SM_20; do not auto-resolve |
| 6. Unique final count? | NOT GENERATED (1,973 byte-unique was expected) | Data conflict gate stopped generation |
| 7. Final train/val/test counts? | NOT AVAILABLE | Split not generated |
| 8. Group leakage? | NOT TESTABLE | Split not generated |
| 9. Near-duplicate leakage? | NOT TESTABLE | Split not generated |
| 10. Final epochs? | 15 | Historical manuscript protocol |
| 11. Batch? | 32 | Historical manuscript protocol |
| 12. LR? | 5e-4 | Historical manuscript protocol |
| 13. Optimizer? | Adam, weight_decay=0.0 | Adam default in historical μ=27 code |
| 14. Model outputs? | SM_0, SM_20, moisture_class; LI branch enabled | Code plus historical classification claim |
| 15. Metric scale? | Regression percentage points 0–100; classification ratios 0–1 | Internal regression/LI divide by 100 |
| 16. Config SHA256? | `40351a9404fcef3a1ba5730c95df2d69b08a880c75ff3e23c55a8097d0e96dd3` | Locked blocked-state config |
| 17. Manifest SHA256? | NOT GENERATED — DATA CONFLICT | No canonical manifest created |
| 18. Split SHA256? | NOT GENERATED — DATA CONFLICT | No split created or opened for metrics |
| 19. GPU ready? | GPU_BLOCKED | CUDA diagnostic only; no driver changes |
| 20. Safe to run final short fine-tune? | NO | Resolve all exact-byte label conflicts with external evidence, issue new experiment ID, then lock manifests; GPU must also be ready |

## Final status

**BLOCKED_DATA_CONFLICT**

Secondary gates: GPU is blocked in the current environment and Git state is `UNCOMMITTED`. Neither changes the primary data-conflict stop.
No training ran. No prospective test metric was read. No raw data/checkpoint was modified. No GitHub push occurred.
