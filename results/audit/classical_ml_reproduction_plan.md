# Classical ML reproduction feasibility

No CNN training or full feature extraction was run in PHASE 2.

| Method | Historical evidence | Validity | Recovery |
|---|---|---|---|
| KNN | No configured implementation, saved estimator, embeddings, or metric row found | MISSING | CHEAP_RECOMPUTE after encoder/config preselection |
| SVM | Local scripts and result rows exist; row-level 80/20 split uses seed 42 | Encoder exposure and sample IDs are not cleanly held out/persisted | CHEAP_RECOMPUTE |
| GBM | Local scripts and result rows exist; row-level 80/20 split uses seed 42 | Same limitation as SVM | CHEAP_RECOMPUTE |
| Decision Tree | No configured implementation/result artifact found | MISSING | CHEAP_RECOMPUTE |
| MLP | Imported in local scripts but not configured in the executed classifier dictionary; no result row | MISSING | CHEAP_RECOMPUTE |

Evidence details:

- Classifier source files found: 2.
- `train_test_split` present: yes; the discovered scripts use `test_size=0.2, random_state=42`.
- Saved `.npy/.npz/.pkl/.joblib` embeddings or classifier objects found in audited roots: none.
- The scripts reference fine-tuned SoilNet checkpoints whose training exposure is not cleanly held out from their row-level test split.

Minimal safe regeneration:

1. Freeze the encoder/checkpoint choice before inspecting prospective test metrics.
2. Extract features once for the fixed 1496/317/244 group-aware manifest.
3. Fit scaler and classifier on train only; use validation only for any hyperparameter choice.
4. Evaluate the 244 test samples exactly once and write predictions plus metadata.
5. If the chosen encoder used the soil image pool, label the protocol transductive; no verified pure-external SSL encoder is currently established.
