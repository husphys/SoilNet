# Final metric specification

This specification is locked before any final training or prospective metric access.

| Task | Target | Metrics | Reported scale |
|---|---|---|---|
| Regression | SM_0 | RMSE, MAE, R2, ME/bias | percentage points, original 0–100 target scale |
| Regression | SM_20 | RMSE, MAE, R2, ME/bias | percentage points, original 0–100 target scale |
| Classification | moisture_class (10 classes, 0–9) | Accuracy, Macro-F1, Macro-Precision, Macro-Recall, confusion matrix | ratios 0–1; confusion matrix as counts |

The model receives regression targets as `value / 100` and LI as `light_value / 100`. Regression predictions and truths are multiplied by 100 exactly once before reporting RMSE, MAE, and ME. Thus an internal error of 0.086 corresponds to 8.6 percentage points; the two values must never be reported interchangeably. R2 is dimensionless. ME is `mean(prediction - truth)` and retains sign.

Metrics are computed separately for SM_0 and SM_20. No combined regression/classification score is a primary manuscript metric. Validation metrics may monitor the fixed protocol but do not change the v3 epoch-60 primary checkpoint. Test metrics are produced once, only by the separate final evaluation command.
