# Metric definitions

## Regression

The two regression targets are `SM_0` and `SM_20`. They are reported separately
using RMSE, MAE, mean error (ME = mean prediction minus truth), and R-squared
when defined. Legacy code divides both targets by 100 for optimization and
multiplies predictions and targets by 100 before RMSE/MAE calculation. The
audited historical RMSE/MAE are therefore on a percentage-point scale, not on
the normalized 0–1 training scale. The physical interpretation beyond that
scale requires dataset documentation not present in the artifacts.

## Classification

The target is `moisture_class`. Report Accuracy, Macro-F1, macro Precision, and
macro Recall. Legacy weighted F1 is retained only with its exact label; it is
not silently relabeled Macro-F1.

Regression and classification metrics are never averaged into a combined
score. Metrics calculated on a training loader are labeled
`TRAINING_SET_METRICS_ONLY`, not validation or test performance.
