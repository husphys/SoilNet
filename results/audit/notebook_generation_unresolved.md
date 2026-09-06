# Notebook-generation unresolved parameters

## Blocking

- **MobileNetV3 variant:** historical architecture code and audited artifacts do
  not identify whether the intended P2 baseline is MobileNetV3-Small or
  MobileNetV3-Large. Both names exist in timm 1.0.29, but availability is not
  provenance. `P2_mobilenetv3.yaml` is therefore
  `SKIPPED_VARIANT_UNRESOLVED`; no variant was silently chosen. This optional
  baseline does not block the manual P0 run.

## Resolved from evidence

- Online train augmentation is deterministic resize, tensor conversion, and
  ImageNet normalization. Historical `ColorJitter(0.2, 0.2)` was only part of
  an invalid-image fallback-generation procedure. PHASE 3B prohibits creating
  new fallbacks, so it is not a training augmentation.
- Loss is `1.0 * MSE(SM_0, SM_20) + 1.0 * CrossEntropy(moisture_class)`.
- Class mapping is the observed and locked integer set 0–9.
- LI and regression targets are divided by 100 internally; regression metrics
  are reported on the restored 0–100 scale.
- AMP defaults to false because it is not part of the locked P0 protocol.
- The optional validation-best checkpoint is disabled; epoch 15 is primary.

## Predeclared classical defaults

Historical values exist for SVM and Gradient Boosting and are reused. KNN,
Decision Tree, and MLP values were not documented; fixed sklearn-style defaults
are declared in `P2_classical_ml.yaml` before any prospective test evaluation.
There is no grid search. Classification and two regression targets remain
separate estimators and result tables.
