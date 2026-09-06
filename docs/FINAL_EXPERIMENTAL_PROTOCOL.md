# Final experimental protocol

The clean dataset contains 1,927 unique labeled images split into 1,407 train,
289 validation, and 231 held-out test samples. The split SHA256 is
`8927b8223b8c4c234d264ad9ea62ac2df6161124a79787a2e71eeb5cd23eae2f`.

All supervised runs use seed 20260905, 60 epochs, batch size 32, Adam,
learning rate `1e-4`, weight decay 0, deterministic historical preprocessing,
and no early stopping. A strict improvement in
`(SM0_RMSE + SM20_RMSE) / 2` on validation replaces
`validation_best_regression.pth`; epoch 60 is also retained. The best checkpoint
is reloaded for final validation artifacts. Classification is auxiliary and is
not a checkpoint-selection input.

P2 uses exact TIMM model `mobilevitv2_050.cvnets_in1k`, ImageNet CVNets
initialization, no VICReg, a normalized scalar LI mapped by
`Linear(1,32)+ReLU` and concatenated with pooled image features, and matched
128-unit regression/classification heads. Relative to P1-noSSL, only the
architecture/parent feature extractor changes.

Notebook 09 never constructs test data. Notebook 10 is the only test-opening
path and validates the complete frozen registry before creating one shared test
dataset/loader. It performs inference only, evaluates all four frozen models on
the same samples, and writes the lock last. Test outcomes cannot be used for
subsequent model changes.
