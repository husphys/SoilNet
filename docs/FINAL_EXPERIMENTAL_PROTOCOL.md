# Final experimental protocol

The locked split contains 1,407 train, 289 validation, and 231 held-out test samples (SHA256 `8927b8223b8c4c234d264ad9ea62ac2df6161124a79787a2e71eeb5cd23eae2f`, seed 20260905). All supervised runs use 60 epochs, batch 32, Adam, learning rate 1e-4, weight decay 0, no early stopping, and strict minimum mean validation regression RMSE. Classification is auxiliary and never selects checkpoints. Notebook 10 alone opens test after all four models are frozen. After `FINAL_TEST_LOCK.json` exists, test outcomes must not drive model changes.
