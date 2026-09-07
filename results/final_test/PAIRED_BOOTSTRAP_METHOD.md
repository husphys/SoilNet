# Paired bootstrap method

All frozen models predict the same 231 held-out samples in the same order. Using seed 20260906, 10,000 bootstrap samples draw shared sample indices with replacement. Each delta is metric(model_a) - metric(model_b); negative error deltas favor model_a. Intervals are the 2.5th and 97.5th percentiles. These are paired test-set bootstrap intervals and do not represent multiple-training-seed uncertainty.
