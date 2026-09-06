# Final clean split report

This is pre-specified descriptive data accounting only. No model was loaded, no test loader was instantiated, and no test metric was read or computed.

## Counts

| Split | Count | Observed ratio | Target ratio |
|---|---|---|---|
| train | 1407 | 0.7302 | 0.73 |
| validation | 289 | 0.1500 | 0.15 |
| test | 231 | 0.1199 | 0.12 |

## Leakage firewall

| Check | Result |
|---|---|
| Exact SHA256 cross-split leakage | 0 |
| Locked exact/near connected-component leakage | 0 |
| Near-duplicate group leakage | 0 |
| Sample ID overlap | 0 |
| Documented capture/session-group leakage | 0 |
| Documented source-group leakage | 0 |

No capture/session or source-group field exists in the historical label table. Thus zero documented groups cross splits, while leakage from an unrecorded latent session remains NOT_TESTABLE. Filename timestamps were not converted into session IDs.

## Moisture-class distributions

| Split | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 142 | 124 | 92 | 100 | 161 | 167 | 162 | 164 | 158 | 137 |
| validation | 29 | 38 | 57 | 29 | 17 | 19 | 22 | 20 | 27 | 31 |
| test | 26 | 21 | 51 | 33 | 9 | 12 | 16 | 16 | 15 | 32 |

## Soil-type distributions

| Split | alluvial |
|---|---|
| train | 1407 |
| validation | 289 |
| test | 231 |

## Numeric summaries

| Split | Field | n | Min | Max | Mean | Median | SD |
|---|---|---|---|---|---|---|---|
| train | SM_0 | 1407 | 0.000 | 100.000 | 52.583 | 55.000 | 28.426 |
| train | SM_20 | 1407 | 1.000 | 100.000 | 54.566 | 56.000 | 28.473 |
| train | light_value | 1407 | 0.000 | 100.000 | 51.157 | 52.000 | 27.060 |
| validation | SM_0 | 289 | 0.000 | 100.000 | 45.450 | 36.000 | 29.641 |
| validation | SM_20 | 289 | 2.000 | 100.000 | 47.758 | 41.000 | 29.592 |
| validation | light_value | 289 | 0.000 | 100.000 | 45.194 | 40.000 | 30.828 |
| test | SM_0 | 231 | 0.000 | 100.000 | 45.329 | 35.000 | 30.411 |
| test | SM_20 | 231 | 1.000 | 100.000 | 47.723 | 38.000 | 30.446 |
| test | light_value | 231 | 0.000 | 100.000 | 37.212 | 29.000 | 30.413 |

Grouping policy: exact SHA256 plus PHASE 2's locked 64-bit difference-hash connected components at Hamming distance <= 5. Perceptual similarity did not remove samples. Assignment used only group sizes, target ratios, the fixed seed, and stable IDs; no model performance was consulted.
