# Final dataset flow

| Stage | Records/images | Interpretation |
|---|---:|---|
| Historical labeled table | 2057 | Strongest-provenance historical labeled records |
| Exact-byte duplicate audit | 46 groups / 130 records | 84 copies beyond one-per-SHA; every group conflicts on SM_0 or SM_20 |
| Auxiliary/class characterization | 43 LI groups / 38 moisture-class groups | Transparency only; not used to decide exclusion |
| Excluded without label adjudication | 130 | All rows in every conflicting exact-SHA group; no representative retained |
| Final conflict-free dataset | 1927 | Unique labeled image SHA256 values |

The final count is **1927**, not 1,973. The 46 otherwise-unique SHA representatives
inside conflict groups have no defensible canonical regression target and are therefore
excluded. The unresolved manuscript count of 2,250 remains a separate historical claim;
it is not silently replaced by this QC result. No source file was deleted or modified.
