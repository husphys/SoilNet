# Dataset count provenance: 2,250 versus 2,057

## Decision

**UNRESOLVED**

The strongest reproducible labeled-sample count is **2,057**: the discovered `labels.csv` has 2,057 data rows, and historical notebook outputs repeatedly report loading 2,057 labeled images.

No source in the searched dataset root, checkpoint root, pinned legacy checkout, current documentation, or readable XLSX cells establishes 2,250 as an image/sample count. The only exact `2250` occurrence in the pinned legacy text is the decimal tail of `MSE: 1319.2250`, not a count. Therefore `2250 - 2057 = 193` is not interpreted as filtered images.

## Search evidence

- DATA_ROOT/legacy/current-document exact 2,057 occurrences: 171
- Exact 2,250 occurrences in a count-like context: 0
- Pinned legacy exact 2250 occurrences: 1 (metric decimal, not count)
- Checkpoint scalar metadata equal to 2,057: 0
- Checkpoint scalar metadata equal to 2,250: 0
- XLSX workbooks inspected: 3
- XLSX exact cells equal to 2,057: 0
- XLSX exact cells equal to 2,250: 0
- XLSX workbook read errors: 0

## Limitation

No manuscript/draft file or historical capture manifest asserting 2,250 was supplied in the audited roots. If such a document is later supplied, it must be reconciled by sample identifiers—not arithmetic difference alone.
