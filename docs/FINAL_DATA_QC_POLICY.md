# Final data QC policy

Policy: `EXACT_DUPLICATE_CONFLICT_EXCLUSION`.

For every group of labeled records that shares the same exact image SHA256,
exclude every record in that group when either `SM_0` or `SM_20` differs among
members. No representative is retained. Conflicts in `light_value` and
`moisture_class` are characterized for transparency, but they do not add a
second decision rule because every affected exact-byte group already conflicts
on the task-critical regression targets.

This rule was locked before any PHASE 3B model training and before any
prospective test evaluation. It does not adjudicate labels, use majority voting,
average targets, rank conflict magnitudes, or use a model to clean labels.
Source image files and the historical label table remain unchanged.

Perceptual similarity is not an exclusion criterion. Non-identical images in a
locked near-duplicate connected component remain in the clean manifest and are
kept in one split by the split firewall.
