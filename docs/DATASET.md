# Dataset

The configured dataset root is external to Git. `labels.csv` is discovered
recursively and never edited. Its historical Windows paths are normalized into
`data/manifests/soilnet_samples.csv`; because paths are relative to the
configured `DATA_ROOT`, the current layout begins with `Soil_Labeled_Data/`.

`dataset_inventory.csv` contains one row per image with byte size, SHA256,
readability, dimensions, UTC mtime, and a 64-bit perceptual difference hash.
`dataset_summary.json` records total files/images, label reconciliation,
missing/corrupt counts, exact duplicates, near-duplicate components, and value
distributions. Hamming-distance matches are screening candidates, not a claim
that two scenes are biologically or experimentally identical; high-impact
groups require visual/session review before a final experimental split.

The historical table contains 2,057 labeled records. All 46 exact-SHA256
duplicate groups (130 records, 84 redundant copies) disagree on `SM_0` or
`SM_20`. Under the pre-training
`EXACT_DUPLICATE_CONFLICT_EXCLUSION` policy, every row in those groups is
excluded without choosing a representative or adjudicating labels. The final
clean manifest therefore contains 1,927 unique labeled image SHA256 values.
The count 1,973 is not used because the 46 potential representatives have no
defensible canonical regression target. The manuscript count 2,250 remains an
unresolved historical claim, not a substitute for either audited count.

The final split assigns connected exact/near-duplicate groups deterministically
with seed 20260905 and target ratios 73%/15%/12%. No documented
capture/session/location or source-group identifier exists in the historical
label columns. Timestamp-like filenames are therefore not treated as session
metadata. Zero documented groups cross splits, but latent unrecorded session
leakage remains `NOT_TESTABLE`. Near similarity is a grouping firewall only;
it does not delete images.

The final split is for a prospective rerun. It must not be described as independent
evaluation of a checkpoint trained on all 2,057 historical labeled rows.

## Mixed filesystem contents

`DATA_ROOT` also contains historical model binaries (`*.pth`) alongside
scientific dataset artifacts. This is a filesystem-organization issue, not a
model/data record: those files are excluded from image counts, label counts,
duplicate-image statistics, split manifests, and any proposed dataset release.
They are inventoried separately as `DATA_ROOT_EMBEDDED_ARTIFACTS` in
`results/audit/data_root_checkpoint_inventory.csv`; source files are not moved,
deleted, copied into Git, or merged into the 969-file `CHECKPOINT_ROOT` scope.
