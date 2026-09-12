# Data provenance and redistribution decision

## REDISTRIBUTABLE

The following research metadata are published in this release:

- `data/manifests/labeled_manifest.csv`: 1,927 unique labeled pairs.
- `data/splits/train_manifest.csv`: 1,407 records.
- `data/splits/validation_manifest.csv`: 289 records.
- `data/splits/test_manifest.csv`: 231 records.
- `data/manifests/unlabeled_manifest.csv`: 11,995 entries.

These files use stable relative paths, sample identifiers where applicable, and
per-file SHA256 values. Here, REDISTRIBUTABLE records the release decision for
the metadata; it is not a blanket license grant for source images.

## NON_REDISTRIBUTABLE_OR_UNVERIFIED

- All labeled raw images are withheld because redistribution rights have not
  been verified for the complete collection.
- All unlabeled raw images are withheld. Third-party/web-derived provenance,
  including any video-derived frames, does not have verified redistribution
  rights in the audited materials.
- Stale/rolling artifacts, local caches, and source-machine state are not data
  release content.

No source image was deleted or modified. A reviewer with lawful local access can
place files under DATA_ROOT and verify each file against the public manifest.
The repository must not be described as making all data publicly available.
