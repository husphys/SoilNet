# Final reproducibility statement

The publication snapshot includes source, configs, final notebooks, audit/QC
evidence, content-addressed manifests and split, per-sample held-out test
predictions, metrics, paired-bootstrap outputs, paper tables/figures, Raspberry
Pi measurements, environment files, citation metadata, and license.

Raw image data and historical/training checkpoint collections are not included.
Checkpoint identity is established by SHA256. The only published model binary
is the 44,895,306-byte frozen P0 deployment checkpoint, deliberately included
to permit dataset-free Raspberry Pi benchmarking. It is verified against SHA256
`eba009dfd45ec21174a8e40b16148e0455e902286c7db55933487221da761379`
before loading.

Final validation and held-out-test regression metrics use original 0-100
percentage points. Historical development values are retained only as labeled
provenance and are never mixed with final evidence.
