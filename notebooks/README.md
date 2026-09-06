# Notebook standard and final sequence

New notebooks must state title, purpose, experiment ID, inputs, outputs, seed,
environment, dataset/split/checkpoint SHA256 values, and manuscript mapping.
Each major code cell requires a preceding Markdown explanation. Legacy
notebooks remain for provenance and are not final-run instructions.

The final automatic sequence is 09 (CUDA training/validation, no test), 10
(one-time frozen test only), 11 (actual Raspberry Pi benchmark plus automatic
edge-only push, no dataset), then 12 (automatic edge-result retrieval,
aggregation, and publication only). Notebooks 09-12 contain no manual
variable toggles. P0 and P1 are already frozen; their completed artifacts are
not overwritten.
