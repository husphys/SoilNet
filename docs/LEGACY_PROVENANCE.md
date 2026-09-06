# Legacy provenance

Source: <https://github.com/diy-hus/SoilNet>

Pinned commit: `312c0a44605de87d31b6c52627cd91b07a4d1f75`

The checkout used for audit is outside the target repository and detached at
the pinned commit. Legacy notebooks and scripts are not modified. Their hashes,
path references, method flags, checkpoint associations, and metrics files are
recorded in `results/audit/`.

Five historical originals (`IMG_0683.JPG`, `IMG_0703.JPG`, `IMG_0746.JPG`,
`IMG_0753.JPG`, and `IMG_0755.JPG`) were replaced by randomly augmented files
in legacy workflows. The current fallback files are inventoried and mapped, but
are never regenerated. Any affected historical experiment is
`NOT_BITWISE_REPRODUCIBLE`: the original random augmentation seed and exact
generation lineage are not established by the retained artifacts.

The static method audit detects fine-tune notebooks that train and calculate
final Accuracy/F1/RMSE/MAE using the same dataloader. Those values are retained
as training behavior only. `strict=False` loads and differing initialization
are recorded rather than treated as proof of compatibility or fairness.
