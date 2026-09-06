# P0 manual-run readiness

No full training or prospective test evaluation was performed during this
implementation phase.

1. **Current environment retained?** YES. The existing `soilnet` environment
   was used without creating, downgrading, or reinstalling anything.
2. **Historical environment documented?** YES. Python 3.10.16, PyTorch
   2.6.0+cu118, and CUDA availability are separated from the current tested
   Python 3.11.15 reproduction environment.
3. **Protocol revised to 60 epochs / 1e-4?** YES. P0 uses epoch 60, batch 32,
   Adam, learning rate 1e-4, and weight decay 0.
4. **Revision occurred before test?** YES. No test loader, prediction, metric,
   or completion marker exists.
5. **Data hashes valid?** YES. Manifest
   `8ff45054d4b8e3df9758d0c112dc16719572b2267906fb3a5ed5b3262a6732bd`
   and split
   `8927b8223b8c4c234d264ad9ea62ac2df6161124a79787a2e71eeb5cd23eae2f`
   remain unchanged; counts are 1,407/289/231.
6. **μ27 SHA valid?** YES:
   `42599ac025b8d8c8c5b8cca36624665f155e0ff588cd90de2fd47b08cd2d60d8`.
7. **577/577 load?** YES, strict normalized load with no missing,
   unexpected, or shape-mismatched canonical key.
8. **Test firewall active?** YES. P0 imports/builds train and validation only;
   notebook 09 remains `RUN_FINAL_TEST = False` with an empty explicit model
   selection.
9. **Resume support ready?** YES. The rolling file is accepted only when
   experiment, architecture, config, manifest, split, SSL, epoch, optimizer,
   and history identities match.
10. **Disk policy ready?** YES. Primary epoch 60 plus at most one optional
    validation-best checkpoint; the single rolling resume file is overwritten
    and removed after success.
11. **P0 notebook manual-run ready?** YES. Notebook/config static validation
    PASS (11 notebooks, 8 configs), compatibility PASS, 26 unit tests PASS, and
    the final repository verifier PASS with no implementation error.
12. **P1 status?** `OPTIONAL_NOT_YET_REQUIRED` for no-LI and no-SSL; neither was
    run.
13. **P2 status?** MobileNetV2, MobileViTv2, EfficientNet-B0, and classical ML
    are `OPTIONAL`; MobileNetV3 is `SKIPPED_VARIANT_UNRESOLVED`. None blocks P0
    or was run.
14. **CUDA status?** PASS in the manually executed Jupyter kernel:
    `torch.cuda.is_available()=True`, one NVIDIA GeForce RTX 3050, CUDA tensor
    multiplication and synthetic SoilNet forward/backward PASS. The separate
    Codex subprocess still cannot initialize NVML, so both execution contexts
    are recorded rather than conflated. Full P0 still refuses CPU fallback.

## Final implementation status

`READY_FOR_MANUAL_P0_RUN`

The verifier has one non-blocking warning: the deleted/empty historical GitHub
snapshot cannot be live re-hashed. Its frozen 10-file inventory remains
checksummed and is not an input to P0 v3. The selected μ27 checkpoint used by
P0 was re-hashed directly and passed.
