# P0–P2 notebook readiness

> Superseded for P0 execution by `P0_MANUAL_RUN_READINESS.md` and protocol v3.
> This file preserves the earlier notebook-generation checkpoint.

This report covers implementation readiness only. No 15-epoch training and no
prospective test evaluation were performed.

1. **Notebooks created:** 11 notebooks, numbered 00–10, under
   `notebooks/final_experiments/`.
2. **Configs created:** 8 explicit experiment configs under
   `config/experiments/`.
3. **Reusable modules created/modified:** shared path/config validation, data
   loaders, SoilNet ablation wrapper, timm baselines, strict checkpoint model
   factory, deep/classical training engines, metric/prediction writers, final
   test gate, and experiment utilities under `src/soilnet/`.
4. **Unresolved scientific parameters:** the exact historical MobileNetV3
   variant (Small versus Large) is not identifiable. Its config and notebook
   stop with `BLOCKED_UNRESOLVED_VARIANT`.
5. **GPU status:** `GPU_BLOCKED`; `torch.cuda.is_available()` is false in the
   current WSL environment. No driver or CUDA package was changed.
6. **Dry-run status:** PASS on CPU for P0 only: exactly one train batch, one
   backward pass, and one validation batch. No full training and no research
   metric were produced.
7. **Test firewall status:** PASS. Notebooks 01–08 do not construct a test
   loader; the bounded smoke loaded zero test samples. Notebook 09 is disabled
   by `RUN_FINAL_TEST = False` and preflights every frozen artifact before test
   access.
8. **Manifest hash verified:** YES —
   `8ff45054d4b8e3df9758d0c112dc16719572b2267906fb3a5ed5b3262a6732bd`.
9. **Split hash verified:** YES —
   `8927b8223b8c4c234d264ad9ea62ac2df6161124a79787a2e71eeb5cd23eae2f`.
10. **SSL checkpoint SHA verified:** YES —
    `42599ac025b8d8c8c5b8cca36624665f155e0ff588cd90de2fd47b08cd2d60d8`;
    strict normalization matched all 577 canonical model keys, with no missing,
    unexpected, or shape-mismatched keys after normalization.
11. **Disk checkpoint policy:** deep runs keep epoch 15 plus at most one
    content-distinct optional validation-best checkpoint; a rolling resume file
    is overwritten and removed after success. All run artifacts remain outside
    Git. The read-only disk reporter deletes nothing.
12. **Unit tests:** PASS — 22 tests.
13. **Repository verifier:** FAIL on an external historical-provenance
    dependency only. The configured temporary legacy snapshot is absent, so 10
    `LEGACY_GITHUB` checkpoint paths cannot be re-read. Nine have byte-identical
    copies in the audited data/checkpoint roots; the remaining historical
    `vicreg_model_final_tinyImageNet_mu_25.0.pth` SHA is unavailable locally,
    from the now-empty GitHub repository, and from its Git LFS object store.
    The verifier reported no Phase 3B or P0–P2 implementation mismatch.
14. **Ready to train P0:** implementation-ready, but real training is blocked
    on this machine until CUDA is available.
15. **Ready to train all P1/P2:** NO. CUDA is unavailable, and P2 MobileNetV3
    remains protocol-blocked pending an evidence-backed variant decision.

## Final status

`BLOCKED_PROTOCOL`

Protocol blocking (the unresolved MobileNetV3 variant and unavailable complete
legacy snapshot) takes precedence over the simultaneous GPU-only execution
block. P0 and all resolved P1/P2 implementations are frozen and statically
ready; no test evidence has been accessed.
