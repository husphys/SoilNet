# PHASE 3B data readiness

1. **Historical labeled records?** 2057.
2. **Duplicate-conflict groups?** 46 exact-SHA256 groups; derived from the manifests, not assumed.
3. **Records excluded?** 130; all members, no representative and no label adjudication.
4. **Final clean labeled count?** 1927 unique labeled images.
5. **Exact duplicate remaining?** 0.
6. **Label conflict remaining?** 0 exact-image target conflicts in the final clean manifest.
7. **Train/validation/test counts?** 1407 / 289 / 231.
8. **Group leakage?** 0 locked connected components cross splits; documented capture/source groups crossing splits: 0. Unrecorded latent sessions are NOT_TESTABLE.
9. **Near-duplicate group leakage?** 0 under the locked dHash Hamming-distance <= 5 connected-component policy.
10. **Manifest SHA256?** `8ff45054d4b8e3df9758d0c112dc16719572b2267906fb3a5ed5b3262a6732bd`.
11. **Split SHA256?** `8927b8223b8c4c234d264ad9ea62ac2df6161124a79787a2e71eeb5cd23eae2f`.
12. **SSL checkpoint unchanged?** Yes: `42599ac025b8d8c8c5b8cca36624665f155e0ff588cd90de2fd47b08cd2d60d8`, `VERIFIED_LINEAGE`.
13. **Protocol unchanged?** Yes: SoilNet + VICReg + ImageNet + mu=27; 15 epochs, batch 32, Adam, lr=5e-4, weight decay 0, epoch 15 primary; outputs SM_0, SM_20, moisture class, with LI auxiliary input.
14. **GPU status?** `CUDA_UNAVAILABLE` / runtime gate `GPU_BLOCKED`.
15. **Safe for final revalidation?** Data and protocol inputs are locked and pass static readiness checks, but execution is not currently safe/possible until the CUDA GPU gate passes. No training or prospective test evaluation occurred in PHASE 3B.

## Final status

**BLOCKED_GPU_ONLY**
