# Final revalidation protocol

PHASE 3B status: **BLOCKED_GPU_ONLY**. Data and scientific protocol inputs are
locked; the current WSL CUDA runtime gate does not pass. No PHASE 3B training or
prospective test evaluation was performed.

## Preselection

- Architecture: SoilNet modified MobileViTV2 + MobileNetV2 with light-intensity input.
- SSL: VICReg, ImageNet-initialized chain, μ=27.
- Source: historical manuscript selection supplied before prospective evaluation.
- Prospective validation/test metrics consulted: no.

## Selected SSL initialization

- Scope/path: `CHECKPOINT_ROOT:checkpoints_VicReg_original_NEW/vicreg_model_final_mu_27.0.pth`
- SHA256: `42599ac025b8d8c8c5b8cca36624665f155e0ff588cd90de2fd47b08cd2d60d8`
- Lineage: `VERIFIED_LINEAGE` for the SSL generation. The raw final state is tensor-identical to the epoch-80 model state; its epoch/loss matches the 80-row μ=27 CSV; the generating notebook is byte-identical in DATA_ROOT and the pinned legacy checkout.
- Compatibility adapter: all 577 canonical keys and shapes are required. Exactly 269 historical `mobilevit_full.*` registration aliases are removed; 261 duplicated stage tensors must equal their `mobilevit_encoder.*` counterparts. Loading remains strict after normalization.
- Load policy: reproduce the historical fine-tune behavior by loading all 577 canonical matching keys, including the regression, classification, and LI branches. These heads were present in the SSL file but were not optimized by the VICReg objective; this limitation is explicit rather than silently reinitializing them.
- Pretraining exposure: soil unlabeled pool; disclose as `TRANSDUCTIVE_PRETRAINING`, not completely held-out inductive pretraining.

## Locked training protocol

- 15 epochs; batch 32; Adam; learning rate 5e-4; weight decay 0.0.
- Loss = MSE(SM_0, SM_20) + CrossEntropy(moisture_class), weights 1.0 and 1.0 from historical μ=27 code.
- Primary checkpoint is the fixed epoch-15 state. No test-based selection and no μ/architecture/SSL search.
- Images resize to 224×224 and use ImageNet normalization. LI and regression targets are divided by 100 internally.

## Data QC and split firewall

The pre-training `EXACT_DUPLICATE_CONFLICT_EXCLUSION` rule excludes all 130
records belonging to the 46 exact-image groups that disagree on `SM_0` or
`SM_20`. It retains no representative and uses no adjudication, voting,
averaging, conflict magnitude, or model-assisted cleaning. The resulting
`final_clean_manifest_v1.csv` has 1,927 rows and one SHA256 per row.

The deterministic 73%/15%/12% split preserves PHASE 2's locked exact/near
connected components. No capture/session or source-group identifier is present
in `labels.csv`, so those latent factors cannot be tested or invented from
filenames. Split generation is metric-free and must not be repeated after
training begins.

Training may instantiate only train and validation loaders after the GPU gate
passes. The test loader exists only in a separate explicit final-evaluation
command. Test evaluation must refuse to run when
`results/final/test_evaluation_completed.json` already exists and must write
that marker with timestamp, Git state, model SHA256, split SHA256, and config
SHA256 after the one permitted evaluation.
