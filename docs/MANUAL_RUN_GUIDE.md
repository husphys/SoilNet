# Manual P0 run guide

1. Activate the existing environment: `conda activate soilnet`.
2. Open `notebooks/final_experiments/00_gpu_environment_check.ipynb` and run
   all cells. Proceed only when CUDA is `True` and `GPU_SMOKE` is `PASS`.
3. Open `01_P0_final_soilnet.ipynb` and run cells sequentially. Before full
   training, confirm locked hashes PASS, GPU PASS, μ27 loads 577/577 keys,
   `PREFLIGHT: PASS`, `optimizer_step_performed=False`, and test loader
   `NOT_CREATED`.
4. In section 13 set `RUN_TRAINING = True`; keep
   `RESUME_IF_AVAILABLE = True`, then run section 14 once.
5. After epoch 60 confirm `TRAINING_COMPLETED`, `epoch_60_final.pth`, its SHA
   file, validation artifacts, and `test_evaluated=false` in metadata.
6. Do **not** run notebook 09 until the paper's exact experiment set is decided
   and every selected model is frozen.

If CUDA is unavailable, a hash fails, resume identities differ, or batch 32
causes OOM, stop. Do not change the environment, batch size, protocol, split,
or checkpoint from inside the notebook.
