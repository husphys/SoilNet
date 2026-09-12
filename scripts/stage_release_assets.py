#!/usr/bin/env python3
"""Stage immutable checkpoint files for a GitHub Release outside the repository."""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path


def sources(run_root: Path, checkpoint_root: Path, data_root: Path):
    return {
    "P1_noSSL_validation_best_regression.pth": (
        run_root / "P1_SOILNET_IMAGENET_LI_NO_SSL_BESTREG/validation_best_regression.pth",
        "50c0f15567ab71a87b04569790b9cc7afd016989e3899d56e1898d8a6fcfb044",
    ),
    "P2_validation_best_regression.pth": (
        run_root / "P2_MOBILEVITV2_IMAGENET_LI_BESTREG/validation_best_regression.pth",
        "0a080d5129a2e57f1c2baf420b5af9aadf913fc0aab0d0050b6d662f24c28a8d",
    ),
    "P3_validation_best_regression.pth": (
        run_root / "P3_MOBILEVITV2_VICREG_LI_BESTREG/supervised/validation_best_regression.pth",
        "e2119a5b7a289925584fba8d3cd57040b7835461fc39d686bda464227de50020",
    ),
    "P3_mobilevitv2_vicreg_encoder_final.pth": (
        run_root / "P3_MOBILEVITV2_VICREG_LI_BESTREG/pretraining/mobilevitv2_vicreg_encoder_final.pth",
        "32a1a62233d39bf6d5de5f803e7781be81e6a96851efc1531f87ec6f80ec5580",
    ),
    "SoilNet_vicreg_mu27_initialization.pth": (
        checkpoint_root / "checkpoints_VicReg_original_NEW/vicreg_model_final_mu_27.0.pth",
        "42599ac025b8d8c8c5b8cca36624665f155e0ff588cd90de2fd47b08cd2d60d8",
    ),
    "SoilNet_pre_vicreg_imagenet_initialization.pth": (
        data_root / "Soil_Labeled_Data/soilNet_di chuyen tu o C/Model/SoilNet_orginal.pth",
        "7fbfac2b637613c6f48b3fef6f2c56be07080eedfba73da1d2b553e0b41f3631",
    ),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/soilnet-manuscript-repro-v1-assets"))
    parser.add_argument("--run-root", type=Path, default=os.environ.get("RUN_ARTIFACT_ROOT"))
    parser.add_argument("--checkpoint-root", type=Path, default=os.environ.get("CHECKPOINT_ROOT"))
    parser.add_argument("--data-root", type=Path, default=os.environ.get("DATA_ROOT"))
    args = parser.parse_args()
    if not all((args.run_root, args.checkpoint_root, args.data_root)):
        raise RuntimeError("RUN_ARTIFACT_ROOT, CHECKPOINT_ROOT, and DATA_ROOT are required")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, (source, expected) in sources(
        args.run_root.resolve(), args.checkpoint_root.resolve(), args.data_root.resolve()
    ).items():
        observed = sha256(source)
        if observed != expected:
            raise RuntimeError(f"Frozen source changed for {name}: {observed}")
        destination = output / name
        shutil.copyfile(source, destination)
        if sha256(destination) != expected:
            raise RuntimeError(f"Staged copy changed for {name}")
        print(f"{expected}  {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
