#!/usr/bin/env python3
"""Reproduce validation/test inference from a frozen checkpoint.

This command never trains or updates weights. P3 is structurally restricted to
validation: its test split cannot be selected and no P3 test dataset is built.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from soilnet.data import SoilNetDataset
from soilnet.evaluation.predictions import prediction_rows_and_metrics, write_prediction_artifacts
from soilnet.io import load_yaml, sha256_file
from soilnet.models import build_frozen_model
from soilnet.models.factory import extract_state_dict
from soilnet.reproducibility import set_determinism


EXPERIMENTS = {
    "P0": "P0_final_soilnet_v4_bestreg.yaml",
    "P1_noLI": "P1_no_li_bestreg.yaml",
    "P1_noSSL": "P1_no_ssl_bestreg.yaml",
    "P2": "P2_mobilevitv2_imagenet_li_bestreg.yaml",
    "P3": "P3_mobilevitv2_vicreg_li_bestreg.yaml",
}


def checkpoint_record(key: str) -> dict:
    manifest = json.loads((REPO / "checkpoints/checkpoint_manifest.json").read_text(encoding="utf-8"))
    return next(row for row in manifest["checkpoints"] if row["key"] == key)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, choices=tuple(EXPERIMENTS))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--instantiate-only", action="store_true")
    args = parser.parse_args()

    if args.experiment == "P3" and args.split == "test":
        raise RuntimeError("P3_TEST_PROHIBITED: validation-only follow-up; test set not opened")
    record = checkpoint_record(args.experiment)
    checkpoint = (args.checkpoint or (REPO / record["location"])).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Download or provide the canonical checkpoint: {checkpoint}")
    observed = sha256_file(checkpoint)
    if observed != record["sha256"]:
        raise RuntimeError(f"Checkpoint SHA256 mismatch: {observed}")
    config = load_yaml(REPO / "config/experiments" / EXPERIMENTS[args.experiment])
    set_determinism(int(config["seed"]))
    model = build_frozen_model(config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {str(key).removeprefix("module."): value for key, value in extract_state_dict(payload).items()}
    result = model.load_state_dict(state, strict=True)
    summary = {
        "experiment": args.experiment,
        "checkpoint_sha256": observed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "strict_load": not result.missing_keys and not result.unexpected_keys,
        "requested_split": args.split,
        "P3_TEST_SET_OPENED": False if args.experiment == "P3" else "NOT_APPLICABLE",
    }
    if args.instantiate_only:
        print(json.dumps(summary, indent=2))
        return 0
    if args.data_root is None or args.output_dir is None:
        raise RuntimeError("--data-root and --output-dir are required for inference")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    split_manifest = REPO / "data/splits/final_clean_split_v1.csv"
    if sha256_file(split_manifest) != config["split_sha256"]:
        raise RuntimeError("Locked split SHA256 mismatch")
    transform = transforms.Compose([
        transforms.Resize(tuple(config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=config["normalization_mean"], std=config["normalization_std"]),
    ])
    dataset = SoilNetDataset(split_manifest, args.data_root.resolve(), transform=transform, split=args.split)
    expected_count = int(config["split_counts"][args.split])
    if len(dataset) != expected_count:
        raise RuntimeError(f"Split count mismatch: {len(dataset)} != {expected_count}")
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=0)
    device = torch.device(args.device)
    model.to(device)
    rows, metrics = prediction_rows_and_metrics(model, loader, device, config)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_prediction_artifacts(
        output / f"{args.experiment}_{args.split}_predictions.csv",
        output / f"{args.experiment}_{args.split}_metrics.json",
        rows,
        metrics,
    )
    summary.update({"sample_count": len(dataset), "output_dir": str(output), "metrics": metrics})
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
