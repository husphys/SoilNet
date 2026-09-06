#!/usr/bin/env python3
"""Train the locked final protocol using train/validation only.

The runtime CUDA gate remains mandatory. This entrypoint never constructs a
prospective test loader.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from soilnet.data import SoilNetDataset
from soilnet.final_protocol import current_git_commit, normalize_historical_soilnet_state, verify_locked_inputs
from soilnet.io import resolve_paths, sha256_file, write_json
from soilnet.models import SoilNetDualHead
from soilnet.reproducibility import set_determinism


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if isinstance(checkpoint.get(key), dict):
                return checkpoint[key]
        if checkpoint and all(hasattr(value, "shape") for value in checkpoint.values()):
            return checkpoint
    raise ValueError("No safe model state_dict found")


def mean_loss(model, loader, device, regression_loss, classification_loss) -> float:
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for image, light, regression, classification in loader:
            predicted_regression, predicted_classification = model(image.to(device), light.to(device))
            loss = regression_loss(predicted_regression, regression.to(device))
            loss = loss + classification_loss(predicted_classification, classification.to(device))
            total += loss.item()
    return total / len(loader)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO / "config/experiments/final_revalidation_v2.yaml")
    parser.add_argument("--lock", type=Path, default=REPO / "results/audit/final_run_lock_v2.json")
    args = parser.parse_args()
    locked = verify_locked_inputs(REPO, args.config, args.lock)
    config, lock = locked["config"], locked["lock"]
    if not torch.cuda.is_available():
        raise RuntimeError("GPU_BLOCKED: final training requires CUDA")
    if config["training"]["epochs"] != 15 or config["training"]["primary_checkpoint"] != "epoch_15_final":
        raise RuntimeError("Primary epoch-15 protocol changed")

    set_determinism(int(config["reproducibility"]["seed"]))
    paths = resolve_paths()
    split_path = REPO / config["data"]["final_split_manifest"]
    transform = transforms.Compose([
        transforms.Resize(tuple(config["input"]["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config["input"]["image_normalization_mean"],
            std=config["input"]["image_normalization_std"],
        ),
    ])
    train_dataset = SoilNetDataset(split_path, paths["data_root"], transform=transform, split="train")
    validation_dataset = SoilNetDataset(split_path, paths["data_root"], transform=transform, split="validation")
    generator = torch.Generator().manual_seed(int(config["reproducibility"]["seed"]))
    train_loader = DataLoader(
        train_dataset, batch_size=int(config["training"]["batch_size"]), shuffle=True,
        num_workers=0, pin_memory=True, generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False,
        num_workers=0, pin_memory=True,
    )
    device = torch.device("cuda")
    model = SoilNetDualHead(num_classes=int(config["model"]["num_classes"]), use_light=True, backbone_pretrained=False)
    ssl_path = paths["checkpoint_root"] / config["initialization"]["relative_path"]
    if sha256_file(ssl_path) != lock["ssl_checkpoint_sha256"]:
        raise RuntimeError("Selected SSL checkpoint SHA256 changed")
    checkpoint = torch.load(ssl_path, map_location="cpu", weights_only=True)
    state = {str(key).removeprefix("module."): value for key, value in extract_state_dict(checkpoint).items()}
    state, _ = normalize_historical_soilnet_state(state, model.state_dict())
    model.load_state_dict(state, strict=True)
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    regression_loss = nn.MSELoss()
    classification_loss = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        total = 0.0
        for image, light, regression, classification in train_loader:
            optimizer.zero_grad(set_to_none=True)
            predicted_regression, predicted_classification = model(image.to(device), light.to(device))
            loss = regression_loss(predicted_regression, regression.to(device))
            loss = loss + classification_loss(predicted_classification, classification.to(device))
            loss.backward()
            optimizer.step()
            total += loss.item()
        history.append({
            "epoch": epoch, "train_loss": total / len(train_loader),
            "validation_loss": mean_loss(model, validation_loader, device, regression_loss, classification_loss),
        })
        print(json.dumps(history[-1]))

    output_dir = REPO / "results/final/FINAL_LEAKAGE_CONTROLLED_REVALIDATION"
    output_dir.mkdir(parents=True, exist_ok=True)
    primary_path = output_dir / "epoch_15_final.pth"
    torch.save({
        "epoch": 15, "model_state_dict": model.state_dict(),
        "config_sha256": sha256_file(args.config), "split_sha256": sha256_file(split_path),
    }, primary_path)
    with (output_dir / "training_validation_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "validation_loss"])
        writer.writeheader()
        writer.writerows(history)
    write_json(output_dir / "training_completed.json", {
        "git_commit": current_git_commit(REPO), "primary_checkpoint": primary_path.relative_to(REPO).as_posix(),
        "model_sha256": sha256_file(primary_path), "config_sha256": sha256_file(args.config),
        "split_sha256": sha256_file(split_path), "epochs_completed": 15,
        "test_loader_instantiated": False, "test_metrics_computed": False,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
