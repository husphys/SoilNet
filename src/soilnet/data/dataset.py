from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class SoilNetDataset(Dataset):
    """Dataset backed by a normalized manifest and a configurable data root."""

    def __init__(self, manifest: Path, data_root: Path, transform=None, split: str | None = None):
        self.data_root = Path(data_root)
        self.transform = transform
        with Path(manifest).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.rows = [row for row in rows if split is None or row.get("split") == split]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        relative = row.get("effective_relative_path") or row["relative_path"]
        with Image.open(self.data_root / relative) as image:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        light = torch.tensor([float(row["light_value"]) / 100.0], dtype=torch.float32)
        regression = torch.tensor([float(row["SM_0"]) / 100.0, float(row["SM_20"]) / 100.0], dtype=torch.float32)
        classification = torch.tensor(int(float(row["moisture_class"])), dtype=torch.long)
        return image, light, regression, classification
