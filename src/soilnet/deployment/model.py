from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms

from soilnet.models.experiment_soilnet import ExperimentSoilNet


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FrozenNoLISoilNet:
    def __init__(
        self,
        checkpoint_path: str | Path,
        expected_sha256: str,
        expected_experiment_id: str,
        *,
        num_classes: int = 10,
        image_size: tuple[int, int] = (224, 224),
        normalization_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalization_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        torch_threads: int = 4,
        device: str = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        observed_sha = sha256_file(self.checkpoint_path)
        if observed_sha.lower() != expected_sha256.lower():
            raise RuntimeError(
                f"Checkpoint SHA256 mismatch. Expected {expected_sha256}, observed {observed_sha}"
            )

        torch.set_num_threads(max(1, int(torch_threads)))
        self.device = torch.device(device)

        self.model = ExperimentSoilNet(
            num_classes=int(num_classes),
            use_li_signal=False,
            backbone_pretrained=False,
        )

        payload: dict[str, Any] = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if payload.get("experiment_id") != expected_experiment_id:
            raise RuntimeError(f"Unexpected experiment_id: {payload.get('experiment_id')!r}")
        if payload.get("li_consumed_by_model") is not False:
            raise RuntimeError("Checkpoint is not the frozen no-LI model.")

        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(tuple(image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=list(normalization_mean),
                std=list(normalization_std),
            ),
        ])

        self.experiment_id = expected_experiment_id
        self.checkpoint_sha256 = observed_sha
        self.checkpoint_epoch = int(payload.get("epoch", -1))

    @torch.inference_mode()
    def predict_pil(self, image: Image.Image) -> dict[str, float]:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        regression, _classification = self.model(tensor, None)
        values = regression.detach().cpu()[0].tolist()

        sm0 = float(values[0]) * 100.0
        sm20 = float(values[1]) * 100.0
        if not (math.isfinite(sm0) and math.isfinite(sm20)):
            raise RuntimeError("Non-finite prediction.")

        return {
            "sm0_raw": sm0,
            "sm20_raw": sm20,
            "sm0_controller": min(100.0, max(0.0, sm0)),
            "sm20_controller": min(100.0, max(0.0, sm20)),
        }

    def predict_path(self, path: str | Path) -> dict[str, float]:
        with Image.open(path) as image:
            return self.predict_pil(image)
