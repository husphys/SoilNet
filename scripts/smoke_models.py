#!/usr/bin/env python3
"""CPU-only construction smoke test; downloads and training are disabled."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import timm
import torch

from soilnet.io import write_json
from soilnet.models.soilnet import SoilNetDualHead


def main() -> int:
    results = {}
    for name in ("mobilenetv2_100.ra_in1k", "mobilevitv2_050"):
        model = timm.create_model(name, pretrained=False)
        results[name] = {"constructed": True, "parameter_count": sum(p.numel() for p in model.parameters()), "pretrained_weights_downloaded": False}
        del model
    model = SoilNetDualHead(backbone_pretrained=False).eval()
    with torch.inference_mode():
        regression, classification = model(torch.zeros(1, 3, 224, 224), torch.zeros(1, 1))
    results["SoilNetDualHead"] = {
        "constructed": True,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "pretrained_weights_downloaded": False,
        "input_shapes": {"image": [1, 3, 224, 224], "light": [1, 1]},
        "output_shapes": {
            "regression": list(regression.shape),
            "classification": list(classification.shape),
        },
    }
    write_json(REPO / "results" / "audit" / "model_smoke_test.json", results)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
