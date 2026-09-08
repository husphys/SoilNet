from __future__ import annotations

import os
from pathlib import Path
import yaml

from soilnet.deployment.fuzzy import SugenoIrrigationController
from soilnet.deployment.model import FrozenNoLISoilNet


def main() -> None:
    config_path = Path(os.environ.get("SOILNET_DEMO_CONFIG", "config/deployment/iphone_demo.yaml"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checkpoint = os.environ.get("SOILNET_P1_CHECKPOINT", "")
    if not checkpoint:
        raise SystemExit("SOILNET_P1_CHECKPOINT is not set.")

    m = config["model"]
    model = FrozenNoLISoilNet(
        checkpoint,
        expected_sha256=m["expected_checkpoint_sha256"],
        expected_experiment_id=m["expected_experiment_id"],
        num_classes=m["num_classes"],
        image_size=tuple(m["image_size"]),
        normalization_mean=tuple(m["normalization_mean"]),
        normalization_std=tuple(m["normalization_std"]),
        torch_threads=m.get("torch_threads", 4),
    )
    fuzzy = SugenoIrrigationController(config["fuzzy"])

    print("Checkpoint/model contract: PASS")
    print("Experiment:", model.experiment_id)
    print("Epoch:", model.checkpoint_epoch)
    print("SHA256:", model.checkpoint_sha256)
    for sm0, sm20 in [(10, 10), (35, 45), (50, 50), (80, 80)]:
        d = fuzzy.evaluate(sm0, sm20)
        print(f"SM0={sm0:>3}, SM20={sm20:>3} -> {d.duration_seconds:>3} s")


if __name__ == "__main__":
    main()
