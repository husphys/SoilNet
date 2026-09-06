from __future__ import annotations

import torch

from soilnet.models.soilnet import SoilNetDualHead


class ExperimentSoilNet(SoilNetDualHead):
    """State-compatible SoilNet with explicit feature and no-LI paths.

    The canonical parent is always constructed with its historical LI branch,
    so P0 state keys and shapes remain identical. The no-LI ablation replaces
    the 32-D sample-dependent LI embedding with zeros; it does not silently
    change head capacity or checkpoint coverage.
    """

    def __init__(self, *, num_classes: int = 10, use_li_signal: bool = True, backbone_pretrained: bool = False):
        super().__init__(num_classes=num_classes, use_light=True, backbone_pretrained=backbone_pretrained)
        self.use_li_signal = use_li_signal

    def extract_features(self, image: torch.Tensor, light: torch.Tensor | None = None) -> torch.Tensor:
        features = self.initial_conv(image)
        features = self.mnv2_block1(features)
        features = self.channel_adapter(features)
        features = self.mobilevit_encoder(features)
        features = self.mvit_to_mnv2(features)
        features = self.mnv2_block2(features)
        features = self.final_conv(features)
        features = torch.flatten(self.pool(features), 1)
        if self.use_li_signal:
            if light is None:
                raise ValueError("light input is required when use_li_signal=True")
            light_features = self.light_dense(light)
        else:
            light_features = torch.zeros((features.shape[0], 32), dtype=features.dtype, device=features.device)
        return torch.cat([features, light_features], dim=1)

    def forward(self, image: torch.Tensor, light: torch.Tensor | None = None):
        features = self.extract_features(image, light)
        return self.reg_head(features), self.cls_head(features)
