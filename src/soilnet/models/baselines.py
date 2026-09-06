from __future__ import annotations

import timm
import torch
from torch import nn


class TimmFusionDualHead(nn.Module):
    """Common ImageNet backbone + 32-D LI fusion + matched dual heads."""

    def __init__(self, model_name: str, *, num_classes: int = 10, pretrained: bool = True, use_light: bool = True):
        super().__init__()
        self.model_name = model_name
        self.use_light = use_light
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        feature_count = int(self.backbone.num_features)
        if use_light:
            self.light_dense = nn.Sequential(nn.Linear(1, 32), nn.ReLU(inplace=True))
            feature_count += 32
        self.reg_head = nn.Sequential(nn.Linear(feature_count, 128), nn.ReLU(inplace=True), nn.Linear(128, 2))
        self.cls_head = nn.Sequential(nn.Linear(feature_count, 128), nn.ReLU(inplace=True), nn.Linear(128, num_classes))

    def extract_features(self, image: torch.Tensor, light: torch.Tensor | None = None) -> torch.Tensor:
        features = self.backbone(image)
        if self.use_light:
            if light is None:
                raise ValueError("light input is required when use_light=True")
            features = torch.cat([features, self.light_dense(light)], dim=1)
        return features

    def forward(self, image: torch.Tensor, light: torch.Tensor | None = None):
        features = self.extract_features(image, light)
        return self.reg_head(features), self.cls_head(features)
