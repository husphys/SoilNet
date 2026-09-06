from __future__ import annotations

import timm
import torch
from torch import nn


class SoilNetDualHead(nn.Module):
    """Canonicalized legacy SoilNet architecture.

    ``backbone_pretrained`` is explicit because legacy SoilNet used ImageNet
    initialization while the legacy baseline script used random initialization.
    """

    def __init__(self, num_classes: int = 10, use_light: bool = True, backbone_pretrained: bool = False):
        super().__init__()
        self.use_light = use_light
        self.initial_conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        mnv2_first = timm.create_model("mobilenetv2_100.ra_in1k", pretrained=backbone_pretrained)
        self.mnv2_block1 = nn.Sequential(*list(mnv2_first.blocks.children())[0:3])
        self.channel_adapter = nn.Conv2d(32, 16, kernel_size=1, bias=False)
        mobilevit = timm.create_model("mobilevitv2_050", pretrained=backbone_pretrained)
        self.mobilevit_encoder = mobilevit.stages
        self.mvit_to_mnv2 = nn.Conv2d(256, 32, kernel_size=1, bias=False)
        mnv2_second = timm.create_model("mobilenetv2_100.ra_in1k", pretrained=backbone_pretrained)
        self.mnv2_block2 = nn.Sequential(*list(mnv2_second.blocks.children())[3:7])
        self.final_conv = nn.Conv2d(320, 1280, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        feature_count = 1280
        if use_light:
            self.light_dense = nn.Sequential(nn.Linear(1, 32), nn.ReLU(inplace=True))
            feature_count += 32
            self.reg_head = nn.Sequential(nn.Linear(feature_count, 128), nn.ReLU(inplace=True), nn.Linear(128, 2))
        self.cls_head = nn.Sequential(nn.Linear(feature_count, 128), nn.ReLU(inplace=True), nn.Linear(128, num_classes))

    def forward(self, image: torch.Tensor, light: torch.Tensor | None = None):
        features = self.initial_conv(image)
        features = self.mnv2_block1(features)
        features = self.channel_adapter(features)
        features = self.mobilevit_encoder(features)
        features = self.mvit_to_mnv2(features)
        features = self.mnv2_block2(features)
        features = self.final_conv(features)
        features = torch.flatten(self.pool(features), 1)
        if self.use_light:
            if light is None:
                raise ValueError("light input is required when use_light=True")
            features = torch.cat([features, self.light_dense(light)], dim=1)
            return self.reg_head(features), self.cls_head(features)
        return self.cls_head(features)
