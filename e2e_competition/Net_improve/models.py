#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from steering_models import ImproveClassifierNet, load_state_dict_flexible
from steering_config import NUM_CLASSES


class DSConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x):
        return self.block(x)


class AutoDriveLegacyNetImprove(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
        )
        self.features = nn.Sequential(
            DSConvBlock(16, 24, stride=2, dropout=0.05),
            DSConvBlock(24, 32, stride=2, dropout=0.08),
            DSConvBlock(32, 48, stride=2, dropout=0.10),
            DSConvBlock(48, 64, stride=1, dropout=0.10),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(p=0.35),
            nn.Linear(32, NUM_CLASSES),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, input_tensor):
        x = input_tensor.view(input_tensor.size(0), 3, 120, 160)
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)


class AutoDriveNetImprove(ImproveClassifierNet):
    def __init__(self, *, use_pretrained: bool = True):
        super().__init__(num_classes=NUM_CLASSES, use_pretrained=use_pretrained)


def build_model_for_checkpoint(state_dict):
    model = AutoDriveNetImprove()
    if load_state_dict_flexible(model, state_dict):
        return model
    legacy = AutoDriveLegacyNetImprove()
    legacy.load_state_dict(state_dict, strict=True)
    return legacy


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
