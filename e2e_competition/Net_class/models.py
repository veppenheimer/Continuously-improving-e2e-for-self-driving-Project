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

from steering_models import PrototypeResidualNet, load_state_dict_flexible
from steering_config import NUM_CLASSES


class AutoDriveLegacyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),
            nn.Dropout(0.5),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(64 * 8 * 13, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.Linear(10, NUM_CLASSES + 1),
        )

    def forward(self, input_tensor):
        x = input_tensor.view(input_tensor.size(0), 3, 120, 160)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.linear_layers(x)


class AutoDriveNet(PrototypeResidualNet):
    def __init__(self, *, use_pretrained: bool = True):
        super().__init__(num_classes=NUM_CLASSES, use_pretrained=use_pretrained)


def build_model_for_checkpoint(state_dict):
    model = AutoDriveNet()
    if load_state_dict_flexible(model, state_dict):
        return model
    legacy = AutoDriveLegacyNet()
    legacy.load_state_dict(state_dict, strict=True)
    return legacy
