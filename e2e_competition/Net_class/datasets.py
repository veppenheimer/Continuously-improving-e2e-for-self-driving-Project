#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from steering_config import angle_to_class_and_delta
from steering_preprocess import NumpyRandomState


def _imread_bgr(path: str):
    p = os.fspath(path)
    buf = np.fromfile(p, dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _is_albumentations_transform(transform: Any) -> bool:
    return hasattr(transform, "__call__") and transform.__class__.__module__.startswith("albumentations")


class AutoDriveDataset(Dataset):
    def __init__(self, data_folder, mode, transform=None, deterministic_seed: int | None = None):
        self.data_folder = data_folder
        self.mode = mode.lower()
        self.transform = transform
        self.deterministic_seed = deterministic_seed
        assert self.mode in {"train", "val", "test"}

        file_path = os.path.join(data_folder, f"{self.mode}.txt")
        self.file_list = []
        self.class_indices = []
        self.raw_angles = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_path, angle_str = line.rsplit(" ", 1)
                raw_angle = float(angle_str)
                cls, delta = angle_to_class_and_delta(raw_angle)
                self.file_list.append((img_path, cls, raw_angle, delta))
                self.class_indices.append(cls)
                self.raw_angles.append(raw_angle)

    def _apply_transform(self, img_bgr: np.ndarray, idx: int):
        if self.transform is None:
            img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            img_hsv = img_hsv.astype(np.float32) / 255.0
            return torch.from_numpy(np.transpose(img_hsv, (2, 0, 1))).float()

        if _is_albumentations_transform(self.transform):
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            if self.deterministic_seed is not None:
                with NumpyRandomState(self.deterministic_seed + idx):
                    return self.transform(image=img_rgb)["image"]
            return self.transform(image=img_rgb)["image"]

        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        img_pil = Image.fromarray(img_hsv)
        return self.transform(img_pil)

    def __getitem__(self, i):
        img_path, class_idx, raw_angle, delta_target = self.file_list[i]
        img = _imread_bgr(img_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {img_path}")
        img = self._apply_transform(img, i)
        return (
            img,
            torch.tensor(class_idx, dtype=torch.long),
            torch.tensor(raw_angle, dtype=torch.float32),
            torch.tensor(delta_target, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.file_list)

