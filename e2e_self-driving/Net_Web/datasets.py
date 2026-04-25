#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from steering_preprocess import NumpyRandomState


def _imread_bgr(path: str):
    p = os.fspath(path)
    buf = np.fromfile(p, dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _is_albumentations_transform(transform: Any) -> bool:
    return hasattr(transform, "__call__") and transform.__class__.__module__.startswith("albumentations")


def _parse_frame_index(path: str | Path) -> int:
    stem = Path(path).stem
    if "_" not in stem:
        raise ValueError(f"missing frame index in filename: {path}")
    return int(stem.split("_", 1)[0])


def _read_list_file(file_path: str):
    file_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                img_path, angle_str = line.rsplit(" ", 1)
                angle = float(angle_str)
            except ValueError:
                continue
            file_list.append((img_path, angle))
    return file_list


class _BaseTemporalDataset(Dataset):
    def __init__(
        self,
        *,
        transform=None,
        deterministic_seed: int | None = None,
        num_frames: int = 1,
        frame_stride: int = 1,
        cache_images: bool = True,
    ):
        self.transform = transform
        self.deterministic_seed = deterministic_seed
        self.num_frames = max(1, int(num_frames))
        self.frame_stride = max(1, int(frame_stride))
        self.cache_images = bool(cache_images)
        self._image_cache: dict[str, np.ndarray] = {}
        self.frame_lookup: dict[tuple[str, int], str] = {}
        self.file_list: list[tuple[str, float]] = []
        self.angles: list[float] = []

    def _read_bgr(self, image_path: str) -> np.ndarray:
        if self.cache_images:
            cached = self._image_cache.get(image_path)
            if cached is not None:
                return cached.copy()
        frame = _imread_bgr(image_path)
        if frame is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")
        if self.cache_images:
            self._image_cache[image_path] = frame
            return frame.copy()
        return frame

    def _build_frame_lookup(self, search_roots: list[Path]) -> None:
        for root in search_roots:
            if not root.exists():
                continue
            files = [root] if root.is_file() else root.rglob("*")
            for image_file in files:
                if not image_file.is_file() or image_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                try:
                    key = (str(image_file.parent), _parse_frame_index(image_file))
                except ValueError:
                    continue
                self.frame_lookup[key] = str(image_file)

    def _load_frame_stack(self, image_path: str) -> list[np.ndarray]:
        current_path = Path(image_path)
        current_index = _parse_frame_index(current_path)
        folder_key = str(current_path.parent)
        frames: list[np.ndarray] = []
        last_valid_path = image_path
        for offset in range(self.num_frames - 1, -1, -1):
            target_index = current_index - offset * self.frame_stride
            candidate_path = self.frame_lookup.get((folder_key, target_index), last_valid_path)
            frame = self._read_bgr(candidate_path)
            frames.append(frame)
            last_valid_path = candidate_path
        return frames

    def _pack_default_frames(self, frames: list[np.ndarray]) -> torch.Tensor:
        tensors = []
        for frame in frames:
            img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
            tensors.append(torch.from_numpy(np.transpose(img_hsv, (2, 0, 1))).float())
        return torch.cat(tensors, dim=0)

    def _apply_transform(self, img_bgr: np.ndarray | list[np.ndarray], idx: int) -> torch.Tensor:
        frames = img_bgr if isinstance(img_bgr, list) else [img_bgr]
        if self.transform is None:
            return self._pack_default_frames(frames)

        if _is_albumentations_transform(self.transform) or getattr(self.transform, "returns_dict", False):
            rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
            payload: dict[str, Any] = {"image": rgb_frames[0]}
            for frame_idx, frame in enumerate(rgb_frames[1:], start=1):
                payload[f"image{frame_idx}"] = frame
            if self.deterministic_seed is not None:
                with NumpyRandomState(self.deterministic_seed + idx):
                    result = self.transform(**payload)
            else:
                result = self.transform(**payload)

            if isinstance(result, dict):
                tensor_list = [result["image"]]
                for frame_idx in range(1, len(frames)):
                    tensor_list.append(result[f"image{frame_idx}"])
                return torch.cat(tensor_list, dim=0)
            if torch.is_tensor(result):
                return result
            raise TypeError(f"unsupported transform result type: {type(result)!r}")

        if len(frames) > 1:
            raise RuntimeError("temporal stacking requires an albumentations-compatible transform")
        img_hsv = cv2.cvtColor(frames[0], cv2.COLOR_BGR2HSV)
        img_pil = Image.fromarray(img_hsv)
        return self.transform(img_pil)

    def __getitem__(self, i):
        img_path, angle = self.file_list[i]
        frames = self._load_frame_stack(img_path)
        img = self._apply_transform(frames if self.num_frames > 1 else frames[-1], i)
        label = torch.tensor([angle], dtype=torch.float32)
        return img, label

    def __len__(self):
        return len(self.file_list)


class AutoDriveDataset(_BaseTemporalDataset):
    def __init__(
        self,
        data_folder,
        mode,
        transform=None,
        deterministic_seed: int | None = None,
        *,
        split_name: str | None = None,
        num_frames: int = 1,
        frame_stride: int = 1,
        cache_images: bool = True,
    ):
        super().__init__(
            transform=transform,
            deterministic_seed=deterministic_seed,
            num_frames=num_frames,
            frame_stride=frame_stride,
            cache_images=cache_images,
        )
        self.data_folder = data_folder
        self.mode = mode.lower()
        assert self.mode in {"train", "val", "test", "train_clean", "val_clean", "test_clean", "val_style_real"}
        file_path = self._resolve_split_file(split_name)
        self.file_list = _read_list_file(file_path)
        self.angles = [angle for _, angle in self.file_list]
        self._build_frame_lookup([Path(self.data_folder)])

    def _resolve_split_file(self, split_name: str | None) -> str:
        def _candidate_names(name: str) -> list[str]:
            key = name.lower()
            if key == "train":
                return ["train_clean.txt", "train.txt"]
            if key == "val":
                return ["val_clean.txt", "val.txt"]
            if key == "test":
                return ["test_clean.txt", "test.txt"]
            if key == "train_clean":
                return ["train_clean.txt", "train.txt"]
            if key == "val_clean":
                return ["val_clean.txt", "val.txt"]
            if key == "test_clean":
                return ["test_clean.txt", "test.txt"]
            if key == "val_style_real":
                return ["val_style_real.txt", "val_style.txt", "val.txt"]
            return [f"{key}.txt"]

        requested = split_name or self.mode
        requested_path = Path(requested)
        if requested_path.is_file():
            return str(requested_path)

        for candidate in _candidate_names(requested):
            file_path = Path(self.data_folder) / candidate
            if file_path.is_file():
                return str(file_path)
        tried = ", ".join(str(Path(self.data_folder) / name) for name in _candidate_names(requested))
        raise FileNotFoundError(f"split file not found for mode={self.mode} split_name={split_name}; tried: {tried}")


class AutoDriveListDataset(_BaseTemporalDataset):
    def __init__(
        self,
        list_file: str,
        transform=None,
        deterministic_seed: int | None = None,
        *,
        num_frames: int = 1,
        frame_stride: int = 1,
        cache_images: bool = True,
    ):
        super().__init__(
            transform=transform,
            deterministic_seed=deterministic_seed,
            num_frames=num_frames,
            frame_stride=frame_stride,
            cache_images=cache_images,
        )
        self.list_file = list_file
        self.file_list = _read_list_file(list_file)
        self.angles = [angle for _, angle in self.file_list]
        parent_dirs = sorted({str(Path(img_path).parent) for img_path, _ in self.file_list})
        self._build_frame_lookup([Path(parent) for parent in parent_dirs])
