"""单张图像预处理 + 推理。

默认兼容旧模型；若 checkpoint 内包含 `preprocess` 元数据，则按该配置进行推理。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
NET_WEB_DIR = CURRENT_DIR.parents[1]
REPO_ROOT = CURRENT_DIR.parents[3]
for candidate in (NET_WEB_DIR, REPO_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from models import AutoDriveNet, build_model_for_checkpoint
from steering_preprocess import (
    DEFAULT_PREPROCESS_CONFIG,
    PreprocessConfig,
    preprocess_bgr_to_tensor,
    preprocess_config_from_dict,
)


def _resolve_preprocess(model: torch.nn.Module) -> PreprocessConfig:
    return getattr(model, "preprocess_config", DEFAULT_PREPROCESS_CONFIG)


def _bgr_to_tensor(bgr: np.ndarray, config: PreprocessConfig) -> torch.Tensor:
    return preprocess_bgr_to_tensor(bgr, config=config)


def _prepare_model_input(model: torch.nn.Module, bgr: np.ndarray) -> torch.Tensor:
    tensor = _bgr_to_tensor(bgr, config=_resolve_preprocess(model))
    num_frames = max(1, int(getattr(model, "num_frames", 1)))
    if num_frames > 1:
        tensor = torch.cat([tensor] * num_frames, dim=1)
    return tensor


def predict_image(model: torch.nn.Module, bgr: np.ndarray, device: torch.device) -> float:
    model.eval()
    x = _prepare_model_input(model, bgr).to(device)
    with torch.no_grad():
        y = model(x)
    return float(y.cpu().numpy().reshape(-1)[0])


def load_checkpoint_model(ckpt_path: Path, device: torch.device) -> AutoDriveNet:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state = ckpt.get("model", ckpt)
    model = build_model_for_checkpoint(state).to(device)
    preprocess = DEFAULT_PREPROCESS_CONFIG
    if isinstance(ckpt, dict):
        preprocess = preprocess_config_from_dict(ckpt.get("preprocess"), fallback=DEFAULT_PREPROCESS_CONFIG)
    setattr(model, "preprocess_config", preprocess)
    model.eval()
    return model


def load_image_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法解码图像")
    return img


def load_image_path(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if img is None:
        raise ValueError("无法读取图像")
    return img
