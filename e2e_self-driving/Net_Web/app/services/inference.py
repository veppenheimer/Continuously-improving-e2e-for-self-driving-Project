"""单张图像预处理 + 推理（与 datasets 中 HSV + Resize + ToTensor 一致）。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from models import AutoDriveNet


def _bgr_to_tensor(bgr: np.ndarray) -> torch.Tensor:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    pil = Image.fromarray(hsv)
    t = transforms.Compose(
        [
            transforms.Resize((120, 160)),
            transforms.ToTensor(),
        ]
    )(pil)
    return t.unsqueeze(0)


def predict_image(model: torch.nn.Module, bgr: np.ndarray, device: torch.device) -> float:
    model.eval()
    x = _bgr_to_tensor(bgr).to(device)
    with torch.no_grad():
        y = model(x)
    return float(y.cpu().numpy().reshape(-1)[0])


def load_checkpoint_model(ckpt_path: Path, device: torch.device) -> AutoDriveNet:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model = AutoDriveNet().to(device)
    state = ckpt.get("model", ckpt)
    if isinstance(state, dict):
        model.load_state_dict(state)
    model.eval()
    return model


def load_image_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法解码图像")
    return img


def load_image_path(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError("无法读取图像")
    return img
