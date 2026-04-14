#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset


def _imread_bgr(path: str):
    """读取 BGR 图像。Windows 上 cv2.imread 对含中文等非 ASCII 路径会失败，改用 imdecode。"""
    p = os.fspath(path)
    buf = np.fromfile(p, dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


class AutoDriveDataset(Dataset):
    """数据集加载器（train.txt / val.txt：每行 `绝对路径 转向角`）"""

    def __init__(self, data_folder, mode, transform=None):
        self.data_folder = data_folder
        self.mode = mode.lower()
        self.transform = transform
        assert self.mode in {"train", "val"}

        if self.mode == "train":
            file_path = os.path.join(data_folder, "train.txt")
        else:
            file_path = os.path.join(data_folder, "val.txt")
        self.file_list = _read_list_file(file_path)

    def __getitem__(self, i):
        img = _imread_bgr(self.file_list[i][0])
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {self.file_list[i][0]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        label = self.file_list[i][1]
        label = torch.from_numpy(np.array([label], dtype=np.float32)).float()
        return img, label

    def __len__(self):
        return len(self.file_list)


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
            file_list.append([img_path, angle])
    return file_list


class AutoDriveListDataset(Dataset):
    """按指定 list 文件加载（每行 `绝对路径 转向角`）。"""

    def __init__(self, list_file: str, transform=None):
        self.transform = transform
        self.file_list = _read_list_file(list_file)

    def __getitem__(self, i):
        img = _imread_bgr(self.file_list[i][0])
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {self.file_list[i][0]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        label = self.file_list[i][1]
        label = torch.from_numpy(np.array([label], dtype=np.float32)).float()
        return img, label

    def __len__(self):
        return len(self.file_list)
