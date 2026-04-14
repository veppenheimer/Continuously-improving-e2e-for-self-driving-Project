#!/usr/bin/env python
# -*- encoding: utf-8 -*-


# 导入系统库
import os
import numpy as np
import cv2
from PIL import Image  # 添加这一行

# 导入PyTorch库
import torch
from torch.utils.data import Dataset

from steering_config import angle_to_class


def _imread_bgr(path: str):
    p = os.fspath(path)
    buf = np.fromfile(p, dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


class AutoDriveDataset(Dataset):
    """
    数据集加载器（标签为 0..NUM_CLASSES-1 的类别索引）
    """

    def __init__(self, data_folder, mode, transform=None):
        """
        :参数 data_folder: # 数据文件所在文件夹根路径(train.txt和val.txt所在文件夹路径)
        :参数 mode: 'train' 或者 'val'
        :参数 transform: 图像变换
        """

        self.data_folder = data_folder
        self.mode = mode.lower()
        self.transform = transform

        assert self.mode in {'train', 'val'}

        # 读取图像列表路径
        if self.mode == 'train':
            file_path = os.path.join(data_folder, 'train.txt')
        else:
            file_path = os.path.join(data_folder, 'val.txt')

        self.file_list = list()
        with open(file_path, 'r', encoding="utf-8") as f:
            files = f.readlines()
            for file in files:
                if not file.strip():
                    continue
                img_path, angle_str = file.strip().rsplit(" ", 1)
                angle = float(angle_str)
                cls = angle_to_class(angle)
                self.file_list.append([img_path, cls])

    def __getitem__(self, i):
        """
        :参数 i: 图像检索号
        :返回: 返回第i个图像和类别标签（long）
        """
        # 读取图像
        img = _imread_bgr(self.file_list[i][0])
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {self.file_list[i][0]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)
        label = self.file_list[i][1]
        label = torch.tensor(label, dtype=torch.long)
        return img, label

    def __len__(self):
        """
        为了使用PyTorch的DataLoader,必须提供该方法.
        :返回: 加载的图像总数
        """
        return len(self.file_list)
