#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
九类转向角定义与标签映射（与竞赛离散档位一致）。
类别索引 0..8 对应下列物理转向角（含义与原回归任务输出一致，仅为离散化）。
"""

import numpy as np

# 按指定顺序：右转大 → 中 → 小 → 直行 → 左转档位
STEERING_CLASSES = (
    1.72,
    1.64,
    1.5,
    0.0,
    -1.5,
    -1.56,
    -1.58,
    -1.6,
    -1.62,
)

NUM_CLASSES = len(STEERING_CLASSES)


def angle_to_class(angle):
    """将任意浮点转向角映射到最近的类别索引。"""
    a = float(angle)
    arr = np.array(STEERING_CLASSES, dtype=np.float64)
    return int(np.argmin(np.abs(arr - a)))


def class_to_angle(class_index):
    """类别索引 → 对应转向角。"""
    return float(STEERING_CLASSES[int(class_index)])
