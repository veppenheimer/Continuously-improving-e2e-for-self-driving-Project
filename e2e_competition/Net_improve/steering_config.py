#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
九类转向角配置与映射工具。
"""

from bisect import bisect_left

STEERING_CLASSES = [1.72, 1.64, 1.5, 0.0, -1.5, -1.56, -1.58, -1.6, -1.62]
NUM_CLASSES = len(STEERING_CLASSES)


def angle_to_class(angle):
    values = sorted(STEERING_CLASSES)
    pos = bisect_left(values, angle)
    if pos == 0:
        nearest = values[0]
    elif pos == len(values):
        nearest = values[-1]
    else:
        left = values[pos - 1]
        right = values[pos]
        nearest = left if abs(angle - left) <= abs(angle - right) else right
    return STEERING_CLASSES.index(nearest)


def class_to_angle(cls_idx):
    return float(STEERING_CLASSES[int(cls_idx)])
