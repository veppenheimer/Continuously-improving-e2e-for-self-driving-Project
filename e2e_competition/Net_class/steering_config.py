#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Steering class definitions and decode helpers for the competition model."""

from __future__ import annotations

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch is available during training/inference
    torch = None


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
MAX_DELTA = 0.04

POS_GROUP = (0, 1, 2)
STRAIGHT_GROUP = (3,)
NEG_GROUP = (4, 5, 6, 7, 8)
STEERING_GROUPS = (POS_GROUP, STRAIGHT_GROUP, NEG_GROUP)

_CLASS_VALUES_NP = np.asarray(STEERING_CLASSES, dtype=np.float32)
_GROUP_VALUES_NP = tuple(_CLASS_VALUES_NP[list(group)] for group in STEERING_GROUPS)


def angle_to_class(angle: float) -> int:
    """Map a raw steering angle to the nearest steering prototype."""
    a = float(angle)
    diff = np.abs(_CLASS_VALUES_NP.astype(np.float64) - a)
    return int(np.argmin(diff))


def class_to_angle(class_index: int) -> float:
    """Return the prototype angle for the given class index."""
    return float(STEERING_CLASSES[int(class_index)])


def clamp_delta(delta: float) -> float:
    """Clamp the in-class steering offset to the supported residual range."""
    return float(np.clip(float(delta), -MAX_DELTA, MAX_DELTA))


def angle_to_class_and_delta(angle: float) -> tuple[int, float]:
    """Return the nearest class index together with the clamped residual."""
    cls = angle_to_class(angle)
    center = class_to_angle(cls)
    return cls, clamp_delta(float(angle) - center)


def split_output(output):
    """Split model output into classification logits and residual head."""
    if torch is not None and isinstance(output, torch.Tensor):
        if output.shape[-1] != NUM_CLASSES + 1:
            raise ValueError(f"expected last dimension {NUM_CLASSES + 1}, got {output.shape[-1]}")
        return output[..., :NUM_CLASSES], output[..., -1]

    arr = np.asarray(output)
    if arr.shape[-1] != NUM_CLASSES + 1:
        raise ValueError(f"expected last dimension {NUM_CLASSES + 1}, got {arr.shape[-1]}")
    return arr[..., :NUM_CLASSES], arr[..., -1]


def _decode_torch(logits, raw_delta):
    decoded = torch.zeros_like(raw_delta, dtype=logits.dtype)
    argmax_cls = torch.argmax(logits, dim=-1)

    for group in STEERING_GROUPS:
        group_index = torch.tensor(group, dtype=torch.long, device=logits.device)
        group_logits = torch.index_select(logits, dim=-1, index=group_index)
        group_probs = torch.softmax(group_logits, dim=-1)
        group_angles = logits.new_tensor([STEERING_CLASSES[idx] for idx in group])
        group_angle = (group_probs * group_angles).sum(dim=-1)

        mask = argmax_cls == group[0]
        for idx in group[1:]:
            mask = mask | (argmax_cls == idx)
        decoded = torch.where(mask, group_angle, decoded)

    delta = torch.tanh(raw_delta) * MAX_DELTA
    return decoded + delta


def _decode_numpy(logits, raw_delta):
    logits = np.asarray(logits, dtype=np.float32)
    raw_delta = np.asarray(raw_delta, dtype=np.float32)
    decoded = np.zeros_like(raw_delta, dtype=np.float32)
    argmax_cls = np.argmax(logits, axis=-1)

    for group, group_angles in zip(STEERING_GROUPS, _GROUP_VALUES_NP):
        group_logits = np.take(logits, group, axis=-1)
        group_logits = group_logits - np.max(group_logits, axis=-1, keepdims=True)
        group_probs = np.exp(group_logits)
        group_probs = group_probs / np.sum(group_probs, axis=-1, keepdims=True)
        group_angle = np.sum(group_probs * group_angles, axis=-1)
        mask = np.isin(argmax_cls, group)
        decoded = np.where(mask, group_angle, decoded)

    delta = np.tanh(raw_delta) * MAX_DELTA
    return decoded + delta


def decode_logits_and_delta(logits, raw_delta):
    """Decode grouped class logits plus residual into a continuous angle."""
    if torch is not None and isinstance(logits, torch.Tensor):
        return _decode_torch(logits, raw_delta)
    return _decode_numpy(logits, raw_delta)


def decode_output(output):
    """Decode the full model output tensor/array into continuous angles."""
    logits, raw_delta = split_output(output)
    return decode_logits_and_delta(logits, raw_delta)
