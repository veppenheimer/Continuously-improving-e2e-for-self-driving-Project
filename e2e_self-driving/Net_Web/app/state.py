"""In-memory training progress and task controls."""

from __future__ import annotations

import threading
from typing import Any, Optional


class TaskControls:
    def __init__(self) -> None:
        self.pause = threading.Event()
        self.stop = threading.Event()


_lock = threading.Lock()
_progress: dict[str, dict[str, Any]] = {}
_controls: dict[str, TaskControls] = {}


def register_task(task_id: str) -> TaskControls:
    with _lock:
        ctrl = TaskControls()
        _controls[task_id] = ctrl
        _progress[task_id] = {
            "status": "pending",
            "currentEpoch": 0,
            "totalEpochs": 0,
            "baseline": {"trainLossSeries": [], "valLossSeries": []},
            "augmented": None,
            "baselineProgress": 0.0,
            "domainAugmentationProgress": None,
            "domainAugmentationText": None,
            "augmentedProgress": None,
            "message": None,
        }
        return ctrl


def unregister_task(task_id: str) -> None:
    with _lock:
        _controls.pop(task_id, None)
        _progress.pop(task_id, None)


def release_controls(task_id: str) -> None:
    with _lock:
        _controls.pop(task_id, None)


def get_controls(task_id: str) -> Optional[TaskControls]:
    with _lock:
        return _controls.get(task_id)


def merge_progress(task_id: str, patch: dict[str, Any]) -> None:
    with _lock:
        base = _progress.setdefault(task_id, {})
        for key, value in patch.items():
            if key in {"baseline", "augmented"}:
                if value is None:
                    base[key] = None
                elif isinstance(value, dict) and isinstance(base.get(key), dict):
                    base[key] = {**base[key], **value}
                else:
                    base[key] = value
            else:
                base[key] = value


def get_progress(task_id: str) -> dict[str, Any]:
    with _lock:
        return dict(_progress.get(task_id, {}))


def append_loss_point(task_id: str, branch: str, epoch: int, train_loss: float, val_loss: float) -> None:
    point = {"epoch": epoch, "trainLoss": train_loss, "valLoss": val_loss}
    with _lock:
        progress = _progress.setdefault(task_id, {})
        current = progress.get(branch)
        if not isinstance(current, dict):
            current = {"trainLossSeries": [], "valLossSeries": []}
            progress[branch] = current
        current["trainLossSeries"].append(point)
        current["valLossSeries"].append(point)
