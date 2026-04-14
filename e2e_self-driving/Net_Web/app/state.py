"""内存中的训练进度与线程同步原语（进程重启后丢失）。"""

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
            "competitionClass": None,
            "competitionLite": None,
            "baselineProgress": 0.0,
            "domainAugmentationProgress": None,
            "domainAugmentationText": None,
            "augmentedProgress": None,
            "competitionClassProgress": None,
            "competitionClassText": None,
            "competitionLiteProgress": None,
            "competitionLiteText": None,
            "message": None,
        }
        return ctrl


def unregister_task(task_id: str) -> None:
    with _lock:
        _controls.pop(task_id, None)
        _progress.pop(task_id, None)


def release_controls(task_id: str) -> None:
    """训练线程结束时调用：移除 pause/stop 句柄，保留进度曲线供 GET /progress 读取。"""
    with _lock:
        _controls.pop(task_id, None)


def get_controls(task_id: str) -> Optional[TaskControls]:
    with _lock:
        return _controls.get(task_id)


def merge_progress(task_id: str, patch: dict[str, Any]) -> None:
    with _lock:
        base = _progress.setdefault(task_id, {})
        for k, v in patch.items():
            if k == "baseline" or k == "augmented":
                if v is None:
                    base[k] = None
                elif isinstance(v, dict) and isinstance(base.get(k), dict):
                    base[k] = {**base[k], **v}
                else:
                    base[k] = v
            else:
                base[k] = v


def get_progress(task_id: str) -> dict[str, Any]:
    with _lock:
        return dict(_progress.get(task_id, {}))


def append_loss_point(
    task_id: str,
    branch: str,
    epoch: int,
    train_loss: float,
    val_loss: float,
) -> None:
    pt = {"epoch": epoch, "trainLoss": train_loss, "valLoss": val_loss}
    with _lock:
        p = _progress.setdefault(task_id, {})
        cur = p.get(branch)
        if not isinstance(cur, dict):
            cur = {"trainLossSeries": [], "valLossSeries": []}
            p[branch] = cur
        cur["trainLossSeries"].append(pt)
        cur["valLossSeries"].append(pt)
