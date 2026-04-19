"""Helpers for persisted training progress and competition TensorBoard logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROGRESS_SNAPSHOT_NAME = "progress.json"


def read_progress_snapshot(task_dir: Path) -> dict[str, Any] | None:
    path = task_dir / PROGRESS_SNAPSHOT_NAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def write_progress_snapshot(task_dir: Path, progress: dict[str, Any]) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    tmp = task_dir / f"{PROGRESS_SNAPSHOT_NAME}.tmp"
    tmp.write_text(json.dumps(progress, ensure_ascii=False), encoding="utf-8")
    tmp.replace(task_dir / PROGRESS_SNAPSHOT_NAME)


def _load_scalars(log_dir: Path) -> dict[str, dict[int, float]]:
    if not log_dir.is_dir():
        return {}
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return {}

    scalars: dict[str, dict[int, float]] = {}
    for event_file in sorted(log_dir.glob("events.out.tfevents.*")):
        try:
            acc = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
            acc.Reload()
            tags = acc.Tags().get("scalars", [])
        except Exception:
            continue

        for tag in tags:
            target = scalars.setdefault(tag, {})
            try:
                events = acc.Scalars(tag)
            except Exception:
                continue
            for item in events:
                target[int(item.step)] = float(item.value)
    return scalars


def _last(values: dict[int, float]) -> float | None:
    if not values:
        return None
    return values[max(values)]


def _read_training_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _loss_bundle(
    train_values: dict[int, float],
    val_values: dict[int, float] | None = None,
) -> dict[str, list[dict[str, float | int]]]:
    if val_values is None:
        val_values = train_values
    if not train_values and val_values:
        train_values = val_values
    epochs = sorted(set(train_values) | set(val_values))
    train_series: list[dict[str, float | int]] = []
    val_series: list[dict[str, float | int]] = []
    last_train: float | None = None
    last_val: float | None = None

    for epoch in epochs:
        if epoch in train_values:
            last_train = train_values[epoch]
        if epoch in val_values:
            last_val = val_values[epoch]
        train_loss = last_train if last_train is not None else last_val
        val_loss = last_val if last_val is not None else train_loss
        if train_loss is None or val_loss is None:
            continue
        point = {"epoch": int(epoch), "trainLoss": float(train_loss), "valLoss": float(val_loss)}
        train_series.append(dict(point))
        val_series.append(dict(point))

    return {"trainLossSeries": train_series, "valLossSeries": val_series}


def load_competition_artifact_snapshot(task_dir: Path) -> dict[str, dict[str, Any]]:
    progress: dict[str, Any] = {}
    metrics: dict[str, Any] = {}

    class_scalars = _load_scalars(task_dir / "competition_class" / "runs")
    class_train_loss = class_scalars.get("CE_Loss", {})
    class_val_loss = class_scalars.get("Val_CE_Loss", {})
    class_train_acc = class_scalars.get("Train_Acc", {})
    class_val_acc = class_scalars.get("Val_Acc", {})
    class_train_mae = class_scalars.get("Train_Angle_MAE", {})
    class_val_mae = class_scalars.get("Val_Angle_MAE", {})
    class_val_stress_mae = class_scalars.get("Val_Stress_Angle_MAE", {})
    class_summary = _read_training_summary(task_dir / "competition_class" / "training_summary.json")
    if class_train_loss or class_val_loss:
        progress["competitionClass"] = _loss_bundle(class_train_loss, class_val_loss)
    if class_train_loss or class_val_loss or class_train_acc or class_val_acc or class_train_mae or class_val_mae or class_val_stress_mae or class_summary:
        metrics["competitionClass"] = {
            "finalTrainLoss": float(_last(class_train_loss) or _last(class_val_loss) or 0.0),
            "finalValLoss": float(_last(class_val_loss) or _last(class_train_loss) or 0.0),
            "steeringError": float(_last(class_val_stress_mae) or _last(class_val_mae) or _last(class_train_mae) or 0.0),
            "finalTrainAcc": _last(class_train_acc),
            "finalValAcc": _last(class_val_acc) if class_val_acc else _last(class_train_acc),
            "finalTrainAngleMAE": _last(class_train_mae),
            "finalValAngleMAE": _last(class_val_mae) if class_val_mae else _last(class_train_mae),
            "finalValStressAngleMAE": _last(class_val_stress_mae) if class_val_stress_mae else None,
        }
        metrics["competitionClass"].update({k: v for k, v in class_summary.items() if k not in metrics["competitionClass"] or v is not None})

    lite_scalars = _load_scalars(task_dir / "competition_lite" / "runs")
    lite_train_loss = lite_scalars.get("Loss/train", {})
    lite_val_loss = lite_scalars.get("Loss/val", {})
    lite_train_acc = lite_scalars.get("Acc/train", {})
    lite_val_acc = lite_scalars.get("Acc/val", {})
    lite_train_mae = lite_scalars.get("AngleMAE/train", {})
    lite_val_mae = lite_scalars.get("AngleMAE/val", {})
    lite_val_stress_mae = lite_scalars.get("Val_Stress_Angle_MAE", {})
    lite_summary = _read_training_summary(task_dir / "competition_lite" / "training_summary.json")
    if lite_train_loss or lite_val_loss:
        progress["competitionLite"] = _loss_bundle(lite_train_loss, lite_val_loss)
    if lite_train_loss or lite_val_loss or lite_train_acc or lite_val_acc or lite_train_mae or lite_val_mae or lite_val_stress_mae or lite_summary:
        metrics["competitionLite"] = {
            "finalTrainLoss": float(_last(lite_train_loss) or _last(lite_val_loss) or 0.0),
            "finalValLoss": float(_last(lite_val_loss) or _last(lite_train_loss) or 0.0),
            "steeringError": float(_last(lite_val_stress_mae) or _last(lite_val_mae) or _last(lite_train_mae) or 0.0),
            "finalTrainAcc": _last(lite_train_acc),
            "finalValAcc": _last(lite_val_acc),
            "finalTrainAngleMAE": _last(lite_train_mae),
            "finalValAngleMAE": _last(lite_val_mae),
            "finalValStressAngleMAE": _last(lite_val_stress_mae),
        }
        metrics["competitionLite"].update({k: v for k, v in lite_summary.items() if k not in metrics["competitionLite"] or v is not None})

    return {"progress": progress, "metrics": metrics}



