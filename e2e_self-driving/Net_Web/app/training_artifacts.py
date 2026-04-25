"""Helpers for persisted training progress and summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROGRESS_SNAPSHOT_NAME = "progress.json"
TRAINING_SUMMARY_NAME = "training_summary.json"


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


def read_training_summary(task_dir: Path) -> dict[str, Any] | None:
    path = task_dir / TRAINING_SUMMARY_NAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def write_training_summary(task_dir: Path, summary: dict[str, Any]) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    tmp = task_dir / f"{TRAINING_SUMMARY_NAME}.tmp"
    tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(task_dir / TRAINING_SUMMARY_NAME)
