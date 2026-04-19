#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train the three generalization-oriented steering models and write a review report."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_training(name: str, work_dir: Path, env: dict[str, str], log_path: Path, python_exe: Path) -> None:
    summary_path = Path(env["VENET_OUTPUT_DIR"]) / "training_summary.json"
    if summary_path.is_file():
        print(f"========== SKIP {name}: found {summary_path} ==========")
        return
    print(f"========== START {name} ==========")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    merged_env.update(env)
    merged_env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            [str(python_exe), "train.py"],
            cwd=str(work_dir),
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            safe_line = line.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8", errors="replace")
            print(safe_line, end="")
            log_file.write(line)
            log_file.flush()
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"{name} training failed with exit code {code}; see {log_path}")
    print(f"========== END {name} ==========")


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_review(run_dir: Path, dataset_dir: Path) -> Path:
    models = [
        ("Net_class", run_dir / "Net_class" / "training_summary.json"),
        ("Net_improve", run_dir / "Net_improve" / "training_summary.json"),
        ("Net_regression", run_dir / "Net_regression" / "training_summary.json"),
    ]
    lines: list[str] = []
    lines.append("# 泛化优先三模型训练 Review")
    lines.append("")
    lines.append(f"- 运行目录: `{run_dir}`")
    lines.append(f"- 数据集目录: `{dataset_dir}`")
    lines.append("- 源数据集: `E:/桌面/data`")
    lines.append("- 批大小: `16`")
    lines.append("- 输入契约: `全图 -> HSV -> Resize(120,160) -> Tensor`")
    lines.append("- ROI: `False`")
    lines.append("- 训练增强: `50% clean / 30% moderate / 20% strong style`")
    lines.append("- 选模指标: `val_stress_mae` 或 `val_stress_angle_mae`")
    lines.append("")

    summary_table: list[dict[str, object]] = []
    keys = [
        "requestedEpochs",
        "completedEpochs",
        "bestEpoch",
        "stoppedEpoch",
        "earlyStopped",
        "modelSelectionMetric",
        "steeringError",
        "finalTrainLoss",
        "finalValLoss",
        "finalTrainMAE",
        "finalValMAE",
        "finalTrainAngleMAE",
        "finalValAngleMAE",
        "finalValStressAngleMAE",
        "finalValStressMAE",
        "finalTestLoss",
        "finalTestMAE",
        "finalTestAngleMAE",
        "finalTestAcc",
        "testBestAngleMAE",
        "testBestAcc",
        "usedDedicatedTestSplit",
        "pretrainedLoaded",
        "usePretrained",
        "freezeBackboneEpochs",
    ]
    for name, path in models:
        summary = read_json(path)
        lines.append(f"## {name}")
        if not summary:
            lines.append(f"- summary: `missing: {path}`")
            lines.append("")
            continue
        for key in keys:
            if key in summary:
                lines.append(f"- {key}: `{summary[key]}`")
        lines.append("")
        summary_table.append(
            {
                "model": name,
                "bestEpoch": summary.get("bestEpoch"),
                "steeringError": summary.get("steeringError"),
                "testMAE": summary.get("finalTestMAE")
                or summary.get("finalTestAngleMAE")
                or summary.get("testBestAngleMAE"),
                "testAcc": summary.get("finalTestAcc") or summary.get("testBestAcc"),
                "earlyStopped": summary.get("earlyStopped"),
                "completedEpochs": summary.get("completedEpochs"),
                "pretrainedLoaded": summary.get("pretrainedLoaded"),
            }
        )

    lines.append("## 汇总表")
    lines.append("")
    lines.append("| model | bestEpoch | steeringError/valStress | testMAE | testAcc | earlyStopped | completedEpochs | pretrainedLoaded |")
    lines.append("|---|---:|---:|---:|---:|---|---:|---|")

    def fmt(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    for row in summary_table:
        lines.append(
            f"| {row['model']} | {fmt(row['bestEpoch'])} | {fmt(row['steeringError'])} | "
            f"{fmt(row['testMAE'])} | {fmt(row['testAcc'])} | {fmt(row['earlyStopped'])} | "
            f"{fmt(row['completedEpochs'])} | {fmt(row['pretrainedLoaded'])} |"
        )
    lines.append("")
    lines.append("## 重要日志位置")
    lines.append("")
    for name, _ in models:
        lines.append(f"- `{name}`: `{run_dir / name / 'terminal.log'}`")
    lines.append("")
    report_path = run_dir / "TRAINING_REVIEW_CN.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Net_class, Net_improve, and Net_regression with the new strategy.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--repo", default=r"E:\桌面\项目")
    parser.add_argument("--python", default=r"E:\桌面\VeNet\ve_env\Scripts\python.exe")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    python_exe = Path(args.python).resolve()
    if not dataset_dir.is_dir():
        raise NotADirectoryError(dataset_dir)
    if not python_exe.is_file():
        raise FileNotFoundError(python_exe)

    run_name = args.run_name or f"generalization_real_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = repo / "training_runs" / run_name
    for branch in ("Net_class", "Net_improve", "Net_regression"):
        (run_dir / branch).mkdir(parents=True, exist_ok=True)

    common = {
        "VENET_DATA_FOLDER": str(dataset_dir),
        "VENET_BATCH_SIZE": "16",
        "VENET_EARLY_STOP_PATIENCE": "12",
        "VENET_EARLY_STOP_MIN_DELTA": "0.0001",
        "VENET_WEIGHT_DECAY": "0.0001",
        "VENET_GRAD_CLIP": "3.0",
        "VENET_USE_PRETRAINED": "1",
        "VENET_FREEZE_BACKBONE_EPOCHS": "5",
        "VENET_BACKBONE_LR_FACTOR": "0.1",
        "VENET_STYLE_MIX_RATIO": "0.5,0.3,0.2",
        "VENET_PREPROCESS_COLOR_SPACE": "hsv",
    }

    run_training(
        "Net_class",
        repo / "e2e_competition" / "Net_class",
        common
        | {
            "VENET_OUTPUT_DIR": str(run_dir / "Net_class"),
            "VENET_LOG_DIR": str(run_dir / "Net_class" / "tb"),
            "VENET_SAVE_NAME": "ve2_generalization_class.pth",
            "VENET_BEST_SAVE_NAME": "best_ve2_generalization_class.pth",
            "VENET_EPOCHS": "100",
            "VENET_LR": "0.0001",
            "VENET_REG_LOSS_WEIGHT": "2.0",
        },
        run_dir / "Net_class" / "terminal.log",
        python_exe,
    )

    run_training(
        "Net_improve",
        repo / "e2e_competition" / "Net_improve",
        common
        | {
            "VENET_OUTPUT_DIR": str(run_dir / "Net_improve"),
            "VENET_LOG_DIR": str(run_dir / "Net_improve" / "tb"),
            "VENET_SAVE_NAME": "ve2_generalization_improve.pth",
            "VENET_BEST_SAVE_NAME": "best_ve2_generalization_improve.pth",
            "VENET_EPOCHS": "80",
            "VENET_LR": "0.0001",
            "VENET_LABEL_SMOOTHING": "0.05",
        },
        run_dir / "Net_improve" / "terminal.log",
        python_exe,
    )

    run_training(
        "Net_regression",
        repo / "e2e_self-driving" / "Net",
        common
        | {
            "VENET_OUTPUT_DIR": str(run_dir / "Net_regression"),
            "VENET_LOG_DIR": str(run_dir / "Net_regression" / "tb"),
            "VENET_SAVE_NAME": "ve2_generalization_regression.pth",
            "VENET_BEST_SAVE_NAME": "best_ve2_generalization_regression.pth",
            "VENET_EPOCHS": "100",
            "VENET_LR": "0.0001",
            "VENET_AUX_CLS_WEIGHT": "0.3",
            "VENET_USE_WEIGHTED_SAMPLER": "1",
        },
        run_dir / "Net_regression" / "terminal.log",
        python_exe,
    )

    report_path = write_review(run_dir, dataset_dir)
    latest_path = repo / "training_runs" / "latest_generalization_real_data_run.txt"
    latest_path.write_text(str(run_dir), encoding="utf-8")
    print(f"REPORT={report_path}")
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()

