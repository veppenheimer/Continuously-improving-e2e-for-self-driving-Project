#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run full-image joint inference on multiple steering datasets.

The script keeps test artifacts inside a selected training run directory so
model review stays tidy and reproducible.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
JOINT_DATASET_SCRIPT = REPO_ROOT / "scripts" / "joint_infer_dataset_compare.py"
RECURSIVE_SCRIPT = REPO_ROOT / "scripts" / "run_joint_infer_recursive.py"


def _default_run_dir() -> Path:
    latest = REPO_ROOT / "training_runs" / "latest_generalization_real_data_run.txt"
    if latest.is_file():
        return Path(latest.read_text(encoding="utf-8-sig").strip()).resolve()
    raise FileNotFoundError(f"latest run file not found: {latest}")


def _safe_dataset_name(path: Path) -> str:
    return path.name.replace(" ", "_")


def _run_command(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            safe_line = line.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
                sys.stdout.encoding or "utf-8",
                errors="replace",
            )
            print(safe_line, end="")
            log_file.write(line)
        return proc.wait()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _run_flat_dataset(
    dataset_name: str,
    dataset_path: Path,
    run_dir: Path,
    output_root: Path,
    test_number: int | None,
) -> dict[str, Any]:
    dataset_output = output_root / dataset_name
    dataset_output.mkdir(parents=True, exist_ok=True)
    json_path = dataset_output / "joint_dataset_compare.json"
    plot_path = dataset_output / "joint_dataset_compare.png"
    log_path = dataset_output / "terminal.log"
    cmd = [
        sys.executable,
        str(JOINT_DATASET_SCRIPT),
        str(dataset_path),
        "--run-dir",
        str(run_dir),
        "--disable-roi",
        "--output-json",
        str(json_path),
        "--output-plot",
        str(plot_path),
    ]
    if test_number is not None and test_number > 0:
        cmd.extend(["--test-number", str(test_number)])

    print(f"\n========== TEST {dataset_name} ==========")
    code = _run_command(cmd, log_path)
    summary = _read_json(json_path)
    return {
        "name": dataset_name,
        "kind": "flat",
        "path": str(dataset_path),
        "outputDir": str(dataset_output),
        "json": str(json_path),
        "plot": str(plot_path),
        "log": str(log_path),
        "returncode": code,
        "numImages": summary.get("numImages"),
        "mae": summary.get("mae", {}),
    }


def _run_recursive_dataset(
    dataset_name: str,
    dataset_path: Path,
    run_dir: Path,
    output_root: Path,
    test_number: int | None,
) -> dict[str, Any]:
    dataset_output = output_root / dataset_name
    dataset_output.mkdir(parents=True, exist_ok=True)
    index_path = dataset_output / "recursive_joint_index.json"
    log_path = dataset_output / "terminal.log"
    cmd = [
        sys.executable,
        str(RECURSIVE_SCRIPT),
        str(dataset_path),
        "--run-dir",
        str(run_dir),
        "--disable-roi",
        "--output-dir",
        str(dataset_output),
    ]
    if test_number is not None and test_number > 0:
        cmd.extend(["--test-number", str(test_number)])

    print(f"\n========== TEST {dataset_name} ==========")
    code = _run_command(cmd, log_path)
    index = _read_json(index_path)
    records = index.get("records", []) if isinstance(index, dict) else []
    ok_records = [record for record in records if record.get("returncode") == 0]

    weighted: dict[str, float] = {}
    total_images = sum(int(record.get("numImages") or 0) for record in ok_records)
    if total_images > 0:
        model_names = ["Net_class", "Net_improve", "Net_regression"]
        for model_name in model_names:
            total_error = 0.0
            for record in ok_records:
                num_images = int(record.get("numImages") or 0)
                mae = (record.get("mae") or {}).get(model_name)
                if mae is not None:
                    total_error += float(mae) * num_images
            weighted[model_name] = total_error / total_images

    return {
        "name": dataset_name,
        "kind": "recursive",
        "path": str(dataset_path),
        "outputDir": str(dataset_output),
        "index": str(index_path),
        "log": str(log_path),
        "returncode": code,
        "numFolders": index.get("numFolders"),
        "successfulFolders": sum(1 for record in records if record.get("returncode") == 0),
        "failedFolders": sum(1 for record in records if record.get("returncode") != 0),
        "numImages": total_images,
        "mae": weighted,
    }


def _write_report(output_root: Path, run_dir: Path, records: list[dict[str, Any]]) -> Path:
    report_path = output_root / "MULTI_DATASET_TEST_REVIEW_CN.md"

    def fmt(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    lines: list[str] = []
    lines.append("# 多数据集联调测试 Review")
    lines.append("")
    lines.append(f"- 训练结果目录: `{run_dir}`")
    lines.append(f"- 测试输出目录: `{output_root}`")
    lines.append("- 推理设置: `全图 / HSV / Resize(120,160) / 无 ROI`")
    lines.append("- 权重来源: 训练目录下最新 `best_ve2_generalization_*.pth`")
    lines.append("")
    lines.append("## 汇总")
    lines.append("")
    lines.append("| dataset | kind | images | folders ok/failed | Net_class MAE | Net_improve MAE | Net_regression MAE | status |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for record in records:
        mae = record.get("mae") or {}
        folder_text = ""
        if record.get("kind") == "recursive":
            folder_text = f"{record.get('successfulFolders', 0)}/{record.get('failedFolders', 0)}"
        lines.append(
            f"| {record['name']} | {record['kind']} | {fmt(record.get('numImages'))} | {folder_text} | "
            f"{fmt(mae.get('Net_class'))} | {fmt(mae.get('Net_improve'))} | {fmt(mae.get('Net_regression'))} | "
            f"{'OK' if record.get('returncode') == 0 else 'FAILED'} |"
        )
    lines.append("")
    lines.append("## 结果文件")
    lines.append("")
    for record in records:
        lines.append(f"- `{record['name']}` 输出目录: `{record['outputDir']}`")
        if record.get("json"):
            lines.append(f"- `{record['name']}` JSON: `{record['json']}`")
        if record.get("index"):
            lines.append(f"- `{record['name']}` 递归索引: `{record['index']}`")
        if record.get("plot"):
            lines.append(f"- `{record['name']}` 曲线图: `{record['plot']}`")
        lines.append(f"- `{record['name']}` 终端日志: `{record['log']}`")
    lines.append("")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple full-image datasets against the latest trained models.")
    parser.add_argument("--run-dir", default=None, help="Training run directory. Defaults to latest generalization run.")
    parser.add_argument("--output-name", default=None, help="Subdirectory name under the run directory.")
    parser.add_argument("--test-number", "--test_number", type=int, default=None, help="Limit each dataset/folder to first n images.")
    parser.add_argument(
        "--flat-dataset",
        action="append",
        default=[],
        help="Flat image folder. Format: name=path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--recursive-dataset",
        action="append",
        default=[],
        help="Recursive image root. Format: name=path. Can be passed multiple times.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve() if args.run_dir else _default_run_dir()
    output_name = args.output_name or f"multi_dataset_tests_full_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_root = run_dir / output_name
    output_root.mkdir(parents=True, exist_ok=True)

    flat_specs = list(args.flat_dataset)
    recursive_specs = list(args.recursive_dataset)
    if not flat_specs and not recursive_specs:
        defaults = [
            ("source_data", Path(r"E:\桌面\data")),
            ("offline_augmented_data", Path(r"E:\桌面\data_aug")),
            ("strong_style_data", Path(r"E:\桌面\data1")),
        ]
        for name, path in defaults:
            if path.is_dir():
                flat_specs.append(f"{name}={path}")
        kunming = Path(r"E:\桌面\kunmingr2")
        if kunming.is_dir():
            recursive_specs.append(f"kunmingr2={kunming}")

    records: list[dict[str, Any]] = []
    for spec in flat_specs:
        name, raw_path = spec.split("=", 1) if "=" in spec else (_safe_dataset_name(Path(spec)), spec)
        path = Path(raw_path).resolve()
        if not path.is_dir():
            records.append({"name": name, "kind": "flat", "path": str(path), "returncode": 1, "error": "missing folder"})
            continue
        records.append(_run_flat_dataset(name, path, run_dir, output_root, args.test_number))

    for spec in recursive_specs:
        name, raw_path = spec.split("=", 1) if "=" in spec else (_safe_dataset_name(Path(spec)), spec)
        path = Path(raw_path).resolve()
        if not path.is_dir():
            records.append({"name": name, "kind": "recursive", "path": str(path), "returncode": 1, "error": "missing folder"})
            continue
        records.append(_run_recursive_dataset(name, path, run_dir, output_root, args.test_number))

    index_path = output_root / "multi_dataset_test_index.json"
    index_path.write_text(
        json.dumps(
            {
                "runDir": str(run_dir),
                "outputRoot": str(output_root),
                "testNumber": args.test_number,
                "fullImage": True,
                "roiEnabled": False,
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    report_path = _write_report(output_root, run_dir, records)
    print(f"\nREPORT={report_path}")
    print(f"INDEX={index_path}")
    print(f"OUTPUT_ROOT={output_root}")


if __name__ == "__main__":
    main()
