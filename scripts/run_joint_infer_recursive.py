#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run batch joint inference for every image-containing subfolder under a root folder."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
REPO_ROOT = Path(__file__).resolve().parents[1]


def _find_image_dirs(root: Path) -> list[Path]:
    dirs = []
    for directory in sorted([p for p in root.rglob("*") if p.is_dir()]):
        has_image = any(child.is_file() and child.suffix.lower() in IMAGE_EXTS for child in directory.iterdir())
        if has_image:
            dirs.append(directory)
    return dirs


def _safe_name(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return "__".join(rel.parts)


def main():
    parser = argparse.ArgumentParser(description="Run joint dataset compare recursively on all image folders.")
    parser.add_argument("root", help="Root folder containing nested image subfolders.")
    parser.add_argument("--run-dir", default=None, help="Training run directory passed through to joint_infer_dataset_compare.py")
    parser.add_argument("--class-ckpt", default=None, help="Override Net_class checkpoint path.")
    parser.add_argument("--improve-ckpt", default=None, help="Override Net_improve checkpoint path.")
    parser.add_argument("--regression-ckpt", default=None, help="Override Net_regression checkpoint path.")
    parser.add_argument("--test-number", "--test_number", type=int, default=None, help="Test only the first n images in each subfolder.")
    parser.add_argument("--disable-roi", action="store_true", help="Disable ROI and use full images.")
    parser.add_argument(
        "--roi-bottom-ratio",
        "--roi_bottom_ratio",
        type=float,
        default=0.6,
        help="Bottom ROI ratio. Default: 0.6",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for plots/json/index. Default: <repo>/training_runs/recursive_joint_<rootname>",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"root not found: {root}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else REPO_ROOT / "training_runs" / f"recursive_joint_{root.name}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    image_dirs = _find_image_dirs(root)
    if not image_dirs:
        raise FileNotFoundError(f"no image-containing subfolders found under: {root}")

    records: list[dict[str, object]] = []
    script_path = REPO_ROOT / "scripts" / "joint_infer_dataset_compare.py"

    for idx, image_dir in enumerate(image_dirs, start=1):
        safe_name = _safe_name(image_dir, root)
        plot_path = output_dir / f"{safe_name}.png"
        json_path = output_dir / f"{safe_name}.json"
        cmd = [sys.executable, str(script_path), str(image_dir), "--output-plot", str(plot_path), "--output-json", str(json_path)]
        if args.run_dir:
            cmd.extend(["--run-dir", str(Path(args.run_dir).expanduser().resolve())])
        if args.class_ckpt:
            cmd.extend(["--class-ckpt", str(Path(args.class_ckpt).expanduser().resolve())])
        if args.improve_ckpt:
            cmd.extend(["--improve-ckpt", str(Path(args.improve_ckpt).expanduser().resolve())])
        if args.regression_ckpt:
            cmd.extend(["--regression-ckpt", str(Path(args.regression_ckpt).expanduser().resolve())])
        if args.test_number is not None and args.test_number > 0:
            cmd.extend(["--test-number", str(args.test_number)])
        if args.disable_roi:
            cmd.append("--disable-roi")
        else:
            cmd.extend(["--roi-bottom-ratio", str(args.roi_bottom_ratio)])

        print(f"[{idx}/{len(image_dirs)}] testing {image_dir}")
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        record = {
            "folder": str(image_dir),
            "plot": str(plot_path),
            "json": str(json_path),
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
        if proc.returncode == 0 and json_path.is_file():
            try:
                summary = json.loads(json_path.read_text(encoding="utf-8"))
                record["mae"] = summary.get("mae", {})
                record["numImages"] = summary.get("numImages")
            except Exception as exc:  # pragma: no cover
                record["jsonReadError"] = str(exc)
        records.append(record)

    index = {
        "root": str(root),
        "outputDir": str(output_dir),
        "numFolders": len(image_dirs),
        "testNumber": args.test_number,
        "classCkpt": args.class_ckpt,
        "improveCkpt": args.improve_ckpt,
        "regressionCkpt": args.regression_ckpt,
        "roiEnabled": not args.disable_roi,
        "roiBottomRatio": None if args.disable_roi else args.roi_bottom_ratio,
        "records": records,
    }
    index_path = output_dir / "recursive_joint_index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_csv = output_dir / "recursive_folder_mae_summary.csv"
    ok_records = [record for record in records if record["returncode"] == 0 and record.get("mae")]
    model_names = ["Net_class", "Net_improve", "Net_regression"]
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["folder", "numImages", "Net_class_mae", "Net_improve_mae", "Net_regression_mae", "plot", "json"],
        )
        writer.writeheader()
        for record in ok_records:
            mae = record.get("mae") or {}
            writer.writerow(
                {
                    "folder": record.get("folder"),
                    "numImages": record.get("numImages"),
                    "Net_class_mae": mae.get("Net_class"),
                    "Net_improve_mae": mae.get("Net_improve"),
                    "Net_regression_mae": mae.get("Net_regression"),
                    "plot": record.get("plot"),
                    "json": record.get("json"),
                }
            )

    summary_plot = output_dir / "recursive_folder_mae_compare.png"
    if ok_records:
        def _sort_value(record):
            mae = record.get("mae") or {}
            return float(mae.get("Net_improve") or mae.get("Net_class") or 0.0)

        ordered = sorted(ok_records, key=_sort_value)
        labels = [Path(str(record["folder"])).relative_to(root).as_posix() for record in ordered]
        y = np.arange(len(ordered))
        height = 0.24
        fig_height = max(6.0, min(28.0, 0.55 * len(ordered) + 2.5))
        fig, ax = plt.subplots(figsize=(14, fig_height))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        offsets = [-height, 0.0, height]
        for model_name, offset, color in zip(model_names, offsets, colors):
            values = [float((record.get("mae") or {}).get(model_name) or 0.0) for record in ordered]
            ax.barh(y + offset, values, height=height, label=model_name, color=color, alpha=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("MAE")
        ax.set_title("Recursive Dataset Folder-Level MAE Compare")
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        ax.legend()
        fig.tight_layout()
        fig.savefig(summary_plot, dpi=170)
        plt.close(fig)

    ok_count = sum(1 for r in records if r["returncode"] == 0)
    print("")
    print(f"root: {root}")
    print(f"output_dir: {output_dir}")
    print(f"tested_folders: {len(image_dirs)}")
    print(f"successful: {ok_count}")
    print(f"failed: {len(image_dirs) - ok_count}")
    print(f"index: {index_path}")
    print(f"summary_csv: {summary_csv}")
    print(f"summary_plot: {summary_plot}")


if __name__ == "__main__":
    main()

