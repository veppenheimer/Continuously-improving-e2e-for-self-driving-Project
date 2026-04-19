#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Batch joint inference and plotting for three trained steering models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from joint_infer_compare import (
    DEFAULT_PREPROCESS_CONFIG,
    ROIConfig,
    _build_roi_config_from_args,
    _checkpoint_preprocess_config,
    _default_paths,
    _fmt,
    _parse_angle_from_name,
    _predict_net_class,
    _predict_net_improve,
    _predict_regression,
    _preprocess,
    _read_latest_run,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _sort_key(path: Path):
    stem = path.stem
    if "_" in stem:
        prefix = stem.split("_", 1)[0]
        if prefix.isdigit():
            return (0, int(prefix), path.name.lower())
    return (1, path.name.lower())


def _collect_images(folder: Path) -> list[Path]:
    images: list[Path] = []
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            _parse_angle_from_name(path)
        except ValueError:
            continue
        images.append(path)
    return sorted(images, key=_sort_key)


def _resolve_paths(args) -> tuple[Path, dict[str, Path]]:
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else _read_latest_run()
    paths = _default_paths(run_dir)
    if args.class_ckpt:
        paths["class_ckpt"] = Path(args.class_ckpt).expanduser().resolve()
    if args.improve_ckpt:
        paths["improve_ckpt"] = Path(args.improve_ckpt).expanduser().resolve()
    if args.regression_ckpt:
        paths["regression_ckpt"] = Path(args.regression_ckpt).expanduser().resolve()

    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} not found: {path}")
    return run_dir, paths


def _build_parser():
    parser = argparse.ArgumentParser(description="Batch joint inference for three trained models.")
    parser.add_argument("folder", help="Folder containing test images named like `index_angle.jpg`.")
    parser.add_argument("--test-number", "--test_number", type=int, default=None, help="Test only the first n images.")
    parser.add_argument("--run-dir", default=None, help="Training run directory. Defaults to latest_data_aug_run.txt or latest_real_data_run.txt.")
    parser.add_argument("--class-ckpt", default=None, help="Override Net_class checkpoint path.")
    parser.add_argument("--improve-ckpt", default=None, help="Override Net_improve checkpoint path.")
    parser.add_argument("--regression-ckpt", default=None, help="Override regression checkpoint path.")
    parser.add_argument("--enable-roi", action="store_true", help="Enable diagnostic ROI cropping.")
    parser.add_argument("--disable-roi", action="store_true", help="Force full-image inference.")
    parser.add_argument(
        "--roi-bottom-ratio",
        "--roi_bottom_ratio",
        type=float,
        default=0.6,
        help="When ROI is enabled, keep the bottom ratio of the image. Default: 0.6",
    )
    parser.add_argument("--output-plot", default=None, help="Output plot path. Defaults to <folder>/joint_dataset_compare.png.")
    parser.add_argument("--output-json", default=None, help="Output json path. Defaults to <folder>/joint_dataset_compare.json.")
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"folder not found: {folder}")

    images = _collect_images(folder)
    if not images:
        raise FileNotFoundError(f"no images found in folder: {folder}")

    if args.test_number is not None and args.test_number > 0:
        images = images[: args.test_number]

    run_dir, paths = _resolve_paths(args)
    roi_config = _build_roi_config_from_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_preprocess = _checkpoint_preprocess_config(paths["class_ckpt"], DEFAULT_PREPROCESS_CONFIG)
    improve_preprocess = _checkpoint_preprocess_config(paths["improve_ckpt"], DEFAULT_PREPROCESS_CONFIG)
    regression_preprocess = _checkpoint_preprocess_config(paths["regression_ckpt"], DEFAULT_PREPROCESS_CONFIG)

    indices: list[int] = []
    gt_values: list[float] = []
    class_preds: list[float] = []
    improve_preds: list[float] = []
    regression_preds: list[float] = []
    rows: list[dict[str, Any]] = []

    for i, image_path in enumerate(images):
        x_class = _preprocess(image_path, device, roi_config=roi_config, base_config=class_preprocess)
        x_improve = _preprocess(image_path, device, roi_config=roi_config, base_config=improve_preprocess)
        x_regression = _preprocess(image_path, device, roi_config=roi_config, base_config=regression_preprocess)
        gt_angle = _parse_angle_from_name(image_path)
        pred_class = _predict_net_class(x_class, paths["class_ckpt"], device)
        pred_improve = _predict_net_improve(x_improve, paths["improve_ckpt"], device)
        pred_regression = _predict_regression(x_regression, paths["regression_ckpt"], device)

        indices.append(i)
        gt_values.append(gt_angle)
        class_preds.append(pred_class)
        improve_preds.append(pred_improve)
        regression_preds.append(pred_regression)
        rows.append(
            {
                "index": i,
                "image": str(image_path),
                "groundTruth": gt_angle,
                "Net_class": pred_class,
                "Net_improve": pred_improve,
                "Net_regression": pred_regression,
            }
        )

    gt_arr = np.asarray(gt_values, dtype=np.float64)
    class_arr = np.asarray(class_preds, dtype=np.float64)
    improve_arr = np.asarray(improve_preds, dtype=np.float64)
    regression_arr = np.asarray(regression_preds, dtype=np.float64)

    summary = {
        "folder": str(folder),
        "runDir": str(run_dir),
        "numImages": len(images),
        "device": str(device),
        "preprocess": {
            "Net_class": {
                "colorSpace": class_preprocess.color_space,
                "inputSize": list(class_preprocess.input_size),
                "useRoi": roi_config.enabled,
            },
            "Net_improve": {
                "colorSpace": improve_preprocess.color_space,
                "inputSize": list(improve_preprocess.input_size),
                "useRoi": roi_config.enabled,
            },
            "Net_regression": {
                "colorSpace": regression_preprocess.color_space,
                "inputSize": list(regression_preprocess.input_size),
                "useRoi": roi_config.enabled,
            },
        },
        "roi": {
            "enabled": roi_config.enabled,
            "mode": roi_config.mode,
            "bottomRatio": roi_config.bottom_ratio,
        },
        "mae": {
            "Net_class": float(np.mean(np.abs(class_arr - gt_arr))),
            "Net_improve": float(np.mean(np.abs(improve_arr - gt_arr))),
            "Net_regression": float(np.mean(np.abs(regression_arr - gt_arr))),
        },
        "results": rows,
    }

    output_plot = Path(args.output_plot).expanduser().resolve() if args.output_plot else folder / "joint_dataset_compare.png"
    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else folder / "joint_dataset_compare.json"

    model_names = ["Net_class", "Net_improve", "Net_regression"]
    pred_map = {
        "Net_class": class_arr,
        "Net_improve": improve_arr,
        "Net_regression": regression_arr,
    }
    colors = {
        "Net_class": "#1f77b4",
        "Net_improve": "#ff7f0e",
        "Net_regression": "#2ca02c",
    }
    abs_errors = {name: np.abs(pred - gt_arr) for name, pred in pred_map.items()}

    # 上千张样本不再画逐图折线，改为更适合 review 的聚合视图。
    unique_angles = np.asarray(sorted(set(float(v) for v in gt_values)), dtype=np.float64)
    x = np.arange(len(unique_angles))
    width = 0.24
    mae_by_angle: dict[str, list[float]] = {name: [] for name in model_names}
    mean_pred_by_angle: dict[str, list[float]] = {name: [] for name in model_names}
    counts_by_angle: list[int] = []
    for angle in unique_angles:
        mask = np.isclose(gt_arr, angle, atol=1e-6)
        counts_by_angle.append(int(mask.sum()))
        for name in model_names:
            mae_by_angle[name].append(float(abs_errors[name][mask].mean()) if mask.any() else 0.0)
            mean_pred_by_angle[name].append(float(pred_map[name][mask].mean()) if mask.any() else 0.0)

    title_suffix = f"ROI bottom {roi_config.bottom_ratio:.0%}" if roi_config.enabled else "Full image"
    fig, axes = plt.subplots(2, 2, figsize=(17, 10))
    fig.suptitle(f"Joint Inference Aggregate Review ({len(images)} images, {title_suffix})", fontsize=15)

    ax = axes[0, 0]
    mae_values = [summary["mae"][name] for name in model_names]
    ax.bar(model_names, mae_values, color=[colors[name] for name in model_names])
    ax.set_title("Overall MAE")
    ax.set_ylabel("MAE")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for idx, value in enumerate(mae_values):
        ax.text(idx, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)

    ax = axes[0, 1]
    ax.boxplot([abs_errors[name] for name in model_names], labels=model_names, showfliers=False)
    ax.set_title("Absolute Error Distribution (outliers hidden)")
    ax.set_ylabel("Absolute Error")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    ax = axes[1, 0]
    for offset, name in zip([-width, 0.0, width], model_names):
        ax.bar(x + offset, mae_by_angle[name], width=width, label=name, color=colors[name], alpha=0.88)
    ax.set_title("MAE by Ground-Truth Angle")
    ax.set_xlabel("Ground-Truth Angle")
    ax.set_ylabel("MAE")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}\n(n={c})" for v, c in zip(unique_angles, counts_by_angle)], rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(unique_angles, unique_angles, color="#111111", linewidth=2.0, label="Ideal")
    for name in model_names:
        ax.plot(unique_angles, mean_pred_by_angle[name], marker="o", linewidth=1.8, label=name, color=colors[name])
    ax.set_title("Mean Prediction by Ground-Truth Angle")
    ax.set_xlabel("Ground-Truth Angle")
    ax.set_ylabel("Mean Prediction")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_plot, dpi=170)
    plt.close(fig)

    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = output_json.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "image",
                "groundTruth",
                "Net_class",
                "Net_class_abs_error",
                "Net_improve",
                "Net_improve_abs_error",
                "Net_regression",
                "Net_regression_abs_error",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "Net_class_abs_error": abs(float(row["Net_class"]) - float(row["groundTruth"])),
                    "Net_improve_abs_error": abs(float(row["Net_improve"]) - float(row["groundTruth"])),
                    "Net_regression_abs_error": abs(float(row["Net_regression"]) - float(row["groundTruth"])),
                }
            )

    angle_summary_path = output_json.with_name(output_json.stem + "_by_angle.csv")
    with angle_summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "groundTruth",
                "count",
                "Net_class_mae",
                "Net_improve_mae",
                "Net_regression_mae",
                "Net_class_mean_pred",
                "Net_improve_mean_pred",
                "Net_regression_mean_pred",
            ],
        )
        writer.writeheader()
        for idx, angle in enumerate(unique_angles):
            writer.writerow(
                {
                    "groundTruth": float(angle),
                    "count": counts_by_angle[idx],
                    "Net_class_mae": mae_by_angle["Net_class"][idx],
                    "Net_improve_mae": mae_by_angle["Net_improve"][idx],
                    "Net_regression_mae": mae_by_angle["Net_regression"][idx],
                    "Net_class_mean_pred": mean_pred_by_angle["Net_class"][idx],
                    "Net_improve_mean_pred": mean_pred_by_angle["Net_improve"][idx],
                    "Net_regression_mean_pred": mean_pred_by_angle["Net_regression"][idx],
                }
            )

    print(f"folder: {folder}")
    print(f"run_dir: {run_dir}")
    print(f"device: {device}")
    print(f"tested_images: {len(images)}")
    print(f"roi_enabled: {roi_config.enabled}")
    print(f"roi_bottom_ratio: {roi_config.bottom_ratio:.3f}")
    print(f"plot: {output_plot}")
    print(f"json: {output_json}")
    print(f"csv: {csv_path}")
    print(f"by_angle_csv: {angle_summary_path}")
    print("")
    print("model           mae")
    print("--------------- ------------")
    for model_name, mae in summary["mae"].items():
        print(f"{model_name:<15} {_fmt(mae):>12}")


if __name__ == "__main__":
    main()
