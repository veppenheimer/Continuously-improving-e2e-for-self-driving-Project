#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Batch inference for regression model on a folder of angle-suffixed images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from joint_infer_compare import _checkpoint_preprocess_config, _fmt, _parse_angle_from_name, _predict_regression, _preprocess


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _sort_key(path: Path):
    stem = path.stem
    if "_" in stem:
        prefix = stem.split("_", 1)[0]
        if prefix.isdigit():
            return (0, int(prefix), path.name.lower())
    return (1, path.name.lower())


def main():
    parser = argparse.ArgumentParser(description="Run regression model on all images in a folder.")
    parser.add_argument("folder")
    parser.add_argument("--ckpt", required=True, help="Regression checkpoint path.")
    parser.add_argument("--output-plot", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--disable-roi", action="store_true")
    parser.add_argument("--roi-bottom-ratio", "--roi_bottom_ratio", type=float, default=0.6)
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    ckpt = Path(args.ckpt).expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"folder not found: {folder}")
    if not ckpt.is_file():
        raise FileNotFoundError(f"ckpt not found: {ckpt}")

    images = []
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            _parse_angle_from_name(path)
        except ValueError:
            continue
        images.append(path)
    images = sorted(images, key=_sort_key)
    if not images:
        raise FileNotFoundError(f"no valid angle-suffixed images found in: {folder}")

    from joint_infer_compare import _build_roi_config_from_args

    roi_config = _build_roi_config_from_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess_config = _checkpoint_preprocess_config(ckpt)

    indices, gt_values, pred_values = [], [], []
    rows = []
    for i, image_path in enumerate(images):
        x = _preprocess(image_path, device, roi_config=roi_config, base_config=preprocess_config)
        gt = _parse_angle_from_name(image_path)
        pred = _predict_regression(x, ckpt, device)
        indices.append(i)
        gt_values.append(gt)
        pred_values.append(pred)
        rows.append(
            {
                "index": i,
                "image": str(image_path),
                "groundTruth": gt,
                "prediction": pred,
                "absoluteError": abs(pred - gt),
            }
        )

    gt_arr = np.asarray(gt_values, dtype=np.float64)
    pred_arr = np.asarray(pred_values, dtype=np.float64)
    summary = {
        "folder": str(folder),
        "ckpt": str(ckpt),
        "numImages": len(images),
        "device": str(device),
        "roiEnabled": roi_config.enabled,
        "roiBottomRatio": roi_config.bottom_ratio,
        "mae": float(np.mean(np.abs(pred_arr - gt_arr))),
        "results": rows,
    }

    output_plot = Path(args.output_plot).expanduser().resolve() if args.output_plot else folder / "regression_compare.png"
    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else folder / "regression_compare.json"

    plt.figure(figsize=(14, 7))
    plt.plot(indices, gt_values, label="Ground Truth", linewidth=2.4, color="#111111")
    plt.plot(indices, pred_values, label="Net_regression", linewidth=1.8, color="#2ca02c")
    title_suffix = f"ROI bottom {roi_config.bottom_ratio:.0%}" if roi_config.enabled else "Full image"
    plt.title(f"Regression Inference Compare ({len(images)} images, {title_suffix})")
    plt.xlabel("Image Index")
    plt.ylabel("Steering Angle")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_plot, dpi=160)
    plt.close()

    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"folder: {folder}")
    print(f"ckpt: {ckpt}")
    print(f"device: {device}")
    print(f"tested_images: {len(images)}")
    print(f"roi_enabled: {roi_config.enabled}")
    print(f"roi_bottom_ratio: {roi_config.bottom_ratio:.3f}")
    print(f"plot: {output_plot}")
    print(f"json: {output_json}")
    print(f"mae: {_fmt(summary['mae'])}")


if __name__ == "__main__":
    main()
