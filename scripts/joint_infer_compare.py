#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run one-image joint inference for the three trained models."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from steering_preprocess import DEFAULT_PREPROCESS_CONFIG, PreprocessConfig, preprocess_config_from_dict, preprocess_path_to_tensor

DEFAULT_RUN_FILES = [
    REPO_ROOT / "training_runs" / "latest_generalization_real_data_run.txt",
    REPO_ROOT / "training_runs" / "latest_data_aug_run.txt",
    REPO_ROOT / "training_runs" / "latest_real_data_run.txt",
]


@dataclass(frozen=True)
class ROIConfig:
    enabled: bool = False
    mode: str = "bottom_ratio"
    bottom_ratio: float = 0.6


def _load_module(module_path: Path, module_name: str):
    module_dir = str(module_path.parent)
    added = False
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
        added = True
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(module_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to load module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if added:
            try:
                sys.path.remove(module_dir)
            except ValueError:
                pass


def _read_latest_run() -> Path:
    for run_file in DEFAULT_RUN_FILES:
        if run_file.is_file():
            return Path(run_file.read_text(encoding="utf-8-sig").strip())
    candidates = ", ".join(str(path) for path in DEFAULT_RUN_FILES)
    raise FileNotFoundError(f"latest run file not found. checked: {candidates}")


def _default_paths(run_dir: Path) -> dict[str, Path]:
    def _pick(base_dir: Path, candidates: list[str]) -> Path:
        for name in candidates:
            path = base_dir / name
            if path.is_file():
                return path
        return base_dir / candidates[0]

    return {
        "class_ckpt": _pick(
            run_dir / "Net_class",
            [
                "best_ve2_generalization_class.pth",
                "best_ve2_data_aug_class.pth",
                "best_ve2_real_class.pth",
                "best_ve2.pth",
                "ve2_data_aug_class_best.pth",
            ],
        ),
        "improve_ckpt": _pick(
            run_dir / "Net_improve",
            [
                "best_ve2_generalization_improve.pth",
                "best_ve2_data_aug_improve.pth",
                "best_ve2_real_improve.pth",
                "ve2_improve_best.pth",
            ],
        ),
        "regression_ckpt": _pick(
            run_dir / "Net_regression",
            [
                "best_ve2_generalization_regression.pth",
                "best_ve2_data_aug_regression.pth",
                "best_ve2_real_regression.pth",
                "best_ve2.pth",
            ],
        ),
    }


def _parse_angle_from_name(image_path: Path) -> float:
    stem = image_path.stem
    if "_" not in stem:
        raise ValueError("image filename does not contain an angle suffix; pass --angle explicitly")
    tail = stem.rsplit("_", 1)[-1]
    try:
        return float(tail)
    except ValueError as exc:
        raise ValueError("failed to parse angle from filename; pass --angle explicitly") from exc


def _checkpoint_preprocess_config(ckpt_path: Path, fallback: PreprocessConfig = DEFAULT_PREPROCESS_CONFIG) -> PreprocessConfig:
    preprocess_dict: dict[str, Any] | None = None
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(ckpt, dict):
            preprocess_dict = ckpt.get("preprocess")
    except Exception:
        preprocess_dict = None

    if preprocess_dict is None:
        summary_path = ckpt_path.parent / "training_summary.json"
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if isinstance(summary, dict):
                    preprocess_dict = summary.get("preprocess")
            except Exception:
                preprocess_dict = None
    return preprocess_config_from_dict(preprocess_dict, fallback=fallback)


def _build_preprocess_config(roi_config: ROIConfig, base_config: PreprocessConfig = DEFAULT_PREPROCESS_CONFIG) -> PreprocessConfig:
    return PreprocessConfig(
        color_space=base_config.color_space,
        input_size=base_config.input_size,
        use_roi=roi_config.enabled,
    )


def _preprocess(
    image_path: Path,
    device: torch.device,
    roi_config: ROIConfig | None = None,
    *,
    base_config: PreprocessConfig = DEFAULT_PREPROCESS_CONFIG,
) -> torch.Tensor:
    roi_cfg = roi_config or ROIConfig()
    preprocess_config = _build_preprocess_config(roi_cfg, base_config=base_config)
    return preprocess_path_to_tensor(
        image_path,
        device=device,
        config=preprocess_config,
        bottom_ratio=roi_cfg.bottom_ratio,
    )


def _load_model_from_checkpoint(module_path: Path, class_name: str, ckpt_path: Path, device: torch.device):
    module = _load_module(module_path, f"{module_path.stem}_{ckpt_path.stem}")
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    builder = getattr(module, "build_model_for_checkpoint", None)
    if callable(builder):
        model = builder(state).to(device)
    else:
        cls = getattr(module, class_name)
        model = cls().to(device)
        model.load_state_dict(state)
    model.eval()
    return module, model


def _predict_net_class(x: torch.Tensor, ckpt_path: Path, device: torch.device) -> float:
    config_mod, model = _load_model_from_checkpoint(
        REPO_ROOT / "e2e_competition" / "Net_class" / "models.py",
        "AutoDriveNet",
        ckpt_path,
        device,
    )
    steering_mod = _load_module(REPO_ROOT / "e2e_competition" / "Net_class" / "steering_config.py", f"joint_net_class_cfg_{ckpt_path.stem}")
    with torch.no_grad():
        output = model(x)
        angle = steering_mod.decode_output(output)
    return float(angle.reshape(-1)[0].item() if torch.is_tensor(angle) else angle)


def _predict_net_improve(x: torch.Tensor, ckpt_path: Path, device: torch.device) -> float:
    _, model = _load_model_from_checkpoint(
        REPO_ROOT / "e2e_competition" / "Net_improve" / "models.py",
        "AutoDriveNetImprove",
        ckpt_path,
        device,
    )
    config_mod = _load_module(REPO_ROOT / "e2e_competition" / "Net_improve" / "steering_config.py", f"joint_net_improve_cfg_{ckpt_path.stem}")
    with torch.no_grad():
        logits = model(x)
        cls_idx = int(torch.argmax(logits, dim=1).item())
    return float(config_mod.class_to_angle(cls_idx))


def _predict_regression(x: torch.Tensor, ckpt_path: Path, device: torch.device) -> float:
    _, model = _load_model_from_checkpoint(
        REPO_ROOT / "e2e_self-driving" / "Net" / "models.py",
        "AutoDriveNet",
        ckpt_path,
        device,
    )
    with torch.no_grad():
        y = model(x)
    return float(y.reshape(-1)[0].item())


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def _build_roi_config_from_args(args) -> ROIConfig:
    enabled = bool(getattr(args, "enable_roi", False)) and not bool(getattr(args, "disable_roi", False))
    return ROIConfig(
        enabled=enabled,
        mode="bottom_ratio",
        bottom_ratio=float(getattr(args, "roi_bottom_ratio", 0.6)),
    )


def main():
    parser = argparse.ArgumentParser(description="Joint single-image inference for three trained models.")
    parser.add_argument("image", help="Path to the image. Angle is parsed from filename suffix unless --angle is set.")
    parser.add_argument("--angle", type=float, default=None, help="Ground-truth steering angle override.")
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
    parser.add_argument("--json", action="store_true", help="Also print a JSON result block.")
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    gt_angle = float(args.angle) if args.angle is not None else _parse_angle_from_name(image_path)
    roi_config = _build_roi_config_from_args(args)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_preprocess = _checkpoint_preprocess_config(paths["class_ckpt"], DEFAULT_PREPROCESS_CONFIG)
    improve_preprocess = _checkpoint_preprocess_config(paths["improve_ckpt"], DEFAULT_PREPROCESS_CONFIG)
    regression_preprocess = _checkpoint_preprocess_config(paths["regression_ckpt"], DEFAULT_PREPROCESS_CONFIG)
    x_class = _preprocess(image_path, device, roi_config=roi_config, base_config=class_preprocess)
    x_improve = _preprocess(image_path, device, roi_config=roi_config, base_config=improve_preprocess)
    x_regression = _preprocess(image_path, device, roi_config=roi_config, base_config=regression_preprocess)

    predictions = [
        ("Net_class", _predict_net_class(x_class, paths["class_ckpt"], device)),
        ("Net_improve", _predict_net_improve(x_improve, paths["improve_ckpt"], device)),
        ("Net_regression", _predict_regression(x_regression, paths["regression_ckpt"], device)),
    ]
    rows: list[dict[str, Any]] = []
    for name, pred in predictions:
        rows.append({
            "model": name,
            "prediction": pred,
            "groundTruth": gt_angle,
            "absoluteError": abs(pred - gt_angle),
        })

    print(f"image: {image_path}")
    print(f"ground_truth: {_fmt(gt_angle)}")
    print(f"device: {device}")
    print(f"run_dir: {run_dir}")
    print(
        "preprocess_by_model: "
        + json.dumps(
            {
                "Net_class": {"colorSpace": class_preprocess.color_space, "inputSize": list(class_preprocess.input_size), "useRoi": roi_config.enabled},
                "Net_improve": {"colorSpace": improve_preprocess.color_space, "inputSize": list(improve_preprocess.input_size), "useRoi": roi_config.enabled},
                "Net_regression": {"colorSpace": regression_preprocess.color_space, "inputSize": list(regression_preprocess.input_size), "useRoi": roi_config.enabled},
            },
            ensure_ascii=False,
        )
    )
    print(f"roi: {json.dumps(asdict(roi_config), ensure_ascii=False)}")
    print("")
    print("model           prediction   abs_error")
    print("--------------- ------------ ------------")
    for row in rows:
        print(f"{row['model']:<15} {_fmt(row['prediction']):>12} {_fmt(row['absoluteError']):>12}")

    if args.json:
        print("")
        print(
            json.dumps(
                {
                    "image": str(image_path),
                    "groundTruth": gt_angle,
                    "preprocessByModel": {
                        "Net_class": {"colorSpace": class_preprocess.color_space, "inputSize": list(class_preprocess.input_size), "useRoi": roi_config.enabled},
                        "Net_improve": {"colorSpace": improve_preprocess.color_space, "inputSize": list(improve_preprocess.input_size), "useRoi": roi_config.enabled},
                        "Net_regression": {"colorSpace": regression_preprocess.color_space, "inputSize": list(regression_preprocess.input_size), "useRoi": roi_config.enabled},
                    },
                    "roi": asdict(roi_config),
                    "results": rows,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()

