"""Run training jobs in a background worker and sync progress to SQLite/memory."""

from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
NET_WEB_DIR = CURRENT_DIR.parents[0]
PROJECT_DIR = CURRENT_DIR.parents[1]
WORKSPACE_ROOT = CURRENT_DIR.parents[2]
for candidate in (NET_WEB_DIR, PROJECT_DIR, WORKSPACE_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from app import database as db
from app.config import settings
from app.state import append_loss_point, get_controls, get_progress, merge_progress, release_controls
from app.training_artifacts import read_training_summary, write_progress_snapshot, write_training_summary
from datasets import AutoDriveDataset, AutoDriveListDataset
from models import AutoDriveLegacyNet, AutoDriveNet, AutoDriveNetTemporal
from steering_augmentations import AugConfig, build_eval_transforms, build_stress_transforms, build_train_transforms
from steering_preprocess import (
    PreprocessConfig,
    build_angle_vocab,
    encode_angles_to_vocab,
    preprocess_config_to_dict,
)
from utils import AverageMeter


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _build_regression_aug_config(*, num_frames: int) -> AugConfig:
    preprocess = PreprocessConfig(
        color_space=os.getenv("VENET_PREPROCESS_COLOR_SPACE", "hsv").strip().lower(),
        input_size=(
            int(os.getenv("VENET_INPUT_HEIGHT", "120")),
            int(os.getenv("VENET_INPUT_WIDTH", "160")),
        ),
        use_roi=False,
    )
    style_mix_ratio = tuple(
        float(part.strip())
        for part in os.getenv("VENET_STYLE_MIX_RATIO", "0.5,0.3,0.2").split(",")
        if part.strip()
    )
    if len(style_mix_ratio) != 3:
        style_mix_ratio = (0.5, 0.3, 0.2)
    return AugConfig(preprocess=preprocess, style_mix_ratio=style_mix_ratio, num_frames=num_frames)


def _build_regression_transforms(*, num_frames: int) -> tuple[Any, Any, Any, PreprocessConfig]:
    cfg = _build_regression_aug_config(num_frames=num_frames)
    return (
        build_train_transforms(cfg),
        build_eval_transforms(cfg),
        build_stress_transforms(cfg),
        cfg.preprocess,
    )


def _extract_batch(batch):
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError(f"unsupported batch type: {type(batch)!r}")


def _forward_model(model: nn.Module, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    try:
        pred, aux_logits = model(imgs, return_aux=True)
    except TypeError:
        pred = model(imgs)
        aux_logits = None
    return pred, aux_logits


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    reg_criterion: nn.Module,
    aux_criterion: nn.Module,
    angle_vocab: list[float],
    aux_cls_weight: float,
    optimizer: torch.optim.Optimizer,
    should_stop,
    grad_clip: float = 0.0,
) -> tuple[float, float]:
    model.train()
    loss_meter = AverageMeter()
    mae_meter = AverageMeter()
    for batch in loader:
        if should_stop():
            break
        imgs, labels = _extract_batch(batch)
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        pred, aux_logits = _forward_model(model, imgs)
        reg_loss = reg_criterion(pred, labels)
        aux_loss = reg_loss.new_tensor(0.0)
        if aux_logits is not None and aux_cls_weight > 0:
            aux_targets = encode_angles_to_vocab(labels.view(-1), angle_vocab, device=device)
            aux_loss = aux_criterion(aux_logits, aux_targets)
        loss = reg_loss + aux_cls_weight * aux_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        mae = (pred - labels).abs().mean().item()
        loss_meter.update(loss.item(), imgs.size(0))
        mae_meter.update(mae, imgs.size(0))
    return loss_meter.avg, mae_meter.avg


def _evaluate_regression(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    reg_criterion: nn.Module,
    aux_criterion: nn.Module,
    angle_vocab: list[float],
    aux_cls_weight: float,
) -> tuple[float, float]:
    model.eval()
    loss_meter = AverageMeter()
    mae_meter = AverageMeter()
    with torch.no_grad():
        for batch in loader:
            imgs, labels = _extract_batch(batch)
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            pred, aux_logits = _forward_model(model, imgs)
            reg_loss = reg_criterion(pred, labels)
            aux_loss = reg_loss.new_tensor(0.0)
            if aux_logits is not None and aux_cls_weight > 0:
                aux_targets = encode_angles_to_vocab(labels.view(-1), angle_vocab, device=device)
                aux_loss = aux_criterion(aux_logits, aux_targets)
            loss = reg_loss + aux_cls_weight * aux_loss
            mae = (pred - labels).abs().mean().item()
            loss_meter.update(loss.item(), imgs.size(0))
            mae_meter.update(mae, imgs.size(0))
    return loss_meter.avg, mae_meter.avg


def _wait_resume(ctrl) -> bool:
    while ctrl.pause.is_set() and not ctrl.stop.is_set():
        time.sleep(0.2)
    return not ctrl.stop.is_set()


def _persist_progress_snapshot(task_id: str, artifacts_dir: Path) -> None:
    try:
        write_progress_snapshot(artifacts_dir, get_progress(task_id))
    except Exception:
        pass


def _read_list_pairs(list_file: Path) -> list[tuple[Path, float]]:
    pairs: list[tuple[Path, float]] = []
    with open(list_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                img_path, angle_str = line.rsplit(" ", 1)
                pairs.append((Path(img_path), float(angle_str)))
            except ValueError:
                continue
    return pairs


def _copy_pairs_as_images(pairs: list[tuple[Path, float]], out_dir: Path, prefix: str) -> list[tuple[str, float]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: list[tuple[str, float]] = []
    for i, (src, angle) in enumerate(pairs):
        ext = src.suffix.lower() if src.suffix else ".jpg"
        name = f"{prefix}_{i:06d}{ext}"
        dst = out_dir / name
        shutil.copy2(src, dst)
        copied.append((name, angle))
    return copied


def _run_subprocess_or_raise(args: list[str], cwd: Path, on_stdout_line=None) -> None:
    proc = subprocess.Popen(
        args,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    out_lines: list[str] = []
    err_lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        out_lines.append(line)
        if on_stdout_line is not None:
            on_stdout_line(line)
    assert proc.stderr is not None
    for line in proc.stderr:
        err_lines.append(line.rstrip("\n"))
    code = proc.wait()
    if code != 0:
        out = "\n".join(out_lines).strip()
        err = "\n".join(err_lines).strip()
        cmd = " ".join(args)
        detail = [
            f"Command failed with exit code {code}",
            f"CMD: {cmd}",
            f"STDERR:\n{err or '(empty)'}",
            f"STDOUT:\n{out or '(empty)'}",
        ]
        raise RuntimeError("\n\n".join(detail)[:6000])


def _ensure_cyclegan_runtime_deps() -> None:
    missing: list[str] = []
    for mod in ("dominate",):
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(mod)
    if missing:
        deps = " ".join(missing)
        raise RuntimeError(
            f"Missing CycleGAN dependencies: {deps}. Please run `pip install {deps}` or `pip install -r requirements.txt` in the Net_Web environment."
        )


def _relative_parent(path: Path, root: Path) -> Path:
    try:
        return path.parent.resolve().relative_to(root.resolve())
    except Exception:
        return Path(path.parent.name)


def _build_c_dataset_with_cyclegan(
    dataset_root_a: Path,
    dataset_root_b: Path,
    artifacts_dir: Path,
    cyclegan_epochs: int,
    cyclegan_decay_epochs: int,
    cyclegan_batch_size: int,
    cyclegan_save_epoch_freq: int,
    cyclegan_save_latest_freq: int,
    cyclegan_load_size: int,
    cyclegan_crop_size: int,
    cyclegan_lambda_identity: float,
    on_train_progress=None,
) -> tuple[Path, Path]:
    project = settings.cyclegan_project_root
    if not project.is_dir():
        raise RuntimeError(f"CycleGAN project directory does not exist: {project}")
    train_py = project / "train.py"
    test_py = project / "test.py"
    if not train_py.is_file() or not test_py.is_file():
        raise RuntimeError("CycleGAN train.py or test.py not found")
    _ensure_cyclegan_runtime_deps()

    job_id = f"task_{uuid.uuid4().hex[:8]}"
    work_dir = artifacts_dir / "domain_aug"
    cyc_data = work_dir / "datasets" / job_id
    train_a_dir = cyc_data / "trainA"
    train_b_dir = cyc_data / "trainB"
    infer_a_dir = cyc_data / "inferA"
    outputs_dir = work_dir / "results"
    c_dir = work_dir / "dataset_c"
    c_dir.mkdir(parents=True, exist_ok=True)

    train_pairs_a = _read_list_pairs(_resolve_split_file(dataset_root_a, "train"))
    train_pairs_b = _read_list_pairs(_resolve_split_file(dataset_root_b, "train"))
    if not train_pairs_a or not train_pairs_b:
        raise RuntimeError("A/B 数据集 train split 为空，无法执行域增强")

    _copy_pairs_as_images(train_pairs_a, train_a_dir, "a")
    _copy_pairs_as_images(train_pairs_b, train_b_dir, "b")

    infer_a_dir.mkdir(parents=True, exist_ok=True)
    infer_pairs: list[dict[str, Any]] = []
    for idx, (src, angle) in enumerate(train_pairs_a):
        ext = src.suffix.lower() if src.suffix else ".jpg"
        infer_name = f"infer_{idx:06d}{ext}"
        shutil.copy2(src, infer_a_dir / infer_name)
        infer_pairs.append(
            {
                "inferName": infer_name,
                "angle": angle,
                "sourcePath": os.fspath(src.resolve()),
                "sourceName": src.name,
                "relativeParent": os.fspath(_relative_parent(src, dataset_root_a)),
            }
        )

    common = [
        "--dataroot",
        str(cyc_data),
        "--name",
        job_id,
        "--model",
        "cycle_gan",
        "--dataset_mode",
        "unaligned",
        "--direction",
        "AtoB",
        "--batch_size",
        str(cyclegan_batch_size),
        "--save_epoch_freq",
        str(cyclegan_save_epoch_freq),
        "--save_latest_freq",
        str(cyclegan_save_latest_freq),
        "--load_size",
        str(cyclegan_load_size),
        "--crop_size",
        str(cyclegan_crop_size),
        "--lambda_identity",
        str(cyclegan_lambda_identity),
    ]
    train_args = [
        sys.executable,
        str(train_py),
        *common,
        "--n_epochs",
        str(cyclegan_epochs),
        "--n_epochs_decay",
        str(cyclegan_decay_epochs),
    ]
    total_train_epochs = cyclegan_epochs + cyclegan_decay_epochs
    epoch_re = re.compile(r"End of epoch (\d+) / (\d+)")

    def _on_train_line(line: str) -> None:
        if on_train_progress is None:
            return
        match = epoch_re.search(line)
        if not match:
            return
        cur = int(match.group(1))
        total = int(match.group(2)) if int(match.group(2)) > 0 else total_train_epochs
        ratio = max(0.0, min(1.0, cur / max(total, 1)))
        on_train_progress(5.0 + ratio * 90.0, f"CycleGAN epoch {cur}/{total}")

    _run_subprocess_or_raise(train_args, project, on_stdout_line=_on_train_line)

    infer_args = [
        sys.executable,
        str(test_py),
        "--dataroot",
        str(infer_a_dir),
        "--name",
        job_id,
        "--model",
        "test",
        "--dataset_mode",
        "single",
        "--direction",
        "AtoB",
        "--model_suffix",
        "_A",
        "--num_test",
        str(len(infer_pairs)),
        "--results_dir",
        str(outputs_dir),
        "--no_dropout",
        "--load_size",
        str(cyclegan_crop_size),
        "--crop_size",
        str(cyclegan_crop_size),
    ]
    if on_train_progress is not None:
        on_train_progress(97.0, "CycleGAN 训练完成，正在生成 C 域数据")
    _run_subprocess_or_raise(infer_args, project)
    if on_train_progress is not None:
        on_train_progress(100.0, "C 域数据生成完成")

    image_dir = outputs_dir / job_id / "test_latest" / "images"
    if not image_dir.is_dir():
        raise RuntimeError("CycleGAN inference output directory does not exist")

    c_list = artifacts_dir / "train_c.txt"
    pairs_meta: list[dict[str, Any]] = []
    with open(c_list, "w", encoding="utf-8") as f:
        for idx, pair in enumerate(infer_pairs):
            source_path = Path(pair["sourcePath"])
            fake_name = f"{Path(pair['inferName']).stem}_fake.png"
            fake_path = image_dir / fake_name
            if not fake_path.is_file():
                raise RuntimeError(f"CycleGAN generated image missing: {fake_name}")

            relative_parent = Path(pair["relativeParent"])
            dst_dir = c_dir / relative_parent
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_name = source_path.with_suffix(".png").name
            dst = dst_dir / dst_name
            shutil.copy2(fake_path, dst)
            f.write(f"{os.fspath(dst.resolve())} {pair['angle']}\n")
            pairs_meta.append(
                {
                    "index": idx,
                    "aName": os.fspath(relative_parent / source_path.name),
                    "cName": os.fspath(relative_parent / dst_name),
                    "aPath": os.fspath(source_path.resolve()),
                    "cPath": os.fspath(dst.resolve()),
                }
            )
    pairs_json = artifacts_dir / "domain_aug_pairs.json"
    with open(pairs_json, "w", encoding="utf-8") as pf:
        pf.write(json.dumps(pairs_meta, ensure_ascii=False, indent=2))
    return c_list, pairs_json


def _normalize_model_variant(raw: str | None) -> str:
    value = (raw or "").strip().lower()
    if value == "legacy":
        return "legacy"
    if value in {"temporal3", "mobilenet_temporal3", "mobilenet_v2_temporal3"}:
        return "temporal3"
    return "mobilenet_v2"


def _resolve_num_frames(model_variant: str) -> int:
    return 3 if model_variant == "temporal3" else 1


def _resolve_split_candidates(split_name: str) -> list[str]:
    key = split_name.lower()
    if key == "train":
        return ["train_clean.txt", "train.txt"]
    if key == "val":
        return ["val_clean.txt", "val.txt"]
    if key == "test":
        return ["test_clean.txt", "test.txt"]
    return [f"{key}.txt"]


def _resolve_split_file(dataset_root: Path, split_name: str) -> Path:
    for candidate in _resolve_split_candidates(split_name):
        path = dataset_root / candidate
        if path.is_file():
            return path
    tried = ", ".join(str(dataset_root / name) for name in _resolve_split_candidates(split_name))
    raise FileNotFoundError(f"split file not found for {split_name}; tried: {tried}")


def _has_split_file(dataset_root: Path, split_name: str) -> bool:
    return any((dataset_root / candidate).is_file() for candidate in _resolve_split_candidates(split_name))


def _build_model_for_variant(
    model_variant: str,
    *,
    num_aux_classes: int,
    use_pretrained: bool,
    num_frames: int,
) -> nn.Module:
    if model_variant == "legacy":
        return AutoDriveLegacyNet()
    if model_variant == "temporal3":
        return AutoDriveNetTemporal(
            num_aux_classes=num_aux_classes,
            use_pretrained=use_pretrained,
            num_frames=num_frames,
        )
    return AutoDriveNet(num_aux_classes=num_aux_classes, use_pretrained=use_pretrained)


def _configure_optimizer(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    backbone_lr_factor: float,
    freeze_backbone: bool,
    lr_patience: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau]:
    if not hasattr(model, "set_backbone_trainable"):
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        model.set_backbone_trainable(not freeze_backbone)
        if freeze_backbone:
            optimizer = torch.optim.AdamW(model.head_parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.AdamW(
                [
                    {"params": list(model.head_parameters()), "lr": learning_rate},
                    {"params": list(model.backbone_parameters()), "lr": learning_rate * backbone_lr_factor},
                ],
                weight_decay=weight_decay,
            )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, lr_patience),
        min_lr=1e-6,
    )
    return optimizer, scheduler


def _make_checkpoint_payload(
    *,
    epoch: int,
    best_epoch: int,
    best_metric: float,
    angle_vocab: list[float],
    preprocess: PreprocessConfig,
    model_variant: str,
    base_model: nn.Module,
    num_frames: int,
    frame_stride: int,
) -> dict[str, Any]:
    state_dict = {key: value.detach().cpu() for key, value in base_model.state_dict().items()}
    return {
        "epoch": epoch,
        "bestEpoch": best_epoch,
        "bestSelectionMetric": best_metric,
        "model": state_dict,
        "modelRaw": state_dict,
        "angleVocab": angle_vocab,
        "modelVariant": model_variant,
        "preprocess": preprocess_config_to_dict(preprocess),
        "pretrainedLoaded": bool(getattr(base_model, "pretrained_loaded", False)),
        "numFrames": int(num_frames),
        "frameStride": int(frame_stride),
    }


def _load_task_params(task_id: str, user_id: str) -> dict[str, Any]:
    row = db.get_task(task_id, user_id)
    if row is None or "task_params_json" not in row.keys() or not row["task_params_json"]:
        return {}
    try:
        payload = json.loads(row["task_params_json"])
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def training_worker(
    task_id: str,
    user_id: str,
    dataset_root: str,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    domain_augmentation: bool,
    dataset_b_root: str | None,
    cyclegan_epochs: int,
    cyclegan_decay_epochs: int,
    cyclegan_batch_size: int,
    cyclegan_save_epoch_freq: int,
    cyclegan_save_latest_freq: int,
    cyclegan_load_size: int,
    cyclegan_crop_size: int,
    cyclegan_lambda_identity: float,
    model_variant: str,
    artifacts_dir: Path,
) -> None:
    ctrl = get_controls(task_id)
    if ctrl is None:
        return

    task_params = _load_task_params(task_id, user_id)
    model_variant = _normalize_model_variant(task_params.get("modelVariant", model_variant))
    num_frames = _resolve_num_frames(model_variant)
    frame_stride = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_ui_epochs = epochs * (2 if domain_augmentation else 1)
    merge_progress(
        task_id,
        {
            "status": "running",
            "totalEpochs": total_ui_epochs,
            "currentEpoch": 0,
            "message": f"设备: {device.type} | 架构: {model_variant}",
            "baselineProgress": 0.0,
            "domainAugmentationProgress": (0.0 if domain_augmentation else None),
            "domainAugmentationText": ("等待开始" if domain_augmentation else None),
            "augmentedProgress": (0.0 if domain_augmentation else None),
            "augmented": ({"trainLossSeries": [], "valLossSeries": []} if domain_augmentation else None),
        },
    )

    train_transform, eval_transform, stress_transform, preprocess = _build_regression_transforms(num_frames=num_frames)
    reg_criterion = nn.SmoothL1Loss().to(device)
    aux_criterion = nn.CrossEntropyLoss().to(device)
    aux_cls_weight = float(os.getenv("VENET_AUX_CLS_WEIGHT", "0.3"))
    use_pretrained = _env_bool("VENET_USE_PRETRAINED", True)
    weight_decay = float(os.getenv("VENET_WEIGHT_DECAY", "1e-4"))
    grad_clip = float(os.getenv("VENET_GRAD_CLIP", "3.0"))
    early_stop_patience = int(os.getenv("VENET_EARLY_STOP_PATIENCE", "12"))
    early_stop_min_delta = float(os.getenv("VENET_EARLY_STOP_MIN_DELTA", "1e-4"))
    freeze_backbone_epochs = int(os.getenv("VENET_FREEZE_BACKBONE_EPOCHS", "5"))
    backbone_lr_factor = float(os.getenv("VENET_BACKBONE_LR_FACTOR", "0.1"))
    lr_patience = max(2, early_stop_patience // 3)

    if model_variant == "legacy":
        use_pretrained = False
        freeze_backbone_epochs = 0
        aux_cls_weight = 0.0

    dataset_root_path = Path(dataset_root)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = artifacts_dir / "baseline.pth"
    augmented_path = artifacts_dir / "augmented.pth"

    last_train_b = 0.0
    last_val_b = 0.0
    last_train_a = 0.0
    last_val_a = 0.0
    mae_b = 0.0
    mae_a = 0.0
    best_baseline_epoch = 0
    best_baseline_loss: float | None = None
    best_baseline_mae = float("inf")
    baseline_completed_epochs = 0
    baseline_stopped_epoch: int | None = None
    baseline_early_stopped = False
    baseline_test_loss = 0.0
    best_augmented_epoch = 0
    best_augmented_loss: float | None = None
    best_augmented_mae = float("inf")
    augmented_completed_epochs = 0
    augmented_stopped_epoch: int | None = None
    augmented_early_stopped = False
    augmented_test_loss = 0.0
    has_test_split = _has_split_file(dataset_root_path, "test")

    try:
        db.update_task_status(task_id, user_id, "running", None)

        dataset_kwargs = {"num_frames": num_frames, "frame_stride": frame_stride}
        train_ds = AutoDriveDataset(
            dataset_root,
            "train",
            train_transform,
            split_name="train_clean",
            **dataset_kwargs,
        )
        val_ds = AutoDriveDataset(
            dataset_root,
            "val",
            eval_transform,
            split_name="val_clean",
            **dataset_kwargs,
        )
        val_stress_ds = AutoDriveDataset(
            dataset_root,
            "val",
            stress_transform,
            deterministic_seed=20260418,
            split_name="val_clean",
            **dataset_kwargs,
        )
        angle_vocab = build_angle_vocab(train_ds.angles)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        val_stress_loader = DataLoader(
            val_stress_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        test_ds = (
            AutoDriveDataset(
                dataset_root,
                "test",
                eval_transform,
                split_name="test_clean",
                **dataset_kwargs,
            )
            if has_test_split
            else val_ds
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        model = _build_model_for_variant(
            model_variant,
            num_aux_classes=len(angle_vocab),
            use_pretrained=use_pretrained,
            num_frames=num_frames,
        ).to(device)
        best_baseline_state = None
        optimizer = None
        scheduler = None
        current_stage: str | None = None
        baseline_no_improve = 0

        for epoch in range(1, epochs + 1):
            if not _wait_resume(ctrl):
                db.update_task_status(task_id, user_id, "stopped", "用户终止")
                merge_progress(task_id, {"status": "stopped"})
                release_controls(task_id)
                return

            desired_stage = "full"
            if hasattr(model, "set_backbone_trainable") and freeze_backbone_epochs > 0:
                desired_stage = "head" if epoch <= freeze_backbone_epochs else "full"
            if optimizer is None or desired_stage != current_stage:
                optimizer, scheduler = _configure_optimizer(
                    model,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    backbone_lr_factor=backbone_lr_factor,
                    freeze_backbone=(desired_stage == "head"),
                    lr_patience=lr_patience,
                )
                current_stage = desired_stage

            last_train_b, _baseline_train_mae = _train_epoch(
                model,
                train_loader,
                device,
                reg_criterion,
                aux_criterion,
                angle_vocab,
                aux_cls_weight,
                optimizer,
                lambda: ctrl.stop.is_set(),
                grad_clip,
            )
            if ctrl.stop.is_set():
                db.update_task_status(task_id, user_id, "stopped", "用户终止")
                merge_progress(task_id, {"status": "stopped"})
                release_controls(task_id)
                return

            last_val_b, _baseline_val_mae = _evaluate_regression(
                model,
                val_loader,
                device,
                reg_criterion,
                aux_criterion,
                angle_vocab,
                aux_cls_weight,
            )
            _, baseline_val_stress_mae = _evaluate_regression(
                model,
                val_stress_loader,
                device,
                reg_criterion,
                aux_criterion,
                angle_vocab,
                aux_cls_weight,
            )
            baseline_completed_epochs = epoch
            assert scheduler is not None
            scheduler.step(baseline_val_stress_mae)
            append_loss_point(task_id, "baseline", epoch, last_train_b, last_val_b)
            merge_progress(
                task_id,
                {
                    "currentEpoch": epoch,
                    "status": "running",
                    "baselineProgress": (epoch / max(epochs, 1)) * 100.0,
                    "message": f"基准阶段 epoch {epoch}/{epochs} | val_stress_mae={baseline_val_stress_mae:.4f}",
                },
            )

            if baseline_val_stress_mae < best_baseline_mae - early_stop_min_delta:
                best_baseline_mae = baseline_val_stress_mae
                best_baseline_epoch = epoch
                best_baseline_loss = last_val_b
                baseline_no_improve = 0
                best_baseline_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                torch.save(
                    _make_checkpoint_payload(
                        epoch=epoch,
                        best_epoch=best_baseline_epoch,
                        best_metric=best_baseline_mae,
                        angle_vocab=angle_vocab,
                        preprocess=preprocess,
                        model_variant=model_variant,
                        base_model=model,
                        num_frames=num_frames,
                        frame_stride=frame_stride,
                    ),
                    baseline_path,
                )
            else:
                baseline_no_improve += 1
                if baseline_no_improve >= early_stop_patience:
                    baseline_early_stopped = True
                    baseline_stopped_epoch = epoch
                    merge_progress(task_id, {"message": f"基准阶段早停于 epoch {epoch}"})
                    break

            _persist_progress_snapshot(task_id, artifacts_dir)

        if best_baseline_state is None:
            best_baseline_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_baseline_mae = baseline_val_stress_mae
            best_baseline_epoch = baseline_completed_epochs
            best_baseline_loss = last_val_b
            torch.save(
                _make_checkpoint_payload(
                    epoch=baseline_completed_epochs,
                    best_epoch=best_baseline_epoch,
                    best_metric=best_baseline_mae,
                    angle_vocab=angle_vocab,
                    preprocess=preprocess,
                    model_variant=model_variant,
                    base_model=model,
                    num_frames=num_frames,
                    frame_stride=frame_stride,
                ),
                baseline_path,
            )

        model.load_state_dict(best_baseline_state)
        baseline_test_loss, baseline_test_mae = _evaluate_regression(
            model,
            test_loader,
            device,
            reg_criterion,
            aux_criterion,
            angle_vocab,
            aux_cls_weight,
        )
        mae_b = baseline_test_mae
        db.update_task_checkpoints(task_id, user_id, baseline_ckpt=str(baseline_path))

        del model, optimizer, train_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if domain_augmentation:
            merge_progress(
                task_id,
                {
                    "augmented": {"trainLossSeries": [], "valLossSeries": []},
                    "message": "基准阶段完成，开始生成 CycleGAN 数据集 C",
                },
            )
            if not dataset_b_root:
                raise RuntimeError("缺少 B 域数据集路径")
            merge_progress(task_id, {"domainAugmentationProgress": 5.0})
            c_list, _pairs_json = _build_c_dataset_with_cyclegan(
                dataset_root_a=dataset_root_path,
                dataset_root_b=Path(dataset_b_root),
                artifacts_dir=artifacts_dir,
                cyclegan_epochs=cyclegan_epochs,
                cyclegan_decay_epochs=cyclegan_decay_epochs,
                cyclegan_batch_size=cyclegan_batch_size,
                cyclegan_save_epoch_freq=cyclegan_save_epoch_freq,
                cyclegan_save_latest_freq=cyclegan_save_latest_freq,
                cyclegan_load_size=cyclegan_load_size,
                cyclegan_crop_size=cyclegan_crop_size,
                cyclegan_lambda_identity=cyclegan_lambda_identity,
                on_train_progress=lambda progress, text: merge_progress(
                    task_id,
                    {
                        "domainAugmentationProgress": progress,
                        "domainAugmentationText": text,
                    },
                ),
            )
            merge_progress(
                task_id,
                {
                    "domainAugmentationProgress": 100.0,
                    "domainAugmentationText": "已完成",
                    "message": "C 域数据集已生成，开始训练增强阶段",
                },
            )

            combined_train = artifacts_dir / "train_augmented.txt"
            with open(_resolve_split_file(dataset_root_path, "train"), "r", encoding="utf-8") as fa, open(
                c_list,
                "r",
                encoding="utf-8",
            ) as fc, open(combined_train, "w", encoding="utf-8") as fout:
                a_content = fa.read()
                c_content = fc.read()
                fout.write(a_content)
                if a_content and not a_content.endswith("\n"):
                    fout.write("\n")
                fout.write(c_content)

            train_ds_a = AutoDriveListDataset(
                str(combined_train),
                train_transform,
                num_frames=num_frames,
                frame_stride=frame_stride,
            )
            val_ds_a = AutoDriveDataset(
                dataset_root,
                "val",
                eval_transform,
                split_name="val_clean",
                **dataset_kwargs,
            )
            val_stress_ds_a = AutoDriveDataset(
                dataset_root,
                "val",
                stress_transform,
                deterministic_seed=20260418,
                split_name="val_clean",
                **dataset_kwargs,
            )
            angle_vocab_a = build_angle_vocab(train_ds_a.angles)
            train_loader_a = DataLoader(
                train_ds_a,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
            )
            val_loader_a = DataLoader(
                val_ds_a,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
            )
            val_stress_loader_a = DataLoader(
                val_stress_ds_a,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
            )
            test_ds_a = (
                AutoDriveDataset(
                    dataset_root,
                    "test",
                    eval_transform,
                    split_name="test_clean",
                    **dataset_kwargs,
                )
                if has_test_split
                else val_ds_a
            )
            test_loader_a = DataLoader(
                test_ds_a,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
            )

            model_a = _build_model_for_variant(
                model_variant,
                num_aux_classes=len(angle_vocab_a),
                use_pretrained=use_pretrained,
                num_frames=num_frames,
            ).to(device)
            best_augmented_state = None
            opt_a = None
            scheduler_a = None
            current_aug_stage: str | None = None
            augmented_no_improve = 0

            for epoch in range(1, epochs + 1):
                if not _wait_resume(ctrl):
                    db.update_task_status(task_id, user_id, "stopped", "用户终止")
                    merge_progress(task_id, {"status": "stopped"})
                    release_controls(task_id)
                    return

                desired_stage = "full"
                if hasattr(model_a, "set_backbone_trainable") and freeze_backbone_epochs > 0:
                    desired_stage = "head" if epoch <= freeze_backbone_epochs else "full"
                if opt_a is None or desired_stage != current_aug_stage:
                    opt_a, scheduler_a = _configure_optimizer(
                        model_a,
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        backbone_lr_factor=backbone_lr_factor,
                        freeze_backbone=(desired_stage == "head"),
                        lr_patience=lr_patience,
                    )
                    current_aug_stage = desired_stage

                global_epoch = epochs + epoch
                last_train_a, _augmented_train_mae = _train_epoch(
                    model_a,
                    train_loader_a,
                    device,
                    reg_criterion,
                    aux_criterion,
                    angle_vocab_a,
                    aux_cls_weight,
                    opt_a,
                    lambda: ctrl.stop.is_set(),
                    grad_clip,
                )
                if ctrl.stop.is_set():
                    db.update_task_status(task_id, user_id, "stopped", "用户终止")
                    merge_progress(task_id, {"status": "stopped"})
                    release_controls(task_id)
                    return

                last_val_a, _augmented_val_mae = _evaluate_regression(
                    model_a,
                    val_loader_a,
                    device,
                    reg_criterion,
                    aux_criterion,
                    angle_vocab_a,
                    aux_cls_weight,
                )
                _, augmented_val_stress_mae = _evaluate_regression(
                    model_a,
                    val_stress_loader_a,
                    device,
                    reg_criterion,
                    aux_criterion,
                    angle_vocab_a,
                    aux_cls_weight,
                )
                augmented_completed_epochs = epoch
                assert scheduler_a is not None
                scheduler_a.step(augmented_val_stress_mae)
                append_loss_point(task_id, "augmented", epoch, last_train_a, last_val_a)
                merge_progress(
                    task_id,
                    {
                        "currentEpoch": global_epoch,
                        "status": "running",
                        "augmentedProgress": (epoch / max(epochs, 1)) * 100.0,
                        "message": f"增强阶段 epoch {epoch}/{epochs} | val_stress_mae={augmented_val_stress_mae:.4f}",
                    },
                )

                if augmented_val_stress_mae < best_augmented_mae - early_stop_min_delta:
                    best_augmented_mae = augmented_val_stress_mae
                    best_augmented_epoch = epoch
                    best_augmented_loss = last_val_a
                    augmented_no_improve = 0
                    best_augmented_state = {k: v.detach().cpu() for k, v in model_a.state_dict().items()}
                    torch.save(
                        _make_checkpoint_payload(
                            epoch=epoch,
                            best_epoch=best_augmented_epoch,
                            best_metric=best_augmented_mae,
                            angle_vocab=angle_vocab_a,
                            preprocess=preprocess,
                            model_variant=model_variant,
                            base_model=model_a,
                            num_frames=num_frames,
                            frame_stride=frame_stride,
                        ),
                        augmented_path,
                    )
                else:
                    augmented_no_improve += 1
                    if augmented_no_improve >= early_stop_patience:
                        augmented_early_stopped = True
                        augmented_stopped_epoch = epoch
                        merge_progress(task_id, {"message": f"增强阶段早停于 epoch {epoch}"})
                        break

                _persist_progress_snapshot(task_id, artifacts_dir)

            if best_augmented_state is None:
                best_augmented_state = {k: v.detach().cpu() for k, v in model_a.state_dict().items()}
                best_augmented_mae = augmented_val_stress_mae
                best_augmented_epoch = augmented_completed_epochs
                best_augmented_loss = last_val_a
                torch.save(
                    _make_checkpoint_payload(
                        epoch=augmented_completed_epochs,
                        best_epoch=best_augmented_epoch,
                        best_metric=best_augmented_mae,
                        angle_vocab=angle_vocab_a,
                        preprocess=preprocess,
                        model_variant=model_variant,
                        base_model=model_a,
                        num_frames=num_frames,
                        frame_stride=frame_stride,
                    ),
                    augmented_path,
                )

            model_a.load_state_dict(best_augmented_state)
            augmented_test_loss, augmented_test_mae = _evaluate_regression(
                model_a,
                test_loader_a,
                device,
                reg_criterion,
                aux_criterion,
                angle_vocab_a,
                aux_cls_weight,
            )
            mae_a = augmented_test_mae
            db.update_task_checkpoints(task_id, user_id, augmented_ckpt=str(augmented_path))

            del model_a, opt_a, train_loader_a
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        result: dict[str, Any] = {
            "baseline": {
                "finalTrainLoss": last_train_b,
                "finalValLoss": last_val_b,
                "steeringError": mae_b,
                "requestedEpochs": epochs,
                "completedEpochs": baseline_completed_epochs,
                "bestEpoch": best_baseline_epoch,
                "stoppedEpoch": baseline_stopped_epoch,
                "earlyStopped": baseline_early_stopped,
                "bestValLoss": best_baseline_loss,
                "valStressMAE": best_baseline_mae,
                "finalTestLoss": baseline_test_loss,
                "usedDedicatedTestSplit": has_test_split,
                "modelVariant": model_variant,
                "numFrames": num_frames,
                "frameStride": frame_stride,
            }
        }
        if domain_augmentation:
            result["augmented"] = {
                "finalTrainLoss": last_train_a,
                "finalValLoss": last_val_a,
                "steeringError": mae_a,
                "requestedEpochs": epochs,
                "completedEpochs": augmented_completed_epochs,
                "bestEpoch": best_augmented_epoch,
                "stoppedEpoch": augmented_stopped_epoch,
                "earlyStopped": augmented_early_stopped,
                "bestValLoss": best_augmented_loss,
                "valStressMAE": best_augmented_mae,
                "finalTestLoss": augmented_test_loss,
                "usedDedicatedTestSplit": has_test_split,
                "modelVariant": model_variant,
                "numFrames": num_frames,
                "frameStride": frame_stride,
            }

        db.update_task_checkpoints(
            task_id,
            user_id,
            result_json=json.dumps(result, ensure_ascii=False),
            status="completed",
        )
        write_training_summary(
            artifacts_dir,
            {
                "taskId": task_id,
                "modelVariant": model_variant,
                "numFrames": num_frames,
                "frameStride": frame_stride,
                "domainAugmentation": domain_augmentation,
                "baselineCkpt": str(baseline_path),
                "augmentedCkpt": (str(augmented_path) if domain_augmentation else None),
                "result": result,
                "params": task_params,
            },
        )
        merge_progress(
            task_id,
            {
                "status": "completed",
                "currentEpoch": total_ui_epochs,
                "baselineProgress": 100.0,
                "domainAugmentationProgress": (100.0 if domain_augmentation else None),
                "augmentedProgress": (100.0 if domain_augmentation else None),
                "message": "训练完成",
            },
        )
        _persist_progress_snapshot(task_id, artifacts_dir)
    except Exception:
        err = traceback.format_exc()
        db.update_task_status(task_id, user_id, "failed", err[:2000])
        merge_progress(task_id, {"status": "failed", "message": err[:500]})
        failure_summary = read_training_summary(artifacts_dir) or {}
        failure_summary.update(
            {
                "taskId": task_id,
                "modelVariant": model_variant,
                "numFrames": num_frames,
                "frameStride": frame_stride,
                "status": "failed",
                "error": err[:4000],
            }
        )
        write_training_summary(artifacts_dir, failure_summary)
        _persist_progress_snapshot(task_id, artifacts_dir)
    finally:
        release_controls(task_id)
