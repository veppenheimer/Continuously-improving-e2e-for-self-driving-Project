#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Net_improve training aligned with the shared full-image HSV preprocessing contract."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from models import AutoDriveNetImprove, count_trainable_params
from steering_config import NUM_CLASSES, angle_to_class, class_to_angle

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from steering_augmentations import AugConfig, build_eval_transforms, build_stress_transforms, build_train_transforms
from steering_preprocess import NumpyRandomState, PreprocessConfig, inverse_frequency_weights

BASE_NET_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "Net_class"))
if BASE_NET_DIR not in sys.path and os.path.isdir(BASE_NET_DIR):
    sys.path.append(BASE_NET_DIR)
from utils import AverageMeter  # noqa: E402


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float_tuple(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    value = os.getenv(name)
    if not value:
        return default
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if len(parts) != len(default):
        return default
    return tuple(float(item) for item in parts)


def _imread_bgr(path: str):
    buf = np.fromfile(os.fspath(path), dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _is_albumentations_transform(transform: Any) -> bool:
    return hasattr(transform, "__call__") and transform.__class__.__module__.startswith("albumentations")


class SampleListDataset(Dataset):
    def __init__(self, samples, transform=None, deterministic_seed: int | None = None):
        self.samples = samples
        self.transform = transform
        self.deterministic_seed = deterministic_seed
        self.class_indices = [label for _, label, _ in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, raw_angle = self.samples[idx]
        img = _imread_bgr(img_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {img_path}")
        if self.transform is None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img = torch.from_numpy(np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))).float()
        elif _is_albumentations_transform(self.transform):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.deterministic_seed is not None:
                with NumpyRandomState(self.deterministic_seed + idx):
                    img = self.transform(image=img_rgb)["image"]
            else:
                img = self.transform(image=img_rgb)["image"]
        else:
            raise TypeError("Net_improve now expects Albumentations transforms or None")
        return img, torch.tensor(label, dtype=torch.long), torch.tensor(raw_angle, dtype=torch.float32)


def load_samples_from_list(list_path: str):
    samples = []
    if not os.path.exists(list_path):
        return samples
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                img_path, angle_str = line.rsplit(" ", 1)
            except ValueError:
                continue
            angle = float(angle_str)
            cls = angle_to_class(angle)
            samples.append((img_path, cls, angle))
    return samples


def evaluate(model, data_loader, criterion, device, metric_prefix="Val"):
    model.eval()
    loss_meter = AverageMeter(f"{metric_prefix}Loss")
    acc_meter = AverageMeter(f"{metric_prefix}Acc")
    mae_meter = AverageMeter(f"{metric_prefix}AngleMAE")
    with torch.no_grad():
        for imgs, labels, raw_angles in data_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            raw_angles = raw_angles.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            pred = logits.argmax(dim=1)
            acc = (pred == labels).float().mean().item()
            pred_angles = torch.tensor([class_to_angle(int(idx)) for idx in pred.detach().cpu().tolist()], dtype=torch.float32, device=device)
            mae = (pred_angles - raw_angles).abs().mean().item()
            loss_meter.update(loss.item(), imgs.size(0))
            acc_meter.update(acc, imgs.size(0))
            mae_meter.update(mae, imgs.size(0))
    return loss_meter.avg, acc_meter.avg, mae_meter.avg


def _write_summary(summary_path: Path, summary: dict):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _configure_optimizer(model: AutoDriveNetImprove, *, lr: float, weight_decay: float, backbone_lr_factor: float, freeze_backbone: bool):
    model.set_backbone_trainable(not freeze_backbone)
    if freeze_backbone:
        optimizer = torch.optim.AdamW(model.head_parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": list(model.head_parameters()), "lr": lr},
                {"params": list(model.backbone_parameters()), "lr": lr * backbone_lr_factor},
            ],
            weight_decay=weight_decay,
        )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(os.getenv("VENET_LR_FACTOR", "0.5")),
        patience=int(os.getenv("VENET_LR_PATIENCE", "4")),
        min_lr=1e-6,
    )
    return optimizer, scheduler


def _build_aug_config() -> AugConfig:
    preprocess = PreprocessConfig(
        color_space=os.getenv("VENET_PREPROCESS_COLOR_SPACE", "hsv").strip().lower(),
        input_size=(
            int(os.getenv("VENET_INPUT_HEIGHT", "120")),
            int(os.getenv("VENET_INPUT_WIDTH", "160")),
        ),
        use_roi=False,
    )
    return AugConfig(
        preprocess=preprocess,
        style_mix_ratio=_env_float_tuple("VENET_STYLE_MIX_RATIO", (0.5, 0.3, 0.2)),
    )


def main():
    data_folder = os.getenv("VENET_DATA_FOLDER", "data/simulate/data")
    batch_size = int(os.getenv("VENET_BATCH_SIZE", "16"))
    start_epoch = 1
    epochs = int(os.getenv("VENET_EPOCHS", "80"))
    lr = float(os.getenv("VENET_LR", "1e-4"))
    weight_decay = float(os.getenv("VENET_WEIGHT_DECAY", "1e-4"))
    early_stop_patience = int(os.getenv("VENET_EARLY_STOP_PATIENCE", "12"))
    min_delta = float(os.getenv("VENET_EARLY_STOP_MIN_DELTA", "1e-4"))
    grad_clip = float(os.getenv("VENET_GRAD_CLIP", "3.0"))
    label_smoothing = float(os.getenv("VENET_LABEL_SMOOTHING", "0.05"))
    use_pretrained = _env_bool("VENET_USE_PRETRAINED", True)
    freeze_backbone_epochs = int(os.getenv("VENET_FREEZE_BACKBONE_EPOCHS", "5"))
    backbone_lr_factor = float(os.getenv("VENET_BACKBONE_LR_FACTOR", "0.1"))
    output_dir = Path(os.getenv("VENET_OUTPUT_DIR", "."))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / os.getenv("VENET_SAVE_NAME", "ve2_improve.pth")
    best_save_path = output_dir / os.getenv("VENET_BEST_SAVE_NAME", "ve2_improve_best.pth")
    summary_path = output_dir / "training_summary.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")

    cudnn.benchmark = True
    writer = SummaryWriter(log_dir=os.getenv("VENET_LOG_DIR", "runs/net_improve"))

    aug_config = _build_aug_config()
    train_transform = build_train_transforms(aug_config)
    eval_transform = build_eval_transforms(aug_config)
    stress_transform = build_stress_transforms(aug_config)

    train_samples = load_samples_from_list(os.path.join(data_folder, "train.txt"))
    val_samples = load_samples_from_list(os.path.join(data_folder, "val.txt"))
    test_samples = load_samples_from_list(os.path.join(data_folder, "test.txt"))
    used_dedicated_test_split = bool(test_samples)
    if not train_samples or not val_samples:
        raise RuntimeError("缺少 train.txt 或 val.txt，无法训练轻量模型。")
    if not test_samples:
        test_samples = list(val_samples)
        print("未检测到 test.txt，测试集暂时复用 val.txt。")

    train_dataset = SampleListDataset(train_samples, transform=train_transform)
    val_dataset = SampleListDataset(val_samples, transform=eval_transform)
    val_stress_dataset = SampleListDataset(val_samples, transform=stress_transform, deterministic_seed=20260418)
    test_dataset = SampleListDataset(test_samples, transform=eval_transform)

    class_weights = inverse_frequency_weights(train_dataset.class_indices, NUM_CLASSES).to(device)
    model = AutoDriveNetImprove(use_pretrained=use_pretrained).to(device)
    print("Model trainable params:", count_trainable_params(model))
    print(
        f"ModelConfig use_pretrained={int(use_pretrained)} pretrained_loaded={int(getattr(model, 'pretrained_loaded', False))} "
        f"freeze_backbone_epochs={freeze_backbone_epochs}"
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing).to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    val_stress_loader = DataLoader(val_stress_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    best_val_stress_mae = float("inf")
    best_epoch = 0
    no_improve_count = 0
    completed_epochs = start_epoch - 1
    stopped_epoch = None
    early_stopped = False
    last_train_loss = 0.0
    last_train_acc = 0.0
    last_train_mae = 0.0
    last_val_loss = 0.0
    last_val_acc = 0.0
    last_val_mae = 0.0
    last_val_stress_mae = 0.0

    current_stage = None
    optimizer = None
    scheduler = None

    for epoch in range(start_epoch, epochs + 1):
        desired_stage = "head" if epoch <= freeze_backbone_epochs else "full"
        if desired_stage != current_stage:
            optimizer, scheduler = _configure_optimizer(
                model,
                lr=lr,
                weight_decay=weight_decay,
                backbone_lr_factor=backbone_lr_factor,
                freeze_backbone=(desired_stage == "head"),
            )
            current_stage = desired_stage
            print(f"TrainingStage epoch={epoch} stage={current_stage} backbone_trainable={int(current_stage == 'full')}")

        model.train()
        train_loss_meter = AverageMeter("TrainLoss")
        train_acc_meter = AverageMeter("TrainAcc")
        train_mae_meter = AverageMeter("TrainAngleMAE")

        for imgs, labels, raw_angles in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            raw_angles = raw_angles.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                acc = (pred == labels).float().mean().item()
                pred_angles = torch.tensor([class_to_angle(int(idx)) for idx in pred.detach().cpu().tolist()], dtype=torch.float32, device=device)
                mae = (pred_angles - raw_angles).abs().mean().item()
            train_loss_meter.update(loss.item(), imgs.size(0))
            train_acc_meter.update(acc, imgs.size(0))
            train_mae_meter.update(mae, imgs.size(0))

        val_loss, val_acc, val_mae = evaluate(model, val_loader, criterion, device, metric_prefix="Val")
        _, _, val_stress_mae = evaluate(model, val_stress_loader, criterion, device, metric_prefix="ValStress")
        scheduler.step(val_stress_mae)
        completed_epochs = epoch

        last_train_loss = train_loss_meter.avg
        last_train_acc = train_acc_meter.avg
        last_train_mae = train_mae_meter.avg
        last_val_loss = val_loss
        last_val_acc = val_acc
        last_val_mae = val_mae
        last_val_stress_mae = val_stress_mae
        current_lr = max(group["lr"] for group in optimizer.param_groups)

        writer.add_scalar("Loss/train", last_train_loss, epoch)
        writer.add_scalar("Acc/train", last_train_acc, epoch)
        writer.add_scalar("AngleMAE/train", last_train_mae, epoch)
        writer.add_scalar("Loss/val", last_val_loss, epoch)
        writer.add_scalar("Acc/val", last_val_acc, epoch)
        writer.add_scalar("AngleMAE/val", last_val_mae, epoch)
        writer.add_scalar("Val_Stress_Angle_MAE", last_val_stress_mae, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        print(
            f"Epoch {epoch:03d} | TrainLoss {last_train_loss:.4f} | TrainAcc {last_train_acc:.4f} | TrainAngleMAE {last_train_mae:.4f} | "
            f"ValLoss {last_val_loss:.4f} | ValAcc {last_val_acc:.4f} | ValAngleMAE {last_val_mae:.4f} | "
            f"ValStressAngleMAE {last_val_stress_mae:.4f} | LR {current_lr:.6f}"
        )

        model_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save({"epoch": epoch, "model": model_state_dict, "bestEpoch": best_epoch}, save_path)

        if val_stress_mae < best_val_stress_mae - min_delta:
            best_val_stress_mae = val_stress_mae
            best_epoch = epoch
            no_improve_count = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model": model_state_dict,
                    "bestEpoch": best_epoch,
                    "bestValStressAngleMAE": best_val_stress_mae,
                },
                best_save_path,
            )
            print(f"Validation stress MAE improved, saved best model to {best_save_path}")
        else:
            no_improve_count += 1
            if no_improve_count >= early_stop_patience:
                early_stopped = True
                stopped_epoch = epoch
                print(
                    f"EarlyStopping triggered at epoch {epoch} | best_epoch {best_epoch} | best_val_stress_angle_mae {best_val_stress_mae:.6f}"
                )
                break

    if best_epoch == 0:
        best_epoch = completed_epochs
        best_val_stress_mae = last_val_stress_mae
        if save_path.is_file() and not best_save_path.is_file():
            torch.save(torch.load(save_path, map_location="cpu"), best_save_path)

    best_checkpoint = torch.load(best_save_path, map_location=device)
    model.load_state_dict(best_checkpoint["model"])
    test_loss, test_acc, test_mae = evaluate(model, test_loader, criterion, device, metric_prefix="Test")
    print(f"Best model | TestLoss {test_loss:.4f} | TestAcc {test_acc:.4f} | TestAngleMAE {test_mae:.4f}")
    writer.add_scalar("Loss/test_best", test_loss, 0)
    writer.add_scalar("Acc/test_best", test_acc, 0)
    writer.add_scalar("AngleMAE/test_best", test_mae, 0)

    summary = {
        "requestedEpochs": epochs,
        "completedEpochs": completed_epochs,
        "bestEpoch": best_epoch,
        "stoppedEpoch": stopped_epoch,
        "earlyStopped": early_stopped,
        "modelSelectionMetric": "val_stress_mae",
        "usePretrained": use_pretrained,
        "pretrainedLoaded": bool(getattr(model, "pretrained_loaded", False)),
        "freezeBackboneEpochs": freeze_backbone_epochs,
        "preprocess": {
            "colorSpace": aug_config.preprocess.color_space,
            "inputSize": list(aug_config.preprocess.input_size),
            "useRoi": aug_config.preprocess.use_roi,
        },
        "finalTrainLoss": float(last_train_loss),
        "finalValLoss": float(last_val_loss),
        "steeringError": float(best_val_stress_mae),
        "finalTrainAcc": float(last_train_acc),
        "finalValAcc": float(last_val_acc),
        "finalTrainAngleMAE": float(last_train_mae),
        "finalValAngleMAE": float(last_val_mae),
        "finalValStressAngleMAE": float(last_val_stress_mae),
        "bestValLoss": float(last_val_loss),
        "testBestLoss": float(test_loss),
        "testBestAcc": float(test_acc),
        "testBestAngleMAE": float(test_mae),
        "usedDedicatedTestSplit": used_dedicated_test_split,
    }
    _write_summary(summary_path, summary)
    print(
        f"TrainingSummary requested_epochs={epochs} completed_epochs={completed_epochs} "
        f"best_epoch={best_epoch} early_stopped={int(early_stopped)} stopped_epoch={stopped_epoch or 0} "
        f"best_val_stress_angle_mae={best_val_stress_mae:.6f}"
    )
    writer.close()


if __name__ == "__main__":
    main()
