#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from datasets import AutoDriveDataset
from models import AutoDriveNet
from steering_config import MAX_DELTA, NUM_CLASSES, decode_output
from utils import AverageMeter

import sys
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from steering_augmentations import AugConfig, build_eval_transforms, build_stress_transforms, build_train_transforms
from steering_preprocess import PreprocessConfig, inverse_frequency_weights


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


def _configure_optimizer(model: AutoDriveNet, *, lr: float, weight_decay: float, backbone_lr_factor: float, freeze_backbone: bool):
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
        factor=0.5,
        patience=max(2, int(os.getenv("VENET_LR_PATIENCE", "4"))),
        min_lr=1e-6,
    )
    return optimizer, scheduler


def _run_epoch(*, model, loader, device, cls_criterion, reg_criterion, reg_loss_weight, optimizer=None, grad_clip=0.0):
    is_train = optimizer is not None
    model.train(is_train)

    ce_meter = AverageMeter("CE")
    total_meter = AverageMeter("Total")
    acc_meter = AverageMeter("Acc")
    mae_meter = AverageMeter("AngleMAE")

    for imgs, class_idx, raw_angle, delta_target in loader:
        imgs = imgs.to(device)
        class_idx = class_idx.to(device)
        raw_angle = raw_angle.to(device)
        delta_target = delta_target.to(device)

        with torch.set_grad_enabled(is_train):
            output = model(imgs)
            logits = output[:, :NUM_CLASSES]
            pred_delta = torch.tanh(output[:, -1]) * MAX_DELTA

            cls_loss = cls_criterion(logits, class_idx)
            reg_loss = reg_criterion(pred_delta, delta_target)
            total_loss = cls_loss + reg_loss_weight * reg_loss

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

        with torch.no_grad():
            pred_class = torch.argmax(logits, dim=1)
            pred_angle = decode_output(output)
            acc = (pred_class == class_idx).float().mean().item()
            mae = (pred_angle - raw_angle).abs().mean().item()

        batch_size = imgs.size(0)
        ce_meter.update(cls_loss.item(), batch_size)
        total_meter.update(total_loss.item(), batch_size)
        acc_meter.update(acc, batch_size)
        mae_meter.update(mae, batch_size)

    return {
        "ce_loss": ce_meter.avg,
        "total_loss": total_meter.avg,
        "acc": acc_meter.avg,
        "mae": mae_meter.avg,
    }


def _write_summary(summary_path: Path, summary: dict):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


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
    checkpoint = os.getenv("VENET_CHECKPOINT", "") or None
    batch_size = int(os.getenv("VENET_BATCH_SIZE", "16"))
    start_epoch = 1
    epochs = int(os.getenv("VENET_EPOCHS", "100"))
    lr = float(os.getenv("VENET_LR", "1e-4"))
    reg_loss_weight = float(os.getenv("VENET_REG_LOSS_WEIGHT", "2.0"))
    weight_decay = float(os.getenv("VENET_WEIGHT_DECAY", "1e-4"))
    grad_clip = float(os.getenv("VENET_GRAD_CLIP", "3.0"))
    early_stop_patience = int(os.getenv("VENET_EARLY_STOP_PATIENCE", "10"))
    early_stop_min_delta = float(os.getenv("VENET_EARLY_STOP_MIN_DELTA", "1e-4"))
    enable_train_aug = not _env_bool("VENET_DISABLE_TRAIN_AUG", False)
    use_pretrained = _env_bool("VENET_USE_PRETRAINED", True)
    freeze_backbone_epochs = int(os.getenv("VENET_FREEZE_BACKBONE_EPOCHS", "5"))
    backbone_lr_factor = float(os.getenv("VENET_BACKBONE_LR_FACTOR", "0.1"))
    log_dir = os.getenv("VENET_LOG_DIR", "")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")

    cudnn.benchmark = True
    writer = SummaryWriter(log_dir=log_dir if log_dir else None)

    aug_config = _build_aug_config()
    train_transform = build_train_transforms(aug_config) if enable_train_aug else build_eval_transforms(aug_config)
    eval_transform = build_eval_transforms(aug_config)
    stress_transform = build_stress_transforms(aug_config)

    train_dataset = AutoDriveDataset(data_folder, mode="train", transform=train_transform)
    val_dataset = AutoDriveDataset(data_folder, mode="val", transform=eval_transform)
    val_stress_dataset = AutoDriveDataset(data_folder, mode="val", transform=stress_transform, deterministic_seed=20260418)
    test_file = Path(data_folder) / "test.txt"
    has_test_split = test_file.is_file()
    test_dataset = AutoDriveDataset(data_folder, mode="test", transform=eval_transform) if has_test_split else val_dataset

    class_weights = inverse_frequency_weights(train_dataset.class_indices, NUM_CLASSES).to(device)
    model = AutoDriveNet(use_pretrained=use_pretrained).to(device)
    print(
        f"ModelConfig use_pretrained={int(use_pretrained)} pretrained_loaded={int(getattr(model, 'pretrained_loaded', False))} "
        f"freeze_backbone_epochs={freeze_backbone_epochs}"
    )
    print(f"ClassWeights {class_weights.detach().cpu().tolist()}")

    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location=device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        model.load_state_dict(ckpt["model"])

    cls_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05).to(device)
    reg_criterion = nn.SmoothL1Loss().to(device)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    val_stress_loader = torch.utils.data.DataLoader(val_stress_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    output_dir = Path(os.getenv("VENET_OUTPUT_DIR", "."))
    save_name = os.getenv("VENET_SAVE_NAME", "ve2.pth")
    best_save_name = os.getenv("VENET_BEST_SAVE_NAME", f"best_{save_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / save_name
    best_path = output_dir / best_save_name
    summary_path = output_dir / "training_summary.json"

    best_val_stress_mae = float("inf")
    best_epoch = 0
    no_improve_count = 0
    completed_epochs = start_epoch - 1
    stopped_epoch = None
    early_stopped = False
    last_train_metrics = {"ce_loss": 0.0, "total_loss": 0.0, "acc": 0.0, "mae": 0.0}
    last_val_metrics = {"ce_loss": 0.0, "total_loss": 0.0, "acc": 0.0, "mae": 0.0}
    last_val_stress_metrics = {"ce_loss": 0.0, "total_loss": 0.0, "acc": 0.0, "mae": 0.0}

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

        last_train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            cls_criterion=cls_criterion,
            reg_criterion=reg_criterion,
            reg_loss_weight=reg_loss_weight,
            optimizer=optimizer,
            grad_clip=grad_clip,
        )
        last_val_metrics = _run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            cls_criterion=cls_criterion,
            reg_criterion=reg_criterion,
            reg_loss_weight=reg_loss_weight,
            optimizer=None,
        )
        last_val_stress_metrics = _run_epoch(
            model=model,
            loader=val_stress_loader,
            device=device,
            cls_criterion=cls_criterion,
            reg_criterion=reg_criterion,
            reg_loss_weight=reg_loss_weight,
            optimizer=None,
        )
        completed_epochs = epoch
        scheduler.step(last_val_stress_metrics["mae"])
        current_lr = max(group["lr"] for group in optimizer.param_groups)

        writer.add_scalar("CE_Loss", last_train_metrics["ce_loss"], epoch)
        writer.add_scalar("Val_CE_Loss", last_val_metrics["ce_loss"], epoch)
        writer.add_scalar("Train_Total_Loss", last_train_metrics["total_loss"], epoch)
        writer.add_scalar("Val_Total_Loss", last_val_metrics["total_loss"], epoch)
        writer.add_scalar("Train_Acc", last_train_metrics["acc"], epoch)
        writer.add_scalar("Val_Acc", last_val_metrics["acc"], epoch)
        writer.add_scalar("Train_Angle_MAE", last_train_metrics["mae"], epoch)
        writer.add_scalar("Val_Angle_MAE", last_val_metrics["mae"], epoch)
        writer.add_scalar("Val_Stress_Angle_MAE", last_val_stress_metrics["mae"], epoch)
        writer.add_scalar("LR", current_lr, epoch)

        print(
            f"epoch:{epoch}"
            f"  CE_Loss:{last_train_metrics['ce_loss']:.6f}"
            f"  Train_Acc:{last_train_metrics['acc']:.6f}"
            f"  Val_CE_Loss:{last_val_metrics['ce_loss']:.6f}"
            f"  Val_Acc:{last_val_metrics['acc']:.6f}"
            f"  Train_Angle_MAE:{last_train_metrics['mae']:.6f}"
            f"  Val_Angle_MAE:{last_val_metrics['mae']:.6f}"
            f"  Val_Stress_Angle_MAE:{last_val_stress_metrics['mae']:.6f}"
            f"  LR:{current_lr:.6f}"
        )

        model_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(
            {
                "epoch": epoch,
                "model": model_state_dict,
                "bestEpoch": best_epoch,
                "bestValStressAngleMAE": best_val_stress_mae,
                "pretrainedLoaded": bool(getattr(model, "pretrained_loaded", False)),
            },
            latest_path,
        )

        if last_val_stress_metrics["mae"] < best_val_stress_mae - early_stop_min_delta:
            best_val_stress_mae = last_val_stress_metrics["mae"]
            best_epoch = epoch
            no_improve_count = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model": model_state_dict,
                    "bestEpoch": best_epoch,
                    "bestValStressAngleMAE": best_val_stress_mae,
                    "pretrainedLoaded": bool(getattr(model, "pretrained_loaded", False)),
                },
                best_path,
            )
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
        best_val_stress_mae = last_val_stress_metrics["mae"]
        if latest_path.is_file() and not best_path.is_file():
            torch.save(torch.load(latest_path, map_location="cpu"), best_path)

    best_checkpoint = torch.load(best_path if best_path.is_file() else latest_path, map_location=device)
    model.load_state_dict(best_checkpoint["model"])
    test_metrics = _run_epoch(
        model=model,
        loader=test_loader,
        device=device,
        cls_criterion=cls_criterion,
        reg_criterion=reg_criterion,
        reg_loss_weight=reg_loss_weight,
        optimizer=None,
    )
    writer.add_scalar("Test_Total_Loss", test_metrics["total_loss"], 0)
    writer.add_scalar("Test_Acc", test_metrics["acc"], 0)
    writer.add_scalar("Test_Angle_MAE", test_metrics["mae"], 0)

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
        "finalTrainLoss": float(last_train_metrics["total_loss"]),
        "finalValLoss": float(last_val_metrics["total_loss"]),
        "steeringError": float(best_val_stress_mae),
        "finalTrainAcc": float(last_train_metrics["acc"]),
        "finalValAcc": float(last_val_metrics["acc"]),
        "finalTrainAngleMAE": float(last_train_metrics["mae"]),
        "finalValAngleMAE": float(last_val_metrics["mae"]),
        "finalValStressAngleMAE": float(last_val_stress_metrics["mae"]),
        "finalTestLoss": float(test_metrics["total_loss"]),
        "finalTestAcc": float(test_metrics["acc"]),
        "finalTestAngleMAE": float(test_metrics["mae"]),
        "usedDedicatedTestSplit": has_test_split,
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
