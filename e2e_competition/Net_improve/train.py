#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Net_improve 训练脚本：更轻量模型 + 更强抗过拟合策略。"""

import os
import sys
import random

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from models import AutoDriveNetImprove, count_trainable_params
from steering_config import angle_to_class

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_NET_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "Net"))
BASE_NET_CLASS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "Net_class"))
if BASE_NET_DIR in sys.path:
    pass
elif os.path.isdir(BASE_NET_DIR):
    sys.path.append(BASE_NET_DIR)
elif os.path.isdir(BASE_NET_CLASS_DIR):
    sys.path.append(BASE_NET_CLASS_DIR)

from utils import AverageMeter  # noqa: E402


class EMA:
    """Exponential Moving Average for model parameters."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            assert name in self.shadow
            new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
            self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.backup[name] = param.data.clone()
            param.data = self.shadow[name].clone()

    def restore(self, model):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            param.data = self.backup[name].clone()
        self.backup = {}


class SampleListDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import cv2
        import numpy as np
        from PIL import Image

        img_path, label = self.samples[idx]
        p = os.fspath(img_path)
        buf = np.fromfile(p, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size > 0 else None
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


def load_samples_from_txt(data_folder):
    txt_paths = [os.path.join(data_folder, "train.txt"), os.path.join(data_folder, "val.txt")]
    all_samples = []
    for txt_path in txt_paths:
        if not os.path.exists(txt_path):
            continue
        with open(txt_path, "r", encoding="utf-8") as f:
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
                all_samples.append((img_path, cls))
    return all_samples


def split_samples_three_way(samples, train_ratio, val_ratio, seed):
    assert 0 < train_ratio < 1
    assert 0 < val_ratio < 1
    assert train_ratio + val_ratio < 1
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    shuffled = [samples[i] for i in indices]
    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    train_samples = shuffled[:n_train]
    val_samples = shuffled[n_train:n_train + n_val]
    test_samples = shuffled[n_train + n_val:]
    return train_samples, val_samples, test_samples


def build_weighted_sampler(samples):
    class_count = {}
    for _, cls in samples:
        class_count[cls] = class_count.get(cls, 0) + 1
    weights = [1.0 / class_count[cls] for _, cls in samples]
    weights_tensor = torch.DoubleTensor(weights)
    return WeightedRandomSampler(weights=weights_tensor, num_samples=len(weights), replacement=True)


def evaluate(model, data_loader, criterion, device, metric_prefix="Val"):
    model.eval()
    loss_meter = AverageMeter(f"{metric_prefix}Loss")
    acc_meter = AverageMeter(f"{metric_prefix}Acc")
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            pred = logits.argmax(dim=1)
            acc = (pred == labels).float().mean().item()
            loss_meter.update(loss.item(), imgs.size(0))
            acc_meter.update(acc, imgs.size(0))
    return loss_meter.avg, acc_meter.avg


def main():
    data_folder = os.getenv("VENET_DATA_FOLDER", "data/simulate/data")
    batch_size = int(os.getenv("VENET_BATCH_SIZE", "32"))
    start_epoch = 1
    epochs = int(os.getenv("VENET_EPOCHS", "80"))
    lr = float(os.getenv("VENET_LR", "1e-3"))
    weight_decay = 1e-4
    ema_decay = 0.999
    early_stop_patience = 12
    min_delta = 1e-4
    train_ratio = 0.70
    val_ratio = 0.15
    split_seed = 3407
    output_dir = os.getenv("VENET_OUTPUT_DIR", ".")
    os.makedirs(output_dir, exist_ok=True)
    save_name = os.path.join(output_dir, os.getenv("VENET_SAVE_NAME", "ve2_improve.pth"))
    best_save_name = os.path.join(output_dir, os.getenv("VENET_BEST_SAVE_NAME", "ve2_improve_best.pth"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("使用 GPU:", torch.cuda.get_device_name(0))
    else:
        print("当前使用 CPU")

    cudnn.benchmark = True
    writer = SummaryWriter(log_dir=os.getenv("VENET_LOG_DIR", "runs/net_improve"))

    model = AutoDriveNetImprove().to(device)
    print("Model trainable params:", count_trainable_params(model))
    ema = EMA(model, decay=ema_decay)
    print(f"EMA enabled with decay={ema_decay}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize((120, 160)),
            transforms.ColorJitter(brightness=0.20, contrast=0.15, saturation=0.15, hue=0.03),
            transforms.RandomAffine(degrees=4, translate=(0.03, 0.03), scale=(0.95, 1.05)),
            transforms.ToTensor(),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((120, 160)),
            transforms.ToTensor(),
        ]
    )

    all_samples = load_samples_from_txt(data_folder)
    if len(all_samples) < 8:
        raise RuntimeError("样本数量过少（<8），无法训练轻量模型，请先扩充数据。")

    # 小数据集时降级为 train/val 二划分，避免直接失败。
    if len(all_samples) < 30:
        rng = random.Random(split_seed)
        shuffled = list(all_samples)
        rng.shuffle(shuffled)
        n_train = max(1, int(len(shuffled) * 0.8))
        n_train = min(n_train, len(shuffled) - 1)
        train_samples = shuffled[:n_train]
        val_samples = shuffled[n_train:]
        test_samples = list(val_samples)
        print(
            f"样本较少，采用二划分: total={len(all_samples)}, "
            f"train={len(train_samples)}, val={len(val_samples)}, test(复用val)={len(test_samples)}"
        )
    else:
        train_samples, val_samples, test_samples = split_samples_three_way(
            all_samples, train_ratio=train_ratio, val_ratio=val_ratio, seed=split_seed
        )
    print(
        f"三划分完成: total={len(all_samples)}, "
        f"train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}"
    )

    train_dataset = SampleListDataset(train_samples, transform=train_transform)
    val_dataset = SampleListDataset(val_samples, transform=val_transform)
    test_dataset = SampleListDataset(test_samples, transform=val_transform)

    train_sampler = build_weighted_sampler(train_samples)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    best_val_loss = float("inf")
    no_improve_count = 0

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss_meter = AverageMeter("TrainLoss")
        train_acc_meter = AverageMeter("TrainAcc")

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            ema.update(model)

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                acc = (pred == labels).float().mean().item()
            train_loss_meter.update(loss.item(), imgs.size(0))
            train_acc_meter.update(acc, imgs.size(0))

        ema.apply_shadow(model)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, metric_prefix="Val")
        ema.restore(model)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Loss/train", train_loss_meter.avg, epoch)
        writer.add_scalar("Acc/train", train_acc_meter.avg, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        print(
            f"Epoch {epoch:03d} | "
            f"TrainLoss {train_loss_meter.avg:.4f} | TrainAcc {train_acc_meter.avg:.4f} | "
            f"ValLoss {val_loss:.4f} | ValAcc {val_acc:.4f} | LR {current_lr:.6f}"
        )

        ema.apply_shadow(model)
        model_state_dict = model.state_dict()
        ema.restore(model)
        torch.save({"epoch": epoch, "model": model_state_dict, "optimizer": optimizer.state_dict()}, save_name)

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            no_improve_count = 0
            torch.save(
                {"epoch": epoch, "model": model_state_dict, "optimizer": optimizer.state_dict()},
                best_save_name,
            )
            print(f"验证集损失下降，保存最优模型到 {best_save_name}")
        else:
            no_improve_count += 1
            if no_improve_count >= early_stop_patience:
                print("触发早停：验证集损失持续未改善。")
                break

    best_checkpoint = torch.load(best_save_name, map_location=device)
    model.load_state_dict(best_checkpoint["model"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, metric_prefix="Test")
    print(f"最佳模型 TestLoss {test_loss:.4f} | TestAcc {test_acc:.4f}")
    writer.add_scalar("Loss/test_best", test_loss, 0)
    writer.add_scalar("Acc/test_best", test_acc, 0)

    writer.close()


if __name__ == "__main__":
    main()
