"""后台线程中执行训练，与 SQLite / 内存进度同步。"""

from __future__ import annotations

import json
import importlib
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
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from app import database as db
from app.config import settings
from app.state import append_loss_point, get_controls, merge_progress, release_controls
from datasets import AutoDriveDataset, AutoDriveListDataset
from models import AutoDriveNet
from utils import AverageMeter


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    should_stop,
) -> float:
    model.train()
    meter = AverageMeter()
    for imgs, labels in loader:
        if should_stop():
            break
        imgs = imgs.to(device)
        labels = labels.to(device)
        pred = model(imgs)
        loss = criterion(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        meter.update(loss.item(), imgs.size(0))
    return meter.avg


def _validate_epoch(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> float:
    model.eval()
    meter = AverageMeter()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            pred = model(imgs)
            loss = criterion(pred, labels)
            meter.update(loss.item(), imgs.size(0))
    return meter.avg


def _steering_mae(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_abs = 0.0
    n = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            pred = model(imgs)
            total_abs += (pred - labels).abs().sum().item()
            n += imgs.size(0)
    return total_abs / max(n, 1)


def _wait_resume(ctrl) -> bool:
    """若返回 False 表示应终止。"""
    while ctrl.pause.is_set() and not ctrl.stop.is_set():
        time.sleep(0.2)
    return not ctrl.stop.is_set()


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
            f"命令执行失败（退出码 {code}）",
            f"CMD: {cmd}",
            f"STDERR:\n{err or '(empty)'}",
            f"STDOUT:\n{out or '(empty)'}",
        ]
        raise RuntimeError("\n\n".join(detail)[:6000])


def _ensure_cyclegan_runtime_deps() -> None:
    """CycleGAN 训练脚本依赖（visualizer 顶层导入）。"""
    missing: list[str] = []
    for mod in ("dominate",):
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(mod)
    if missing:
        deps = " ".join(missing)
        raise RuntimeError(
            f"缺少 CycleGAN 依赖: {deps}。请在 Net_Web 环境执行: "
            f"`pip install {deps}` 或 `pip install -r requirements.txt`。"
        )


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
        raise RuntimeError(f"CycleGAN 项目目录不存在: {project}")
    train_py = project / "train.py"
    test_py = project / "test.py"
    if not train_py.is_file() or not test_py.is_file():
        raise RuntimeError("未找到 CycleGAN train.py 或 test.py")
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

    train_pairs_a = _read_list_pairs(dataset_root_a / "train.txt")
    train_pairs_b = _read_list_pairs(dataset_root_b / "train.txt")
    if not train_pairs_a or not train_pairs_b:
        raise RuntimeError("A/B 数据集 train.txt 为空，无法执行域增强")

    _copy_pairs_as_images(train_pairs_a, train_a_dir, "a")
    _copy_pairs_as_images(train_pairs_b, train_b_dir, "b")
    infer_pairs = _copy_pairs_as_images(train_pairs_a, infer_a_dir, "infer")

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
        m = epoch_re.search(line)
        if not m:
            return
        cur = int(m.group(1))
        total = int(m.group(2)) if int(m.group(2)) > 0 else total_train_epochs
        ratio = max(0.0, min(1.0, cur / max(total, 1)))
        # Domain augmentation stage allocates 5%~95% for CycleGAN train.
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
        on_train_progress(97.0, "CycleGAN训练完成，正在生成 C 数据")
    _run_subprocess_or_raise(infer_args, project)
    if on_train_progress is not None:
        on_train_progress(100.0, "C 数据生成完成")

    image_dir = outputs_dir / job_id / "test_latest" / "images"
    if not image_dir.is_dir():
        raise RuntimeError("CycleGAN 推理输出目录不存在")

    c_list = artifacts_dir / "train_c.txt"
    pairs_meta: list[dict[str, Any]] = []
    with open(c_list, "w", encoding="utf-8") as f:
        for idx, (src_name, angle) in enumerate(infer_pairs):
            fake_name = f"{Path(src_name).stem}_fake.png"
            fake_path = image_dir / fake_name
            if not fake_path.is_file():
                raise RuntimeError(f"未找到生成图像: {fake_name}")
            dst = c_dir / fake_name
            shutil.copy2(fake_path, dst)
            f.write(f"{os.fspath(dst.resolve())} {angle}\n")
            pairs_meta.append(
                {
                    "index": idx,
                    "aName": src_name,
                    "cName": fake_name,
                    "aPath": os.fspath((infer_a_dir / src_name).resolve()),
                    "cPath": os.fspath(dst.resolve()),
                }
            )
    pairs_json = artifacts_dir / "domain_aug_pairs.json"
    with open(pairs_json, "w", encoding="utf-8") as pf:
        pf.write(json.dumps(pairs_meta, ensure_ascii=False))
    return c_list, pairs_json


def _run_competition_model_training(
    *,
    model_name: str,
    script_dir: Path,
    dataset_root: str,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    out_dir: Path,
    ckpt_name: str,
    branch_name: str,
    task_id: str,
    on_progress,
) -> dict[str, float | None]:
    train_py = script_dir / "train.py"
    if not train_py.is_file():
        raise RuntimeError(f"{model_name} 训练脚本不存在: {train_py}")
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "VENET_DATA_FOLDER": dataset_root,
            "VENET_BATCH_SIZE": str(batch_size),
            "VENET_EPOCHS": str(epochs),
            "VENET_LR": str(learning_rate),
            "VENET_OUTPUT_DIR": os.fspath(out_dir.resolve()),
            "VENET_SAVE_NAME": ckpt_name,
            "VENET_BEST_SAVE_NAME": f"best_{ckpt_name}",
            "VENET_LOG_DIR": os.fspath((out_dir / "runs").resolve()),
        }
    )
    r_classic = re.compile(r"epoch:\s*(\d+)")
    r_lite = re.compile(r"Epoch\s+(\d+)\s+\|")
    r_classic_loss = re.compile(r"CE_Loss:\s*([0-9.eE+-]+)")
    r_classic_acc = re.compile(r"Train_Acc:\s*([0-9.eE+-]+)")
    r_lite_train_loss = re.compile(r"TrainLoss\s+([0-9.eE+-]+)")
    r_lite_train_acc = re.compile(r"TrainAcc\s+([0-9.eE+-]+)")
    r_lite_val_loss = re.compile(r"ValLoss\s+([0-9.eE+-]+)")
    r_lite_val_acc = re.compile(r"ValAcc\s+([0-9.eE+-]+)")
    metrics: dict[str, float | None] = {
        "finalTrainLoss": None,
        "finalValLoss": None,
        "finalTrainAcc": None,
        "finalValAcc": None,
    }

    def _on_line(line: str) -> None:
        m = r_classic.search(line) or r_lite.search(line)
        if not m:
            return
        cur = int(m.group(1))
        ratio = max(0.0, min(1.0, cur / max(epochs, 1)))
        on_progress(ratio * 100.0, f"{model_name} epoch {cur}/{epochs}")
        t_loss = None
        v_loss = None
        t_acc = None
        v_acc = None
        m1 = r_classic_loss.search(line)
        if m1:
            t_loss = float(m1.group(1))
            v_loss = t_loss
        m2 = r_classic_acc.search(line)
        if m2:
            t_acc = float(m2.group(1))
            v_acc = t_acc
        m3 = r_lite_train_loss.search(line)
        if m3:
            t_loss = float(m3.group(1))
        m4 = r_lite_val_loss.search(line)
        if m4:
            v_loss = float(m4.group(1))
        m5 = r_lite_train_acc.search(line)
        if m5:
            t_acc = float(m5.group(1))
        m6 = r_lite_val_acc.search(line)
        if m6:
            v_acc = float(m6.group(1))
        if t_loss is not None:
            metrics["finalTrainLoss"] = t_loss
        if v_loss is not None:
            metrics["finalValLoss"] = v_loss
        if t_acc is not None:
            metrics["finalTrainAcc"] = t_acc
        if v_acc is not None:
            metrics["finalValAcc"] = v_acc
        if t_loss is not None and v_loss is not None:
            append_loss_point(task_id, branch_name, cur, float(t_loss), float(v_loss))

    proc = subprocess.Popen(
        [sys.executable, str(train_py)],
        cwd=str(script_dir),
        env=env,
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
        s = line.rstrip("\n")
        out_lines.append(s)
        _on_line(s)
    assert proc.stderr is not None
    for line in proc.stderr:
        err_lines.append(line.rstrip("\n"))
    code = proc.wait()
    if code != 0:
        raise RuntimeError(
            f"{model_name} 训练失败(exit={code})\nSTDERR:\n{chr(10).join(err_lines)[:3000]}"
            f"\nSTDOUT:\n{chr(10).join(out_lines)[:3000]}"
        )
    on_progress(100.0, f"{model_name} 训练完成")
    if metrics["finalTrainLoss"] is None:
        metrics["finalTrainLoss"] = 0.0
    if metrics["finalValLoss"] is None:
        metrics["finalValLoss"] = float(metrics["finalTrainLoss"] or 0.0)
    return metrics


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
    use_competition_class_model: bool,
    use_competition_lite_model: bool,
    artifacts_dir: Path,
) -> None:
    ctrl = get_controls(task_id)
    if ctrl is None:
        return

    # 以数据库中的任务快照为准，避免线程参数与持久化配置不一致。
    row0 = db.get_task(task_id, user_id)
    if row0 is not None and "task_params_json" in row0.keys() and row0["task_params_json"]:
        try:
            p0 = json.loads(row0["task_params_json"])
            use_competition_class_model = bool(p0.get("useCompetitionClassModel", use_competition_class_model))
            use_competition_lite_model = bool(p0.get("useCompetitionLiteModel", use_competition_lite_model))
        except Exception:
            pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_ui_epochs = epochs * (2 if domain_augmentation else 1)
    if use_competition_class_model:
        total_ui_epochs += epochs
    if use_competition_lite_model:
        total_ui_epochs += epochs
    patch: dict[str, Any] = {
        "status": "running",
        "totalEpochs": total_ui_epochs,
        "currentEpoch": 0,
        "message": f"设备: {device}",
        "baselineProgress": 0.0,
        "domainAugmentationProgress": (0.0 if domain_augmentation else None),
        "domainAugmentationText": ("等待开始" if domain_augmentation else None),
        "augmentedProgress": (0.0 if domain_augmentation else None),
        "competitionClassProgress": (0.0 if use_competition_class_model else None),
        "competitionClassText": ("等待开始" if use_competition_class_model else None),
        "competitionLiteProgress": (0.0 if use_competition_lite_model else None),
        "competitionLiteText": ("等待开始" if use_competition_lite_model else None),
    }
    if domain_augmentation:
        patch["augmented"] = {"trainLossSeries": [], "valLossSeries": []}
    else:
        patch["augmented"] = None
    merge_progress(task_id, patch)

    transform_baseline = transforms.Compose(
        [
            transforms.Resize((120, 160)),
            transforms.ToTensor(),
        ]
    )
    criterion = nn.MSELoss().to(device)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = artifacts_dir / "baseline.pth"
    augmented_path = artifacts_dir / "augmented.pth"

    last_train_b = 0.0
    last_val_b = 0.0
    last_train_a = 0.0
    last_val_a = 0.0
    mae_b = 0.0
    mae_a = 0.0
    comp_class_metrics: dict[str, float | None] | None = None
    comp_lite_metrics: dict[str, float | None] | None = None
    comp_class_note: str | None = None
    comp_lite_note: str | None = None

    try:
        db.update_task_status(task_id, user_id, "running", None)

        # ----- 基准模型 -----
        train_ds = AutoDriveDataset(dataset_root, "train", transform_baseline)
        val_ds = AutoDriveDataset(dataset_root, "val", transform_baseline)
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

        model = AutoDriveNet().to(device)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
        )

        for epoch in range(1, epochs + 1):
            if not _wait_resume(ctrl):
                db.update_task_status(task_id, user_id, "stopped", "用户终止")
                merge_progress(task_id, {"status": "stopped"})
                release_controls(task_id)
                return

            last_train_b = _train_epoch(
                model,
                train_loader,
                device,
                criterion,
                optimizer,
                lambda: ctrl.stop.is_set(),
            )
            if ctrl.stop.is_set():
                db.update_task_status(task_id, user_id, "stopped", "用户终止")
                merge_progress(task_id, {"status": "stopped"})
                release_controls(task_id)
                return

            last_val_b = _validate_epoch(model, val_loader, device, criterion)
            append_loss_point(task_id, "baseline", epoch, last_train_b, last_val_b)
            merge_progress(
                task_id,
                {
                    "currentEpoch": epoch,
                    "status": "running",
                    "baselineProgress": (epoch / max(epochs, 1)) * 100.0,
                },
            )

        mae_b = _steering_mae(model, val_loader, device)
        torch.save({"model": model.state_dict(), "epoch": epochs}, baseline_path)
        db.update_task_checkpoints(task_id, user_id, baseline_ckpt=str(baseline_path))

        del model, optimizer, train_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ----- 域增强模型（可选）-----
        if domain_augmentation:
            merge_progress(
                task_id,
                {
                    "augmented": {"trainLossSeries": [], "valLossSeries": []},
                    "message": "基准模型已完成，开始 CycleGAN 生成 C 数据集",
                },
            )
            if not dataset_b_root:
                raise RuntimeError("缺少 B 域数据集路径")
            merge_progress(task_id, {"domainAugmentationProgress": 5.0})
            c_list, _pairs_json = _build_c_dataset_with_cyclegan(
                dataset_root_a=Path(dataset_root),
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
                on_train_progress=lambda p, txt: merge_progress(
                    task_id, {"domainAugmentationProgress": p, "domainAugmentationText": txt}
                ),
            )
            merge_progress(
                task_id,
                {
                    "domainAugmentationProgress": 100.0,
                    "domainAugmentationText": "已完成",
                    "message": "C 数据集生成完成，开始增强模型训练",
                },
            )
            combined_train = artifacts_dir / "train_augmented.txt"
            with open(Path(dataset_root) / "train.txt", "r", encoding="utf-8") as fa, open(
                c_list, "r", encoding="utf-8"
            ) as fc, open(combined_train, "w", encoding="utf-8") as fout:
                a_content = fa.read()
                c_content = fc.read()
                fout.write(a_content)
                if a_content and not a_content.endswith("\n"):
                    fout.write("\n")
                fout.write(c_content)

            train_ds_a = AutoDriveListDataset(str(combined_train), transform_baseline)
            val_ds_a = AutoDriveDataset(dataset_root, "val", transform_baseline)
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

            model_a = AutoDriveNet().to(device)
            opt_a = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model_a.parameters()),
                lr=learning_rate,
            )

            for epoch in range(1, epochs + 1):
                if not _wait_resume(ctrl):
                    db.update_task_status(task_id, user_id, "stopped", "用户终止")
                    merge_progress(task_id, {"status": "stopped"})
                    release_controls(task_id)
                    return

                g_ep = epochs + epoch
                last_train_a = _train_epoch(
                    model_a,
                    train_loader_a,
                    device,
                    criterion,
                    opt_a,
                    lambda: ctrl.stop.is_set(),
                )
                if ctrl.stop.is_set():
                    db.update_task_status(task_id, user_id, "stopped", "用户终止")
                    merge_progress(task_id, {"status": "stopped"})
                    release_controls(task_id)
                    return

                last_val_a = _validate_epoch(model_a, val_loader_a, device, criterion)
                append_loss_point(task_id, "augmented", epoch, last_train_a, last_val_a)
                merge_progress(
                    task_id,
                    {
                        "currentEpoch": g_ep,
                        "status": "running",
                        "augmentedProgress": (epoch / max(epochs, 1)) * 100.0,
                    },
                )

            mae_a = _steering_mae(model_a, val_loader_a, device)
            torch.save({"model": model_a.state_dict(), "epoch": epochs}, augmented_path)
            db.update_task_checkpoints(task_id, user_id, augmented_ckpt=str(augmented_path))

        comp_root = settings.competition_project_root
        if use_competition_class_model:
            merge_progress(task_id, {"competitionClass": {"trainLossSeries": [], "valLossSeries": []}})
            merge_progress(task_id, {"competitionClassText": "开始训练分类模型"})
            try:
                comp_class_metrics = _run_competition_model_training(
                    model_name="分类模型",
                    script_dir=comp_root / "Net_class",
                    dataset_root=dataset_root,
                    batch_size=batch_size,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    out_dir=artifacts_dir / "competition_class",
                    ckpt_name="ve2_competition_class.pth",
                    branch_name="competitionClass",
                    task_id=task_id,
                    on_progress=lambda p, txt: merge_progress(
                        task_id,
                        {
                            "competitionClassProgress": p,
                            "competitionClassText": txt,
                            "currentEpoch": min(
                                total_ui_epochs,
                                int(round((epochs * (2 if domain_augmentation else 1)) + (p / 100.0) * epochs)),
                            ),
                        },
                    ),
                )
            except RuntimeError as e:
                msg = str(e)
                comp_class_note = f"失败已跳过: {msg[:160]}"
                merge_progress(
                    task_id,
                    {
                        "competitionClassProgress": 100.0,
                        "competitionClassText": f"失败已跳过：{msg[:80]}",
                    },
                )

        if use_competition_lite_model:
            merge_progress(task_id, {"competitionLite": {"trainLossSeries": [], "valLossSeries": []}})
            base_ep = epochs * (2 if domain_augmentation else 1) + (epochs if use_competition_class_model else 0)
            merge_progress(task_id, {"competitionLiteText": "开始训练轻量模型"})
            try:
                comp_lite_metrics = _run_competition_model_training(
                    model_name="轻量模型",
                    script_dir=comp_root / "Net_improve",
                    dataset_root=dataset_root,
                    batch_size=batch_size,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    out_dir=artifacts_dir / "competition_lite",
                    ckpt_name="ve2_competition_lite.pth",
                    branch_name="competitionLite",
                    task_id=task_id,
                    on_progress=lambda p, txt: merge_progress(
                        task_id,
                        {
                            "competitionLiteProgress": p,
                            "competitionLiteText": txt,
                            "currentEpoch": min(total_ui_epochs, int(round(base_ep + (p / 100.0) * epochs))),
                        },
                    ),
                )
            except RuntimeError as e:
                msg = str(e)
                comp_lite_note = f"失败已跳过: {msg[:160]}"
                merge_progress(
                    task_id,
                    {
                        "competitionLiteProgress": 100.0,
                        "competitionLiteText": f"失败已跳过：{msg[:80]}",
                    },
                )

        result = {
            "baseline": {
                "finalTrainLoss": last_train_b,
                "finalValLoss": last_val_b,
                "steeringError": mae_b,
            }
        }
        if domain_augmentation:
            result["augmented"] = {
                "finalTrainLoss": last_train_a,
                "finalValLoss": last_val_a,
                "steeringError": mae_a,
            }
        row_end = db.get_task(task_id, user_id)
        params_end: dict[str, Any] = {}
        if row_end is not None and "task_params_json" in row_end.keys() and row_end["task_params_json"]:
            try:
                params_end = json.loads(row_end["task_params_json"])
            except Exception:
                params_end = {}
        use_comp_class_final = bool(params_end.get("useCompetitionClassModel", use_competition_class_model))
        use_comp_lite_final = bool(params_end.get("useCompetitionLiteModel", use_competition_lite_model))
        if use_comp_class_final:
            m = comp_class_metrics or {}
            result["competitionClass"] = {
                "finalTrainLoss": float(m.get("finalTrainLoss") or 0.0),
                "finalValLoss": float(m.get("finalValLoss") or 0.0),
                "steeringError": 0.0,
                "finalTrainAcc": m.get("finalTrainAcc"),
                "finalValAcc": m.get("finalValAcc"),
                "note": comp_class_note or ("未采集到训练指标，可能被跳过或日志未匹配" if not comp_class_metrics else None),
            }
        if use_comp_lite_final:
            m = comp_lite_metrics or {}
            result["competitionLite"] = {
                "finalTrainLoss": float(m.get("finalTrainLoss") or 0.0),
                "finalValLoss": float(m.get("finalValLoss") or 0.0),
                "steeringError": 0.0,
                "finalTrainAcc": m.get("finalTrainAcc"),
                "finalValAcc": m.get("finalValAcc"),
                "note": comp_lite_note or ("未采集到训练指标，可能被跳过或日志未匹配" if not comp_lite_metrics else None),
            }

        db.update_task_checkpoints(
            task_id,
            user_id,
            result_json=json.dumps(result, ensure_ascii=False),
            status="completed",
        )
        merge_progress(
            task_id,
            {
                "status": "completed",
                "currentEpoch": total_ui_epochs,
                "baselineProgress": 100.0,
                "domainAugmentationProgress": (100.0 if domain_augmentation else None),
                "augmentedProgress": (100.0 if domain_augmentation else None),
                "competitionClassProgress": (100.0 if use_competition_class_model else None),
                "competitionClassText": ("已完成" if use_competition_class_model else None),
                "competitionLiteProgress": (100.0 if use_competition_lite_model else None),
                "competitionLiteText": ("已完成" if use_competition_lite_model else None),
                "message": "训练完成",
            },
        )
    except Exception:
        err = traceback.format_exc()
        db.update_task_status(task_id, user_id, "failed", err[:2000])
        merge_progress(task_id, {"status": "failed", "message": err[:500]})
    finally:
        release_controls(task_id)
