"""Run training jobs in a background worker and sync progress to SQLite/memory."""

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
from torch.utils.data import DataLoader

from app import database as db
from app.config import settings
from app.state import append_loss_point, get_controls, get_progress, merge_progress, release_controls
from app.training_artifacts import write_progress_snapshot
from datasets import AutoDriveDataset, AutoDriveListDataset
from models import AutoDriveNet
from utils import AverageMeter

CURRENT_DIR = Path(__file__).resolve().parent
NET_WEB_DIR = CURRENT_DIR.parents[0]
REPO_ROOT = CURRENT_DIR.parents[2]
for candidate in (NET_WEB_DIR, REPO_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from steering_augmentations import AugConfig, build_eval_transforms, build_stress_transforms, build_train_transforms
from steering_preprocess import PreprocessConfig, build_angle_vocab, encode_angles_to_vocab


def _build_regression_aug_config() -> AugConfig:
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
    return AugConfig(preprocess=preprocess, style_mix_ratio=style_mix_ratio)


def _build_regression_transforms() -> tuple[Any, Any, Any]:
    cfg = _build_regression_aug_config()
    return build_train_transforms(cfg), build_eval_transforms(cfg), build_stress_transforms(cfg)


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
    for imgs, labels in loader:
        if should_stop():
            break
        imgs = imgs.to(device)
        labels = labels.to(device)
        aux_targets = encode_angles_to_vocab(labels.view(-1), angle_vocab, device=device)
        pred, aux_logits = model(imgs, return_aux=True)
        reg_loss = reg_criterion(pred, labels)
        aux_loss = aux_criterion(aux_logits, aux_targets)
        loss = reg_loss + aux_cls_weight * aux_loss
        optimizer.zero_grad()
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
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            aux_targets = encode_angles_to_vocab(labels.view(-1), angle_vocab, device=device)
            pred, aux_logits = model(imgs, return_aux=True)
            reg_loss = reg_criterion(pred, labels)
            aux_loss = aux_criterion(aux_logits, aux_targets)
            loss = reg_loss + aux_cls_weight * aux_loss
            mae = (pred - labels).abs().mean().item()
            loss_meter.update(loss.item(), imgs.size(0))
            mae_meter.update(mae, imgs.size(0))
    return loss_meter.avg, mae_meter.avg


def _read_training_summary(summary_path: Path) -> dict[str, Any]:
    if not summary_path.is_file():
        return {}
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _wait_resume(ctrl) -> bool:
    """Return False if training should stop while paused."""
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
    """Ensure optional CycleGAN Python dependencies are available."""
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
        raise RuntimeError(f"CycleGAN 椤圭洰鐩綍涓嶅瓨鍦? {project}")
    train_py = project / "train.py"
    test_py = project / "test.py"
    if not train_py.is_file() or not test_py.is_file():
        raise RuntimeError("鏈壘鍒?CycleGAN train.py 鎴?test.py")
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
        raise RuntimeError("A/B 鏁版嵁闆?train.txt 涓虹┖锛屾棤娉曟墽琛屽煙澧炲己")

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
        on_train_progress(97.0, "CycleGAN璁粌瀹屾垚锛屾鍦ㄧ敓鎴?C 鏁版嵁")
    _run_subprocess_or_raise(infer_args, project)
    if on_train_progress is not None:
        on_train_progress(100.0, "C 鏁版嵁鐢熸垚瀹屾垚")

    image_dir = outputs_dir / job_id / "test_latest" / "images"
    if not image_dir.is_dir():
        raise RuntimeError("CycleGAN inference output directory does not exist")

    c_list = artifacts_dir / "train_c.txt"
    pairs_meta: list[dict[str, Any]] = []
    with open(c_list, "w", encoding="utf-8") as f:
        for idx, (src_name, angle) in enumerate(infer_pairs):
            fake_name = f"{Path(src_name).stem}_fake.png"
            fake_path = image_dir / fake_name
            if not fake_path.is_file():
                raise RuntimeError(f"鏈壘鍒扮敓鎴愬浘鍍? {fake_name}")
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
) -> dict[str, Any]:
    train_py = script_dir / "train.py"
    if not train_py.is_file():
        raise RuntimeError(f"{model_name} training script not found: {train_py}")
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
            "VENET_EARLY_STOP_PATIENCE": os.getenv("VENET_EARLY_STOP_PATIENCE", "12"),
            "VENET_EARLY_STOP_MIN_DELTA": os.getenv("VENET_EARLY_STOP_MIN_DELTA", "1e-4"),
            "VENET_WEIGHT_DECAY": os.getenv("VENET_WEIGHT_DECAY", "1e-4"),
            "VENET_GRAD_CLIP": os.getenv("VENET_GRAD_CLIP", "3.0"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    r_classic = re.compile(r"epoch:\s*(\d+)")
    r_lite = re.compile(r"Epoch\s+(\d+)\s+\|")
    r_classic_train_loss = re.compile(r"CE_Loss:\s*([0-9.eE+-]+)")
    r_classic_val_loss = re.compile(r"Val_CE_Loss:\s*([0-9.eE+-]+)")
    r_classic_train_acc = re.compile(r"Train_Acc:\s*([0-9.eE+-]+)")
    r_classic_val_acc = re.compile(r"Val_Acc:\s*([0-9.eE+-]+)")
    r_classic_train_mae = re.compile(r"Train_Angle_MAE:\s*([0-9.eE+-]+)")
    r_classic_val_mae = re.compile(r"Val_Angle_MAE:\s*([0-9.eE+-]+)")
    r_classic_val_stress_mae = re.compile(r"Val_Stress_Angle_MAE:\s*([0-9.eE+-]+)")
    r_lite_train_loss = re.compile(r"TrainLoss\s+([0-9.eE+-]+)")
    r_lite_train_acc = re.compile(r"TrainAcc\s+([0-9.eE+-]+)")
    r_lite_val_loss = re.compile(r"ValLoss\s+([0-9.eE+-]+)")
    r_lite_val_acc = re.compile(r"ValAcc\s+([0-9.eE+-]+)")
    r_lite_train_mae = re.compile(r"TrainAngleMAE\s+([0-9.eE+-]+)")
    r_lite_val_mae = re.compile(r"ValAngleMAE\s+([0-9.eE+-]+)")
    r_lite_val_stress_mae = re.compile(r"ValStressAngleMAE\s+([0-9.eE+-]+)")
    metrics: dict[str, Any] = {
        "finalTrainLoss": None,
        "finalValLoss": None,
        "finalTrainAcc": None,
        "finalValAcc": None,
        "finalTrainAngleMAE": None,
        "finalValAngleMAE": None,
        "finalValStressAngleMAE": None,
        "steeringError": None,
        "requestedEpochs": epochs,
        "completedEpochs": None,
        "bestEpoch": None,
        "stoppedEpoch": None,
        "earlyStopped": False,
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
        t_mae = None
        v_mae = None

        m1 = r_classic_train_loss.search(line)
        if m1:
            t_loss = float(m1.group(1))
        m2 = r_classic_val_loss.search(line)
        if m2:
            v_loss = float(m2.group(1))
        m3 = r_classic_train_acc.search(line)
        if m3:
            t_acc = float(m3.group(1))
        m4 = r_classic_val_acc.search(line)
        if m4:
            v_acc = float(m4.group(1))
        m5 = r_classic_train_mae.search(line)
        if m5:
            t_mae = float(m5.group(1))
        m6 = r_classic_val_mae.search(line)
        if m6:
            v_mae = float(m6.group(1))

        m7 = r_lite_train_loss.search(line)
        if m7:
            t_loss = float(m7.group(1))
        m8 = r_lite_val_loss.search(line)
        if m8:
            v_loss = float(m8.group(1))
        m9 = r_lite_train_acc.search(line)
        if m9:
            t_acc = float(m9.group(1))
        m10 = r_lite_val_acc.search(line)
        if m10:
            v_acc = float(m10.group(1))

        if t_loss is not None:
            metrics["finalTrainLoss"] = t_loss
        if v_loss is not None:
            metrics["finalValLoss"] = v_loss
        if t_acc is not None:
            metrics["finalTrainAcc"] = t_acc
        if v_acc is not None:
            metrics["finalValAcc"] = v_acc
        if t_mae is not None:
            metrics["finalTrainAngleMAE"] = t_mae
        if v_mae is not None:
            metrics["finalValAngleMAE"] = v_mae
        if v_mae is not None:
            metrics["steeringError"] = v_mae
        elif t_mae is not None:
            metrics["steeringError"] = t_mae

        use_train_loss = t_loss
        use_val_loss = v_loss if v_loss is not None else t_loss
        if use_train_loss is not None and use_val_loss is not None:
            append_loss_point(task_id, branch_name, cur, float(use_train_loss), float(use_val_loss))

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
            f"{model_name} training failed(exit={code})\nSTDERR:\n{chr(10).join(err_lines)[:3000]}"
            f"\nSTDOUT:\n{chr(10).join(out_lines)[:3000]}"
        )
    on_progress(100.0, f"{model_name} training completed")
    if metrics["finalTrainLoss"] is None:
        metrics["finalTrainLoss"] = 0.0
    if metrics["finalValLoss"] is None:
        metrics["finalValLoss"] = float(metrics["finalTrainLoss"] or 0.0)
    if metrics["steeringError"] is None:
        metrics["steeringError"] = float(metrics["finalValAngleMAE"] or metrics["finalTrainAngleMAE"] or 0.0)
    summary = _read_training_summary(out_dir / "training_summary.json")
    for key in (
        "requestedEpochs",
        "completedEpochs",
        "bestEpoch",
        "stoppedEpoch",
        "earlyStopped",
        "finalTrainLoss",
        "finalValLoss",
        "steeringError",
        "finalTrainAcc",
        "finalValAcc",
        "finalTrainAngleMAE",
        "finalValAngleMAE",
        "bestValLoss",
        "testBestLoss",
        "testBestAcc",
    ):
        if key in summary:
            metrics[key] = summary[key]
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

    # 浠ユ暟鎹簱涓殑浠诲姟蹇収涓哄噯锛岄伩鍏嶇嚎绋嬪弬鏁颁笌鎸佷箙鍖栭厤缃笉涓€鑷淬€?
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
        "message": f"璁惧: {device}",
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

    train_transform, eval_transform, stress_transform = _build_regression_transforms()
    reg_criterion = nn.SmoothL1Loss().to(device)
    aux_criterion = nn.CrossEntropyLoss().to(device)
    aux_cls_weight = float(os.getenv("VENET_AUX_CLS_WEIGHT", "0.3"))
    weight_decay = float(os.getenv("VENET_WEIGHT_DECAY", "1e-4"))
    grad_clip = float(os.getenv("VENET_GRAD_CLIP", "3.0"))
    early_stop_patience = int(os.getenv("VENET_EARLY_STOP_PATIENCE", "12"))
    early_stop_min_delta = float(os.getenv("VENET_EARLY_STOP_MIN_DELTA", "1e-4"))
    freeze_backbone_epochs = int(os.getenv("VENET_FREEZE_BACKBONE_EPOCHS", "5"))
    backbone_lr_factor = float(os.getenv("VENET_BACKBONE_LR_FACTOR", "0.1"))
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
    baseline_completed_epochs = 0
    baseline_stopped_epoch: int | None = None
    baseline_early_stopped = False
    baseline_test_loss = 0.0
    best_augmented_epoch = 0
    best_augmented_loss: float | None = None
    augmented_completed_epochs = 0
    augmented_stopped_epoch: int | None = None
    augmented_early_stopped = False
    augmented_test_loss = 0.0
    comp_class_metrics: dict[str, Any] | None = None
    comp_lite_metrics: dict[str, Any] | None = None
    comp_class_note: str | None = None
    comp_lite_note: str | None = None

    try:
        db.update_task_status(task_id, user_id, "running", None)

                # ----- 基准模型 -----
        train_ds = AutoDriveDataset(dataset_root, "train", train_transform)
        val_ds = AutoDriveDataset(dataset_root, "val", eval_transform)
        val_stress_ds = AutoDriveDataset(dataset_root, "val", stress_transform, deterministic_seed=20260418)
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
        test_file = Path(dataset_root) / "test.txt"
        has_test_split = test_file.is_file()
        test_ds = AutoDriveDataset(dataset_root, "test", eval_transform) if has_test_split else val_ds
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        model = AutoDriveNet(num_aux_classes=len(angle_vocab), use_pretrained=True).to(device)
        best_baseline_mae = float("inf")
        best_baseline_epoch = 0
        best_baseline_state = None
        best_baseline_loss = None
        baseline_completed_epochs = 0
        baseline_stopped_epoch = None
        baseline_early_stopped = False
        baseline_train_mae = 0.0
        baseline_val_mae = 0.0
        baseline_val_stress_mae = 0.0
        baseline_no_improve = 0
        optimizer = None
        scheduler = None
        current_stage = None

        for epoch in range(1, epochs + 1):
            if not _wait_resume(ctrl):
                db.update_task_status(task_id, user_id, "stopped", "用户终止")
                merge_progress(task_id, {"status": "stopped"})
                release_controls(task_id)
                return

            desired_stage = "head" if epoch <= freeze_backbone_epochs else "full"
            if desired_stage != current_stage:
                model.set_backbone_trainable(desired_stage != "head")
                if desired_stage == "head":
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
                    patience=max(2, early_stop_patience // 3),
                    min_lr=1e-6,
                )
                current_stage = desired_stage

            last_train_b, baseline_train_mae = _train_epoch(
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

            last_val_b, baseline_val_mae = _evaluate_regression(
                model, val_loader, device, reg_criterion, aux_criterion, angle_vocab, aux_cls_weight
            )
            _, baseline_val_stress_mae = _evaluate_regression(
                model, val_stress_loader, device, reg_criterion, aux_criterion, angle_vocab, aux_cls_weight
            )
            baseline_completed_epochs = epoch
            scheduler.step(baseline_val_stress_mae)
            append_loss_point(task_id, "baseline", epoch, last_train_b, last_val_b)
            merge_progress(
                task_id,
                {
                    "currentEpoch": epoch,
                    "status": "running",
                    "baselineProgress": (epoch / max(epochs, 1)) * 100.0,
                    "message": f"基准模型 epoch {epoch}/{epochs} | val_stress_mae={baseline_val_stress_mae:.4f}",
                },
            )

            if baseline_val_stress_mae < best_baseline_mae - early_stop_min_delta:
                best_baseline_mae = baseline_val_stress_mae
                best_baseline_epoch = epoch
                best_baseline_loss = last_val_b
                baseline_no_improve = 0
                best_baseline_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                torch.save({"model": best_baseline_state, "epoch": epoch, "angleVocab": angle_vocab}, baseline_path)
            else:
                baseline_no_improve += 1
                if baseline_no_improve >= early_stop_patience:
                    baseline_early_stopped = True
                    baseline_stopped_epoch = epoch
                    merge_progress(task_id, {"message": f"基准模型早停于 epoch {epoch}"})
                    break

        if best_baseline_state is None:
            best_baseline_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_baseline_mae = baseline_val_stress_mae
            best_baseline_epoch = baseline_completed_epochs
            best_baseline_loss = last_val_b
            torch.save({"model": best_baseline_state, "epoch": baseline_completed_epochs, "angleVocab": angle_vocab}, baseline_path)

        model.load_state_dict(best_baseline_state)
        baseline_test_loss, baseline_test_mae = _evaluate_regression(
            model, test_loader, device, reg_criterion, aux_criterion, angle_vocab, aux_cls_weight
        )
        mae_b = baseline_test_mae
        db.update_task_checkpoints(task_id, user_id, baseline_ckpt=str(baseline_path))

        del model, optimizer, train_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ----- 鍩熷寮烘ā鍨嬶紙鍙€夛級-----
        if domain_augmentation:
            merge_progress(
                task_id,
                {
                    "augmented": {"trainLossSeries": [], "valLossSeries": []},
                    "message": "基准模型已完成，开始生成 CycleGAN 数据集 C",
                },
            )
            if not dataset_b_root:
                raise RuntimeError("缂哄皯 B 鍩熸暟鎹泦璺緞")
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
                    "message": "C 数据集已生成，开始训练增强模型",
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

            train_ds_a = AutoDriveListDataset(str(combined_train), train_transform)
            val_ds_a = AutoDriveDataset(dataset_root, "val", eval_transform)
            val_stress_ds_a = AutoDriveDataset(dataset_root, "val", stress_transform, deterministic_seed=20260418)
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
            test_ds_a = AutoDriveDataset(dataset_root, "test", eval_transform) if has_test_split else val_ds_a
            test_loader_a = DataLoader(
                test_ds_a,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
            )

            model_a = AutoDriveNet(num_aux_classes=len(angle_vocab_a), use_pretrained=True).to(device)
            best_augmented_mae = float("inf")
            best_augmented_epoch = 0
            best_augmented_state = None
            best_augmented_loss = None
            augmented_completed_epochs = 0
            augmented_stopped_epoch = None
            augmented_early_stopped = False
            augmented_train_mae = 0.0
            augmented_val_mae = 0.0
            augmented_val_stress_mae = 0.0
            augmented_no_improve = 0
            opt_a = None
            scheduler_a = None
            current_aug_stage = None

            for epoch in range(1, epochs + 1):
                if not _wait_resume(ctrl):
                    db.update_task_status(task_id, user_id, "stopped", "用户终止")
                    merge_progress(task_id, {"status": "stopped"})
                    release_controls(task_id)
                    return

                desired_stage = "head" if epoch <= freeze_backbone_epochs else "full"
                if desired_stage != current_aug_stage:
                    model_a.set_backbone_trainable(desired_stage != "head")
                    if desired_stage == "head":
                        opt_a = torch.optim.AdamW(model_a.head_parameters(), lr=learning_rate, weight_decay=weight_decay)
                    else:
                        opt_a = torch.optim.AdamW(
                            [
                                {"params": list(model_a.head_parameters()), "lr": learning_rate},
                                {"params": list(model_a.backbone_parameters()), "lr": learning_rate * backbone_lr_factor},
                            ],
                            weight_decay=weight_decay,
                        )
                    scheduler_a = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        opt_a,
                        mode="min",
                        factor=0.5,
                        patience=max(2, early_stop_patience // 3),
                        min_lr=1e-6,
                    )
                    current_aug_stage = desired_stage

                g_ep = epochs + epoch
                last_train_a, augmented_train_mae = _train_epoch(
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

                last_val_a, augmented_val_mae = _evaluate_regression(
                    model_a, val_loader_a, device, reg_criterion, aux_criterion, angle_vocab_a, aux_cls_weight
                )
                _, augmented_val_stress_mae = _evaluate_regression(
                    model_a, val_stress_loader_a, device, reg_criterion, aux_criterion, angle_vocab_a, aux_cls_weight
                )
                augmented_completed_epochs = epoch
                scheduler_a.step(augmented_val_stress_mae)
                append_loss_point(task_id, "augmented", epoch, last_train_a, last_val_a)
                merge_progress(
                    task_id,
                    {
                        "currentEpoch": g_ep,
                        "status": "running",
                        "augmentedProgress": (epoch / max(epochs, 1)) * 100.0,
                        "message": f"增强模型 epoch {epoch}/{epochs} | val_stress_mae={augmented_val_stress_mae:.4f}",
                    },
                )

                if augmented_val_stress_mae < best_augmented_mae - early_stop_min_delta:
                    best_augmented_mae = augmented_val_stress_mae
                    best_augmented_epoch = epoch
                    best_augmented_loss = last_val_a
                    augmented_no_improve = 0
                    best_augmented_state = {k: v.detach().cpu() for k, v in model_a.state_dict().items()}
                    torch.save({"model": best_augmented_state, "epoch": epoch, "angleVocab": angle_vocab_a}, augmented_path)
                else:
                    augmented_no_improve += 1
                    if augmented_no_improve >= early_stop_patience:
                        augmented_early_stopped = True
                        augmented_stopped_epoch = epoch
                        merge_progress(task_id, {"message": f"增强模型早停于 epoch {epoch}"})
                        break

            if best_augmented_state is None:
                best_augmented_state = {k: v.detach().cpu() for k, v in model_a.state_dict().items()}
                best_augmented_mae = augmented_val_stress_mae
                best_augmented_epoch = augmented_completed_epochs
                best_augmented_loss = last_val_a
                torch.save({"model": best_augmented_state, "epoch": augmented_completed_epochs, "angleVocab": angle_vocab_a}, augmented_path)

            model_a.load_state_dict(best_augmented_state)
            augmented_test_loss, augmented_test_mae = _evaluate_regression(
                model_a, test_loader_a, device, reg_criterion, aux_criterion, angle_vocab_a, aux_cls_weight
            )
            mae_a = augmented_test_mae
            db.update_task_checkpoints(task_id, user_id, augmented_ckpt=str(augmented_path))

        comp_root = settings.competition_project_root
        if use_competition_class_model:
            merge_progress(task_id, {"competitionClass": {"trainLossSeries": [], "valLossSeries": []}})
            merge_progress(task_id, {"competitionClassText": "开始训练分类模型"})
            try:
                comp_class_metrics = _run_competition_model_training(
                    model_name="鍒嗙被妯″瀷",
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
                comp_class_note = f"澶辫触宸茶烦杩? {msg[:160]}"
                merge_progress(
                    task_id,
                    {
                        "competitionClassProgress": 100.0,
                        "competitionClassText": f"澶辫触宸茶烦杩囷細{msg[:80]}",
                    },
                )

        if use_competition_lite_model:
            merge_progress(task_id, {"competitionLite": {"trainLossSeries": [], "valLossSeries": []}})
            base_ep = epochs * (2 if domain_augmentation else 1) + (epochs if use_competition_class_model else 0)
            merge_progress(task_id, {"competitionLiteText": "开始训练轻量模型"})
            try:
                comp_lite_metrics = _run_competition_model_training(
                    model_name="杞婚噺妯″瀷",
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
                comp_lite_note = f"澶辫触宸茶烦杩? {msg[:160]}"
                merge_progress(
                    task_id,
                    {
                        "competitionLiteProgress": 100.0,
                        "competitionLiteText": f"澶辫触宸茶烦杩囷細{msg[:80]}",
                    },
                )

        result = {
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
                "steeringError": float(m.get("steeringError") or 0.0),
                "valStressMAE": m.get("finalValStressAngleMAE"),
                "finalTrainAcc": m.get("finalTrainAcc"),
                "finalValAcc": m.get("finalValAcc"),
                "requestedEpochs": m.get("requestedEpochs"),
                "completedEpochs": m.get("completedEpochs"),
                "bestEpoch": m.get("bestEpoch"),
                "stoppedEpoch": m.get("stoppedEpoch"),
                "earlyStopped": m.get("earlyStopped"),
                "note": comp_class_note or ("鏈噰闆嗗埌璁粌鎸囨爣锛屽彲鑳借璺宠繃鎴栨棩蹇楁湭鍖归厤" if not comp_class_metrics else None),
            }
        if use_comp_lite_final:
            m = comp_lite_metrics or {}
            result["competitionLite"] = {
                "finalTrainLoss": float(m.get("finalTrainLoss") or 0.0),
                "finalValLoss": float(m.get("finalValLoss") or 0.0),
                "steeringError": float(m.get("steeringError") or 0.0),
                "valStressMAE": m.get("finalValStressAngleMAE"),
                "finalTrainAcc": m.get("finalTrainAcc"),
                "finalValAcc": m.get("finalValAcc"),
                "requestedEpochs": m.get("requestedEpochs"),
                "completedEpochs": m.get("completedEpochs"),
                "bestEpoch": m.get("bestEpoch"),
                "stoppedEpoch": m.get("stoppedEpoch"),
                "earlyStopped": m.get("earlyStopped"),
                "note": comp_lite_note or ("鏈噰闆嗗埌璁粌鎸囨爣锛屽彲鑳借璺宠繃鎴栨棩蹇楁湭鍖归厤" if not comp_lite_metrics else None),
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
                "message": "璁粌瀹屾垚",
            },
        )
        _persist_progress_snapshot(task_id, artifacts_dir)
    except Exception:
        err = traceback.format_exc()
        db.update_task_status(task_id, user_id, "failed", err[:2000])
        merge_progress(task_id, {"status": "failed", "message": err[:500]})
        _persist_progress_snapshot(task_id, artifacts_dir)
    finally:
        release_controls(task_id)




















