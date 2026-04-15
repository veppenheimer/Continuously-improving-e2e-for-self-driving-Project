"""训练任务：创建、进度、控制、结果、推理、下载、WebSocket。"""

from __future__ import annotations

import asyncio
import json
import shutil
import threading
import importlib.util
from pathlib import Path

import torch
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, WebSocket, status
from fastapi.responses import FileResponse

from app import database as db
from app.config import settings
from app.deps import CurrentUser
from app.progress_view import build_task_progress
from app.schemas import (
    CompareInferOut,
    CreateTaskBody,
    DomainAugPairOut,
    ModelMetrics,
    TaskProgress,
    TaskResultSummary,
    TrainingTaskSummary,
)
from app.security import safe_decode
from app.services import inference as infer_svc
from app import state
from app.training_artifacts import load_competition_artifact_snapshot, read_progress_snapshot
from app.training_runner import training_worker

router = APIRouter(tags=["tasks"])


def _fallback_from_row(row) -> dict:
    dom = bool(row["domain_augmentation"])
    extra_class = False
    extra_lite = False
    try:
        raw = row["task_params_json"]
        if raw:
            p = json.loads(raw)
            extra_class = bool(p.get("useCompetitionClassModel"))
            extra_lite = bool(p.get("useCompetitionLiteModel"))
    except Exception:
        pass
    total = int(row["epochs"]) * (2 if dom else 1)
    if extra_class:
        total += int(row["epochs"])
    if extra_lite:
        total += int(row["epochs"])
    return {
        "status": row["status"],
        "message": row["message"],
        "domain_augmentation": dom,
        "totalEpochs": total,
        "baselineProgress": (100.0 if row["status"] == "completed" else 0.0),
        "domainAugmentationProgress": (100.0 if row["status"] == "completed" and dom else (0.0 if dom else None)),
        "domainAugmentationText": ("已完成" if row["status"] == "completed" and dom else None),
        "augmentedProgress": (100.0 if row["status"] == "completed" and dom else (0.0 if dom else None)),
        "competitionClassProgress": (100.0 if row["status"] == "completed" and extra_class else (0.0 if extra_class else None)),
        "competitionClassText": ("已完成" if row["status"] == "completed" and extra_class else None),
        "competitionLiteProgress": (100.0 if row["status"] == "completed" and extra_lite else (0.0 if extra_lite else None)),
        "competitionLiteText": ("已完成" if row["status"] == "completed" and extra_lite else None),
    }


def _fallback_progress_dict(fb: dict) -> dict:
    dom = fb["domain_augmentation"]
    total = fb["totalEpochs"]
    return {
        "status": fb["status"],
        "currentEpoch": total if fb["status"] == "completed" else 0,
        "totalEpochs": total,
        "baseline": {"trainLossSeries": [], "valLossSeries": []},
        "augmented": ({"trainLossSeries": [], "valLossSeries": []} if dom else None),
        "competitionClass": (
            {"trainLossSeries": [], "valLossSeries": []}
            if fb.get("competitionClassProgress") is not None
            else None
        ),
        "competitionLite": (
            {"trainLossSeries": [], "valLossSeries": []}
            if fb.get("competitionLiteProgress") is not None
            else None
        ),
        "baselineProgress": fb.get("baselineProgress", 0.0),
        "domainAugmentationProgress": fb.get("domainAugmentationProgress"),
        "domainAugmentationText": fb.get("domainAugmentationText"),
        "augmentedProgress": fb.get("augmentedProgress"),
        "competitionClassProgress": fb.get("competitionClassProgress"),
        "competitionClassText": fb.get("competitionClassText"),
        "competitionLiteProgress": fb.get("competitionLiteProgress"),
        "competitionLiteText": fb.get("competitionLiteText"),
        "message": fb.get("message"),
    }


def _empty_loss_bundle(value) -> bool:
    if not isinstance(value, dict):
        return True
    return not (value.get("trainLossSeries") or value.get("valLossSeries"))


def _progress_for_task(task_id: str, fb: dict) -> dict:
    task_dir = settings.data_dir / "tasks" / task_id
    raw = state.get_progress(task_id)
    if not raw:
        raw = read_progress_snapshot(task_dir) or {}
    raw = dict(raw) if raw else _fallback_progress_dict(fb)

    needs_class = fb.get("competitionClassProgress") is not None and _empty_loss_bundle(raw.get("competitionClass"))
    needs_lite = fb.get("competitionLiteProgress") is not None and _empty_loss_bundle(raw.get("competitionLite"))
    if needs_class or needs_lite:
        artifact_progress = load_competition_artifact_snapshot(task_dir).get("progress", {})
        if needs_class and artifact_progress.get("competitionClass"):
            raw["competitionClass"] = artifact_progress["competitionClass"]
        if needs_lite and artifact_progress.get("competitionLite"):
            raw["competitionLite"] = artifact_progress["competitionLite"]
    return raw


_STEERING_CLASSES = [1.72, 1.64, 1.5, 0.0, -1.5, -1.56, -1.58, -1.6, -1.62]


def _load_comp_model(model_py: Path, class_name: str, ckpt: Path, device: torch.device):
    spec = importlib.util.spec_from_file_location(f"comp_{class_name}_{ckpt.stem}", str(model_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模型定义: {model_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = getattr(mod, class_name)
    model = cls().to(device)
    data = torch.load(str(ckpt), map_location=device)
    state = data.get("model", data) if isinstance(data, dict) else data
    model.load_state_dict(state)
    model.eval()
    return model


def _predict_comp_steering(model: torch.nn.Module, bgr, device: torch.device) -> float:
    x = infer_svc._bgr_to_tensor(bgr).to(device)  # noqa: SLF001
    with torch.no_grad():
        logits = model(x)
    cls = int(torch.argmax(logits, dim=1).item())
    cls = max(0, min(cls, len(_STEERING_CLASSES) - 1))
    return float(_STEERING_CLASSES[cls])


@router.get("/tasks", response_model=list[TrainingTaskSummary])
def list_tasks(user: CurrentUser):
    rows = db.list_tasks_for_user(user["id"])
    return [TrainingTaskSummary(**db.task_row_to_summary(r)) for r in rows]


@router.get("/tasks/{task_id}", response_model=TrainingTaskSummary)
def get_task_detail(task_id: str, user: CurrentUser):
    row = db.get_task(task_id, user["id"])
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
    return TrainingTaskSummary(**db.task_row_to_summary(row))


@router.post("/tasks", response_model=TrainingTaskSummary)
def create_task(body: CreateTaskBody, user: CurrentUser):
    ds = db.get_dataset(body.dataset_id, user["id"])
    if ds is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="数据集不存在")
    ds_b = None
    if body.domain_augmentation:
        if not body.domain_b_dataset_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="开启域增强时必须选择 B 域数据集")
        ds_b = db.get_dataset(body.domain_b_dataset_id, user["id"])
        if ds_b is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="B 域数据集不存在")
        if body.domain_b_dataset_id == body.dataset_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="A 域和 B 域数据集不能相同")

    tid = db.insert_task(
        user["id"],
        body.dataset_id,
        body.learning_rate,
        body.batch_size,
        body.epochs,
        body.domain_augmentation,
        body.name,
        {
            "learningRate": body.learning_rate,
            "batchSize": body.batch_size,
            "epochs": body.epochs,
            "datasetId": body.dataset_id,
            "datasetName": ds["name"],
            "domainAugmentation": body.domain_augmentation,
            "domainBDatasetId": body.domain_b_dataset_id,
            "domainBDatasetName": (ds_b["name"] if ds_b is not None else None),
            "cycleGanEpochs": body.cyclegan_epochs,
            "cycleGanDecayEpochs": body.cyclegan_decay_epochs,
            "cycleGanBatchSize": body.cyclegan_batch_size,
            "cycleGanSaveEpochFreq": body.cyclegan_save_epoch_freq,
            "cycleGanSaveLatestFreq": body.cyclegan_save_latest_freq,
            "cycleGanLoadSize": body.cyclegan_load_size,
            "cycleGanCropSize": body.cyclegan_crop_size,
            "cycleGanLambdaIdentity": body.cyclegan_lambda_identity,
            "useCompetitionClassModel": body.use_competition_class_model,
            "useCompetitionLiteModel": body.use_competition_lite_model,
        },
    )
    artifacts = settings.data_dir / "tasks" / tid
    artifacts.mkdir(parents=True, exist_ok=True)

    state.register_task(tid)
    thread = threading.Thread(
        target=training_worker,
        kwargs={
            "task_id": tid,
            "user_id": user["id"],
            "dataset_root": ds["root_dir"],
            "learning_rate": body.learning_rate,
            "batch_size": body.batch_size,
            "epochs": body.epochs,
            "domain_augmentation": body.domain_augmentation,
            "dataset_b_root": (ds_b["root_dir"] if ds_b is not None else None),
            "cyclegan_epochs": body.cyclegan_epochs,
            "cyclegan_decay_epochs": body.cyclegan_decay_epochs,
            "cyclegan_batch_size": body.cyclegan_batch_size,
            "cyclegan_save_epoch_freq": body.cyclegan_save_epoch_freq,
            "cyclegan_save_latest_freq": body.cyclegan_save_latest_freq,
            "cyclegan_load_size": body.cyclegan_load_size,
            "cyclegan_crop_size": body.cyclegan_crop_size,
            "cyclegan_lambda_identity": body.cyclegan_lambda_identity,
            "use_competition_class_model": body.use_competition_class_model,
            "use_competition_lite_model": body.use_competition_lite_model,
            "artifacts_dir": artifacts,
        },
        daemon=True,
    )
    thread.start()

    row = db.get_task(tid, user["id"])
    assert row is not None
    return TrainingTaskSummary(**db.task_row_to_summary(row))


@router.delete("/tasks/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_task(task_id: str, user: CurrentUser):
    row = db.get_task(task_id, user["id"])
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
    ctrl = state.get_controls(task_id)
    if ctrl is not None:
        ctrl.stop.set()
    state.unregister_task(task_id)
    db.delete_task(task_id, user["id"])
    art = settings.data_dir / "tasks" / task_id
    if art.is_dir():
        shutil.rmtree(art, ignore_errors=True)
    return None


@router.get("/tasks/{task_id}/progress", response_model=TaskProgress)
def get_task_progress(task_id: str, user: CurrentUser):
    row = db.get_task(task_id, user["id"])
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
    fb = _fallback_from_row(row)
    raw = _progress_for_task(task_id, fb)
    return build_task_progress(raw, fb)


@router.post("/tasks/{task_id}/pause", status_code=status.HTTP_204_NO_CONTENT)
def pause_or_resume(task_id: str, user: CurrentUser):
    row = db.get_task(task_id, user["id"])
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
    ctrl = state.get_controls(task_id)
    if ctrl is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="任务当前不可暂停（未在运行或已结束）")
    st = row["status"]
    if st == "running":
        ctrl.pause.set()
        db.update_task_status(task_id, user["id"], "paused", None)
    elif st == "paused":
        ctrl.pause.clear()
        db.update_task_status(task_id, user["id"], "running", None)
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="当前状态不支持暂停/继续")
    return None


@router.post("/tasks/{task_id}/stop", status_code=status.HTTP_204_NO_CONTENT)
def stop_task(task_id: str, user: CurrentUser):
    row = db.get_task(task_id, user["id"])
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
    ctrl = state.get_controls(task_id)
    if ctrl is None:
        db.update_task_status(task_id, user["id"], "stopped", "用户终止")
        return None
    if row["status"] in ("completed", "failed", "stopped"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="任务已结束")
    ctrl.stop.set()
    return None


@router.get("/tasks/{task_id}/results", response_model=TaskResultSummary)
def get_results(task_id: str, user: CurrentUser):
    row = db.get_task(task_id, user["id"])
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
    if row["status"] != "completed":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="任务尚未完成")
    if not row["result_json"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="无结果数据")
    data = json.loads(row["result_json"])
    params = {}
    try:
        if row["task_params_json"]:
            params = json.loads(row["task_params_json"])
    except Exception:
        params = {}

    artifact_metrics = load_competition_artifact_snapshot(settings.data_dir / "tasks" / task_id).get("metrics", {})

    def _has_metric_values(value) -> bool:
        if not isinstance(value, dict):
            return False
        return any(
            value.get(k) not in (None, 0, 0.0)
            for k in ("finalTrainLoss", "finalValLoss", "finalTrainAcc", "finalValAcc")
        )

    def _hydrate_metric(key: str) -> None:
        artifact_value = artifact_metrics.get(key)
        if artifact_value and not _has_metric_values(data.get(key)):
            data[key] = artifact_value

    if params.get("useCompetitionClassModel"):
        _hydrate_metric("competitionClass")
    if params.get("useCompetitionLiteModel"):
        _hydrate_metric("competitionLite")

    if params.get("useCompetitionClassModel") and "competitionClass" not in data:
        data["competitionClass"] = {
            "finalTrainLoss": 0.0,
            "finalValLoss": 0.0,
            "steeringError": 0.0,
            "finalTrainAcc": None,
            "finalValAcc": None,
            "note": "该任务创建时尚未记录该模型指标，请重新训练以获取完整统计。",
        }
    if params.get("useCompetitionLiteModel") and "competitionLite" not in data:
        data["competitionLite"] = {
            "finalTrainLoss": 0.0,
            "finalValLoss": 0.0,
            "steeringError": 0.0,
            "finalTrainAcc": None,
            "finalValAcc": None,
            "note": "该任务创建时尚未记录该模型指标，请重新训练以获取完整统计。",
        }
    base = data["baseline"]
    aug = data.get("augmented")
    comp_class = data.get("competitionClass")
    comp_lite = data.get("competitionLite")
    return TaskResultSummary(
        baseline=ModelMetrics(
            final_train_loss=float(base["finalTrainLoss"]),
            final_val_loss=float(base["finalValLoss"]),
            steering_error=float(base["steeringError"]),
        ),
        augmented=(
            ModelMetrics(
                final_train_loss=float(aug["finalTrainLoss"]),
                final_val_loss=float(aug["finalValLoss"]),
                steering_error=float(aug["steeringError"]),
            )
            if aug
            else None
        ),
        competition_class=(
            ModelMetrics(
                final_train_loss=float(comp_class.get("finalTrainLoss", 0.0)),
                final_val_loss=float(comp_class.get("finalValLoss", 0.0)),
                steering_error=float(comp_class.get("steeringError", 0.0)),
                final_train_acc=float(comp_class["finalTrainAcc"]) if comp_class.get("finalTrainAcc") is not None else None,
                final_val_acc=float(comp_class["finalValAcc"]) if comp_class.get("finalValAcc") is not None else None,
                note=(str(comp_class["note"]) if comp_class.get("note") else None),
            )
            if comp_class
            else None
        ),
        competition_lite=(
            ModelMetrics(
                final_train_loss=float(comp_lite.get("finalTrainLoss", 0.0)),
                final_val_loss=float(comp_lite.get("finalValLoss", 0.0)),
                steering_error=float(comp_lite.get("steeringError", 0.0)),
                final_train_acc=float(comp_lite["finalTrainAcc"]) if comp_lite.get("finalTrainAcc") is not None else None,
                final_val_acc=float(comp_lite["finalValAcc"]) if comp_lite.get("finalValAcc") is not None else None,
                note=(str(comp_lite["note"]) if comp_lite.get("note") else None),
            )
            if comp_lite
            else None
        ),
    )


@router.post("/tasks/{task_id}/infer/compare", response_model=CompareInferOut)
async def infer_compare(task_id: str, user: CurrentUser, file: UploadFile = File(...)):
    row = db.get_task(task_id, user["id"])
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
    if row["status"] != "completed":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="请先完成训练")
    if not row["baseline_ckpt"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="未找到基准模型")
    data = await file.read()
    try:
        bgr = infer_svc.load_image_bytes(data)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_path = Path(row["baseline_ckpt"])
    model_b = infer_svc.load_checkpoint_model(base_path, device)
    b_angle = infer_svc.predict_image(model_b, bgr, device)

    aug_angle: float | None = None
    if row["augmented_ckpt"]:
        aug_path = Path(row["augmented_ckpt"])
        if aug_path.is_file():
            model_a = infer_svc.load_checkpoint_model(aug_path, device)
            aug_angle = infer_svc.predict_image(model_a, bgr, device)

    comp_class_angle: float | None = None
    comp_lite_angle: float | None = None
    task_art = settings.data_dir / "tasks" / task_id
    comp_root = settings.competition_project_root
    class_ckpt = task_art / "competition_class" / "ve2_competition_class.pth"
    lite_ckpt = task_art / "competition_lite" / "ve2_competition_lite.pth"
    try:
        if class_ckpt.is_file():
            m = _load_comp_model(comp_root / "Net_class" / "models.py", "AutoDriveNet", class_ckpt, device)
            comp_class_angle = _predict_comp_steering(m, bgr, device)
    except Exception:
        comp_class_angle = None
    try:
        if lite_ckpt.is_file():
            m = _load_comp_model(comp_root / "Net_improve" / "models.py", "AutoDriveNetImprove", lite_ckpt, device)
            comp_lite_angle = _predict_comp_steering(m, bgr, device)
    except Exception:
        comp_lite_angle = None

    return CompareInferOut(
        baseline_steering=b_angle,
        augmented_steering=aug_angle,
        competition_class_steering=comp_class_angle,
        competition_lite_steering=comp_lite_angle,
    )


@router.get("/tasks/{task_id}/download")
def download_model(
    task_id: str,
    user: CurrentUser,
    model: str = Query(..., description="baseline 或 augmented"),
):
    row = db.get_task(task_id, user["id"])
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
    if row["status"] != "completed":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="任务未完成")
    if model not in ("baseline", "augmented"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="model 须为 baseline 或 augmented")
    path_str = row["baseline_ckpt"] if model == "baseline" else row["augmented_ckpt"]
    if not path_str:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="该模型不存在")
    path = Path(path_str)
    if not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="文件已丢失")
    name = f"{model}_model.pth"
    return FileResponse(path, filename=name, media_type="application/octet-stream")


@router.websocket("/tasks/{task_id}/stream")
async def task_progress_ws(websocket: WebSocket, task_id: str, token: str | None = Query(None)):
    await websocket.accept()
    payload = safe_decode(token or "")
    if payload is None or "sub" not in payload:
        await websocket.close(code=4401)
        return
    user_id = str(payload["sub"])
    row = db.get_task(task_id, user_id)
    if row is None:
        await websocket.close(code=4404)
        return

    try:
        while True:
            row = db.get_task(task_id, user_id)
            if row is None:
                break
            fb = _fallback_from_row(row)
            raw = _progress_for_task(task_id, fb)
            prog = build_task_progress(raw, fb)
            await websocket.send_text(prog.model_dump_json(by_alias=True))
            if row["status"] in ("completed", "failed", "stopped"):
                await asyncio.sleep(1.0)
                break
            await asyncio.sleep(1.5)
    except Exception:
        pass


@router.get("/tasks/{task_id}/domain-aug/pairs", response_model=list[DomainAugPairOut])
def list_domain_aug_pairs(task_id: str, user: CurrentUser):
    row = db.get_task(task_id, user["id"])
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
    if not bool(row["domain_augmentation"]):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="该任务未开启域增强")
    pairs_path = settings.data_dir / "tasks" / task_id / "domain_aug_pairs.json"
    if not pairs_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="暂无域增强对比数据")
    try:
        data = json.loads(pairs_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="域增强对比数据损坏") from e
    out: list[DomainAugPairOut] = []
    for x in data:
        out.append(DomainAugPairOut(index=int(x["index"]), a_name=str(x["aName"]), c_name=str(x["cName"])))
    return out


@router.get("/tasks/{task_id}/domain-aug/image")
def get_domain_aug_image(
    task_id: str,
    user: CurrentUser,
    index: int = Query(..., ge=0),
    kind: str = Query(..., description="a 或 c"),
):
    row = db.get_task(task_id, user["id"])
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
    pairs_path = settings.data_dir / "tasks" / task_id / "domain_aug_pairs.json"
    if not pairs_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="暂无域增强对比数据")
    data = json.loads(pairs_path.read_text(encoding="utf-8"))
    target = next((x for x in data if int(x.get("index", -1)) == index), None)
    if target is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="对比索引不存在")
    if kind not in ("a", "c"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="kind 必须为 a 或 c")
    path_key = "aPath" if kind == "a" else "cPath"
    name_key = "aName" if kind == "a" else "cName"
    p = Path(str(target[path_key]))
    if not p.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="图像文件不存在")
    media = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(p, media_type=media, filename=str(target[name_key]))
