"""Training task routes: create, inspect, progress, inference, download, and websocket updates."""

from __future__ import annotations

import asyncio
import json
import shutil
import threading
from pathlib import Path

import torch
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, WebSocket, status
from fastapi.responses import FileResponse

from app import database as db
from app import state
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
from app.training_artifacts import read_progress_snapshot, read_training_summary
from app.training_runner import training_worker
from app.config import settings

router = APIRouter(tags=["tasks"])


def _fallback_from_row(row) -> dict:
    dom = bool(row["domain_augmentation"])
    total = int(row["epochs"]) * (2 if dom else 1)
    return {
        "status": row["status"],
        "message": row["message"],
        "domain_augmentation": dom,
        "totalEpochs": total,
        "baselineProgress": (100.0 if row["status"] == "completed" else 0.0),
        "domainAugmentationProgress": (100.0 if row["status"] == "completed" and dom else (0.0 if dom else None)),
        "domainAugmentationText": ("已完成" if row["status"] == "completed" and dom else None),
        "augmentedProgress": (100.0 if row["status"] == "completed" and dom else (0.0 if dom else None)),
    }


def _fallback_progress_dict(fallback: dict) -> dict:
    dom = fallback["domain_augmentation"]
    total = fallback["totalEpochs"]
    return {
        "status": fallback["status"],
        "currentEpoch": total if fallback["status"] == "completed" else 0,
        "totalEpochs": total,
        "baseline": {"trainLossSeries": [], "valLossSeries": []},
        "augmented": ({"trainLossSeries": [], "valLossSeries": []} if dom else None),
        "baselineProgress": fallback.get("baselineProgress", 0.0),
        "domainAugmentationProgress": fallback.get("domainAugmentationProgress"),
        "domainAugmentationText": fallback.get("domainAugmentationText"),
        "augmentedProgress": fallback.get("augmentedProgress"),
        "message": fallback.get("message"),
    }


def _progress_for_task(task_id: str, fallback: dict) -> dict:
    task_dir = settings.data_dir / "tasks" / task_id
    raw = state.get_progress(task_id)
    if not raw:
        raw = read_progress_snapshot(task_dir) or {}
    return dict(raw) if raw else _fallback_progress_dict(fallback)


def _load_result_payload(task_id: str, row) -> dict[str, object]:
    if row["result_json"]:
        try:
            data = json.loads(row["result_json"])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    summary = read_training_summary(settings.data_dir / "tasks" / task_id)
    if isinstance(summary, dict) and isinstance(summary.get("result"), dict):
        return summary["result"]
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="没有结果数据")


def _build_metrics(data: dict[str, object]) -> ModelMetrics:
    return ModelMetrics(
        final_train_loss=float(data.get("finalTrainLoss") or 0.0),
        final_val_loss=float(data.get("finalValLoss") or 0.0),
        steering_error=float(data.get("steeringError") or 0.0),
        requested_epochs=(int(data["requestedEpochs"]) if data.get("requestedEpochs") is not None else None),
        completed_epochs=(int(data["completedEpochs"]) if data.get("completedEpochs") is not None else None),
        best_epoch=(int(data["bestEpoch"]) if data.get("bestEpoch") is not None else None),
        stopped_epoch=(int(data["stoppedEpoch"]) if data.get("stoppedEpoch") is not None else None),
        early_stopped=(bool(data["earlyStopped"]) if data.get("earlyStopped") is not None else None),
        best_val_loss=(float(data["bestValLoss"]) if data.get("bestValLoss") is not None else None),
        final_test_loss=(float(data["finalTestLoss"]) if data.get("finalTestLoss") is not None else None),
        used_dedicated_test_split=(
            bool(data["usedDedicatedTestSplit"]) if data.get("usedDedicatedTestSplit") is not None else None
        ),
        val_stress_mae=(float(data["valStressMAE"]) if data.get("valStressMAE") is not None else None),
        model_variant=(str(data["modelVariant"]) if data.get("modelVariant") is not None else None),
        num_frames=(int(data["numFrames"]) if data.get("numFrames") is not None else None),
        frame_stride=(int(data["frameStride"]) if data.get("frameStride") is not None else None),
        note=(str(data["note"]) if data.get("note") else None),
    )


@router.get("/tasks", response_model=list[TrainingTaskSummary])
def list_tasks(user: CurrentUser, project_id: str | None = Query(default=None, alias="projectId")):
    rows = db.list_tasks_for_user(user["id"], project_id)
    return [TrainingTaskSummary(**db.task_row_to_summary(row)) for row in rows]


@router.get("/tasks/{task_id}", response_model=TrainingTaskSummary)
def get_task_detail(task_id: str, user: CurrentUser):
    row = db.get_task(task_id, user["id"])
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
    return TrainingTaskSummary(**db.task_row_to_summary(row))


@router.post("/tasks", response_model=TrainingTaskSummary)
def create_task(body: CreateTaskBody, user: CurrentUser):
    project = db.get_project(body.project_id, user["id"])
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="项目不存在")

    dataset = db.get_dataset(body.dataset_id, user["id"], body.project_id)
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="数据集不存在或不属于该项目")

    dataset_b = None
    if body.domain_augmentation:
        if not body.domain_b_dataset_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="开启域增强时必须选择 B 域数据集")
        dataset_b = db.get_dataset(body.domain_b_dataset_id, user["id"], body.project_id)
        if dataset_b is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="B 域数据集不存在或不属于该项目")
        if body.domain_b_dataset_id == body.dataset_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="A 域和 B 域数据集不能相同")

    task_id = db.insert_task(
        user["id"],
        body.project_id,
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
            "datasetName": dataset["name"],
            "projectId": body.project_id,
            "projectName": project["name"],
            "modelVariant": body.model_variant,
            "domainAugmentation": body.domain_augmentation,
            "domainBDatasetId": body.domain_b_dataset_id,
            "domainBDatasetName": (dataset_b["name"] if dataset_b is not None else None),
            "cycleGanEpochs": body.cyclegan_epochs,
            "cycleGanDecayEpochs": body.cyclegan_decay_epochs,
            "cycleGanBatchSize": body.cyclegan_batch_size,
            "cycleGanSaveEpochFreq": body.cyclegan_save_epoch_freq,
            "cycleGanSaveLatestFreq": body.cyclegan_save_latest_freq,
            "cycleGanLoadSize": body.cyclegan_load_size,
            "cycleGanCropSize": body.cyclegan_crop_size,
            "cycleGanLambdaIdentity": body.cyclegan_lambda_identity,
        },
    )
    artifacts = settings.data_dir / "tasks" / task_id
    artifacts.mkdir(parents=True, exist_ok=True)

    state.register_task(task_id)
    thread = threading.Thread(
        target=training_worker,
        kwargs={
            "task_id": task_id,
            "user_id": user["id"],
            "dataset_root": dataset["root_dir"],
            "learning_rate": body.learning_rate,
            "batch_size": body.batch_size,
            "epochs": body.epochs,
            "domain_augmentation": body.domain_augmentation,
            "dataset_b_root": (dataset_b["root_dir"] if dataset_b is not None else None),
            "cyclegan_epochs": body.cyclegan_epochs,
            "cyclegan_decay_epochs": body.cyclegan_decay_epochs,
            "cyclegan_batch_size": body.cyclegan_batch_size,
            "cyclegan_save_epoch_freq": body.cyclegan_save_epoch_freq,
            "cyclegan_save_latest_freq": body.cyclegan_save_latest_freq,
            "cyclegan_load_size": body.cyclegan_load_size,
            "cyclegan_crop_size": body.cyclegan_crop_size,
            "cyclegan_lambda_identity": body.cyclegan_lambda_identity,
            "model_variant": body.model_variant,
            "artifacts_dir": artifacts,
        },
        daemon=True,
    )
    thread.start()

    row = db.get_task(task_id, user["id"])
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
    artifacts = settings.data_dir / "tasks" / task_id
    if artifacts.is_dir():
        shutil.rmtree(artifacts, ignore_errors=True)
    return None


@router.get("/tasks/{task_id}/progress", response_model=TaskProgress)
def get_task_progress(task_id: str, user: CurrentUser):
    row = db.get_task(task_id, user["id"])
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
    fallback = _fallback_from_row(row)
    raw = _progress_for_task(task_id, fallback)
    return build_task_progress(raw, fallback)


@router.post("/tasks/{task_id}/pause", status_code=status.HTTP_204_NO_CONTENT)
def pause_or_resume(task_id: str, user: CurrentUser):
    row = db.get_task(task_id, user["id"])
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
    ctrl = state.get_controls(task_id)
    if ctrl is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="任务当前不可暂停")
    if row["status"] == "running":
        ctrl.pause.set()
        db.update_task_status(task_id, user["id"], "paused", None)
        return None
    if row["status"] == "paused":
        ctrl.pause.clear()
        db.update_task_status(task_id, user["id"], "running", None)
        return None
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="当前状态不支持暂停/继续")


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
    data = _load_result_payload(task_id, row)

    baseline = data.get("baseline")
    if not isinstance(baseline, dict):
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="结果数据缺少 baseline")
    augmented = data.get("augmented")

    return TaskResultSummary(
        baseline=_build_metrics(baseline),
        augmented=(_build_metrics(augmented) if isinstance(augmented, dict) else None),
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
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_model = infer_svc.load_checkpoint_model(Path(row["baseline_ckpt"]), device)
    baseline_angle = infer_svc.predict_image(baseline_model, bgr, device)

    augmented_angle: float | None = None
    if row["augmented_ckpt"]:
        augmented_path = Path(row["augmented_ckpt"])
        if augmented_path.is_file():
            augmented_model = infer_svc.load_checkpoint_model(augmented_path, device)
            augmented_angle = infer_svc.predict_image(augmented_model, bgr, device)

    return CompareInferOut(
        baseline_steering=baseline_angle,
        augmented_steering=augmented_angle,
    )


@router.get("/tasks/{task_id}/download")
def download_model(
    task_id: str,
    user: CurrentUser,
    stage: str = Query(..., description="baseline or augmented"),
):
    row = db.get_task(task_id, user["id"])
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
    if row["status"] != "completed":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="任务未完成")
    if stage not in ("baseline", "augmented"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="stage 必须为 baseline 或 augmented")
    path_str = row["baseline_ckpt"] if stage == "baseline" else row["augmented_ckpt"]
    if not path_str:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="该阶段模型不存在")
    path = Path(path_str)
    if not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="文件已丢失")
    return FileResponse(path, filename=f"{stage}_checkpoint.pth", media_type="application/octet-stream")


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
            fallback = _fallback_from_row(row)
            raw = _progress_for_task(task_id, fallback)
            progress = build_task_progress(raw, fallback)
            await websocket.send_text(progress.model_dump_json(by_alias=True))
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
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="域增强对比数据损坏") from exc
    return [
        DomainAugPairOut(index=int(item["index"]), a_name=str(item["aName"]), c_name=str(item["cName"]))
        for item in data
    ]


@router.get("/tasks/{task_id}/domain-aug/image")
def get_domain_aug_image(
    task_id: str,
    user: CurrentUser,
    index: int = Query(..., ge=0),
    kind: str = Query(..., description="a or c"),
):
    row = db.get_task(task_id, user["id"])
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
    pairs_path = settings.data_dir / "tasks" / task_id / "domain_aug_pairs.json"
    if not pairs_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="暂无域增强对比数据")
    data = json.loads(pairs_path.read_text(encoding="utf-8"))
    target = next((item for item in data if int(item.get("index", -1)) == index), None)
    if target is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="对比索引不存在")
    if kind not in ("a", "c"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="kind 必须为 a 或 c")
    path_key = "aPath" if kind == "a" else "cPath"
    name_key = "aName" if kind == "a" else "cName"
    path = Path(str(target[path_key]))
    if not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="图像文件不存在")
    media = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(path, media_type=media, filename=str(target[name_key]))
