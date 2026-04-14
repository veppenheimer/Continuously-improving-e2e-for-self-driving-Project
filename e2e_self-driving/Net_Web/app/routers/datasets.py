"""数据集上传与列表。"""

import shutil
import uuid
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app import database as db
from app.config import settings
from app.deps import CurrentUser
from app.schemas import DatasetItemOut
from app.services.dataset_ingest import ingest_zip_to_folder

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("", response_model=list[DatasetItemOut])
def list_datasets(user: CurrentUser):
    rows = db.list_datasets_for_user(user["id"])
    return [
        DatasetItemOut(
            id=r["id"],
            name=r["name"],
            image_count=r["imageCount"],
            created_at=r["createdAt"],
        )
        for r in rows
    ]


@router.post("/upload", response_model=DatasetItemOut)
async def upload_dataset(
    user: CurrentUser,
    file: Annotated[UploadFile, File(...)],
    name: Annotated[str | None, Form()] = None,
):
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="请上传 .zip 文件")
    did = str(uuid.uuid4())
    base = settings.data_dir / "datasets" / did
    try:
        meta = await ingest_zip_to_folder(base, file, name or "")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    row = db.insert_dataset(
        user["id"],
        meta["name"],
        meta["root_dir"],
        meta["image_count"],
        dataset_id=did,
    )
    return DatasetItemOut(
        id=row["id"],
        name=row["name"],
        image_count=row["imageCount"],
        created_at=row["createdAt"],
    )


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(dataset_id: str, user: CurrentUser):
    row = db.get_dataset(dataset_id, user["id"])
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="数据集不存在")

    using_count = db.dataset_task_count(dataset_id, user["id"])
    if using_count > 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="该数据集已被训练任务引用，暂不支持删除",
        )

    db.delete_dataset(dataset_id, user["id"])
    root_dir = row["root_dir"]
    if root_dir:
        shutil.rmtree(root_dir, ignore_errors=True)
    return None
