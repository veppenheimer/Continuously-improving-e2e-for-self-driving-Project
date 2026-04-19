"""项目管理：增删改查。"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from app import database as db
from app.config import settings
from app.deps import CurrentUser
from app import state
from app.schemas import CreateProjectBody, ProjectItemOut, RenameProjectBody

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("", response_model=list[ProjectItemOut])
def list_projects(user: CurrentUser):
    rows = db.list_projects_for_user(user["id"])
    return [ProjectItemOut(id=r["id"], name=r["name"], created_at=r["createdAt"]) for r in rows]


@router.post("", response_model=ProjectItemOut)
def create_project(body: CreateProjectBody, user: CurrentUser):
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="项目名不能为空")
    try:
        row = db.insert_project(user["id"], name)
    except sqlite3.IntegrityError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="项目名已存在") from e
    return ProjectItemOut(id=row["id"], name=row["name"], created_at=row["createdAt"])


@router.patch("/{project_id}", response_model=ProjectItemOut)
def rename_project(project_id: str, body: RenameProjectBody, user: CurrentUser):
    row = db.get_project(project_id, user["id"])
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="项目不存在")
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="项目名不能为空")
    try:
        ok = db.rename_project(project_id, user["id"], name)
    except sqlite3.IntegrityError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="项目名已存在") from e
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="项目不存在")
    updated = db.get_project(project_id, user["id"])
    assert updated is not None
    return ProjectItemOut(id=updated["id"], name=updated["name"], created_at=updated["created_at"])


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project_id: str, user: CurrentUser):
    row = db.get_project(project_id, user["id"])
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="项目不存在")
    if int(row["is_default"]) == 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="默认项目不允许删除")

    task_rows = db.list_tasks_for_project(user["id"], project_id)
    dataset_rows = db.list_datasets_for_project(user["id"], project_id)

    for t in task_rows:
        task_id = str(t["id"])
        ctrl = state.get_controls(task_id)
        if ctrl is not None:
            ctrl.stop.set()
        state.unregister_task(task_id)
        db.delete_task(task_id, user["id"])
        art = settings.data_dir / "tasks" / task_id
        if art.is_dir():
            shutil.rmtree(art, ignore_errors=True)

    for d in dataset_rows:
        did = str(d["id"])
        db.delete_dataset(did, user["id"], project_id)
        root = Path(str(d["root_dir"]))
        if root.is_dir():
            shutil.rmtree(root, ignore_errors=True)

    ok = db.delete_project(project_id, user["id"])
    if not ok:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="项目删除失败")
    return None

