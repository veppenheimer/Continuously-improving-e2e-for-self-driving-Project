"""SQLite 持久化（用户、数据集、任务元数据）。"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from app.config import settings


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    path = settings.sqlite_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=MEMORY")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db() -> Iterator[sqlite3.Connection]:
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with get_db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                is_default INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id),
                UNIQUE (user_id, name)
            );
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                name TEXT NOT NULL,
                root_dir TEXT NOT NULL,
                image_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (project_id) REFERENCES projects(id)
            );
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                status TEXT NOT NULL,
                learning_rate REAL NOT NULL,
                batch_size INTEGER NOT NULL,
                epochs INTEGER NOT NULL,
                domain_augmentation INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                display_name TEXT NOT NULL DEFAULT '',
                task_params_json TEXT,
                message TEXT,
                baseline_ckpt TEXT,
                augmented_ckpt TEXT,
                result_json TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (project_id) REFERENCES projects(id),
                FOREIGN KEY (dataset_id) REFERENCES datasets(id)
            );
            """
        )
        _migrate_tasks_display_name(conn)
        _migrate_tasks_params_json(conn)
        _migrate_datasets_project_id(conn)
        _migrate_tasks_project_id(conn)
        _backfill_default_projects(conn)


def _migrate_tasks_display_name(conn: sqlite3.Connection) -> None:
    cur = conn.execute("PRAGMA table_info(tasks)")
    cols = {row[1] for row in cur.fetchall()}
    if "display_name" not in cols:
        conn.execute("ALTER TABLE tasks ADD COLUMN display_name TEXT NOT NULL DEFAULT ''")


def _migrate_tasks_params_json(conn: sqlite3.Connection) -> None:
    cur = conn.execute("PRAGMA table_info(tasks)")
    cols = {row[1] for row in cur.fetchall()}
    if "task_params_json" not in cols:
        conn.execute("ALTER TABLE tasks ADD COLUMN task_params_json TEXT")


def _migrate_datasets_project_id(conn: sqlite3.Connection) -> None:
    cur = conn.execute("PRAGMA table_info(datasets)")
    cols = {row[1] for row in cur.fetchall()}
    if "project_id" not in cols:
        conn.execute("ALTER TABLE datasets ADD COLUMN project_id TEXT")


def _migrate_tasks_project_id(conn: sqlite3.Connection) -> None:
    cur = conn.execute("PRAGMA table_info(tasks)")
    cols = {row[1] for row in cur.fetchall()}
    if "project_id" not in cols:
        conn.execute("ALTER TABLE tasks ADD COLUMN project_id TEXT")


def _ensure_default_project(conn: sqlite3.Connection, user_id: str) -> str:
    cur = conn.execute(
        "SELECT id FROM projects WHERE user_id = ? AND is_default = 1 ORDER BY created_at ASC LIMIT 1",
        (user_id,),
    )
    row = cur.fetchone()
    if row is not None:
        return str(row["id"])
    pid = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO projects (id, user_id, name, is_default, created_at)
           VALUES (?, ?, ?, 1, ?)""",
        (pid, user_id, "默认项目", _utc_now()),
    )
    return pid


def _backfill_default_projects(conn: sqlite3.Connection) -> None:
    users = conn.execute("SELECT id FROM users").fetchall()
    for u in users:
        user_id = str(u["id"])
        default_pid = _ensure_default_project(conn, user_id)
        conn.execute(
            "UPDATE datasets SET project_id = ? WHERE user_id = ? AND (project_id IS NULL OR project_id = '')",
            (default_pid, user_id),
        )
        conn.execute(
            """
            UPDATE tasks
            SET project_id = (
                SELECT d.project_id FROM datasets d WHERE d.id = tasks.dataset_id
            )
            WHERE user_id = ? AND (project_id IS NULL OR project_id = '')
            """,
            (user_id,),
        )
        conn.execute(
            "UPDATE tasks SET project_id = ? WHERE user_id = ? AND (project_id IS NULL OR project_id = '')",
            (default_pid, user_id),
        )


def create_user(username: str, password_hash: str, email: Optional[str]) -> dict[str, Any]:
    uid = str(uuid.uuid4())
    with get_db() as conn:
        conn.execute(
            "INSERT INTO users (id, username, password_hash, email, created_at) VALUES (?,?,?,?,?)",
            (uid, username, password_hash, email, _utc_now()),
        )
        _ensure_default_project(conn, uid)
    return {"id": uid, "username": username, "email": email}


def list_projects_for_user(user_id: str) -> list[dict[str, Any]]:
    with get_db() as conn:
        _ensure_default_project(conn, user_id)
        rows = conn.execute(
            "SELECT id, name, created_at FROM projects WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
    return [{"id": r["id"], "name": r["name"], "createdAt": r["created_at"]} for r in rows]


def get_project(project_id: str, user_id: str) -> Optional[sqlite3.Row]:
    with get_db() as conn:
        cur = conn.execute("SELECT * FROM projects WHERE id = ? AND user_id = ?", (project_id, user_id))
        return cur.fetchone()


def get_default_project(user_id: str) -> dict[str, Any]:
    with get_db() as conn:
        pid = _ensure_default_project(conn, user_id)
        row = conn.execute(
            "SELECT id, name, created_at FROM projects WHERE id = ?",
            (pid,),
        ).fetchone()
    assert row is not None
    return {"id": row["id"], "name": row["name"], "createdAt": row["created_at"]}


def insert_project(user_id: str, name: str) -> dict[str, Any]:
    pid = str(uuid.uuid4())
    ts = _utc_now()
    with get_db() as conn:
        _ensure_default_project(conn, user_id)
        conn.execute(
            "INSERT INTO projects (id, user_id, name, is_default, created_at) VALUES (?, ?, ?, 0, ?)",
            (pid, user_id, name.strip(), ts),
        )
    return {"id": pid, "name": name.strip(), "createdAt": ts}


def rename_project(project_id: str, user_id: str, name: str) -> bool:
    with get_db() as conn:
        cur = conn.execute(
            "UPDATE projects SET name = ? WHERE id = ? AND user_id = ?",
            (name.strip(), project_id, user_id),
        )
        return cur.rowcount > 0


def delete_project(project_id: str, user_id: str) -> bool:
    with get_db() as conn:
        cur = conn.execute(
            "DELETE FROM projects WHERE id = ? AND user_id = ? AND is_default = 0",
            (project_id, user_id),
        )
        return cur.rowcount > 0


def get_user_by_username(username: str) -> Optional[sqlite3.Row]:
    with get_db() as conn:
        cur = conn.execute("SELECT * FROM users WHERE username = ?", (username,))
        return cur.fetchone()


def get_user_by_id(user_id: str) -> Optional[sqlite3.Row]:
    with get_db() as conn:
        cur = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        return cur.fetchone()


def insert_dataset(
    user_id: str,
    project_id: str,
    name: str,
    root_dir: str,
    image_count: int,
    dataset_id: str | None = None,
) -> dict[str, Any]:
    did = dataset_id or str(uuid.uuid4())
    ts = _utc_now()
    with get_db() as conn:
        conn.execute(
            """INSERT INTO datasets (id, user_id, project_id, name, root_dir, image_count, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (did, user_id, project_id, name, root_dir, image_count, ts),
        )
    return {
        "id": did,
        "projectId": project_id,
        "name": name,
        "imageCount": image_count,
        "createdAt": ts,
        "root_dir": root_dir,
    }


def list_datasets_for_user(user_id: str, project_id: str | None = None) -> list[dict[str, Any]]:
    with get_db() as conn:
        _ensure_default_project(conn, user_id)
        if project_id:
            cur = conn.execute(
                """SELECT id, project_id, name, image_count, created_at
                   FROM datasets
                   WHERE user_id = ? AND project_id = ?
                   ORDER BY created_at DESC""",
                (user_id, project_id),
            )
        else:
            cur = conn.execute(
                """SELECT id, project_id, name, image_count, created_at
                   FROM datasets
                   WHERE user_id = ?
                   ORDER BY created_at DESC""",
                (user_id,),
            )
        rows = cur.fetchall()
    return [
        {
            "id": r["id"],
            "projectId": r["project_id"],
            "name": r["name"],
            "imageCount": r["image_count"],
            "createdAt": r["created_at"],
        }
        for r in rows
    ]


def get_dataset(dataset_id: str, user_id: str, project_id: str | None = None) -> Optional[sqlite3.Row]:
    with get_db() as conn:
        if project_id:
            cur = conn.execute(
                "SELECT * FROM datasets WHERE id = ? AND user_id = ? AND project_id = ?",
                (dataset_id, user_id, project_id),
            )
        else:
            cur = conn.execute(
                "SELECT * FROM datasets WHERE id = ? AND user_id = ?",
                (dataset_id, user_id),
            )
        return cur.fetchone()


def dataset_task_count(dataset_id: str, user_id: str, project_id: str | None = None) -> int:
    with get_db() as conn:
        if project_id:
            cur = conn.execute(
                "SELECT COUNT(1) AS cnt FROM tasks WHERE dataset_id = ? AND user_id = ? AND project_id = ?",
                (dataset_id, user_id, project_id),
            )
        else:
            cur = conn.execute(
                "SELECT COUNT(1) AS cnt FROM tasks WHERE dataset_id = ? AND user_id = ?",
                (dataset_id, user_id),
            )
        row = cur.fetchone()
        return int(row["cnt"]) if row is not None else 0


def delete_dataset(dataset_id: str, user_id: str, project_id: str | None = None) -> bool:
    with get_db() as conn:
        if project_id:
            cur = conn.execute(
                "DELETE FROM datasets WHERE id = ? AND user_id = ? AND project_id = ?",
                (dataset_id, user_id, project_id),
            )
        else:
            cur = conn.execute("DELETE FROM datasets WHERE id = ? AND user_id = ?", (dataset_id, user_id))
        return cur.rowcount > 0


def insert_task(
    user_id: str,
    project_id: str,
    dataset_id: str,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    domain_augmentation: bool,
    display_name: Optional[str] = None,
    task_params: Optional[dict[str, Any]] = None,
) -> str:
    tid = str(uuid.uuid4())
    label = (display_name or "").strip()[:128]
    stored_name = label if label else f"训练 {tid[:8]}"
    with get_db() as conn:
        conn.execute(
            """INSERT INTO tasks (
                id, user_id, project_id, dataset_id, status, learning_rate, batch_size, epochs,
                domain_augmentation, created_at, display_name, task_params_json, message
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                tid,
                user_id,
                project_id,
                dataset_id,
                "pending",
                learning_rate,
                batch_size,
                epochs,
                1 if domain_augmentation else 0,
                _utc_now(),
                stored_name,
                (json.dumps(task_params, ensure_ascii=False) if task_params else None),
                None,
            ),
        )
    return tid


def delete_task(task_id: str, user_id: str) -> bool:
    with get_db() as conn:
        cur = conn.execute("DELETE FROM tasks WHERE id = ? AND user_id = ?", (task_id, user_id))
        return cur.rowcount > 0


def update_task_status(task_id: str, user_id: str, status: str, message: Optional[str] = None) -> None:
    with get_db() as conn:
        if message is None:
            conn.execute(
                "UPDATE tasks SET status = ? WHERE id = ? AND user_id = ?",
                (status, task_id, user_id),
            )
        else:
            conn.execute(
                "UPDATE tasks SET status = ?, message = ? WHERE id = ? AND user_id = ?",
                (status, message, task_id, user_id),
            )


def update_task_checkpoints(
    task_id: str,
    user_id: str,
    baseline_ckpt: Optional[str] = None,
    augmented_ckpt: Optional[str] = None,
    result_json: Optional[str] = None,
    status: Optional[str] = None,
) -> None:
    fields: list[str] = []
    vals: list[Any] = []
    if baseline_ckpt is not None:
        fields.append("baseline_ckpt = ?")
        vals.append(baseline_ckpt)
    if augmented_ckpt is not None:
        fields.append("augmented_ckpt = ?")
        vals.append(augmented_ckpt)
    if result_json is not None:
        fields.append("result_json = ?")
        vals.append(result_json)
    if status is not None:
        fields.append("status = ?")
        vals.append(status)
    if not fields:
        return
    vals.extend([task_id, user_id])
    sql = f"UPDATE tasks SET {', '.join(fields)} WHERE id = ? AND user_id = ?"
    with get_db() as conn:
        conn.execute(sql, vals)


def get_task(task_id: str, user_id: str) -> Optional[sqlite3.Row]:
    with get_db() as conn:
        cur = conn.execute(
            """SELECT t.*, d.name AS dataset_name, p.name AS project_name FROM tasks t
               JOIN datasets d ON d.id = t.dataset_id
               LEFT JOIN projects p ON p.id = t.project_id
               WHERE t.id = ? AND t.user_id = ?""",
            (task_id, user_id),
        )
        return cur.fetchone()


def list_tasks_for_user(user_id: str, project_id: str | None = None) -> list[sqlite3.Row]:
    with get_db() as conn:
        _ensure_default_project(conn, user_id)
        if project_id:
            cur = conn.execute(
                """SELECT t.*, d.name AS dataset_name, p.name AS project_name FROM tasks t
                   JOIN datasets d ON d.id = t.dataset_id
                   LEFT JOIN projects p ON p.id = t.project_id
                   WHERE t.user_id = ? AND t.project_id = ?
                   ORDER BY t.created_at DESC""",
                (user_id, project_id),
            )
        else:
            cur = conn.execute(
                """SELECT t.*, d.name AS dataset_name, p.name AS project_name FROM tasks t
               JOIN datasets d ON d.id = t.dataset_id
               LEFT JOIN projects p ON p.id = t.project_id
               WHERE t.user_id = ?
               ORDER BY t.created_at DESC""",
                (user_id,),
            )
        return cur.fetchall()


def list_tasks_for_project(user_id: str, project_id: str) -> list[sqlite3.Row]:
    with get_db() as conn:
        cur = conn.execute(
            "SELECT * FROM tasks WHERE user_id = ? AND project_id = ? ORDER BY created_at DESC",
            (user_id, project_id),
        )
        return cur.fetchall()


def list_datasets_for_project(user_id: str, project_id: str) -> list[sqlite3.Row]:
    with get_db() as conn:
        cur = conn.execute(
            "SELECT * FROM datasets WHERE user_id = ? AND project_id = ? ORDER BY created_at DESC",
            (user_id, project_id),
        )
        return cur.fetchall()


def _task_display_name(row: sqlite3.Row) -> str:
    try:
        raw = row["display_name"]
    except (KeyError, IndexError):
        raw = ""
    s = (raw or "").strip()
    return s if s else str(row["id"])


def task_row_to_summary(row: sqlite3.Row) -> dict[str, Any]:
    result_summary = None
    if row["result_json"]:
        try:
            result_summary = json.loads(row["result_json"])
        except json.JSONDecodeError:
            result_summary = None
    params = {
        "learningRate": row["learning_rate"],
        "batchSize": row["batch_size"],
        "epochs": row["epochs"],
        "datasetId": row["dataset_id"],
        "datasetName": row["dataset_name"],
        "projectId": row["project_id"],
        "projectName": row["project_name"],
    }
    raw_params = row["task_params_json"] if "task_params_json" in row.keys() else None
    if raw_params:
        try:
            decoded = json.loads(raw_params)
            if isinstance(decoded, dict):
                params.update(decoded)
        except json.JSONDecodeError:
            pass
    return {
        "id": row["id"],
        "project_id": row["project_id"],
        "name": _task_display_name(row),
        "status": row["status"],
        "created_at": row["created_at"],
        "domain_augmentation": bool(row["domain_augmentation"]),
        "params": params,
        "result_summary": result_summary,
    }
