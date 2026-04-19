"""FastAPI 入口：与 web 前端 `docs/BACKEND_API.md` 对齐。"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db
from app.routers import auth, datasets, projects, tasks


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "datasets").mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "tasks").mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(projects.router)
app.include_router(datasets.router)
app.include_router(tasks.router)


@app.get("/health")
def health():
    return {"status": "ok"}
