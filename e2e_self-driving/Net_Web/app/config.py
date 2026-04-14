"""应用配置（可用环境变量覆盖）。"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "E2E AutoDrive API"
    secret_key: str = "change-me-in-production-use-long-random-string"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 * 7

    data_dir: Path = Path(__file__).resolve().parent.parent / "data_storage"
    database_url: str = "sqlite:///./data_storage/app.db"
    cyclegan_project_root: Path = Path(__file__).resolve().parent.parent.parent.parent / "pytorch-CycleGAN-and-pix2pix"
    competition_project_root: Path = Path(__file__).resolve().parent.parent.parent.parent / "e2e_competition"

    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def sqlite_path(self) -> Path:
        root = Path(__file__).resolve().parent.parent
        if self.database_url.startswith("sqlite:///./"):
            return root / self.database_url.replace("sqlite:///./", "")
        if self.database_url.startswith("sqlite:///"):
            return Path(self.database_url.replace("sqlite:///", ""))
        return root / "data_storage" / "app.db"

    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
