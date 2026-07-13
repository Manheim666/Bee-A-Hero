from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central config, read from environment / .env."""

    secret_key: str = "dev-secret-change-me-please-0123456789"
    anthropic_api_key: str = ""
    gemini_api_key: str = ""            # read from .env (git-ignored); never committed
    frontend_origin: str = "http://localhost:5173"

    # JWT
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24  # 1 day

    # Storage
    database_url: str = f"sqlite:///{BACKEND_ROOT / 'bee.db'}"
    uploads_dir: Path = BACKEND_ROOT / "uploads"

    # Uploads
    max_upload_mb: int = 200
    allowed_video_ext: tuple[str, ...] = (".mp4", ".mov", ".avi")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
settings.uploads_dir.mkdir(parents=True, exist_ok=True)
