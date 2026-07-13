from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central config, read from environment / .env."""

    secret_key: str = "dev-secret-change-me-please-0123456789"
    anthropic_api_key: str = ""
    gemini_api_key: str = ""            # read from .env (git-ignored); never committed
    hf_api_token: str = ""             # Hugging Face Inference token (free tier); .env only
    hf_model: str = "meta-llama/Llama-3.1-8B-Instruct"  # open-source assistant model
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

    # Absolute path so keys load no matter where uvicorn is launched from (the sh file, an
    # IDE, etc.). Keys live only in this git-ignored file and are never printed or committed.
    model_config = SettingsConfigDict(env_file=str(BACKEND_ROOT / ".env"), extra="ignore")


settings = Settings()
settings.uploads_dir.mkdir(parents=True, exist_ok=True)
