from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    droidcam_url: str = "http://192.168.1.100:4747/video"
    model_paths: str = "yolov8n.pt"
    model_labels: str = ""
    conf_threshold: float = 0.35
    img_size: int = 640
    reconnect_delay: float = 3.0
    jpeg_quality: int = 80
    device: str = "cpu"

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent / ".env"),
        extra="ignore",
    )

    def model_path_list(self) -> list[str]:
        return [p.strip() for p in self.model_paths.split(",") if p.strip()]

    def model_label_list(self) -> list[str]:
        raw = [p.strip() for p in self.model_labels.split(",") if p.strip()]
        paths = self.model_path_list()
        # Pad/truncate to match paths.
        if len(raw) < len(paths):
            raw += ["" for _ in range(len(paths) - len(raw))]
        return raw[: len(paths)]


settings = Settings()
