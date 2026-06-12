from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    project_root: Path = Path(__file__).resolve().parents[3]
    data_dir: Path = project_root / "data"
    cache_dir: Path = project_root / ".cache"
    upload_dir: Path = data_dir / "uploads"
    evidence_dir: Path = data_dir / "evidence"
    plate_dir: Path = data_dir / "plates"
    preview_dir: Path = data_dir / "previews"
    metadata_dir: Path = data_dir / "metadata"
    database_path: Path = project_root / "database" / "saferide.db"
    object_model_path: Path = project_root / "models" / "yolo11s.pt"
    helmet_model_path: Path = project_root / "models" / "helmet-yolov8n.pt"
    plate_model_path: Path = project_root / "models" / "license-plate-yolo11n.pt"
    allowed_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    sample_every_seconds: float = 1
    live_preview_fps: int = 12
    realtime_preview: bool = True
    violation_cooldown_seconds: int = 4
    plate_aggregation_seconds: float = 2
    plate_aggregation_min_samples: int = 3
    tracker_high_confidence: float = 0.25
    tracker_low_confidence: float = 0.10
    tracker_new_track_confidence: float = 0.25
    tracker_match_threshold: float = 0.25
    tracker_max_lost_seconds: float = 3
    max_violations_per_video: int = 25
    preview_every_samples: int = 1
    object_imgsz: int = 960
    helmet_imgsz: int = 960
    plate_imgsz: int = 960
    object_confidence: float = 0.35
    helmet_confidence: float = 0.35
    plate_confidence: float = 0.30
    enable_ocr: bool = True
    ocr_languages: list[str] = ["th", "en"]
    ocr_gpu: bool = False

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    for path in [
        settings.cache_dir,
        settings.cache_dir / "ultralytics",
        settings.cache_dir / "easyocr",
        settings.upload_dir,
        settings.evidence_dir,
        settings.plate_dir,
        settings.preview_dir,
        settings.metadata_dir,
        settings.database_path.parent,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    return settings


settings = get_settings()
