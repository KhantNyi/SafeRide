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
    video_orientation_auto: bool = True
    sample_every_seconds: float = 1
    adaptive_sampling: bool = True
    adaptive_sample_divisor: int = 5
    adaptive_hold_seconds: float = 2.5
    live_preview_fps: int = 12
    realtime_preview: bool = False
    metadata_write_seconds: float = 1.0
    violation_cooldown_seconds: int = 4
    plate_collection_seconds: float = 6
    plate_candidate_limit: int = 5
    plate_ocr_candidate_limit: int = 3
    tracker_high_confidence: float = 0.25
    tracker_low_confidence: float = 0.10
    tracker_new_track_confidence: float = 0.25
    tracker_match_threshold: float = 0.25
    tracker_max_lost_seconds: float = 3
    tracker_appearance_weight: float = 0.30
    live_max_seconds: int = 900
    rider_dedupe_seconds: float = 12
    rider_dedupe_match_threshold: float = 0.22
    min_helmet_person_score: float = 0.24
    min_person_motorcycle_score: float = 0.18
    min_helmet_motorcycle_score: float = 0.30
    min_no_helmet_association_score: float = 0.38
    min_plate_motorcycle_score: float = 0.28
    min_no_helmet_votes: int = 2
    plate_min_aspect: float = 0.55
    plate_max_aspect: float = 2.0
    plate_horizontal_slop: float = 0.22
    plate_assignment_margin: float = 0.04
    plate_min_track_sightings: int = 2
    max_violations_per_video: int = 25
    preview_every_samples: int = 1
    object_imgsz: int = 960
    helmet_imgsz: int = 960
    helmet_crop_inference: bool = True
    helmet_crop_imgsz: int = 640
    plate_imgsz: int = 960
    object_confidence: float = 0.35
    helmet_confidence: float = 0.35
    plate_confidence: float = 0.30
    model_device: str = "auto"
    enable_ocr: bool = True
    ocr_languages: list[str] = ["th", "en"]
    ocr_gpu: bool | None = None

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
