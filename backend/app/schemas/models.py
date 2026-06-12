from pydantic import BaseModel, Field


class Job(BaseModel):
    id: str
    filename: str
    status: str
    message: str | None = None
    progress: float = 0
    current_frame: int = 0
    total_frames: int = 0
    sampled_frames: int = 0
    violation_count: int = 0
    elapsed_seconds: float = 0
    processing_fps: float = 0
    eta_seconds: float = 0
    preview_image: str | None = None
    source_video: str | None = None
    result: str | None = None
    created_at: str
    updated_at: str


class Violation(BaseModel):
    id: str
    job_id: str
    detected_at: str
    helmet_status: str
    helmet_confidence: float
    plate_text: str | None = None
    plate_confidence: float | None = None
    evidence_image: str
    plate_image: str | None = None
    frame_number: int | None = None
    track_id: int | None = None
    review_status: str = "pending"


class ReviewUpdate(BaseModel):
    review_status: str


class DetectionSettings(BaseModel):
    object_confidence: float
    helmet_confidence: float
    plate_confidence: float
    sample_every_seconds: float
    max_violations_per_video: int
    enable_ocr: bool


class DetectionSettingsUpdate(BaseModel):
    object_confidence: float | None = Field(default=None, ge=0.05, le=0.95)
    helmet_confidence: float | None = Field(default=None, ge=0.05, le=0.95)
    plate_confidence: float | None = Field(default=None, ge=0.05, le=0.95)
    sample_every_seconds: float | None = Field(default=None, ge=0.25, le=10)
    max_violations_per_video: int | None = Field(default=None, ge=1, le=200)
    enable_ocr: bool | None = None
