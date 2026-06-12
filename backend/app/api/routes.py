import json

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.schemas.models import DetectionSettings, DetectionSettingsUpdate, Job, ReviewUpdate, Violation
from app.services.pipeline import detection_metadata_path, process_uploaded_video
from app.services.repository import (
    create_job,
    delete_all_jobs,
    delete_job,
    delete_violation,
    get_job,
    get_job_storage,
    get_violation,
    list_job_storage,
    list_jobs,
    list_violations,
    update_violation_review,
)
from app.services.storage import delete_job_media, delete_violation_media, save_upload
from app.services.streaming import frame_hub

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def current_detection_settings() -> DetectionSettings:
    return DetectionSettings(
        object_confidence=settings.object_confidence,
        helmet_confidence=settings.helmet_confidence,
        plate_confidence=settings.plate_confidence,
        sample_every_seconds=settings.sample_every_seconds,
        max_violations_per_video=settings.max_violations_per_video,
        enable_ocr=settings.enable_ocr,
    )


@router.get("/settings", response_model=DetectionSettings)
def get_detection_settings() -> DetectionSettings:
    return current_detection_settings()


@router.patch("/settings", response_model=DetectionSettings)
def update_detection_settings(update: DetectionSettingsUpdate) -> DetectionSettings:
    updates = update.model_dump(exclude_unset=True)
    for key, value in updates.items():
        setattr(settings, key, value)
    return current_detection_settings()


@router.post("/videos/upload", response_model=Job)
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> Job:
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Please upload a video file")

    job_id, source_path = await save_upload(file)
    create_job(job_id, file.filename or source_path.name, str(source_path))
    background_tasks.add_task(process_uploaded_video, job_id, str(source_path))

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=500, detail="Job was not created")
    return Job(**job)


@router.get("/jobs", response_model=list[Job])
def jobs() -> list[Job]:
    return [Job(**job) for job in list_jobs()]


@router.delete("/jobs")
def clear_jobs() -> dict[str, int]:
    records = list_job_storage()
    deleted = delete_all_jobs()
    for record in records:
        delete_job_media(record["id"], record.get("source_path"))
        frame_hub.close(record["id"])
    return {"deleted": deleted}


@router.get("/jobs/{job_id}", response_model=Job)
def job(job_id: str) -> Job:
    record = get_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    return Job(**record)


@router.delete("/jobs/{job_id}")
def remove_job(job_id: str) -> dict[str, str]:
    record = get_job_storage(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    if not delete_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    delete_job_media(job_id, record.get("source_path"))
    frame_hub.close(job_id)
    return {"status": "deleted"}


@router.get("/jobs/{job_id}/stream")
def job_stream(job_id: str) -> StreamingResponse:
    if not get_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    return StreamingResponse(
        frame_hub.stream(job_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/jobs/{job_id}/detections")
def job_detections(job_id: str) -> dict:
    if not get_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")

    path = detection_metadata_path(job_id)
    if not path.exists():
        return {"frames": []}
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/violations", response_model=list[Violation])
def violations(limit: int = 50) -> list[Violation]:
    return [Violation(**record) for record in list_violations(limit=limit)]


@router.delete("/violations/{violation_id}")
def remove_violation(violation_id: str) -> dict[str, str]:
    record = get_violation(violation_id)
    if not record:
        raise HTTPException(status_code=404, detail="Violation not found")
    if not delete_violation(violation_id):
        raise HTTPException(status_code=404, detail="Violation not found")
    delete_violation_media(record.get("evidence_image"), record.get("plate_image"))
    return {"status": "deleted"}


@router.patch("/violations/{violation_id}/review", response_model=Violation)
def review_violation(violation_id: str, update: ReviewUpdate) -> Violation:
    allowed = {"pending", "confirmed", "false_positive"}
    if update.review_status not in allowed:
        raise HTTPException(status_code=400, detail="review_status must be pending, confirmed, or false_positive")
    record = update_violation_review(violation_id, update.review_status)
    if not record:
        raise HTTPException(status_code=404, detail="Violation not found")
    return Violation(**record)
