from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.core.config import settings


def safe_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    return suffix if suffix else ".mp4"


async def save_upload(file: UploadFile) -> tuple[str, Path]:
    job_id = uuid4().hex
    target = settings.upload_dir / f"{job_id}{safe_suffix(file.filename or 'upload.mp4')}"
    with target.open("wb") as output:
        while chunk := await file.read(1024 * 1024):
            output.write(chunk)
    return job_id, target


def delete_job_media(job_id: str, source_path: str | None = None) -> None:
    candidates: set[Path] = set()
    if source_path:
        candidates.add(Path(source_path))

    for directory in [
        settings.upload_dir,
        settings.preview_dir,
        settings.evidence_dir,
        settings.plate_dir,
        settings.metadata_dir,
    ]:
        candidates.update(directory.glob(f"{job_id}*"))

    for path in candidates:
        delete_file_if_safe(path)


def delete_violation_media(evidence_image: str | None, plate_image: str | None) -> None:
    for media_path in [evidence_image, plate_image]:
        if media_path:
            delete_file_if_safe(path_from_media_url(media_path))


def path_from_media_url(media_path: str) -> Path:
    relative = media_path.removeprefix("/media/").lstrip("/\\")
    return settings.data_dir / relative


def delete_file_if_safe(path: Path) -> None:
    try:
        resolved = path.resolve()
    except OSError:
        return

    roots = [
        settings.upload_dir,
        settings.preview_dir,
        settings.evidence_dir,
        settings.plate_dir,
        settings.metadata_dir,
    ]
    if not any(is_relative_to(resolved, root.resolve()) for root in roots):
        return
    if resolved.is_file():
        resolved.unlink(missing_ok=True)


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
