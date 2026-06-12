from pathlib import Path

from app.core.config import settings
from app.core.database import get_connection, utc_now


def media_url_for_path(path: str | None) -> str | None:
    if not path:
        return None
    try:
        relative = Path(path).resolve().relative_to(settings.data_dir.resolve())
    except ValueError:
        return None
    return f"/media/{relative.as_posix()}"


def with_source_video(record: dict | None) -> dict | None:
    if not record:
        return None
    record["source_video"] = media_url_for_path(record.get("source_path"))
    record.pop("source_path", None)
    return record


def create_job(job_id: str, filename: str, source_path: str) -> None:
    now = utc_now()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO jobs (id, filename, source_path, status, message, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (job_id, filename, source_path, "queued", "Waiting to process", now, now),
        )


def update_job(
    job_id: str,
    status: str,
    message: str | None = None,
    *,
    progress: float | None = None,
    current_frame: int | None = None,
    total_frames: int | None = None,
    sampled_frames: int | None = None,
    violation_count: int | None = None,
    elapsed_seconds: float | None = None,
    processing_fps: float | None = None,
    eta_seconds: float | None = None,
    preview_image: str | None = None,
    result: str | None = None,
) -> None:
    fields = ["status = ?", "message = ?", "updated_at = ?"]
    values: list = [status, message, utc_now()]

    optional_fields = {
        "progress": progress,
        "current_frame": current_frame,
        "total_frames": total_frames,
        "sampled_frames": sampled_frames,
        "violation_count": violation_count,
        "elapsed_seconds": elapsed_seconds,
        "processing_fps": processing_fps,
        "eta_seconds": eta_seconds,
        "preview_image": preview_image,
        "result": result,
    }
    for name, value in optional_fields.items():
        if value is not None:
            fields.append(f"{name} = ?")
            values.append(value)
    values.append(job_id)

    with get_connection() as conn:
        conn.execute(
            f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?",
            values,
        )


def get_job(job_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT id, filename, source_path, status, message, progress, current_frame, total_frames,
                   sampled_frames, violation_count, elapsed_seconds, processing_fps, eta_seconds,
                   preview_image, result, created_at, updated_at
            FROM jobs
            WHERE id = ?
            """,
            (job_id,),
        ).fetchone()
    return with_source_video(dict(row)) if row else None


def get_job_storage(job_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT id, source_path, preview_image
            FROM jobs
            WHERE id = ?
            """,
            (job_id,),
        ).fetchone()
    return dict(row) if row else None


def list_jobs() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, filename, source_path, status, message, progress, current_frame, total_frames,
                   sampled_frames, violation_count, elapsed_seconds, processing_fps, eta_seconds,
                   preview_image, result, created_at, updated_at
            FROM jobs
            ORDER BY created_at DESC
            """
        ).fetchall()
    return [with_source_video(dict(row)) for row in rows]


def list_job_storage() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, source_path, preview_image
            FROM jobs
            """
        ).fetchall()
    return [dict(row) for row in rows]


def delete_job(job_id: str) -> bool:
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        conn.execute("DELETE FROM violations WHERE job_id = ?", (job_id,))
    return cursor.rowcount > 0


def delete_all_jobs() -> int:
    with get_connection() as conn:
        row = conn.execute("SELECT COUNT(*) AS count FROM jobs").fetchone()
        count = int(row["count"] if row else 0)
        conn.execute("DELETE FROM violations")
        conn.execute("DELETE FROM jobs")
    return count


def create_violation(record: dict) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO violations (
                id, job_id, detected_at, helmet_status, helmet_confidence,
                plate_text, plate_confidence, evidence_image, plate_image, frame_number, track_id, review_status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["id"],
                record["job_id"],
                record["detected_at"],
                record["helmet_status"],
                record["helmet_confidence"],
                record.get("plate_text"),
                record.get("plate_confidence"),
                record["evidence_image"],
                record.get("plate_image"),
                record.get("frame_number"),
                record.get("track_id"),
                record.get("review_status", "pending"),
            ),
        )


def list_violations(limit: int = 50) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, job_id, detected_at, helmet_status, helmet_confidence,
                   plate_text, plate_confidence, evidence_image, plate_image, frame_number, track_id, review_status
            FROM violations
            ORDER BY detected_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def update_violation_review(violation_id: str, review_status: str) -> dict | None:
    now = utc_now()
    with get_connection() as conn:
        cursor = conn.execute(
            """
            UPDATE violations
            SET review_status = ?
            WHERE id = ?
            """,
            (review_status, violation_id),
        )
        if not cursor.rowcount:
            return None
        row = conn.execute(
            """
            SELECT id, job_id, detected_at, helmet_status, helmet_confidence,
                   plate_text, plate_confidence, evidence_image, plate_image, frame_number, track_id, review_status
            FROM violations
            WHERE id = ?
            """,
            (violation_id,),
        ).fetchone()
        if row:
            conn.execute("UPDATE jobs SET updated_at = ? WHERE id = ?", (now, row["job_id"]))
    return dict(row) if row else None


def get_violation(violation_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT id, job_id, evidence_image, plate_image
            FROM violations
            WHERE id = ?
            """,
            (violation_id,),
        ).fetchone()
    return dict(row) if row else None


def delete_violation(violation_id: str) -> bool:
    with get_connection() as conn:
        job_id = get_violation_job_id(conn, violation_id)
        cursor = conn.execute("DELETE FROM violations WHERE id = ?", (violation_id,))
        if cursor.rowcount and job_id:
            conn.execute(
                """
                UPDATE jobs
                SET violation_count = MAX(violation_count - 1, 0), updated_at = ?
                WHERE id = ?
                """,
                (utc_now(), job_id),
            )
    return cursor.rowcount > 0


def get_violation_job_id(conn, violation_id: str) -> str | None:
    row = conn.execute("SELECT job_id FROM violations WHERE id = ?", (violation_id,)).fetchone()
    return row["job_id"] if row else None
