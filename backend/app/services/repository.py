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


VIOLATION_COLUMNS = (
    "id, job_id, detected_at, helmet_status, helmet_confidence, plate_text, plate_confidence, "
    "evidence_image, plate_image, frame_number, track_id, review_status, source, note, miss_reason"
)


def create_violation(record: dict) -> None:
    with get_connection() as conn:
        conn.execute(
            f"""
            INSERT INTO violations ({VIOLATION_COLUMNS})
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                record.get("source", "detected"),
                record.get("note"),
                record.get("miss_reason"),
            ),
        )


def list_violations(limit: int = 50) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT {VIOLATION_COLUMNS}
            FROM violations
            ORDER BY detected_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def increment_job_violation_count(job_id: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE jobs SET violation_count = violation_count + 1, updated_at = ? WHERE id = ?",
            (utc_now(), job_id),
        )


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
            f"""
            SELECT {VIOLATION_COLUMNS}
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


def review_metrics() -> dict:
    """Aggregate human review decisions into precision and recall metrics.

    Precision is confirmed / (confirmed + false_positive) over pipeline-detected
    records only, i.e. how often a saved violation survived human review.
    Pending records are excluded from precision but counted.

    Manual records (reviewer-reported misses) are counted separately and drive
    recall: confirmed / (confirmed + manual). Reviewers only report misses they
    notice, so this is an upper bound on true recall.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT v.job_id, v.review_status, v.helmet_confidence, v.source, j.filename
            FROM violations v
            LEFT JOIN jobs j ON j.id = v.job_id
            """
        ).fetchall()

    def empty_bucket() -> dict:
        return {"total": 0, "pending": 0, "confirmed": 0, "false_positive": 0, "manual": 0}

    overall = empty_bucket()
    jobs: dict[str, dict] = {}
    bands = {band: empty_bucket() for band in CONFIDENCE_BANDS}

    for row in rows:
        job_bucket = jobs.setdefault(
            row["job_id"],
            {"job_id": row["job_id"], "filename": row["filename"], **empty_bucket()},
        )
        if row["source"] == "manual":
            overall["manual"] += 1
            job_bucket["manual"] += 1
            continue
        status = row["review_status"]
        if status not in ("confirmed", "false_positive"):
            status = "pending"
        band_bucket = bands[confidence_band(row["helmet_confidence"])]
        for bucket in (overall, job_bucket, band_bucket):
            bucket["total"] += 1
            bucket[status] += 1

    for bucket in [overall, *jobs.values(), *bands.values()]:
        bucket["precision"] = review_precision(bucket)
        bucket["recall"] = review_recall(bucket)

    return {
        "overall": overall,
        "jobs": sorted(jobs.values(), key=lambda job: job["total"] + job["manual"], reverse=True),
        "confidence_bands": bands,
    }


CONFIDENCE_BANDS = ["under_50", "50_to_65", "65_to_80", "80_plus"]


def confidence_band(helmet_confidence: float | None) -> str:
    value = helmet_confidence or 0.0
    if value < 0.50:
        return "under_50"
    if value < 0.65:
        return "50_to_65"
    if value < 0.80:
        return "65_to_80"
    return "80_plus"


def review_precision(bucket: dict) -> float | None:
    reviewed = bucket["confirmed"] + bucket["false_positive"]
    if reviewed == 0:
        return None
    return round(bucket["confirmed"] / reviewed, 4)


def review_recall(bucket: dict) -> float | None:
    known_events = bucket["confirmed"] + bucket["manual"]
    if known_events == 0:
        return None
    return round(bucket["confirmed"] / known_events, 4)
