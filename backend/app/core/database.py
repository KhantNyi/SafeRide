import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from app.core.config import settings


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(settings.database_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                source_path TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT,
                progress REAL NOT NULL DEFAULT 0,
                current_frame INTEGER NOT NULL DEFAULT 0,
                total_frames INTEGER NOT NULL DEFAULT 0,
                sampled_frames INTEGER NOT NULL DEFAULT 0,
                violation_count INTEGER NOT NULL DEFAULT 0,
                elapsed_seconds REAL NOT NULL DEFAULT 0,
                processing_fps REAL NOT NULL DEFAULT 0,
                eta_seconds REAL NOT NULL DEFAULT 0,
                preview_image TEXT,
                result TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        migrate_jobs_table(conn)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS violations (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                detected_at TEXT NOT NULL,
                helmet_status TEXT NOT NULL,
                helmet_confidence REAL NOT NULL,
                plate_text TEXT,
                plate_confidence REAL,
                evidence_image TEXT NOT NULL,
                plate_image TEXT,
                frame_number INTEGER,
                track_id INTEGER,
                review_status TEXT NOT NULL DEFAULT 'pending',
                FOREIGN KEY(job_id) REFERENCES jobs(id)
            )
            """
        )
        migrate_violations_table(conn)


def migrate_jobs_table(conn: sqlite3.Connection) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    migrations = {
        "progress": "ALTER TABLE jobs ADD COLUMN progress REAL NOT NULL DEFAULT 0",
        "current_frame": "ALTER TABLE jobs ADD COLUMN current_frame INTEGER NOT NULL DEFAULT 0",
        "total_frames": "ALTER TABLE jobs ADD COLUMN total_frames INTEGER NOT NULL DEFAULT 0",
        "sampled_frames": "ALTER TABLE jobs ADD COLUMN sampled_frames INTEGER NOT NULL DEFAULT 0",
        "violation_count": "ALTER TABLE jobs ADD COLUMN violation_count INTEGER NOT NULL DEFAULT 0",
        "elapsed_seconds": "ALTER TABLE jobs ADD COLUMN elapsed_seconds REAL NOT NULL DEFAULT 0",
        "processing_fps": "ALTER TABLE jobs ADD COLUMN processing_fps REAL NOT NULL DEFAULT 0",
        "eta_seconds": "ALTER TABLE jobs ADD COLUMN eta_seconds REAL NOT NULL DEFAULT 0",
        "preview_image": "ALTER TABLE jobs ADD COLUMN preview_image TEXT",
        "result": "ALTER TABLE jobs ADD COLUMN result TEXT",
    }
    for column, sql in migrations.items():
        if column not in columns:
            conn.execute(sql)


def migrate_violations_table(conn: sqlite3.Connection) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(violations)").fetchall()}
    migrations = {
        "review_status": "ALTER TABLE violations ADD COLUMN review_status TEXT NOT NULL DEFAULT 'pending'",
        "track_id": "ALTER TABLE violations ADD COLUMN track_id INTEGER",
    }
    for column, sql in migrations.items():
        if column not in columns:
            conn.execute(sql)
