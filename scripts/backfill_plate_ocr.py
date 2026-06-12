import sqlite3
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import settings
from app.services.pipeline import read_plate_text


def media_path_to_file(path: str) -> Path:
    relative = path.removeprefix("/media/")
    return settings.data_dir / relative


def main() -> None:
    attempted = 0
    updated = 0

    with sqlite3.connect(settings.database_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, plate_image
            FROM violations
            WHERE plate_image IS NOT NULL
              AND (plate_text IS NULL OR TRIM(plate_text) = '')
            """
        ).fetchall()

        for row in rows:
            plate_path = media_path_to_file(row["plate_image"])
            image = cv2.imread(str(plate_path))
            if image is None:
                continue

            attempted += 1
            text, confidence = read_plate_text(image)
            if not text:
                continue

            conn.execute(
                """
                UPDATE violations
                SET plate_text = ?, plate_confidence = ?
                WHERE id = ?
                """,
                (text, confidence, row["id"]),
            )
            updated += 1

    print(f"attempted={attempted} updated={updated}")


if __name__ == "__main__":
    main()
