import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.config import settings
from app.core.database import init_db, migrate_violations_table
from app.schemas.models import Violation
from app.services.repository import create_job, create_violation, list_violations


class PlateStatusStorageTests(unittest.TestCase):
    def test_legacy_migration_is_repeatable_and_preserves_records(self):
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE violations (id TEXT, plate_text TEXT)")
        conn.execute("INSERT INTO violations VALUES ('old', '1กข 1234')")
        migrate_violations_table(conn)
        migrate_violations_table(conn)
        record = dict(conn.execute("SELECT * FROM violations").fetchone())
        self.assertEqual(record["plate_text"], "1กข 1234")
        self.assertIsNone(record["plate_ocr_status"])

    def test_uncertainty_survives_storage_and_api_serialization(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(settings, "database_path", Path(directory) / "test.db"):
            init_db()
            create_job("job", "test.mov", "/test.mov")
            base = {"job_id": "job", "detected_at": "2026-09-07T00:00:00Z", "helmet_status": "no_helmet",
                    "helmet_confidence": .8, "plate_text": "1กข 1234", "plate_confidence": .12,
                    "evidence_image": "/test.jpg"}
            create_violation({**base, "id": "new", "plate_ocr_status": "uncertain"})
            create_violation({**base, "id": "manual", "source": "manual"})
            rows = {row["id"]: Violation(**row).model_dump() for row in list_violations()}
            self.assertEqual(rows["new"]["plate_ocr_status"], "uncertain")
            self.assertEqual(rows["new"]["plate_text"], "1กข 1234")
            self.assertIsNone(rows["manual"]["plate_ocr_status"])
