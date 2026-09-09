import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi import BackgroundTasks, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.api import routes


class PlaybackRouteTests(unittest.TestCase):
    def test_completed_record_uses_job_status_and_ignores_stale_processing_record(self):
        stale = {"status": "processing", "updated_at": "2020-01-01T00:00:00+00:00"}
        with patch.object(routes, "get_job_storage", return_value={"id": "job", "source_path": "video.mov"}), \
             patch.object(routes, "get_job", return_value={"status": "completed"}), \
             patch.object(routes, "list_jobs", return_value=[stale]), \
             patch.object(routes, "reserve_playback") as reserve:
            tasks = BackgroundTasks()
            self.assertEqual(routes.start_playback("job", tasks)["status"], "processing")
            reserve.assert_called_once_with("job")
            self.assertEqual(len(tasks.tasks), 1)

    def test_recent_processing_job_blocks_enhancement(self):
        recent = {"status": "processing", "updated_at": datetime.now(timezone.utc).isoformat()}
        with patch.object(routes, "get_job_storage", return_value={"source_path": "video.mov"}), \
             patch.object(routes, "get_job", return_value={"status": "completed"}), \
             patch.object(routes, "list_jobs", return_value=[recent]), \
             patch.object(routes, "reserve_playback") as reserve:
            with self.assertRaises(HTTPException) as error:
                routes.start_playback("job", BackgroundTasks())
            self.assertEqual(error.exception.status_code, 409)
            reserve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
