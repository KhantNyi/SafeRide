import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.config import settings
from app.schemas.models import DetectionSettingsUpdate
from app.services import plate_ocr
from app.services.paddle_ocr import PaddleReader, paddle_reader


class OCREngineTests(unittest.TestCase):
    def test_engine_selection_keeps_existing_easyocr_reader(self):
        easy = Mock()
        with patch.object(plate_ocr, "_ocr_reader", easy):
            with patch.object(settings, "ocr_engine", "paddleocr"):
                self.assertIs(plate_ocr.get_reader(), paddle_reader)
            with patch.object(settings, "ocr_engine", "easyocr"):
                self.assertIs(plate_ocr.get_reader(), easy)

    def test_disabled_ocr_does_not_start_either_engine(self):
        with patch.object(settings, "enable_ocr", False), patch.object(plate_ocr, "get_reader") as reader:
            self.assertEqual(plate_ocr.read_plate_text(np.zeros((10, 10, 3), np.uint8)), (None, None))
            reader.assert_not_called()

    def test_worker_failure_is_reported_instead_of_empty_reading(self):
        with patch.object(settings, "enable_ocr", True), patch.object(settings, "ocr_engine", "paddleocr"):
            with patch.object(plate_ocr, "get_reader", side_effect=RuntimeError("worker failed")):
                with self.assertLogs("app.services.plate_ocr", level="ERROR"):
                    with self.assertRaisesRegex(RuntimeError, "worker failed"):
                        plate_ocr.read_plate_text(np.zeros((10, 10, 3), np.uint8))

    def test_unknown_engine_rejected(self):
        with self.assertRaises(ValidationError):
            DetectionSettingsUpdate(ocr_engine="unknown")

    def test_worker_timeout_closes_process(self):
        import queue
        reader = PaddleReader()
        reader._process = Mock()
        reader._responses = Mock()
        reader._responses.get.side_effect = queue.Empty
        with patch.object(reader, "_start"), patch.object(reader, "close") as close:
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                reader.readtext(np.zeros((10, 10, 3), np.uint8))
            close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
