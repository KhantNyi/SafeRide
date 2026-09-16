"""Persistent local PaddleOCR worker, isolated from the PyTorch environment."""
import atexit
import base64
import json
import os
import queue
import subprocess
import threading

import cv2

from app.core.config import settings

PREFIX = "SAFERIDE_OCR:"


class PaddleReader:
    def __init__(self):
        self._process = None
        self._lock = threading.Lock()
        self._responses = None

    def close(self):
        process, self._process = self._process, None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            for stream in (process.stdin, process.stdout):
                if stream:
                    stream.close()

    def _start(self):
        if self._process is not None and self._process.poll() is None:
            return
        self.close()
        root = settings.project_root
        executable = root / ".cache/paddleocr-venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not executable.exists():
            raise RuntimeError("PaddleOCR environment is missing. See docs/paddleocr-setup.md.")
        responses = queue.Queue()
        self._responses = responses
        with (settings.cache_dir / "paddleocr-worker.log").open("a", encoding="utf-8") as log:
            self._process = subprocess.Popen(
                [str(executable), "-u", str(root / "scripts/paddleocr_worker.py")],
                cwd=str(root), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                text=True, encoding="utf-8",
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        process = self._process

        def receive():
            try:
                for line in process.stdout:
                    if line.startswith(PREFIX):
                        responses.put(json.loads(line[len(PREFIX):]))
            except Exception as exc:
                responses.put({"error": str(exc)})
            finally:
                responses.put({"error": "PaddleOCR worker stopped. See .cache/paddleocr-worker.log."})

        threading.Thread(target=receive, daemon=True).start()

    def readtext(self, image, **kwargs):
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise ValueError("Cannot encode the plate crop")
        with self._lock:
            try:
                self._start()
                request = {"image": base64.b64encode(encoded).decode("ascii")}
                self._process.stdin.write(json.dumps(request) + "\n")
                self._process.stdin.flush()
                response = self._responses.get(timeout=90)
                if "error" in response:
                    raise RuntimeError(response["error"])
                return response["lines"]
            except queue.Empty as exc:
                self.close()
                raise RuntimeError("PaddleOCR timed out. See .cache/paddleocr-worker.log.") from exc
            except Exception:
                self.close()
                raise


paddle_reader = PaddleReader()
atexit.register(paddle_reader.close)
