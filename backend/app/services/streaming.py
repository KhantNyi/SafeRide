from collections import defaultdict
from threading import Condition
from time import monotonic, sleep


class FrameHub:
    def __init__(self) -> None:
        self._conditions: dict[str, Condition] = defaultdict(Condition)
        self._frames: dict[str, bytes] = {}
        self._versions: dict[str, int] = defaultdict(int)
        self._closed: set[str] = set()

    def publish(self, job_id: str, frame: bytes) -> None:
        condition = self._conditions[job_id]
        with condition:
            self._frames[job_id] = frame
            self._versions[job_id] += 1
            self._closed.discard(job_id)
            condition.notify_all()

    def close(self, job_id: str) -> None:
        condition = self._conditions[job_id]
        with condition:
            self._closed.add(job_id)
            condition.notify_all()

    def stream(self, job_id: str):
        last_version = -1
        idle_started = monotonic()
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-store\r\n\r\n"

        while True:
            condition = self._conditions[job_id]
            with condition:
                condition.wait_for(
                    lambda: self._versions[job_id] != last_version or job_id in self._closed,
                    timeout=1,
                )
                frame = self._frames.get(job_id)
                version = self._versions[job_id]
                closed = job_id in self._closed

            if frame is not None and version != last_version:
                last_version = version
                idle_started = monotonic()
                yield boundary + frame + b"\r\n"
                continue

            if closed:
                if frame is not None and version == last_version:
                    yield boundary + frame + b"\r\n"
                break

            if monotonic() - idle_started > 30:
                break

            sleep(0.05)


frame_hub = FrameHub()
