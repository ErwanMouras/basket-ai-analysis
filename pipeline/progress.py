"""Atomic, phase-aware progress through publication and integrity checks.

The global percentage is a weighted work estimate, never a time estimate.
Frame counters remain exact and the final 100% requires a successful health check.
"""

import os
import time
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Lock, Thread

from training.common.files import write_json

PHASES = [
    ("loading", 1),
    ("perception", 69),
    ("identity_colors", 8),
    ("identity_ocr", 12),
    ("identity_resolution", 1),
    ("identity_handoffs", 2),
    ("identity_collisions", 1),
    ("statistics", 4),
    ("publication", 1),
    ("health", 1),
]


class Progress:
    def __init__(self, output, run_id, *, identity=True, heartbeat_s=5):
        self.path = Path(output) / "progress.json"
        self.phases = [
            (name, weight)
            for name, weight in PHASES
            if identity or not name.startswith("identity_")
        ]
        self.started = time.perf_counter()
        self.stage_started = self.started
        self.lock = Lock()
        self.stop = Event()
        self.state = {
            "run_id": run_id,
            "pid": os.getpid(),
            "status": "running",
            "frames": 0,
            "total_frames": None,
            "stage": "loading",
            "stage_frames": 0,
            "stage_total_frames": None,
            "stage_percent": None,
            "overall_percent": 0.0,
            "overall_basis": "estimated_weighted_work_not_elapsed_time",
            "phase_weights": dict(self.phases),
        }
        self.update("loading")
        self.thread = Thread(target=self._heartbeat, args=(heartbeat_s,), daemon=True)
        self.thread.start()

    def _write(self):
        self.state["elapsed_s"] = time.perf_counter() - self.started
        self.state["stage_elapsed_s"] = time.perf_counter() - self.stage_started
        self.state["updated_at"] = datetime.now(UTC).isoformat()
        write_json(self.path, self.state)

    def _heartbeat(self, interval):
        while not self.stop.wait(interval):
            with self.lock:
                self._write()

    def update(
        self,
        stage,
        completed=0,
        total=None,
        *,
        frames=None,
        total_frames=None,
        status="running",
        error=None,
    ):
        with self.lock:
            if stage != self.state["stage"]:
                self.stage_started = time.perf_counter()
            if frames is not None:
                self.state["frames"] = frames
            if total_frames is not None:
                self.state["total_frames"] = total_frames
            self.state.update(
                stage=stage,
                stage_frames=completed,
                stage_total_frames=total,
                status=status,
            )
            self.state["stage_percent"] = (
                min(100.0, 100 * completed / total) if total else None
            )
            if error:
                self.state["error"] = error
            names = [name for name, _ in self.phases]
            if stage in names:
                index = names.index(stage)
                weight = self.phases[index][1]
                fraction = min(1.0, completed / total) if total else 0.0
                percent = (
                    100
                    * (sum(w for _, w in self.phases[:index]) + weight * fraction)
                    / sum(w for _, w in self.phases)
                )
                self.state["overall_percent"] = max(
                    self.state["overall_percent"], min(99.9, percent)
                )
            if status in ("completed", "partial"):
                self.state["overall_percent"] = 100.0
            self._write()

    def close(self):
        self.stop.set()
        self.thread.join()

    def metrics(self, **values):
        with self.lock:
            self.state.update(values)
            self._write()
