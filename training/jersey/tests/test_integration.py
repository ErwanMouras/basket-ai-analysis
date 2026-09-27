import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

import cv2
import numpy as np

from training.jersey.config import settings
from training.jersey.reader import ParseqReader, ResourceDeferred
from training.jersey.setup import install
from training.jersey.temporal import JerseyRecognizer
from training.jersey.tests.test_temporal import Clock, Reader
from training.players.predict import DEFAULTS, predict_video


class IntegrationTests(unittest.TestCase):
    def run_video(self, root, enabled=True, stop=None):
        source = root / "video.mp4"
        writer = cv2.VideoWriter(
            str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (100, 160)
        )
        image = np.random.default_rng(9).integers(0, 256, (160, 100, 3), dtype=np.uint8)
        for _ in range(20):
            writer.write(image)
        writer.release()

        class Detector:
            family = "yolo"
            provenance: ClassVar = {"fixture": True}

            def predict(self, image, **kwargs):
                return [
                    {"bbox": [0.0, 0.0, 100.0, 160.0], "confidence": 0.9, "class_id": 0}
                ]

        class Tracker:
            provenance: ClassVar = {"fixture": True}
            segment_id = 0

            def __init__(self, *a, **kw):
                pass

            def update(self, detections, image, **kw):
                return [{**d, "track_id": 1} for d in detections]

        clock = Clock()
        reader = Reader(clock, duration=0)
        # No wall-clock throttling in this deterministic artifact fixture.
        recognizer = JerseyRecognizer(
            settings(), reader=reader, clock=lambda: clock.value
        )
        recognizer.config["min_batch_interval_seconds"] = 0
        cfg = {
            **DEFAULTS,
            "video": str(source),
            "output": str(root / "out"),
            "tracking": {"enabled": True},
            "jersey": {"enabled": enabled},
        }
        with (
            patch("training.players.predict.runtime", return_value={}),
            patch("training.players.predict.PlayerTracker", Tracker),
        ):
            return predict_video(
                cfg,
                detector=Detector(),
                jersey_recognizer=recognizer,
                stop_after_frame=stop,
            )

    def test_sidecars_preserve_tracking_and_disabled_mode(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as t:
                root = Path(t)
                result = self.run_video(root, enabled)
                self.assertEqual(result["jersey"]["enabled"], enabled)
                self.assertEqual((root / "out/jerseys.jsonl").exists(), enabled)
                if not enabled:
                    continue
                rows = [
                    json.loads(x)
                    for x in (root / "out/jerseys.jsonl").read_text().splitlines()
                ]
                tracks = [
                    json.loads(x)
                    for x in (root / "out/tracks.jsonl").read_text().splitlines()
                ]
                self.assertEqual(len(rows), 20)
                for row, track in zip(rows, tracks):
                    self.assertEqual(row["tracking_run_id"], track["tracking_run_id"])
                    self.assertEqual(row["frame_index"], track["frame_index"])
                    self.assertTrue(
                        all(
                            row["detections"][0][k] == v
                            for k, v in track["detections"][0].items()
                        )
                    )
                self.assertEqual(rows[-1]["detections"][0]["jersey"]["number"], "23")
                summaries = [
                    json.loads(x)
                    for x in (root / "out/jersey_tracks.jsonl").read_text().splitlines()
                ]
                self.assertEqual(summaries[0]["jersey"]["number"], "23")
                for name, digest in result["artifacts"].items():
                    self.assertEqual(
                        hashlib.sha256((root / "out" / name).read_bytes()).hexdigest(),
                        digest,
                    )

    def test_interruption_keeps_partial_files(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            with self.assertRaises(KeyboardInterrupt):
                self.run_video(root, stop=8)
            self.assertFalse((root / "out/jerseys.jsonl").exists())
            self.assertEqual(
                len((root / "out/jerseys.partial.jsonl").read_text().splitlines()), 8
            )
            self.assertEqual(
                json.loads((root / "out/progress.json").read_text())["status"], "KILLED"
            )

    def test_setup_hash_atomicity_and_idempotence(self):
        with tempfile.TemporaryDirectory() as t:
            src, out = Path(t) / "source", Path(t) / "model"
            src.write_bytes(b"fixture")
            with self.assertRaises(ValueError):
                install(out, src)
            self.assertFalse(out.exists())
            with patch(
                "training.jersey.setup.MODEL_SHA256",
                hashlib.sha256(b"fixture").hexdigest(),
            ):
                install(out, src)
                install(out, src)
            self.assertEqual(out.read_bytes(), b"fixture")
            with self.assertRaises(ValueError):
                install(out, src)

    def test_gpu_guard_refuses_pressure_and_absent_device(self):
        reader = ParseqReader.__new__(ParseqReader)
        reader.config = settings({"device": "cuda:0"})
        reader.model = None
        reader.failed_cuda = False
        cuda = SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            mem_get_info=lambda _: (512 * 1024**2, 16 * 1024**3),
        )
        with self.assertRaises(ResourceDeferred):
            reader._gpu_guard(SimpleNamespace(cuda=cuda))
        cuda.is_available = lambda: False
        with self.assertRaises(RuntimeError):
            reader._gpu_guard(SimpleNamespace(cuda=cuda))
        cuda.is_available = lambda: True
        cuda.mem_get_info = lambda _: (8 * 1024**3, 16 * 1024**3)
        reader._gpu_guard(SimpleNamespace(cuda=cuda))
        reader.failed_cuda = True
        with self.assertRaises(ResourceDeferred):
            reader._gpu_guard(SimpleNamespace(cuda=cuda))


@unittest.skipUnless(
    os.environ.get("JERSEY_MODEL_TESTS"),
    "Set JERSEY_MODEL_TESTS=1 for local PARSeq inference",
)
class RealReaderTests(unittest.TestCase):
    def test_known_digits_and_blank_abstention(self):
        from PIL import Image, ImageDraw, ImageFont

        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 42
        )
        crops = []
        for text in ("23", "00"):
            image = Image.new("RGB", (128, 64), "white")
            ImageDraw.Draw(image).text((20, 5), text, font=font, fill="black")
            crops.append(np.array(image.resize((128, 32), Image.Resampling.BICUBIC)))
        reader = ParseqReader(
            settings({"device": os.environ.get("JERSEY_DEVICE", "cpu")})
        )
        results = reader.read(crops)
        self.assertEqual([r["number"] for r in results], ["23", "00"])
        blank = reader.read([np.zeros((32, 128, 3), dtype=np.uint8)])[0]
        self.assertTrue(blank["number"] is None or blank["confidence"] < 0.9)
