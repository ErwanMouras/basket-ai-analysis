import json
import tempfile
import unittest
from pathlib import Path

from mlops.health import build_health
from training.common.files import write_json, write_jsonl
from training.common.provenance import file_hash


class HealthTests(unittest.TestCase):
    def test_empty_detection_support_is_unavailable_and_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / "statistics.json", {"schema_version": 1, "metric": "observed_distance", "players": []})
            write_jsonl(root / "observations.jsonl", [{"schema_version": 1, "artifact_type": "video_observation",
                "frame_index": 0, "segment_id": 0, "timestamp_s": 0., "width": 100, "height": 80,
                "coordinate_space": "source_pixels", "distance_unit": "m", "persons": [],
                "ball": {"position_px": None, "confidence": None}, "court": None, "stages": {}}])
            write_json(root / "run.json", {"status": "completed", "run_id": "fixture", "frames": 1,
                "artifacts": {"observations.jsonl": {"sha256": file_hash(root / "observations.jsonl")}}})
            report = build_health(root)
            self.assertIsNone(report["ratios"]["named_identity_availability"])
            self.assertEqual(report["ratios"]["calibration_availability"], 0.)
            with (root / "observations.jsonl").open("a") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(ValueError, "integrity"):
                build_health(root)
