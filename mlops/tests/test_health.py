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

    def test_named_availability_uses_resolved_identities_and_excludes_suppression(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            person = {"observation_id": "a", "track_id": 1, "role": "player",
                      "bbox": [1, 1, 10, 20], "position_m": None, "confidence": .9,
                      "jersey": {"status": "ambiguous", "team_id": "home", "player_name": "Uncertain"}}
            rows = []
            for i in range(4):
                current = {**person, "observation_id": str(i)}
                if i:
                    current["resolved_identity"] = {"team_id": "home", "number": "00", "player_name": "Known"}
                if i == 3:
                    current["jersey_number_suppressed"] = True
                rows.append({"schema_version": 1, "artifact_type": "video_observation",
                    "frame_index": i, "segment_id": 0, "timestamp_s": i / 10,
                    "width": 100, "height": 80, "coordinate_space": "source_pixels",
                    "distance_unit": "m", "persons": [current], "ball": {"position_px": None},
                    "court": None, "stages": {}})
            write_jsonl(root / "observations.jsonl", rows)
            write_json(root / "statistics.json", {"schema_version": 1, "metric": "observed_distance", "players": []})
            write_json(root / "run.json", {"status": "completed", "run_id": "fixture", "frames": 4})
            report = build_health(root)
            self.assertEqual(report["counts"]["named_player_observations"], 2)
            self.assertEqual(report["ratios"]["named_identity_availability"], .5)
