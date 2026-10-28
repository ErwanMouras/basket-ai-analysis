"""Completion is gated on source coverage and unambiguous exported IDs."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from pipeline.cuda.core import CudaOnly, tensor
from pipeline.cuda.integrity import observation_metadata
from pipeline.cuda.run import health
from pipeline.numeric import cuda_numbers
from pipeline.progress import Progress


def frame(index):
    return {
        "run_id": "integrity-test",
        "frame_index": index,
        "segment_id": 0,
        "timestamp_s": index * 0.04,
        "included": True,
        "scene_cut": False,
        "court": {"image_to_court": None},
        "ball": {"position_px": None},
        "stages": {"players": {"status": "ok"}},
        "persons": [
            {
                "observation_id": f"integrity-test:{index}:0",
                "track_id": 1,
                "role": "player",
                "position_m": None,
            }
        ],
    }


class MetadataTests(unittest.TestCase):
    def test_missing_or_repeated_frame_cannot_pass_same_total(self):
        with self.assertRaisesRegex(ValueError, "sequence/run mismatch"):
            observation_metadata(frame(2), "integrity-test", 1)

    def test_duplicate_tracks_and_resolved_players_are_rejected(self):
        row = frame(0)
        duplicate = {**row["persons"][0], "observation_id": "integrity-test:0:1"}
        row["persons"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "track ID"):
            observation_metadata(row, "integrity-test", 0)
        duplicate["track_id"] = 2
        for person in row["persons"]:
            person["resolved_identity"] = {"team_id": "a", "number": "00"}
        with self.assertRaisesRegex(ValueError, "resolved identity"):
            observation_metadata(row, "integrity-test", 0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CoverageTests(unittest.TestCase):
    def check(self, rows):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "observations.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            (output / "statistics.json").write_text('{"players": []}')
            progress = Progress(output, "integrity-test")
            try:
                with torch.inference_mode(), cuda_numbers("cuda:0"), CudaOnly():
                    return health(
                        output,
                        {
                            "frames": 3,
                            "run_id": "integrity-test",
                            "status": "validating",
                            "artifacts": {},
                        },
                        "cuda:0",
                        progress,
                        expected_timestamps=tensor([0.0, 0.04, 0.08]),
                    )
            finally:
                progress.close()

    def test_source_timestamp_equality_and_complete_sequence(self):
        result = self.check([frame(i) for i in range(3)])
        self.assertTrue(result["integrity"]["timestamps_match_source"])
        self.assertEqual(result["integrity"]["verified_frames"], 3)

    def test_repeated_index_is_rejected_even_with_three_rows(self):
        rows = [frame(i) for i in range(3)]
        rows[1]["frame_index"] = 0
        with self.assertRaisesRegex(ValueError, "sequence/run mismatch"):
            self.check(rows)

    def test_ordered_but_wrong_timestamps_are_rejected(self):
        rows = [frame(i) for i in range(3)]
        rows[1]["timestamp_s"] = 0.05
        with self.assertRaisesRegex(ValueError, "timestamps"):
            self.check(rows)

    def test_truncated_export_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "frame count mismatch"):
            self.check([frame(0), frame(1)])
