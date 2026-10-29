"""Adaptive uniforms must recover shaded jerseys without bridging cuts or gaps."""

import copy
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path

import torch

from pipeline.cuda.colors import (
    UniformCalibration,
    anchor_labels,
    color_consensus,
    torso_features,
)
from pipeline.cuda.core import CudaOnly, tensor
from pipeline.cuda.integrity import geometry_bytes
from pipeline.numeric import cuda_numbers
from training.jersey.roster import Roster


class ColorReplayIntegrityTests(unittest.TestCase):
    def test_geometry_fingerprint_ignores_identity_changes_but_detects_body_changes(
        self,
    ):
        row = {
            "frame_index": 0,
            "timestamp_s": 0.0,
            "persons": [
                {
                    "bbox": [0, 0, 10, 30],
                    "confidence": 0.5,
                    "role": "player",
                    "observation_id": "old:0:0",
                    "cuda_evidence": {"team_id": "old"},
                }
            ],
        }
        changed = copy.deepcopy(row)
        changed["persons"][0].update(
            observation_id="new:0:0",
            cuda_evidence={"team_id": "new"},
            resolved_identity={"number": "1"},
        )
        self.assertEqual(geometry_bytes(row), geometry_bytes(changed))
        changed["persons"][0]["bbox"] = [1, 0, 10, 30]
        self.assertNotEqual(geometry_bytes(row), geometry_bytes(changed))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class AdaptiveColorTests(unittest.TestCase):
    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        for ctx in (
            torch.inference_mode(),
            torch.device("cuda:0"),
            cuda_numbers("cuda:0"),
            CudaOnly(),
        ):
            self.context.enter_context(ctx)
        self.device = "cuda:0"

    def sample_person(self):
        return {
            "bbox": [0, 0, 100, 200],
            "pose": {"keypoints": [[0, 0]] * 17, "valid": [False] * 17},
            "segmentation": {
                "mask": {
                    "encoding": "roi_rle_c",
                    "roi": [0, 0, 100, 200],
                    "counts": [0, 20000],
                }
            },
        }

    def test_color_anchors_reject_single_digits_and_require_wide_crop_support(self):
        roster = Roster(
            {
                "schema_version": 1,
                "teams": [
                    {
                        "team_id": "dark",
                        "name": "Dark",
                        "players": [{"number": "11", "name": "A"}],
                    },
                    {
                        "team_id": "light",
                        "name": "Light",
                        "players": [
                            {"number": "1", "name": "B"},
                            {"number": "24", "name": "C"},
                            {"number": "30", "name": "D"},
                        ],
                    },
                ],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reads, tracks = [], {}
            for track, number, region in (
                (1, "1", "bbox_upper"),
                (2, "24", "pose_number_center"),
                (3, "30", "bbox_upper"),
            ):
                for frame in (0, 2, 4):
                    reads.append(
                        {
                            "segment_id": 0,
                            "track_id": track,
                            "frame_index": frame,
                            "number": number,
                            "confidence": 0.99,
                            "region": region,
                        }
                    )
                tracks[f"0:{track}"] = {
                    "frames": [0, 1, 2, 3, 4],
                    "episodes": [
                        {"number_majority": number, "start_frame": 0, "end_frame": 4}
                    ],
                }
            (root / "jersey_reads.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in reads)
            )
            (root / "team_votes.json").write_text(json.dumps({"tracks": tracks}))
            labels = anchor_labels(root, roster, self.device)
            self.assertEqual({key[1] for key in labels}, {3})

    def test_grey_torso_features_are_kept_for_calibration(self):
        image = torch.full((3, 200, 100), 140, device=self.device, dtype=torch.uint8)
        features, status = torso_features(image, self.sample_person())
        self.assertEqual(status, "usable")
        self.assertTrue(
            torch.equal(
                features,
                torch.full((2, 3), 140, device=self.device, dtype=torch.float32),
            )
        )

    def test_missing_foreground_and_colored_skin_remain_unknown(self):
        p = self.sample_person()
        p["segmentation"]["mask"]["counts"] = [20000]
        image = torch.full((3, 200, 100), 140, device=self.device, dtype=torch.uint8)
        self.assertIsNone(torso_features(image, p)[0])
        image[:] = tensor([210, 140, 80], device=self.device, dtype=torch.uint8)[
            :, None, None
        ]
        self.assertIsNone(torso_features(image, self.sample_person())[0])

    def training_data(self):
        noise = (
            torch.arange(200, device=self.device, dtype=torch.float32).remainder(7) - 3
        )
        centers = tensor(
            [[[30, 40, 50]] * 2, [[130, 145, 160]] * 2],
            device=self.device,
            dtype=torch.float32,
        )
        labels = (torch.arange(200, device=self.device) >= 100).long()
        return (
            centers[labels] + noise[:, None, None],
            labels,
            torch.arange(200, device=self.device),
        )

    def test_match_calibration_accepts_shaded_white_and_holds_out_tracks(self):
        values, labels, groups = self.training_data()
        model, audit = UniformCalibration.fit(values, labels, groups, ["dark", "light"])
        predicted, _ = model.predict(
            tensor([[[130, 145, 160]] * 2], device=self.device, dtype=torch.float32)
        )
        self.assertEqual(predicted.tolist(), [1])
        self.assertEqual(int(audit["heldout_samples"]), 40)
        self.assertEqual(int(audit["training_samples"]), 160)

    def test_conflicting_rois_and_out_of_distribution_values_are_rejected(self):
        values, labels, groups = self.training_data()
        model, _ = UniformCalibration.fit(values, labels, groups, ["dark", "light"])
        predicted, _ = model.predict(
            tensor(
                [[[30, 40, 50], [130, 145, 160]], [[250, 250, 250]] * 2],
                device=self.device,
                dtype=torch.float32,
            )
        )
        self.assertEqual(predicted.tolist(), [-1, -1])

    def test_overlapping_team_distributions_cannot_be_published(self):
        values, labels, groups = self.training_data()
        values[labels == 1] -= 100
        with self.assertRaisesRegex(ValueError, "not separated"):
            UniformCalibration.fit(values, labels, groups, ["dark", "light"])

    def test_temporal_consensus_never_crosses_source_track_gaps(self):
        frames = [0, 1, 2, 3, 4, 10, 11, 12]
        result = color_consensus(frames, {0: 0, 2: 0, 4: 0, 10: 0, 12: 0}, self.device)
        self.assertEqual(set(result), {0, 1, 2, 3, 4, 10, 12})
        self.assertNotIn(11, result)

    def test_supported_color_switch_is_not_smoothed_into_previous_team(self):
        frames = list(range(20))
        result = color_consensus(
            frames, {0: 0, 2: 0, 4: 0, 10: 1, 12: 1, 14: 1}, self.device
        )
        self.assertTrue(all(result[i] == 0 for i in range(10)))
        self.assertTrue(all(result[i] == 1 for i in range(10, 20)))


if __name__ == "__main__":
    unittest.main()
