"""Refinement rejects ambiguous seeds, stale annotations and duplicate bodies."""

import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path

import torch

from pipeline.config import load_config
from pipeline.cuda.core import CudaOnly, tensor
from pipeline.cuda.demo import reviewed_detections, verified_lineup
from pipeline.cuda.identity import replay
from pipeline.cuda.lineup import LineupTracker, seed_assignment
from pipeline.cuda.recovery import decode_number, deduplicate_persons, saved_tensors
from pipeline.cuda.vision import candidate, ocr_due
from pipeline.numeric import cuda_numbers
from pipeline.progress import Progress
from pipeline.tests.test_pipeline import person, row
from training.jersey.roster import Roster


class TokenTests(unittest.TestCase):
    def test_review_keeps_original_detection_and_confidence(self):
        detection = {
            "observation_id": "source:1:0",
            "role": "player",
            "confidence": 0.26,
            "bbox": [1, 2, 3, 4],
        }
        result, dropped = reviewed_detections(
            {"persons": [detection]}, {"source:1:0": {"action": "keep"}}
        )
        self.assertEqual(result["persons"][0]["confidence"], 0.26)
        self.assertEqual(result["persons"][0]["bbox"], detection["bbox"])
        self.assertNotIn("resolved_identity", result["persons"][0])
        self.assertFalse(dropped)

    def test_number_label_preserves_zero_and_leading_zero(self):
        self.assertEqual(decode_number([1, 0, 0]), "0")
        self.assertEqual(decode_number([1, 1, 0]), "00")
        self.assertEqual(decode_number([3, 5, 0]), "24")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class RefinementTests(unittest.TestCase):
    def test_bbox_number_crop_survives_a_narrow_pose(self):
        image = (
            (
                (
                    (
                        torch.arange(200, device="cuda:0")[:, None]
                        + torch.arange(200, device="cuda:0")[None]
                    )
                    % 2
                )
                * 255
            )
            .float()[None]
            .expand(3, -1, -1)
        )
        boxes = tensor([[30, 20, 130, 180, 0.95, 2]])
        points = tensor([[75, 60]] * 17, dtype=torch.float32)
        points[6] = tensor([80, 60], dtype=torch.float32)
        points[11] = tensor([75, 120], dtype=torch.float32)
        points[12] = tensor([80, 120], dtype=torch.float32)
        valid = torch.ones(17, device="cuda:0", dtype=torch.bool)
        masks = torch.ones((1, 200, 200), device="cuda:0", dtype=torch.bool)
        config = load_config()["jersey"]
        self.assertIsNone(
            candidate(image, boxes, 0, points, valid, masks, config, "center")
        )
        crop = candidate(image, boxes, 0, points, valid, masks, config, "bbox_upper")
        self.assertIsNotNone(crop)
        self.assertTrue(crop["image"].is_cuda)
        self.assertEqual(crop["region"], "bbox_upper")

    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        for ctx in (torch.inference_mode(), cuda_numbers("cuda:0"), CudaOnly()):
            self.context.enter_context(ctx)

    def test_confirmed_identity_is_rechecked(self):
        s = {"start": 0, "confirmed": True, "last_read_s": tensor(0)}
        cfg = {"confirmed_interval_seconds": 3.0}
        self.assertFalse(ocr_due(s, 50, tensor(2), cfg))
        self.assertTrue(ocr_due(s, 76, tensor(3.04), cfg))
        self.assertFalse(ocr_due(s, 77, tensor(3.08), cfg))
        s["confirmed"] = False
        self.assertTrue(ocr_due(s, 2, tensor(0.08), cfg))

    def mask_person(self, x, confidence=0.9, role="player"):
        return {
            "bbox": [x, 0, x + 4, 4],
            "confidence": confidence,
            "role": role,
            "pose": {"keypoints": [[x, 0]] * 17, "valid": [True] * 17},
            "segmentation": {
                "mask": {
                    "encoding": "roi_rle_c",
                    "roi": [x, 0, x + 4, 4],
                    "counts": [0, 16],
                }
            },
        }

    def test_same_mask_is_deduplicated_across_roles_but_distinct_body_is_kept(self):
        a = self.mask_person(0, 0.7)
        b = self.mask_person(0, 0.95, "referee")
        c = self.mask_person(6)
        result, dropped = deduplicate_persons(
            {"persons": [a, b, c], "width": 10, "height": 4}, "cuda:0"
        )
        self.assertEqual(result["persons"], [b, c])
        self.assertEqual(dropped[0]["bbox"], a["bbox"])

    def test_invalid_saved_mask_does_not_reach_ocr(self):
        p = self.mask_person(0)
        p["segmentation"]["mask"]["counts"] = [0, 17]
        with self.assertRaisesRegex(ValueError, "RLE lengths"):
            saved_tensors({"persons": [p], "width": 10, "height": 4}, "cuda:0")

    def test_reviewed_real_body_keeps_source_confidence_when_masks_merge(self):
        first = self.mask_person(0, 0.95)
        weak = {**self.mask_person(0, 0.26), "reviewed_keep": True}
        result, dropped = deduplicate_persons(
            {"persons": [first, weak], "width": 10, "height": 4}, "cuda:0", 0.4
        )
        self.assertEqual(len(result["persons"]), 2)
        self.assertEqual(result["persons"][1]["confidence"], 0.26)
        self.assertFalse(dropped)

    def test_ambiguous_lineup_is_not_filled_in_arbitrary_order(self):
        players = [{"team_id": "a", "number": "1"}, {"team_id": "a", "number": "2"}]
        with self.assertRaisesRegex(ValueError, "uniquely identify"):
            seed_assignment([10, 20], ["a", "a"], players, [], "cuda:0")

    def test_motion_links_a_fragment_without_using_a_new_name(self):
        players = [{"team_id": "a", "number": "1"}, {"team_id": "b", "number": "2"}]
        persons = [
            {
                "role": "player",
                "track_id": 10,
                "bbox": [0, 0, 10, 30],
                "confidence": 0.9,
            },
            {
                "role": "player",
                "track_id": 20,
                "bbox": [50, 0, 60, 30],
                "confidence": 0.9,
            },
        ]
        tracker = LineupTracker(persons, players, {10: 0, 20: 1}, "cuda:0")
        next_persons = [
            {**persons[0], "track_id": None, "bbox": [1, 0, 11, 30]},
            persons[1],
        ]
        assignments, _ = tracker.update(next_persons, 1, {20: "b"})
        self.assertEqual(assignments, {0: 0, 1: 1})

    def test_substitution_invalidates_starting_lineup_constraint(self):
        teams = [
            {
                "team_id": t,
                "name": t,
                "players": [{"number": str(i), "name": t + str(i)} for i in range(5)],
            }
            for t in ("a", "b")
        ]
        roster = Roster({"schema_version": 1, "teams": teams})
        official = {
            "boxscore": {
                "players": [
                    {
                        "statistics": [
                            {
                                "athletes": [
                                    {
                                        "athlete": {"displayName": t + str(i)},
                                        "starter": True,
                                    }
                                    for i in range(5)
                                ]
                            }
                        ]
                    }
                    for t in ("a", "b")
                ]
            },
            "plays": [
                {
                    "period": {"number": 1},
                    "text": "X enters the game",
                    "type": {"text": "Substitution"},
                    "clock": {"displayValue": "6:11"},
                }
            ],
        }
        review = {
            "period": 1,
            "clock_start": "9:26",
            "clock_end": "9:18",
            "seeds": [
                {"source_track_id": i, "team_id": p["team_id"], "number": p["number"]}
                for i, p in enumerate(
                    [
                        {"team_id": t, "number": str(i)}
                        for t in ("a", "b")
                        for i in range(5)
                    ]
                )
            ],
        }
        players, seeds = verified_lineup(review, roster, official, "cuda:0")
        self.assertEqual(len(players), 10)
        self.assertEqual(len(seeds), 10)
        review["clock_end"] = "5:00"
        with self.assertRaisesRegex(ValueError, "substitution precedes"):
            verified_lineup(review, roster, official, "cuda:0")

    def test_replay_clears_a_previous_collision_before_applying_new_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            r = row(0, [person()])
            p = r["persons"][0]
            p.update(
                jersey_number_suppressed=True,
                spatial_handoff={"stale": True},
                resolved_team_id="old",
            )
            (path / "observations.partial.jsonl").write_text(json.dumps(r) + "\n")
            progress = Progress(path, "r")
            self.addCleanup(progress.close)
            player = {
                "team_id": "a",
                "team_name": "A",
                "number": "1",
                "player_name": "One",
            }
            replay(
                path,
                None,
                load_config(
                    overrides={
                        "distance": {**load_config()["distance"], "device": "cuda:0"}
                    }
                ),
                "r",
                progress,
                1,
                ({(0, p["track_id"], 0): player}, {}, {}, {}, set()),
                {
                    "teams": {"a": "#ffffff"},
                    "unassigned": "#999999",
                    "referee": "#ffaa00",
                },
            )
            updated = json.loads((path / "observations.partial.jsonl").read_text())[
                "persons"
            ][0]
            self.assertEqual(updated["resolved_identity"]["number"], "1")
            self.assertNotIn("jersey_number_suppressed", updated)
            self.assertNotIn("spatial_handoff", updated)
