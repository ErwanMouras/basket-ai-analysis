import itertools
import unittest
from unittest.mock import patch

import numpy as np

from training.jersey.config import number, settings
from training.jersey.crops import candidate
from training.jersey.reader import ResourceDeferred, decode
from training.jersey.roster import Roster
from training.jersey.temporal import JerseyRecognizer


def fixture_roster():
    return Roster(
        {
            "schema_version": 1,
            "teams": [
                {
                    "team_id": "a",
                    "name": "Team A",
                    "players": [
                        {"number": "23", "name": "Player A"},
                        {"number": "7", "name": "Player B"},
                        {"number": "0", "name": "Player C"},
                    ],
                },
                {
                    "team_id": "b",
                    "name": "Team B",
                    "players": [
                        {"number": "00", "name": "Player D"},
                        {"number": "01", "name": "Player E"},
                    ],
                },
            ],
        }
    )


def observation(tid=1):
    return {
        "bbox": [0.0, 0.0, 100.0, 160.0],
        "class_id": 0,
        "confidence": 0.9,
        "track_id": tid,
    }


class Clock:
    value = 100.0

    def __call__(self):
        return self.value


class Reader:
    def __init__(self, clock, text="23", duration=0.01):
        self.provenance = {"model": "fixture"}
        self.clock, self.text, self.duration = clock, text, duration
        self.batches = []

    def read(self, crops):
        self.batches.append(len(crops))
        self.clock.value += self.duration
        return [
            {
                "text": self.text,
                "number": number(self.text),
                "confidence": 0.99,
                "eos": True,
            }
            for _ in crops
        ]


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(4).integers(
            0, 256, (160, 100, 3), dtype=np.uint8
        )
        self.clock = Clock()
        self.reader = Reader(self.clock)
        self.engine = JerseyRecognizer(
            settings(), roster=fixture_roster(), reader=self.reader, clock=self.clock
        )

    def step(self, frame, detections=None, segment=0, wall=True):
        if wall:
            self.clock.value += 0.1
        return self.engine.update(
            self.image,
            [observation()] if detections is None else detections,
            frame_index=frame,
            timestamp_seconds=frame / 10,
            segment_id=segment,
        )

    def test_strict_numbers_preserve_zero(self):
        for v in ["0", "00", "01", "7", "99"]:
            self.assertEqual(number(v), v)
        for v in [" 7", "7 ", "2a3", "１２", "123", "", None, 7, "1\n"]:
            self.assertIsNone(number(v))

    def test_config_budgets_and_tracking_requirements(self):
        for v in [
            {"batch_size": 5},
            {"max_tracks": 257},
            {"max_duty_cycle": 0},
            {"min_votes": 1},
            {"min_agreement": 0.5},
            {"enabled": 1},
            {"device": "cuda"},
            {"crops_per_video_second": float("nan")},
        ]:
            with self.assertRaises(ValueError):
                settings(v)
        from training.players.predict import load_config

        with self.assertRaisesRegex(ValueError, "requires tracking"):
            load_config(
                None,
                video="dummy.mp4",
                checkpoint="dummy.pt",
                jersey=True,
                tracker="none",
            )

    def test_multiple_independent_reads_confirm_and_slow_down(self):
        reads = []
        for f in range(80):
            displayed, new, _ = self.step(f)
            reads += new
        self.assertEqual(displayed[0]["jersey"]["number"], "23")
        times = [r["timestamp_seconds"] for r in reads]
        self.assertTrue(all(b - a >= 0.75 for a, b in itertools.pairwise(times)))
        self.assertLess(len(reads), 6)
        self.assertTrue(any(b - a >= 3 for a, b in itertools.pairwise(times)))

    def test_contradiction_abstains_then_consistent_evidence_can_replace(self):
        for f in range(30):
            displayed, _, _ = self.step(f)
        self.assertEqual(displayed[0]["jersey"]["number"], "23")
        self.reader.text = "7"
        seen = False
        for f in range(30, 140):
            displayed, readings, _ = self.step(f)
            if readings and not seen:
                self.assertIsNone(displayed[0]["jersey"]["number"])
                seen = True
        self.assertTrue(seen)
        self.assertEqual(displayed[0]["jersey"]["number"], "7")

    def test_no_votes_from_low_confidence_or_letters(self):
        self.reader.text = "A23"
        for f in range(40):
            displayed, _, _ = self.step(f)
        self.assertIsNone(displayed[0]["jersey"]["number"])
        self.assertEqual(len(self.engine.states[1]["votes"]), 0)

    def test_empty_untracked_and_low_quality_skip_ocr(self):
        self.image[:] = 0
        for f in range(20):
            self.step(f)
        self.step(20, [])
        displayed, _, _ = self.step(21, [observation(None)])
        self.assertEqual(displayed[0]["jersey"]["status"], "untracked")
        self.assertEqual(self.reader.batches, [])

    def test_evidence_and_track_expiration(self):
        for f in range(30):
            self.step(f)
        # Visible but unreadable: old identity must expire, not persist forever.
        self.image[:] = 0
        for f in range(30, 160):
            displayed, _, _ = self.step(f)
        self.assertIsNone(displayed[0]["jersey"]["number"])
        _, _, ended = self.step(300, [])
        self.assertEqual(ended[0]["reason"], "expired")
        self.assertEqual(self.engine.states, {})

    def test_scene_reset_never_inherits_number(self):
        for f in range(30):
            self.step(f)
        displayed, _, ended = self.step(30, segment=1)
        self.assertEqual(ended[0]["jersey"]["number"], "23")
        self.assertEqual(ended[0]["segment_id"], 0)
        self.assertIsNone(displayed[0]["jersey"]["number"])

    def test_bounded_state_crops_and_source_budget_over_long_sequence(self):
        self.engine.config.update(max_tracks=3, max_overlap=1)
        # Candidate fixture isolates scheduling from overlap geometry.
        crop = candidate(self.image, observation(), [], settings())
        with patch(
            "training.jersey.temporal.candidate", side_effect=lambda *args: dict(crop)
        ):
            for f in range(1000):
                self.step(f, [observation(i) for i in range(1, 10)])
        self.assertLessEqual(len(self.engine.states), 3)
        self.assertLessEqual(self.engine.stats["peak_crop_bytes"], 3 * 32 * 128 * 3)
        self.assertTrue(all(len(s["votes"]) <= 8 for s in self.engine.states.values()))
        self.assertLessEqual(self.engine.stats["ocr_crops"], 4 + 8 * 99.9)
        self.assertLessEqual(max(self.reader.batches), 4)
        self.assertGreater(self.engine.stats["capacity_rejections"], 0)

    def test_wall_time_cooldown_blocks_fast_video(self):
        self.reader.duration = 1.0
        for f in range(10):
            self.step(f, wall=False)
        self.assertEqual(len(self.reader.batches), 1)
        next_allowed = self.engine.next_wall
        for f in range(10, 80):
            self.step(f, wall=False)
        self.assertEqual(len(self.reader.batches), 1)
        self.clock.value = next_allowed
        for f in range(80, 85):
            self.step(f, wall=False)
        self.assertEqual(len(self.reader.batches), 2)

    def test_resource_pressure_defers_without_cpu_fallback_or_votes(self):
        with patch.object(
            self.reader, "read", side_effect=ResourceDeferred("gpu_memory_reserve")
        ):
            for f in range(30):
                displayed, _, _ = self.step(f)
        self.assertGreater(self.engine.stats["resource_deferrals"], 0)
        self.assertEqual(self.engine.stats["ocr_crops"], 0)
        self.assertIsNone(displayed[0]["jersey"]["number"])

    def test_crops_are_bounded_and_reject_overlap(self):
        d = observation()
        crop = candidate(self.image, d, [d], settings())
        self.assertEqual(crop["image"].shape, (32, 128, 3))
        self.assertEqual(crop["region"], "center_chest")
        self.assertIsNone(candidate(self.image, d, [d, observation(2)], settings()))
        d["pose"] = {"keypoints": [[0, 0]] * 17, "valid": [False] * 17}
        for i, p in zip((5, 6, 11, 12), ([20, 30], [80, 30], [25, 100], [75, 100])):
            d["pose"]["keypoints"][i] = p
            d["pose"]["valid"][i] = True
        self.assertEqual(
            candidate(self.image, d, [d], settings())["region"], "pose_torso"
        )

    def test_decoder_requires_eos_full_string_and_confidence(self):
        p = np.full((4, 95), 0.00001)
        # 00 then EOS, trailing arbitrary token is irrelevant.
        for i, k in enumerate([1, 1, 0, 11]):
            p[i, k] = 0.99
        result = decode(p)
        self.assertEqual(result["number"], "00")
        self.assertAlmostEqual(result["confidence"], 0.99**3)
        p[2, 0] = 0
        p[2, 2] = 0.99
        self.assertIsNone(decode(p)["number"])

    def test_low_confidence_numeric_read_is_not_evidence(self):
        with patch.object(
            self.reader,
            "read",
            return_value=[{"text": "23", "confidence": 0.89, "eos": True}],
        ):
            for f in range(30):
                displayed, _, _ = self.step(f)
        self.assertIsNone(displayed[0]["jersey"]["number"])
        self.assertEqual(len(self.engine.states[1]["votes"]), 0)

    def test_capacity_prefers_visible_tracks_and_ignores_weak_new_tracks(self):
        self.engine.config["max_tracks"] = 1
        self.step(0)
        displayed, _, ended = self.step(1, [observation(2)])
        self.assertEqual(ended[0]["reason"], "capacity_eviction")
        self.assertEqual(set(self.engine.states), {2})
        weak = observation(3)
        weak["confidence"] = 0.2
        displayed, _, _ = self.step(2, [weak])
        self.assertEqual(displayed[0]["jersey"]["status"], "below_detection_confidence")
        self.assertEqual(set(self.engine.states), {2})

    def test_nonmonotonic_frames_and_duplicate_ids_fail(self):
        self.step(1)
        with self.assertRaises(ValueError):
            self.step(1)
        with self.assertRaises(ValueError):
            self.step(2, [observation(), observation()])
