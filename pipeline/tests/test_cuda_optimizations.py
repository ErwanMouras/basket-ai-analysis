"""Cross-check fused CUDA paths against the established numerical rules."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from pipeline.config import load_config
from pipeline.cuda.core import CudaOnly, export, project, tensor
from pipeline.cuda.distance import CudaDistance
from pipeline.cuda.geometry import dlt
from pipeline.cuda.identity import (
    _collision_frames_reference,
    _minimal,
    _minimal_batch,
    _split_episodes_reference,
    collision_frames,
    split_episodes,
)
from pipeline.cuda.resume import copy_prefix, inspect_prefix
from pipeline.cuda.tracking import Kalman, Track
from pipeline.cuda.vision import _dominant_color_reference, dominant_color
from pipeline.distance import Distance
from pipeline.numeric import CudaCounter, cuda_numbers
from pipeline.tests.test_pipeline import person, row


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class OptimizationTests(unittest.TestCase):
    def setUp(self):
        from contextlib import ExitStack

        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for context in (torch.inference_mode(), cuda_numbers("cuda:0"), CudaOnly()):
            self.stack.enter_context(context)
        self.device = "cuda:0"
        with torch.device(self.device):
            __import__("kornia.color")

    def test_four_point_solver_matches_normalized_dlt(self):
        from pipeline.cuda.kernels import four_point

        torch.manual_seed(37)
        first = torch.rand((128, 4, 2), device=self.device, dtype=torch.float64) * 30
        truth = tensor([[35, 4, 150], [-3, 40, 180], [0.001, 0.002, 1]])
        second = project(truth, first)
        result = four_point(first, second)
        self.assertTrue(bool(torch.isfinite(result).all()))
        self.assertLess(float((project(result, first) - second).abs().max()), 1.0e-6)
        reference = dlt(first, second)
        self.assertLess(float((result - reference).abs().max()), 1.0e-5)

    def test_fused_colors_preserve_categories_and_circular_hues(self):
        torch.manual_seed(23)
        for color in (
            [255, 255, 255],
            [0, 0, 0],
            [240, 25, 35],
            [25, 230, 50],
            [25, 35, 240],
        ):
            image = tensor(color, dtype=torch.float32)[:, None, None].expand(3, 71, 53)
            noise = torch.rand(image.shape, device=self.device) * 10 - 5
            image = (image + noise).clamp(0, 255)
            mask = torch.rand((71, 53), device=self.device) > 0.15
            for foreground in (None, mask, torch.zeros_like(mask)):
                expected = _dominant_color_reference(image, foreground)
                actual = dominant_color(image, foreground)
                self.assertEqual(actual is None, expected is None)
                if actual:
                    self.assertEqual(actual[0], expected[0])
                    diff = (actual[1] - expected[1]).abs()
                    self.assertLess(float(torch.minimum(diff, 180 - diff)), 0.001)

    def test_batched_kalman_matches_individual_filters(self):
        kalman = Kalman(self.device)
        tracks, reference = [], []
        for index in range(12):
            mean, covariance = kalman.initiate(tensor([50 + index, 90, 0.5, 100]))
            state = "lost" if index % 3 == 0 else "tracked"
            tracks.append(
                Track(index + 1, 2, mean.clone(), covariance.clone(), 0, 0, state, True)
            )
            reference.append(kalman.predict(mean, covariance, lost=state == "lost"))
        kalman.predict_many(tracks)
        rows = tensor(
            [[40 + index, 40, 90 + index, 140, 0.9, 2] for index in range(12)]
        )
        from pipeline.cuda.tracking import xyah

        expected = [
            kalman.update(mean, covariance, xyah(rows[i, :4]))
            for i, (mean, covariance) in enumerate(reference)
        ]
        kalman.update_many([(track, i) for i, track in enumerate(tracks)], rows)
        for track, (mean, covariance) in zip(tracks, expected, strict=True):
            self.assertTrue(torch.allclose(track.mean, mean, atol=1.0e-10, rtol=0))
            self.assertTrue(
                torch.allclose(track.covariance, covariance, atol=1.0e-10, rtol=0)
            )

    def test_episodes_and_histograms_preserve_gaps_unknowns_and_ties(self):
        frames = [i for i in range(300) if i % 37 != 0]
        colors = {i: ("a" if i % 30 < 14 else "b") for i in frames if i % 7 != 0}
        self.assertEqual(
            split_episodes(frames, colors, self.device),
            _split_episodes_reference(frames, colors, self.device),
        )
        for colors in ({}, {frames[-1]: "a"}):
            self.assertEqual(
                split_episodes(frames, colors, self.device),
                _split_episodes_reference(frames, colors, self.device),
            )
        counter = CudaCounter(["b", "a", "a", "b", None], device=self.device)
        self.assertEqual(export(dict(counter)), {"b": 2.0, "a": 2.0, None: 1.0})
        self.assertEqual([key for key, _ in counter.most_common()], ["b", "a", None])
        counter.update({"a": 3})
        self.assertEqual(float(counter.get("a")), 5.0)
        self.assertEqual(float(counter.get("missing")), 0.0)

    def test_cached_constants_remain_independent_when_mutated(self):
        first = tensor([0.1, 0.2, 0.3])
        first.zero_()
        second = tensor([0.1, 0.2, 0.3])
        self.assertEqual(second.tolist(), [0.1, 0.2, 0.3])

    def test_gap_corridor_accepts_cuda_boxes_and_blocks_unknown_body(self):
        from pipeline.spatial_identity import _crowded_gap

        first = {"bbox": tensor([100, 100, 150, 250])}
        last = {"bbox": tensor([102, 100, 152, 250])}
        rows = {
            4: {
                "frame_index": 4,
                "segment_id": 0,
                "persons": [
                    {
                        "role": "player",
                        "track_id": None,
                        "bbox": tensor([100, 100, 150, 250]),
                    }
                ],
            }
        }
        self.assertTrue(_crowded_gap(first, last, rows, 3, 5, {}, {}, "a"))
        rows[4]["persons"][0]["bbox"] = tensor([700, 100, 750, 250])
        self.assertFalse(_crowded_gap(first, last, rows, 3, 5, {}, {}, "a"))

    def test_compact_geometry_and_collision_batches_match_reference(self):
        batch = [{**row(i, [person(i / 10)]), "scene_cut": False} for i in range(3)]
        for item in batch:
            item["persons"][0]["bbox"] = [10, 20, 40, 100]
        self.assertEqual(
            export(_minimal_batch(batch, self.device)),
            export([_minimal(item, self.device) for item in batch]),
        )
        player = {"team_id": "a", "number": "1"}
        rows, direct = {}, {}
        for i in range(515):
            rows[i] = {
                "frame_index": i,
                "segment_id": 0,
                "persons": [{"role": "player", "track_id": track} for track in (1, 2)],
            }
            direct[(0, 1, i)] = player
            if i % 3 == 0:
                direct[(0, 2, i)] = player
        self.assertEqual(
            collision_frames(rows, direct, {}, {}, self.device),
            _collision_frames_reference(rows, direct, {}, {}, self.device),
        )

    def test_fused_metrics_match_every_contribution_through_breaks_and_conflicts(self):
        config = {
            **load_config()["distance"],
            "device": self.device,
            "smoothing_tau_s": 0.12,
            "deadband_m": 0.025,
        }
        reference, fused = Distance(config, "test"), CudaDistance(config, "test")
        for index in range(160):
            people = (
                []
                if index % 19 == 0
                else [person(None if index % 13 == 0 else index * 0.05, name="A")]
            )
            if index % 31 == 0:
                people.append(person(index * 0.05, name="A", tid=2))
            frame = row(index, people, segment=index // 45, included=index % 17 != 0)
            expected, actual = reference.update(frame), export(fused.update(frame))
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(expected, actual, strict=True):
                for key in ("reason", "subject_id", "frame_index"):
                    self.assertEqual(a[key], b[key])
                for key in ("distance_m", "measured_duration_s", "interval_start_s"):
                    if a[key] is None:
                        self.assertIsNone(b[key])
                    else:
                        self.assertAlmostEqual(a[key], b[key], places=12)
        for a, b in zip(
            reference.result()["players"],
            export(fused.result())["players"],
            strict=True,
        ):
            self.assertEqual(a["warnings"], b["warnings"])
            self.assertEqual(a["excluded_intervals"], b["excluded_intervals"])
            for key in (
                "observed_distance_m",
                "tracked_duration_s",
                "measured_duration_s",
                "coverage_ratio",
            ):
                self.assertAlmostEqual(a[key], b[key], places=12)


class ResumeTests(unittest.TestCase):
    def test_retains_only_complete_segments_and_never_mutates_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            output = Path(temporary) / "output"
            source.mkdir()
            output.mkdir()
            config = {"mode": "sequential", "max_frames": None, "test": 1}
            (source / "run.json").write_text(
                json.dumps(
                    {
                        "backend": "native_cuda",
                        "run_id": "test",
                        "roster_sha256": "roster",
                        "config": config,
                    }
                )
            )
            observations = [
                {
                    "frame_index": i,
                    "run_id": "test",
                    "scene_cut": i == 3,
                    "segment_id": 0 if i < 3 else 1,
                    "persons": [{"track_id": 5 if i < 3 else 9}],
                }
                for i in range(6)
            ]
            text = "".join(json.dumps(item) + "\n" for item in observations)
            (source / "observations.partial.jsonl").write_text(text)
            (source / "jersey_reads.partial.jsonl").write_text(
                "".join(json.dumps({"frame_index": i}) + "\n" for i in range(6))
            )
            checkpoint = inspect_prefix(
                source, {**config, "mode": "parallel"}, "roster"
            )
            self.assertEqual((checkpoint["start"], checkpoint["track_counter"]), (3, 5))
            copy_prefix(checkpoint, output)
            kept = [
                json.loads(line)
                for line in (output / "observations.partial.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(kept, observations[:3])
            self.assertEqual((source / "observations.partial.jsonl").read_text(), text)
            with self.assertRaisesRegex(ValueError, "config mismatch"):
                inspect_prefix(source, {**config, "test": 2}, "roster")
