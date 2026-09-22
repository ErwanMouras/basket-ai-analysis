"""Geometry, overlap, scoring and review contracts without a model dependency."""

import csv
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from training.ball.evaluation.config import load_config
from training.ball.evaluation.frames import (
    frame_result,
    merge_windows,
    report,
    summarize,
)
from training.ball.evaluation.video import write_review, write_videos


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "threshold": 0.5,
            "tolerance_px": 4,
            "tolerance_space": "source",
            "unknown_position": "ignore",
        }
        self.geometry = {"input_width": 100, "input_height": 100}
        self.source = {
            "video": "val/game/clip.mp4",
            "sidecar": "val/game/clip.mp4.ballann.json",
            "geometry": {"left": 0, "top": 50, "scale_x": 0.5, "scale_y": 0.5},
            "video_metadata": {"width": 400, "height": 200, "fps": 25},
        }
        self.frame = {
            "image": "images/val/clip_a/0.png",
            "clip_id": "clip_a",
            "match_id": "game",
            "split": "val",
            "frame_index": 0,
            "timestamp_seconds": 0,
            "width": 200,
            "height": 200,
            "status": "visible",
            "occluded": False,
            "position_excluded": False,
            "has_position": True,
            "source_annotation": {"cx": 100, "cy": 60, "radius": 4, "occluded": False},
            "position": [50, 80],
        }

    def result(self, point=(25, 40), frame=None):
        prediction = {
            "position_input": point,
            "confidence": 0.9,
            "window_count": 2,
            "future_context_frames": 2,
        }
        return frame_result(
            frame or self.frame, self.source, prediction, self.geometry, self.config
        )

    def test_letterbox_inverse_and_stretch_use_separate_axes(self):
        row = self.result()
        self.assertEqual(row["position_export"], [50, 80])
        self.assertEqual(row["position_source"], [100, 60])
        self.assertEqual(row["error_source_px"], 0)
        self.source["geometry"].update(scale_y=1, top=20)
        self.assertEqual(self.result()["position_source"], [100, 60])

    def test_padding_prediction_is_not_clipped_onto_source(self):
        row = self.result((25, 0))
        self.assertEqual(row["position_source"], [100, -100])
        self.assertFalse(row["prediction_in_source"])
        self.assertEqual(row["outcome"], "displaced")

    def test_tolerance_is_in_the_selected_coordinate_space(self):
        self.assertEqual(self.result((27, 40))["outcome"], "displaced")
        self.config["tolerance_space"] = "export"
        self.assertEqual(self.result((27, 40))["outcome"], "tp")

    def test_fusion_averages_heatmaps_including_absent_votes_before_decoding(self):
        frames = [{"image": str(i), "frame_index": i} for i in range(4)]
        samples = [({}, frames[:3]), ({}, frames[1:])]
        first, second = np.zeros((3, 4, 4)), np.zeros((3, 4, 4))
        first[:, 1, 2] = 0.8
        second[1:, 1, 2] = 0.6
        merged = dict(
            merge_windows(
                samples, iter([(frames[:3], first), (frames[1:], second)]), 0.5
            )
        )
        self.assertEqual(len(merged), 4)
        self.assertIsNone(merged["1"]["position_input"])
        self.assertEqual(merged["2"]["position_input"], [2, 1])
        self.assertEqual(
            [merged[str(i)]["window_count"] for i in range(4)], [1, 2, 2, 1]
        )
        self.assertEqual(merged["1"]["future_context_frames"], 2)
        self.assertEqual(merged["3"]["future_context_frames"], 0)
        with self.assertRaises(ValueError):
            dict(merge_windows(samples, iter([(frames[:3], first)]), 0.5))

    def test_displacement_is_one_fp_and_one_fn_and_errors_include_failures(self):
        rows = [self.result(), self.result((35, 40)), self.result(None)]
        result = summarize(rows)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 2))
        self.assertEqual(result["localized_pairs"], 2)
        self.assertEqual(result["error_source_mean_px"], 20)
        self.assertEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["recall"], 1 / 3)

    def test_unknown_and_excluded_annotations_are_not_implicit_negatives(self):
        frame = deepcopy(self.frame)
        frame.update(status="unknown_position", has_position=False, position=None)
        frame["source_annotation"] = {
            "cx": None,
            "cy": None,
            "radius": None,
            "occluded": True,
        }
        self.assertEqual(self.result(frame=frame)["outcome"], "ignored")
        self.config["unknown_position"] = "negative"
        self.assertEqual(self.result(frame=frame)["outcome"], "fp")
        self.assertEqual(self.result(None, frame)["outcome"], "tn")
        frame.update(status="occluded_with_position", position_excluded=True)
        self.assertEqual(self.result(frame=frame)["outcome"], "ignored")

    def test_uncovered_frames_report_coverage_without_becoming_model_misses(self):
        row = frame_result(self.frame, self.source, None, self.geometry, self.config)
        metrics = report([row, self.result()])
        self.assertEqual(metrics["overall"]["coverage"], 0.5)
        self.assertEqual(metrics["overall"]["known_coverage"], 0.5)
        self.assertEqual(metrics["overall"]["fn"], 0)
        self.assertEqual(metrics["by_match_id"]["game"], metrics["overall"])
        self.assertIsNone(summarize([row])["recall"])

    def test_review_video_timeline_preserves_source_frame_identity_and_gaps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / self.frame["image"]
            image.parent.mkdir(parents=True)
            cv2.imwrite(str(image), np.zeros((200, 200, 3), np.uint8))
            first = self.result()
            second = {**self.result(None), "frame_index": 3, "timestamp_seconds": 0.12}
            output = root / "review"
            output.mkdir()
            write_videos(root, output, [first, second], {"clip_a": self.source}, 200)
            write_review(output, [first, second])
            with (output / "video_frames.csv").open() as handle:
                timeline = list(csv.DictReader(handle))
            self.assertEqual(timeline[1]["frame_index"], "3")
            self.assertEqual(timeline[1]["omitted_source_frames"], "2")
            video = cv2.VideoCapture(str(output / "videos/clip_a.mp4"))
            try:
                self.assertEqual(int(video.get(cv2.CAP_PROP_FRAME_COUNT)), 2)
                ok, decoded = video.read()
                self.assertTrue(ok)
                self.assertTrue(decoded.any())
            finally:
                video.release()
            with (output / "review.csv").open() as handle:
                candidates = list(csv.DictReader(handle))
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]["outcome"], "fn")
            self.assertEqual(candidates[0]["review_decision"], "")

    def test_config_rejects_test_and_invalid_thresholds(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.yaml"
            for text in (
                "split: test",
                "threshold: .nan",
                "tolerance_px: 0",
                "workers: -1",
                "threshold_typo: 0.5",
            ):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    load_config(path, "checkpoint.pt")


if __name__ == "__main__":
    unittest.main()
