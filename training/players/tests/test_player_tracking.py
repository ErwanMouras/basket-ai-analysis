"""Real online tracker regressions, without detector weights or downloads."""

import importlib.metadata
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from training.players.predict import DEFAULTS as VIDEO_DEFAULTS
from training.players.predict import load_config, predict_video
from training.players.tracking import PlayerTracker, settings


def available():
    try:
        return (
            importlib.metadata.version("ultralytics") == "8.4.39"
            and importlib.metadata.version("lap") == "0.5.12"
        )
    except importlib.metadata.PackageNotFoundError:
        return False


def detection(x=10, score=0.9):
    return {
        "class_id": 0,
        "bbox": [float(x), 10.0, float(x + 15), 50.0],
        "confidence": score,
    }


def frame():
    return np.zeros((80, 120, 3), dtype=np.uint8)


class ConfigTests(unittest.TestCase):
    def test_invalid_settings_and_score_floor(self):
        for values in (
            {"tracker": "unknown"},
            {"enabled": 1},
            {"track_high_thresh": float("nan")},
            {"track_buffer_seconds": 0},
            {"track_low_thresh": 0.3},
            {"new_track_thresh": 0.1},
            {"reset_frames": [5, 4]},
            {"reset_frames": [4, 4]},
            {"reset_frames": [True]},
            {"with_reid": True},
            {"gmc_method": "bad"},
        ):
            with (
                self.subTest(values=values),
                self.assertRaises((ValueError, TypeError)),
            ):
                settings(values)
        with self.assertRaisesRegex(ValueError, "score_floor"):
            settings({"enabled": True}, score_floor=0.2)

    def test_recipe_cli_and_defaults_do_not_mutate(self):
        config = load_config(
            None,
            video="example.mp4",
            checkpoint="example.pt",
            tracker="bytetrack",
            output="runs/example",
            max_frames=5,
        )
        self.assertTrue(config["tracking"]["enabled"])
        self.assertEqual(config["tracking"]["tracker"], "bytetrack")
        self.assertEqual(config["max_frames"], 5)
        self.assertFalse(VIDEO_DEFAULTS["tracking"]["enabled"])
        disabled = load_config(
            "training/players/configs/predict_video.yaml", tracker="none"
        )
        self.assertFalse(disabled["tracking"]["enabled"])

    def test_missing_lap_fails_before_backend_import(self):
        real = importlib.metadata.version

        def version(package):
            if package == "lap":
                raise importlib.metadata.PackageNotFoundError(package)
            return "8.4.39" if package == "ultralytics" else real(package)

        with (
            patch(
                "training.players.tracking.importlib.metadata.version",
                side_effect=version,
            ),
            self.assertRaisesRegex(RuntimeError, "lap==0.5.12"),
        ):
            PlayerTracker(settings(), fps=30.0)


@unittest.skipUnless(available(), "Install the dedicated players lock including lap")
class TrackerTests(unittest.TestCase):
    def make_tracker(self, name="botsort", **overrides):
        return PlayerTracker(settings({"tracker": name, **overrides}), fps=10.0)

    def test_ids_follow_boxes_when_order_changes_and_low_scores_are_recovered(self):
        for name in ("botsort", "bytetrack"):
            with self.subTest(tracker=name):
                tracker = self.make_tracker(name)
                first = [detection(10), detection(80)]
                initial = tracker.update(first, frame(), frame_index=0)
                a, b = [d["track_id"] for d in initial]
                self.assertNotEqual(a, b)
                # A rejected detection before both confidence subsets must not
                # shift their exported association back to index zero.
                second = [detection(50, 0.01), detection(79, 0.2), detection(11, 0.9)]
                result = tracker.update(second, frame(), frame_index=1)
                self.assertEqual([d["track_id"] for d in result], [None, b, a])
                self.assertEqual(
                    [d["bbox"] for d in result], [d["bbox"] for d in second]
                )
                self.assertNotIn("track_id", first[0])

    def test_short_occlusion_retains_id_and_expiration_creates_new_id(self):
        for name in ("botsort", "bytetrack"):
            with self.subTest(tracker=name):
                tracker = self.make_tracker(name, track_buffer_seconds=0.3)
                old = tracker.update([detection()], frame(), frame_index=0)[0][
                    "track_id"
                ]
                self.assertEqual(tracker.update([], frame(), frame_index=1), [])
                self.assertEqual(
                    tracker.update([detection()], frame(), frame_index=2)[0][
                        "track_id"
                    ],
                    old,
                )
                for index in range(3, 7):
                    tracker.update([], frame(), frame_index=index)
                self.assertIsNone(
                    tracker.update([detection()], frame(), frame_index=7)[0]["track_id"]
                )
                new = tracker.update([detection()], frame(), frame_index=8)[0][
                    "track_id"
                ]
                self.assertIsInstance(new, int)
                self.assertNotEqual(old, new)

    def test_new_objects_need_confirmation_after_first_frame(self):
        tracker = self.make_tracker()
        tracker.update([], frame(), frame_index=0)
        self.assertIsNone(
            tracker.update([detection()], frame(), frame_index=1)[0]["track_id"]
        )
        self.assertIsInstance(
            tracker.update([detection()], frame(), frame_index=2)[0]["track_id"], int
        )

    def test_explicit_cut_never_reuses_an_id(self):
        tracker = self.make_tracker(reset_frames=[2])
        ids = [
            tracker.update([detection()], frame(), frame_index=i)[0]["track_id"]
            for i in range(4)
        ]
        self.assertEqual(ids[0], ids[1])
        self.assertEqual(ids[2], ids[3])
        self.assertGreater(ids[2], ids[0])
        self.assertEqual(tracker.segment_id, 1)

    def test_interleaved_tracker_instances_have_independent_id_counters(self):
        a = self.make_tracker()
        self.assertEqual(
            a.update([detection()], frame(), frame_index=0)[0]["track_id"], 1
        )
        b = self.make_tracker("bytetrack")
        self.assertEqual(
            b.update([detection()], frame(), frame_index=0)[0]["track_id"], 1
        )
        for tracker in (a, b):
            tracker.update([detection(), detection(80)], frame(), frame_index=1)
            result = tracker.update(
                [detection(80), detection()], frame(), frame_index=2
            )
            self.assertEqual([d["track_id"] for d in result], [2, 1])

    def test_bad_frames_boxes_and_version_fail_explicitly(self):
        tracker = self.make_tracker()
        for index in (-1, 1, True):
            with self.assertRaisesRegex(ValueError, "consecutive"):
                tracker.update([], frame(), frame_index=index)
        with self.assertRaisesRegex(ValueError, "Invalid source"):
            tracker.update([detection(-1)], frame(), frame_index=0)
        with self.assertRaisesRegex(ValueError, "uint8"):
            tracker.update([], frame().astype(float), frame_index=0)
        tracker.update([], frame(), frame_index=0)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            tracker.update([], frame()[:40], frame_index=1)
        with (
            patch(
                "training.players.tracking.importlib.metadata.version",
                return_value="999",
            ),
            self.assertRaisesRegex(RuntimeError, "requires ultralytics"),
        ):
            self.make_tracker()

    def test_buffer_is_in_source_frames_at_actual_fps(self):
        tracker = PlayerTracker(settings({"track_buffer_seconds": 1.0}), fps=29.97)
        self.assertEqual(tracker.buffer_frames, 30)
        self.assertEqual(tracker.provenance["with_reid"], False)

    def test_long_empty_tail_does_not_accumulate_tracks(self):
        tracker = self.make_tracker("bytetrack", track_buffer_seconds=0.1)
        tracker.update([detection()], frame(), frame_index=0)
        for index in range(1, 200):
            tracker.update([], frame(), frame_index=index)
        self.assertFalse(tracker._backend.tracked_stracks)
        self.assertFalse(tracker._backend.lost_stracks)
        self.assertLessEqual(len(tracker._backend.removed_stracks), 1000)


@unittest.skipUnless(available(), "Install the dedicated players lock including lap")
class TrackingVideoTests(unittest.TestCase):
    def run_video(self, root, *, tracker="botsort", stop=None, family="yolo"):
        video = root / "input.mp4"
        if not video.exists():
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (120, 80)
            )
            self.assertTrue(writer.isOpened())
            for _ in range(6):
                writer.write(frame())
            writer.release()

        class Detector:
            def __init__(self):
                self.provenance = {"family": family, "checkpoint_sha256": "fixture"}
                self.index = 0

            def predict(self, image, **kwargs):
                self.index += 1
                return [] if self.index == 3 else [detection(10 + self.index)]

        config = {
            **VIDEO_DEFAULTS,
            "video": str(video),
            "output": str(root / family),
            "tracking": settings(
                {"enabled": tracker is not None, "tracker": tracker or "botsort"}
            ),
        }
        with patch("training.players.predict.runtime", return_value={}):
            return predict_video(config, detector=Detector(), stop_after_frame=stop)

    def test_streamed_sidecar_overlay_provenance_and_detector_independence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for family, tracker in (("yolo", "botsort"), ("rfdetr", "bytetrack")):
                with (
                    self.subTest(family=family),
                    patch(
                        "training.players.predict.cv2.putText", wraps=cv2.putText
                    ) as draw,
                ):
                    result = self.run_video(root, family=family, tracker=tracker)
                    out = root / family
                    rows = [
                        json.loads(s)
                        for s in (out / "tracks.jsonl").read_text().splitlines()
                    ]
                    raw = [
                        json.loads(s)
                        for s in (out / "predictions.jsonl").read_text().splitlines()
                    ]
                    self.assertEqual(len(rows), 6)
                    self.assertEqual(rows[2]["detections"], [])
                    ids = [d["track_id"] for row in rows for d in row["detections"]]
                    self.assertEqual(ids, [1] * 5)
                    for row, original in zip(rows, raw):
                        self.assertEqual(
                            row["source_sha256"], original["source_sha256"]
                        )
                        self.assertEqual(
                            [
                                {k: v for k, v in d.items() if k != "track_id"}
                                for d in row["detections"]
                            ],
                            original["detections"],
                        )
                    self.assertTrue(
                        all(
                            call.args[1].startswith("#1 ")
                            for call in draw.call_args_list
                        )
                    )
                    self.assertEqual(result["tracking"]["tracked_observations"], 5)
                    self.assertEqual(
                        set(result["artifacts"]),
                        {
                            "tracks.jsonl",
                            "tracking.json",
                            "predictions.jsonl",
                            "annotated.mp4",
                        },
                    )
                    metadata = json.loads((out / "tracking.json").read_text())
                    self.assertEqual(
                        metadata["tracking_run_id"], rows[0]["tracking_run_id"]
                    )
                    self.assertEqual(metadata["detector"]["family"], family)
                    self.assertFalse(list(out.glob("*.partial.*")))

    def test_interruption_keeps_only_partial_tracks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(KeyboardInterrupt):
                self.run_video(root, stop=2)
            out = root / "yolo"
            self.assertEqual(
                len((out / "tracks.partial.jsonl").read_text().splitlines()), 2
            )
            self.assertFalse((out / "tracks.jsonl").exists())
            self.assertFalse((out / "result.json").exists())
            self.assertEqual(
                json.loads((out / "progress.json").read_text())["status"], "KILLED"
            )

    def test_disabled_tracking_preserves_detection_only_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = self.run_video(root, tracker=None)
            self.assertFalse(result["tracking"]["enabled"])
            self.assertFalse((root / "yolo/tracks.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
