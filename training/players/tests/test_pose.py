"""Pose geometry, association, batching and video artifacts without model downloads."""

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from training.players.pose import (
    PlayerPose,
    create_session,
    decode,
    draw_pose,
    preprocess,
    settings,
)
from training.players.pose_setup import install
from training.players.predict import DEFAULTS as VIDEO_DEFAULTS
from training.players.predict import load_config, predict_video


def detection(x=10, score=0.9, track_id=7):
    return {
        "class_id": 0,
        "bbox": [float(x), 10.0, float(x + 20), 60.0],
        "confidence": score,
        "track_id": track_id,
    }


def frame():
    return np.zeros((80, 120, 3), dtype=np.uint8)


class Session:
    def __init__(self, *, fixed=False):
        self.fixed = fixed
        self.batches = []

    def get_inputs(self):
        return [
            SimpleNamespace(
                name="input",
                type="tensor(float)",
                shape=[1 if self.fixed else "batch", 3, 256, 192],
            )
        ]

    def get_outputs(self):
        return [SimpleNamespace(name=n) for n in ("simcc_x", "simcc_y")]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, names, inputs):
        batch = inputs["input"]
        assert batch.dtype == np.float32
        self.batches.append(len(batch))
        x = np.zeros((len(batch), 17, 384), dtype=np.float32)
        y = np.zeros((len(batch), 17, 512), dtype=np.float32)
        x[:, :, 192] = 0.9
        y[:, :, 256] = 0.7
        return [x, y]


def estimator(**overrides):
    model = PlayerPose.__new__(PlayerPose)
    model.config = settings(overrides)
    model.session = Session()
    model._initialize()
    model.provenance = {"model": "rtmpose-m-fixture", "format": "coco17"}
    return model


class PoseTests(unittest.TestCase):
    def test_invalid_configuration_and_cli(self):
        for override in (
            {"enabled": 1},
            {"batch_size": 0},
            {"device": "cuda"},
            {"min_bbox_area": float("nan")},
            {"max_players": True},
            {"keypoint_min_confidence": 2},
            {"weights": "https://example/model.onnx"},
        ):
            with (
                self.subTest(override=override),
                self.assertRaises((TypeError, ValueError)),
            ):
                settings(override)
        with self.assertRaisesRegex(ValueError, "score_floor"):
            settings({"enabled": True}, score_floor=0.9)
        config = load_config(
            None,
            video="video.mp4",
            checkpoint="model.pt",
            pose=True,
            pose_device="cuda:0",
        )
        self.assertTrue(config["pose"]["enabled"])
        self.assertEqual(config["pose"]["device"], "cuda:0")
        self.assertFalse(VIDEO_DEFAULTS["pose"]["enabled"])

    def test_preprocess_preserves_bgr_and_restores_box_center(self):
        image = np.full((80, 120, 3), [10, 20, 30], dtype=np.uint8)
        batch, center, scale = preprocess(image, [20.0, 20.0, 60.0, 60.0])
        self.assertEqual(batch.shape, (3, 256, 192))
        self.assertEqual(batch.dtype, np.float32)
        np.testing.assert_allclose(
            batch[:, 128, 96],
            (np.array([10, 20, 30]) - [123.675, 116.28, 103.53])
            / [58.395, 57.12, 57.375],
        )
        raw = Session().run([], {"input": batch[None]})
        points, scores = decode(raw, [center], [scale])
        np.testing.assert_allclose(points[0], np.tile([40.0, 40.0], (17, 1)))
        np.testing.assert_allclose(scores, 0.8)

    def test_candidates_keep_detection_index_and_track_id_with_bounded_batches(self):
        model = estimator(batch_size=2, max_players=3)
        detections = [
            detection(1, 0.01),
            detection(20, track_id=12),
            detection(50, track_id=3),
            detection(80, track_id=None),
        ]
        original = json.dumps(detections)
        result = model.predict(frame(), detections)
        self.assertEqual(model.session.batches, [2, 1])
        self.assertEqual(result[0]["pose_status"], "low_detection_confidence")
        self.assertEqual([d["track_id"] for d in result], [7, 12, 3, None])
        self.assertEqual([d["detection_index"] for d in result], [0, 1, 2, 3])
        for i in range(1, 4):
            np.testing.assert_allclose(
                result[i]["pose"]["keypoints"][0], [detections[i]["bbox"][0] + 10, 35]
            )
        self.assertEqual(json.dumps(detections), original)

    def test_empty_frame_and_filtered_detections_never_infer_whole_image(self):
        model = estimator()
        self.assertEqual(model.predict(frame(), []), [])
        self.assertIsNone(model.predict(frame(), [detection(score=0.01)])[0]["pose"])
        tiny = {**detection(), "bbox": [1.0, 1.0, 3.0, 3.0]}
        self.assertEqual(model.predict(frame(), [tiny])[0]["pose_status"], "small_bbox")
        self.assertEqual(model.session.batches, [])

    def test_largest_boxes_selected_without_changing_output_order(self):
        model = estimator(max_players=1)
        larger = {**detection(track_id=90), "bbox": [30.0, 1.0, 70.0, 79.0]}
        result = model.predict(frame(), [detection(), larger])
        self.assertEqual(result[0]["pose_status"], "player_limit")
        self.assertEqual(result[1]["track_id"], 90)
        self.assertIsNotNone(result[1]["pose"])

    def test_fixed_batch_one_and_signature_validation(self):
        model = estimator()
        model.session = Session(fixed=True)
        model._initialize()
        model.predict(frame(), [detection(), detection(50)])
        self.assertEqual(model.session.batches, [1, 1])
        with (
            patch.object(model.session, "get_inputs", return_value=[]),
            self.assertRaisesRegex(ValueError, "signature"),
        ):
            model._initialize()

    def test_bad_outputs_fail_instead_of_misaligning_players(self):
        for output in (
            [np.zeros((1, 16, 384)), np.zeros((1, 17, 512))],
            [np.full((1, 17, 384), np.nan), np.zeros((1, 17, 512))],
        ):
            model = estimator()
            with (
                patch.object(model.session, "run", return_value=output),
                self.assertRaises(RuntimeError),
            ):
                model.predict(frame(), [detection()])

    def test_low_confidence_and_outside_points_are_not_drawn(self):
        model = estimator(keypoint_min_confidence=0.9)
        result = model.predict(frame(), [detection()])[0]
        self.assertEqual(result["pose_status"], "low_keypoint_confidence")
        with (
            patch("training.players.pose.cv2.line") as line,
            patch("training.players.pose.cv2.circle") as circle,
        ):
            draw_pose(frame(), result["pose"], (0, 255, 0))
            line.assert_not_called()
            circle.assert_not_called()
        x, y = np.zeros((1, 17, 384)), np.zeros((1, 17, 512))
        x[:, :, 0] = y[:, :, 0] = 0.9
        model = estimator()
        with patch.object(model.session, "run", return_value=[x, y]):
            pose = model.predict(
                frame(), [{**detection(), "bbox": [0.0, 0.0, 120.0, 80.0]}]
            )[0]["pose"]
        self.assertEqual(pose["valid_keypoints"], 0)
        self.assertLess(pose["keypoints"][0][0], 0)

    def test_missing_response_has_null_coordinates_and_safe_json(self):
        model = estimator()
        outputs = [np.zeros((1, 17, 384)), np.zeros((1, 17, 512))]
        with patch.object(model.session, "run", return_value=outputs):
            result = model.predict(frame(), [detection()])
        self.assertEqual(result[0]["pose"]["keypoints"], [None] * 17)
        json.dumps(result, allow_nan=False)

    def test_model_hash_checked_before_session_and_install_is_atomic(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.onnx"
            source.write_bytes(b"fixture")
            target = root / "model.onnx"
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                install(target, source=source)
            self.assertFalse(target.exists())
            with patch(
                "training.players.pose_setup.MODEL_SHA256",
                hashlib.sha256(b"fixture").hexdigest(),
            ):
                self.assertEqual(install(target, source=source), target)
                self.assertEqual(install(target), target)
            with self.assertRaisesRegex(ValueError, "different checksum"):
                install(target, source=source)
            self.assertEqual(target.read_bytes(), b"fixture")
            with (
                patch("training.players.pose.create_session") as create,
                self.assertRaisesRegex(ValueError, "SHA-256"),
            ):
                PlayerPose(settings({"weights": str(source)}))
            create.assert_not_called()
            self.assertFalse(list(root.glob("*.tmp")))

    def test_model_missing_and_bad_input_fail(self):
        with self.assertRaises(FileNotFoundError):
            PlayerPose(settings({"weights": "/missing/pose.onnx"}))
        model = estimator()
        with self.assertRaisesRegex(ValueError, "uint8"):
            model.predict(frame().astype(float), [])
        with self.assertRaisesRegex(ValueError, "source-space"):
            model.predict(frame(), [detection(-1)])

    def test_cuda_session_cannot_silently_fall_back_to_cpu(self):
        runtime = SimpleNamespace(
            __version__="1.27.0",
            SessionOptions=SimpleNamespace,
            preload_dlls=lambda: None,
            get_available_providers=lambda: [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
            InferenceSession=lambda *args, **kwargs: Session(),
        )
        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
        )

        def version(package):
            if package == "onnxruntime-gpu":
                return "1.27.0"
            from importlib.metadata import PackageNotFoundError

            raise PackageNotFoundError(package)

        with (
            patch.dict(sys.modules, {"onnxruntime": runtime, "torch": torch}),
            patch(
                "training.players.pose.importlib.metadata.version", side_effect=version
            ),
            self.assertRaisesRegex(RuntimeError, "silent CPU fallback"),
        ):
            create_session("fixture.onnx", settings({"device": "cuda:0"}))


class VideoPoseTests(unittest.TestCase):
    def run_video(self, root, *, stop=None, enabled=True):
        path = root / "input.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (120, 80)
        )
        self.assertTrue(writer.isOpened())
        for _ in range(4):
            writer.write(frame())
        writer.release()

        class Detector:
            def __init__(self):
                self.provenance = {"family": "fixture"}
                self.index = 0

            def predict(self, image, **kwargs):
                self.index += 1
                row = detection()
                row.pop("track_id")
                return [] if self.index == 2 else [row]

        cfg = {
            **VIDEO_DEFAULTS,
            "video": str(path),
            "output": str(root / "output"),
            "pose": settings({"enabled": enabled}),
        }
        with patch("training.players.predict.runtime", return_value={}):
            return predict_video(
                cfg,
                detector=Detector(),
                pose_estimator=estimator(),
                stop_after_frame=stop,
            )

    def test_video_pose_sidecar_and_disabled_path(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                result = self.run_video(root, enabled=enabled)
                self.assertEqual(result["pose"]["enabled"], enabled)
                path = root / "output/poses.jsonl"
                if not enabled:
                    self.assertFalse(path.exists())
                    continue
                rows = [json.loads(s) for s in path.read_text().splitlines()]
                self.assertEqual([r["frame_index"] for r in rows], list(range(4)))
                self.assertEqual(rows[1]["detections"], [])
                self.assertIsNone(rows[0]["detections"][0]["track_id"])
                self.assertEqual(result["pose"]["posed_observations"], 3)
                self.assertEqual(result["pose"]["valid_keypoints"], 51)
                raw = json.loads(
                    (root / "output/predictions.jsonl").read_text().splitlines()[0]
                )
                self.assertNotIn("pose", raw["detections"][0])
                self.assertEqual(rows[0]["source_sha256"], raw["source_sha256"])
                self.assertFalse(list((root / "output").glob("*.partial.*")))

    def test_interruption_retains_partial_pose(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(KeyboardInterrupt):
                self.run_video(root, stop=2)
            self.assertEqual(
                len((root / "output/poses.partial.jsonl").read_text().splitlines()), 2
            )
            self.assertFalse((root / "output/poses.jsonl").exists())
            self.assertEqual(
                json.loads((root / "output/progress.json").read_text())["status"],
                "KILLED",
            )


@unittest.skipUnless(
    os.environ.get("PLAYERS_POSE_WEIGHTS"),
    "Set PLAYERS_POSE_WEIGHTS for real local ONNX inference",
)
class RealPoseTests(unittest.TestCase):
    def test_real_model_batch_matches_single(self):
        cfg = {
            "weights": os.environ["PLAYERS_POSE_WEIGHTS"],
            "batch_size": 1,
            "device": os.environ.get("PLAYERS_POSE_DEVICE", "cpu"),
        }
        single = PlayerPose(settings(cfg))
        batched = PlayerPose(settings({**cfg, "batch_size": 2}))
        image = np.random.default_rng(2).integers(0, 256, (80, 120, 3), dtype=np.uint8)
        detections = [detection(), detection(60, track_id=8)]
        a, b = single.predict(image, detections), batched.predict(image, detections)
        for left, right in zip(a, b):
            self.assertEqual(left["track_id"], right["track_id"])
            np.testing.assert_allclose(
                left["pose"]["scores"], right["pose"]["scores"], atol=1e-4
            )
            np.testing.assert_allclose(
                left["pose"]["keypoints"], right["pose"]["keypoints"], atol=0.5
            )


if __name__ == "__main__":
    unittest.main()
