import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from training.court.calibration import CourtCalibrator, fit_keypoints, motion_between
from training.court.config import settings
from training.court.geometry import (
    CORNERS, CORNER_DISTANCE, FREE_THROW_X, LANDMARKS, LANE_WIDTH, LENGTH,
    LINE_WIDTH, RESTRICTED_RADIUS, RIM_X, THREE_RADIUS, WIDTH, markings, project,
)
from training.court.setup import install
from training.players.predict import DEFAULTS, load_config, predict_video

MATRIX = np.array([[30., 8., 100.], [-2., 30., 130.], [.001, .006, 1.]])
SHAPE = (720, 1280, 3)


def keypoints(matrix=MATRIX):
    return np.column_stack((project(matrix, LANDMARKS), np.full(18, .95)))


def texture():
    rng = np.random.default_rng(51)
    image = np.full(SHAPE, 90, np.uint8)
    for x, y, intensity in rng.integers([0, 0, 20], [1280, 720, 240], size=(1000, 3)):
        cv2.circle(image, (int(x), int(y)), 3, (int(intensity),) * 3, -1)
    return image


class Model:
    provenance = {"model": "analytical_fixture"}

    def __init__(self, sequence=None):
        self.sequence = sequence or [keypoints()]
        self.calls = 0

    def predict(self, image):
        result = self.sequence[min(self.calls, len(self.sequence) - 1)]
        self.calls += 1
        return result.copy()


def calibrator(**overrides):
    return CourtCalibrator(settings(overrides), fps=30, model=Model())


class GeometryTests(unittest.TestCase):
    def test_exact_nba_regulation_dimensions_and_landmark_order(self):
        for actual, expected in [(LENGTH, 28.6512), (WIDTH, 15.24), (LANE_WIDTH, 4.8768),
                                 (FREE_THROW_X, 5.7912), (RIM_X, 1.6002),
                                 (THREE_RADIUS, 7.239), (CORNER_DISTANCE, 6.7056),
                                 (RESTRICTED_RADIUS, 1.2192), (LINE_WIDTH, .0508)]:
            self.assertAlmostEqual(actual, expected, places=8)
        np.testing.assert_allclose(LANDMARKS[8], [5.7912, 5.1816])
        np.testing.assert_allclose(LANDMARKS[17], [22.86, 10.0584])
        self.assertEqual(LANDMARKS.shape, (18, 2))
        for line in markings():
            self.assertTrue(np.isfinite(line).all())
            self.assertTrue(((line >= -1e-6) & (line <= [LENGTH + 1e-6, WIDTH + 1e-6])).all())

    def test_ransac_rejects_outlier_and_returns_metres_not_pixels(self):
        values = keypoints()
        values[4, :2] += [130, -100]
        state, report = fit_keypoints(values, SHAPE, settings())
        self.assertIsNotNone(state)
        self.assertNotIn(4, report["inlier_ids"])
        expected = np.array([[3.8, 4.1], [23.1, 9.6], [14.3256, 7.62]])
        recovered = project(state["image_to_court"], project(MATRIX, expected))
        np.testing.assert_allclose(recovered, expected, atol=1e-5)

    def test_mirrored_model_labels_canonicalized_without_flipping_y(self):
        reflection = np.array([[-1., 0, LENGTH], [0, 1, 0], [0, 0, 1]])
        state, report = fit_keypoints(keypoints(MATRIX @ reflection), SHAPE, settings())
        self.assertTrue(report["reflected_model_labels_x"])
        np.testing.assert_allclose(project(state["image_to_court"], project(MATRIX, CORNERS)), CORNERS, atol=1e-5)

    def test_insufficient_collinear_and_nonfinite_points_rejected(self):
        values = keypoints()
        values[:, 2] = 0
        values[:4, 2] = .95
        self.assertIsNone(fit_keypoints(values, SHAPE, settings())[0])
        values[:6, 2] = .95  # all six on the same baseline
        self.assertIsNone(fit_keypoints(values, SHAPE, settings())[0])
        values[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            fit_keypoints(values, SHAPE, settings())

    def test_duplicate_image_points_cannot_count_as_independent_support(self):
        values = keypoints()
        values[:, :2] = [200, 300]
        state, report = fit_keypoints(values, SHAPE, settings())
        self.assertIsNone(state)
        self.assertEqual(len(report["candidate_ids"]), 1)

    def test_horizon_points_are_not_clamped_to_court(self):
        matrix = np.array([[1., 0, 0], [0, 1, 0], [1, 0, -1]])
        self.assertTrue(np.isnan(project(matrix, [[1, 4]])).all())

    def test_settings_cli_and_install_checksum(self):
        for value in ({"enabled": 1}, {"detect_every": 0}, {"min_inliers": 4},
                      {"device": "gpu"}, {"keypoint_confidence": 1.1},
                      {"max_propagation_seconds": float("nan")},
                      {"excluded_regions": [[0, 0, 2, 1]]}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                settings(value)
        cfg = load_config(None, video="input.mp4", checkpoint="players.pt", court=True, court_device="cuda:0")
        self.assertTrue(cfg["court"]["enabled"])
        self.assertFalse(DEFAULTS["court"]["enabled"])
        with tempfile.TemporaryDirectory() as temp:
            src, dst = Path(temp) / "input", Path(temp) / "model"
            src.write_bytes(b"fixture")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                install(dst, source=src)
            self.assertFalse(dst.exists())
            with patch("training.court.setup.MODEL_SHA256", hashlib.sha256(b"fixture").hexdigest()):
                self.assertEqual(install(dst, source=src), dst)
                self.assertEqual(install(dst), dst)
            self.assertFalse(list(Path(temp).glob("*.tmp")))


class TemporalTests(unittest.TestCase):
    def test_pan_and_zoom_propagate_in_the_correct_direction(self):
        model = Model([keypoints(), np.zeros((18, 3))])
        c = CourtCalibrator(settings({"detect_every": 100}), fps=30, model=model)
        first = texture()
        c.update(first, [], frame_index=0)
        affine = np.array([[1.008, -.002, 3], [.002, 1.008, -2], [0, 0, 1.]])
        second = cv2.warpPerspective(first, affine, (1280, 720))
        record = c.update(second, [], frame_index=1)
        self.assertEqual(record["status"], "propagated")
        expected = np.array([[5., 7.], [25., 12.]])
        source = project(affine @ MATRIX, expected)
        np.testing.assert_allclose(project(record["image_to_court"], source), expected, atol=.025)
        self.assertEqual(model.calls, 1)

    def test_no_motion_no_fit_never_reuses_stale_homography(self):
        c = CourtCalibrator(settings({"auto_cuts": False}), fps=30,
                            model=Model([keypoints(), np.zeros((18, 3))]))
        image = np.zeros(SHAPE, np.uint8)
        self.assertEqual(c.update(image, [], frame_index=0)["status"], "fit")
        row = c.update(image, [], frame_index=1)
        self.assertEqual(row["status"], "unavailable")
        self.assertIsNone(row["image_to_court"])

    def test_propagation_expires_without_absolute_recalibration(self):
        c = CourtCalibrator(settings({"detect_every": 100, "max_propagation_seconds": .04}), fps=30,
                            model=Model([keypoints(), np.zeros((18, 3))]))
        image = texture()
        c.update(image, [], frame_index=0)
        self.assertEqual(c.update(image, [], frame_index=1)["status"], "propagated")
        self.assertEqual(c.update(image, [], frame_index=2)["status"], "unavailable")

    def test_automatic_and_explicit_cuts_reset_state(self):
        for explicit in (True, False):
            c = CourtCalibrator(settings(), fps=30, model=Model([keypoints(), np.zeros((18, 3))]))
            c.update(np.zeros(SHAPE, np.uint8), [], frame_index=0)
            row = c.update(np.full(SHAPE, 255, np.uint8), [], frame_index=1, force_cut=explicit)
            self.assertTrue(row["scene_cut"])
            self.assertEqual(row["segment_id"], 1)
            self.assertIsNone(row["image_to_court"])
            self.assertIsNone(row["last_fit_frame"])

    def test_temporal_jump_is_rejected_while_motion_is_good(self):
        shifted = np.array([[1., 0, 150.], [0, 1, 0], [0, 0, 1.]]) @ MATRIX
        c = CourtCalibrator(settings({"detect_every": 1}), fps=30,
                            model=Model([keypoints(), keypoints(shifted)]))
        image = texture()
        c.update(image, [], frame_index=0)
        row = c.update(image, [], frame_index=1)
        self.assertEqual(row["status"], "propagated")
        self.assertEqual(row["fit_attempt"]["rejection"], "temporal_disagreement")

    def test_arena_flash_does_not_reset_identities(self):
        c = calibrator()
        image = texture()
        c.update(image, [], frame_index=0)
        flash = np.clip(image.astype(float) * 1.2 + 65, 0, 255).astype(np.uint8)
        row = c.update(flash, [], frame_index=1)
        self.assertFalse(row["scene_cut"])
        self.assertEqual(row["segment_id"], 0)
        self.assertGreater(row["cut_structure_correlation"], .75)

    def test_players_and_scoreboard_excluded_from_motion_mask(self):
        c = calibrator()
        image = texture()
        c.update(image, [{"bbox": [300, 300, 400, 500], "confidence": .9}], frame_index=0)
        self.assertEqual(c.previous_mask[300, 262], 0)
        self.assertEqual(c.previous_mask[-10, 300], 0)
        self.assertGreater(np.count_nonzero(c.previous_mask), 1000)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.assertIsNone(motion_between(gray, gray, np.zeros_like(gray), settings())[0])

    def test_order_enforced(self):
        with self.assertRaisesRegex(ValueError, "consecutive"):
            calibrator().update(texture(), [], frame_index=1)


class ProjectionTests(unittest.TestCase):
    def test_outside_court_and_extrapolation_are_null_not_clipped(self):
        c = calibrator(max_extrapolation_m=.5)
        record = {"image_to_court": np.linalg.inv(MATRIX).tolist(),
                  "support_polygon_m": [[1., 1.], [6., 1.], [6., 8.], [1., 8.]]}
        detections = []
        for point in [[-2., 6.], [12., 6.]]:
            x, y = project(MATRIX, [point])[0]
            detections.append({"bbox": [x - 10, y - 80, x + 10, y], "confidence": .9})
        rows = c.project_players(detections, record, SHAPE)
        self.assertEqual([r["status"] for r in rows], ["outside_court", "outside_calibrated_support"])
        self.assertTrue(all(r["position_m"] is None for r in rows))

    def test_ankles_are_explicit_and_fall_back_to_bbox(self):
        c = calibrator(footpoint="ankles")
        record = c.update(texture(), [], frame_index=0)
        x, y = project(MATRIX, [[5., 7.]])[0]
        valid = [False] * 15 + [True, True]
        row = {"bbox": [x - 20, y - 100, x + 20, y + 10], "confidence": .9,
               "pose": {"format": "coco17", "valid": valid,
                        "keypoints": [None] * 15 + [[x - 5, y], [x + 5, y]]}}
        projected = c.project_players([row], record, SHAPE)[0]
        np.testing.assert_allclose(projected["position_m"], [5., 7.], atol=1e-5)
        self.assertEqual(projected["ground_point_method"], "ankles_midpoint_approximation")
        row["pose"]["valid"][15] = False
        projected = c.project_players([row], record, SHAPE)[0]
        self.assertEqual(projected["ground_point_method"], "bbox_bottom_center")

    def test_ids_null_positions_bounds_and_feet_preserved(self):
        c = calibrator(max_extrapolation_m=0)
        record = c.update(texture(), [], frame_index=0)
        px = project(MATRIX, [[5, 7]])[0]
        valid = {"bbox": [px[0] - 20, px[1] - 100, px[0] + 20, px[1]], "confidence": .9, "track_id": 17}
        detections = [valid, {**valid, "confidence": .1}, {**valid, "bbox": [1, 1, 50, 720]}]
        original = json.dumps(detections)
        rows = c.project_players(detections, record, SHAPE)
        np.testing.assert_allclose(rows[0]["position_m"], [5, 7], atol=1e-5)
        self.assertEqual(rows[0]["track_id"], 17)
        self.assertEqual(rows[1]["status"], "low_player_confidence")
        self.assertEqual(rows[2]["status"], "feet_out_of_frame")
        self.assertIsNone(rows[2]["position_m"])
        self.assertEqual(json.dumps(detections), original)
        json.dumps(rows, allow_nan=False)
        rows = c.project_players([valid], {**record, "image_to_court": None}, SHAPE)
        self.assertIsNone(rows[0]["position_m"])


class IntegrationTests(unittest.TestCase):
    def test_scene_cut_and_explicit_reset_do_not_reuse_ids_or_double_reset(self):
        from training.players.tracking import PlayerTracker
        tracker = PlayerTracker({"enabled": True, "tracker": "bytetrack", "reset_frames": [1, 3]}, fps=30)
        image = texture()
        detection = {"class_id": 0, "bbox": [300., 200., 350., 350.], "confidence": .9}
        first = tracker.update([detection], image, frame_index=0)[0]["track_id"]
        second = tracker.update([detection], image, frame_index=1, scene_cut=True)[0]["track_id"]
        self.assertNotEqual(first, second)
        self.assertEqual(tracker.segment_id, 1)
        tracker.update([detection], image, frame_index=2)
        tracker.update([detection], image, frame_index=3)
        self.assertEqual(tracker.segment_id, 2)

    def run_video(self, root, *, enabled=True, stop=None):
        path = root / "input.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
        self.assertTrue(writer.isOpened())
        for _ in range(3):
            writer.write(texture())
        writer.release()
        class Detector:
            provenance = {"family": "fixture"}
            def predict(self, image, **kwargs):
                return [{"class_id": 0, "bbox": [300., 200., 350., 350.], "confidence": .9}]
        cfg = {**DEFAULTS, "video": str(path), "output": str(root / "output"),
               "court": settings({"enabled": enabled})}
        with patch("training.players.predict.runtime", return_value={}):
            return predict_video(cfg, detector=Detector(), court_calibrator=calibrator(), stop_after_frame=stop)

    def test_video_sidecars_geometry_and_optional_path(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                result = self.run_video(root, enabled=enabled)
                self.assertEqual(result["court"]["enabled"], enabled)
                path = root / "output/calibrations.jsonl"
                if not enabled:
                    self.assertFalse(path.exists())
                    continue
                rows = [json.loads(s) for s in path.read_text().splitlines()]
                self.assertEqual([r["frame_index"] for r in rows], [0, 1, 2])
                self.assertTrue(all(r["image_to_court"] is not None for r in rows))
                positions = [json.loads(s) for s in (root / "output/court_positions.jsonl").read_text().splitlines()]
                self.assertEqual(len(positions), 3)
                self.assertEqual(rows[0]["source_sha256"], positions[0]["source_sha256"])
                self.assertIn("tactical.mp4", result["artifacts"])
                cap = cv2.VideoCapture(str(root / "output/tactical.mp4"))
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
                cap.release()

    def test_interrupt_keeps_partial_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(KeyboardInterrupt):
                self.run_video(root, stop=2)
            out = root / "output"
            self.assertFalse((out / "calibrations.jsonl").exists())
            self.assertEqual(len((out / "calibrations.partial.jsonl").read_text().splitlines()), 2)
            self.assertTrue((out / "tactical.partial.mp4").exists())
            self.assertEqual(json.loads((out / "progress.json").read_text())["status"], "KILLED")


if __name__ == "__main__":
    unittest.main()
