import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from pipeline.config import load_config
from pipeline.distance import Distance
from pipeline.run import analyze, recompute
from pipeline.runtime import Runtime
from training.jersey.config import settings as jersey_settings
from training.jersey.temporal import JerseyRecognizer


def person(x=0.0, tid=1, role="player", name=None, method="bbox"):
    jersey = {"status": "confirmed", "number": "23", "identity_status": "unique_number",
              "team_id": "a", "player_name": name} if name else {}
    return {"track_id": tid, "role": role, "position_m": None if x is None else [x, 0.0],
            "position_status": "calibration_unavailable" if x is None else "projected",
            "ground_point_method": method, "jersey": jersey}


def row(i, people, segment=0, included=True, timestamp=None):
    return {"frame_index": i, "timestamp_s": i * .1 if timestamp is None else timestamp,
            "segment_id": segment, "included": included, "persons": people}


class DistanceTests(unittest.TestCase):
    def metric(self, **overrides):
        return Distance({**load_config()["distance"], "smoothing_tau_s": 0.0,
                         "deadband_m": 0.0, **overrides}, "test")

    def test_known_line(self):
        m = self.metric()
        for i in range(11):
            m.update(row(i, [person(i * .1)]))
        p = m.result()["players"][0]
        self.assertAlmostEqual(p["observed_distance_m"], 1.0)
        self.assertAlmostEqual(p["coverage_ratio"], 1.0)

    def test_stationary_noise_deadband(self):
        m = self.metric(deadband_m=.03, smoothing_tau_s=.12)
        for i in range(100):
            m.update(row(i, [person(.01 * (-1) ** i)]))
        self.assertEqual(m.result()["players"][0]["observed_distance_m"], 0.0)

    def test_missing_position_breaks_pairs(self):
        m = self.metric()
        for i, x in enumerate([0, .1, None, .3, .4]):
            m.update(row(i, [person(x)]))
        p = m.result()["players"][0]
        self.assertAlmostEqual(p["observed_distance_m"], .2)
        self.assertAlmostEqual(p["coverage_ratio"], .5)
        self.assertEqual(p["measurement_status"], "partial")

    def test_missing_detection_never_bridged(self):
        m = self.metric()
        for i in range(5):
            m.update(row(i, [] if i == 2 else [person(i * .1)]))
        self.assertAlmostEqual(m.result()["players"][0]["observed_distance_m"], .2)

    def test_cuts_sum_without_joining_coordinates(self):
        m = self.metric()
        for i, x in enumerate([0, .1, 10, 10.1]):
            m.update(row(i, [person(x, name="A")], segment=i // 2))
        people = m.result()["players"]
        self.assertEqual(len(people), 1)
        self.assertAlmostEqual(people[0]["observed_distance_m"], .2)
        self.assertEqual(len(people[0]["track_refs"]), 2)

    def test_null_is_not_zero(self):
        m = self.metric()
        m.update(row(0, [person()]))
        self.assertIsNone(m.result()["players"][0]["observed_distance_m"])
        m.update(row(1, [person()]))
        self.assertEqual(m.result()["players"][0]["observed_distance_m"], 0)

    def test_referee_and_unknown_excluded(self):
        m = self.metric()
        for i in range(2):
            m.update(row(i, [person(i, role="referee"), person(i, tid=2, role="unknown")]))
        self.assertEqual(m.result()["players"], [])

    def test_duplicate_identity_conflicts(self):
        m = self.metric()
        for i in range(3):
            m.update(row(i, [person(i * .1, name="A"), person(2 + i * .1, tid=2, name="A")]))
        people = m.result()["players"]
        self.assertTrue(all(p["player_id"] is None and p["observed_distance_m"] is None for p in people))
        self.assertTrue(all("identity_conflict" in p["warnings"] for p in people))

    def test_identity_confirmation_does_not_relabel_past(self):
        m = self.metric()
        for i in range(4):
            m.update(row(i, [person(i * .1, name="A" if i >= 2 else None)]))
        people = m.result()["players"]
        self.assertEqual(len(people), 2)
        self.assertTrue(all(abs(p["observed_distance_m"] - .1) < 1e-9 for p in people))

    def test_anonymous_number_is_preserved_but_not_used_as_identity(self):
        m = self.metric()
        for i in range(2):
            p = person(i * .1)
            p["jersey"] = {"number": "00", "status": "confirmed", "identity_status": "anonymous"}
            m.update(row(i, [p]))
        result = m.result()["players"][0]
        self.assertEqual(result["jersey_number"], "00")
        self.assertIsNone(result["player_id"])

    def test_anonymous_color_group_is_retained_without_roster_identity(self):
        m = self.metric()
        for i in range(3):
            p = person(i * .1)
            p["jersey"] = {"number": "28" if i else None,
                           "status": "confirmed" if i else "unknown",
                           "identity_status": "ambiguous" if i else "unresolved",
                           "team_group": "team_1" if i else None}
            m.update(row(i, [p]))
        result = m.result()["players"][0]
        self.assertEqual(result["team_group"], "team_1")
        self.assertEqual(result["jersey_number"], "28")
        self.assertIsNone(result["team_id"])

    def test_color_conflict_clears_statistic_group(self):
        m = self.metric()
        for i in range(3):
            p = person(i * .1)
            p["jersey"] = {"team_group": "team_1" if i < 2 else None,
                           "team_group_conflict": i == 2}
            m.update(row(i, [p]))
        result = m.result()["players"][0]
        self.assertIsNone(result["team_group"])
        self.assertIn("team_group_conflict", result["warnings"])

    def test_large_jump_method_and_time_gaps(self):
        for change in ("jump", "method", "time"):
            with self.subTest(change=change):
                m = self.metric()
                m.update(row(0, [person()]))
                m.update(row(1, [person(10 if change == "jump" else .1,
                                        method="ankles" if change == "method" else "bbox")],
                             timestamp=1 if change == "time" else .1))
                self.assertIsNone(m.result()["players"][0]["observed_distance_m"])

    def test_curve(self):
        m = self.metric(max_speed_m_s=20)
        for i, point in enumerate(([0, 0], [1, 0], [1, 1], [0, 1])):
            p = person()
            p["position_m"] = point
            m.update(row(i, [p]))
        self.assertAlmostEqual(m.result()["players"][0]["observed_distance_m"], 3)

    def test_order_rejected(self):
        m = self.metric()
        with self.assertRaises(ValueError):
            m.update(row(1, [person()]))

    def test_variable_timestamps_drive_speed_filter(self):
        m = self.metric(max_speed_m_s=1.5)
        for i, t in enumerate((0, .1, .15)):
            m.update(row(i, [person(i * .1)], timestamp=t))
        p = m.result()["players"][0]
        self.assertAlmostEqual(p["observed_distance_m"], .1)
        self.assertAlmostEqual(p["measured_duration_s"], .1)
        self.assertIn("implausible_speed", p["warnings"])


class RuntimeTests(unittest.TestCase):
    def test_independent_cpu_work_overlaps(self):
        rt = Runtime(load_config())
        barrier = threading.Barrier(2, timeout=3)
        try:
            a = rt.call("a", "cpu", barrier.wait)
            b = rt.call("b", "cpu", barrier.wait)
            self.assertEqual({a.result(), b.result()}, {0, 1})
        finally:
            rt.close()

    def test_role_tracking_cannot_cross_associate(self):
        from pipeline.adapters import RoleTracker
        tracker = RoleTracker(load_config()["tracking"], fps=30)
        image = np.zeros((100, 100, 3), np.uint8)
        box = {"bbox": [10, 10, 30, 50], "confidence": .99, "class_id": 0}
        first = tracker.update([{**box, "role": "player"}], image, frame_index=0)
        second = tracker.update([{**box, "role": "referee"}], image, frame_index=1)
        third = tracker.update([{**box, "role": "referee"}], image, frame_index=2)
        self.assertIsNotNone(first[0]["track_id"])
        self.assertIsNotNone(third[0]["track_id"])
        self.assertNotEqual(first[0]["track_id"], third[0]["track_id"])

    def test_device_admission(self):
        rt = Runtime(load_config())
        active = 0
        peak = 0
        lock = threading.Lock()

        def work():
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(.01)
            with lock:
                active -= 1
        try:
            futures = [rt.call(str(i), "cuda:0", work) for i in range(8)]
            for f in futures:
                f.result()
            self.assertEqual(peak, 1)
        finally:
            rt.close()


class FakePeople:
    def predict(self, image):
        x = 10 + round(float(image.mean()) / 10)
        return [{"bbox": [x, 10, x + 10, 40], "confidence": .99, "role": role, "class_id": 0}
                for role in ("player", "referee")]


class FakeTracker:
    def update(self, detections, image, **kwargs):
        return [{**d, "track_id": i + 1} for i, d in enumerate(detections)]


class FakeCourt:
    def predict(self, image):
        return np.zeros((18, 3))

    def update(self, image, detections, *, frame_index, **kwargs):
        return {"scene_cut": frame_index == 4, "image_to_court": np.eye(3).tolist()}

    def project_players(self, detections, record, shape):
        return [{"position_m": [d["bbox"][0] / 10, 0], "status": "projected",
                 "ground_point_px": [d["bbox"][0], 40], "ground_point_method": "bbox"} for d in detections]


class FakeBall:
    def predict(self, frames):
        return {"status": "ok", "position_px": [20, 20], "confidence": .9,
                "source_frames": [f.index for f in frames]}


def fake_models(config, fps, roster, runtime):
    return SimpleNamespace(people=FakePeople(), tracker=FakeTracker(), court=FakeCourt(),
                           court_points=FakeCourt(), ball=FakeBall(), jersey=None, pose=None,
                           capabilities={"players": "ok", "referees": "ok", "court": "ok",
                                         "ball": "ok", "pose": "disabled", "jersey": "disabled"},
                           provenance={}, errors={}, close=lambda: None)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root / "video.avi"
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64))
        self.assertTrue(writer.isOpened())
        for i in range(8):
            writer.write(np.full((64, 64, 3), i * 10, np.uint8))
        writer.release()
        self.cfg = load_config(overrides={"timestamp_policy": "fps"})
        self.cfg["distance"].update(smoothing_tau_s=0, deadband_m=0)

    def run_video(self, name, factory=fake_models, roster=None):
        return analyze(self.video, self.root / name, self.cfg, roster, models_factory=factory)

    def test_video_without_roster_and_recompute(self):
        result = self.run_video("out")
        self.assertEqual(len(result["players"]), 2)  # Anonymous tracks stay separate across cuts.
        self.assertAlmostEqual(sum(p["observed_distance_m"] for p in result["players"]), .6)
        again = recompute(self.root / "out", self.root / "again.json")
        self.assertEqual(result["players"], again["players"])
        rows = [json.loads(s) for s in (self.root / "out/observations.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[4]["ball"]["status"], "unavailable")
        self.assertEqual(rows[6]["ball"]["source_frames"], [4, 5, 6])
        self.assertFalse(list((self.root / "out").glob("*.partial.*")))

    def test_sequential_parallel_same_measurements(self):
        a = self.run_video("parallel")
        self.cfg["mode"] = "sequential"
        b = self.run_video("sequential")
        for result in (a, b):
            for p in result["players"]:
                for ref in p["track_refs"]:
                    ref.pop("run_id")
        self.assertEqual(a["players"], b["players"])

    def test_pose_is_joined_by_detection_not_its_track_id(self):
        class Pose:
            def predict(self, image, detections):
                return [{**d, "track_id": 999, "detection_index": i, "pose": None, "pose_status": "low_keypoint_confidence"}
                        for i, d in enumerate(detections)]

        def factory(*args):
            models = fake_models(*args)
            models.pose = Pose()
            models.capabilities["pose"] = "ok"
            return models
        self.run_video("out", factory)
        first = json.loads((self.root / "out/observations.jsonl").read_text().splitlines()[0])
        self.assertEqual([p["track_id"] for p in first["persons"]], [1, 2])

    def test_invalid_roster_before_loading(self):
        with self.assertRaises(FileNotFoundError):
            self.run_video("out", roster=self.root / "absent.json")
        self.assertFalse((self.root / "out").exists())

    def test_roster_snapshot(self):
        roster = self.root / "roster.json"
        payload = {"schema_version": 1, "teams": [
            {"team_id": t, "name": t, "players": [{"name": t, "number": "23"}]} for t in ("a", "b")]}
        roster.write_text(json.dumps(payload))
        self.run_video("out", roster=roster)
        self.assertEqual(json.loads((self.root / "out/roster.json").read_text()), payload)

    def test_failure_keeps_partial_outputs(self):
        class BadPeople:
            def predict(self, image):
                if image.mean() >= 30:
                    raise RuntimeError("injected failure")
                return FakePeople().predict(image)

        def factory(*args):
            models = fake_models(*args)
            models.people = BadPeople()
            return models
        with self.assertRaisesRegex(RuntimeError, "injected failure"):
            self.run_video("out", factory)
        self.assertFalse((self.root / "out/statistics.json").exists())
        self.assertTrue((self.root / "out/observations.partial.jsonl").exists())
        self.assertEqual(json.loads((self.root / "out/run.json").read_text())["status"], "failed")

    def test_optional_failure_is_explicit(self):
        def fail(frames):
            raise RuntimeError("ball failed")

        def factory(*args):
            models = fake_models(*args)
            models.ball = SimpleNamespace(predict=fail)
            return models
        result = self.run_video("out", factory)
        self.assertEqual(result["capabilities"]["ball"], "error")
        self.assertTrue(result["players"])

    def test_tampered_artifact_rejected(self):
        self.run_video("out")
        with (self.root / "out/observations.jsonl").open("a") as f:
            f.write("{}\n")
        with self.assertRaisesRegex(ValueError, "hash"):
            recompute(self.root / "out", self.root / "again.json")

    def test_exclusion_between_frames_breaks_distance(self):
        self.cfg["exclude_intervals"] = [[.14, .16]]
        result = self.run_video("out")
        self.assertAlmostEqual(sum(p["observed_distance_m"] or 0 for p in result["players"]), .5)


class ConfigurationTests(unittest.TestCase):
    def test_real_config(self):
        self.assertEqual(load_config("configs/analysis.yaml")["court"]["detect_every"], 5)

    def test_unknown_key_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.yaml"
            p.write_text("distance:\n  typo: 1\n")
            with self.assertRaises(ValueError):
                load_config(p)

    def test_anonymous_ocr(self):
        from collections import deque
        reader = SimpleNamespace(provenance={})
        recognizer = JerseyRecognizer(jersey_settings(), reader=reader)
        self.assertTrue(recognizer.roster.allows("00"))
        self.assertFalse(recognizer.roster.allows("234"))
        self.assertEqual(recognizer.roster.identity("23")["identity_status"], "anonymous")
        state = {"votes": deque([{"number": "00", "weight": .99, "timestamp_seconds": t} for t in (0, .8)])}
        decision = recognizer._decision(state, 1.0)
        self.assertEqual(decision["number"], "00")
        self.assertEqual(decision["identity_status"], "anonymous")


if __name__ == "__main__":
    unittest.main()
