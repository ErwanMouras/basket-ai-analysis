"""Shot behavior: known camera, synthetic flights, negative actions and cuts."""

import copy
import math
import unittest

from viewer.shot_geometry import inverse, multiply, project, quadratic, rim_anchors
from viewer.shots import ShotIndex, hand_contact

GEOMETRY = {"league": "NBA", "length": 28.6512, "width": 15.24, "units": "metres"}
ROSTER = {
    "teams": [
        {"team_id": "a", "name": "Alpha", "players": [{"number": "7", "name": "Seven"}]},
        {"team_id": "b", "name": "Beta", "players": []},
    ]
}


def camera():
    # Independently constructed pinhole camera, square pixels, level roll.
    f, cx, cy = 1800.0, 960.0, 540.0
    centre = [14.3256, -20.0, 15.0]
    v = [0.0, 27.62, -15.0]
    norm = math.sqrt(sum(x * x for x in v))
    v = [x / norm for x in v]
    right = [1.0, 0.0, 0.0]
    down = [0.0, v[2], -v[1]]
    rotation = [right, down, v]
    trans = [-sum(a * b for a, b in zip(row, centre)) for row in rotation]
    k = [[f, 0, cx], [0, f, cy], [0, 0, 1]]
    columns = [multiply(k, [rotation[i][j] for i in range(3)]) for j in range(3)]
    shift = multiply(k, trans)
    h = [[columns[0][i], columns[1][i], shift[i]] for i in range(3)]
    rim = [
        sum(columns[j][i] * p for j, p in enumerate([27.051, 7.62, 3.048])) + shift[i]
        for i in range(3)
    ]
    return inverse(h), (rim[0] / rim[2], rim[1] / rim[2])


def rows(kind="shot", motion=False, identity=True):
    h, rim = camera()
    start = (rim[0] - 350, rim[1] + 80)
    result = []
    for i in range(46):
        t = max(0, (i - 5) / 30)
        x, y = start[0] + 360 * t, start[1] - 500 * t + 600 * t * t
        if kind == "held":
            x, y = start
        if kind == "pass":
            y = start[1]
        if kind == "dribble":
            y = start[1] + 100 + 120 * abs(math.sin(t * 8))
        shift = 220 * i / 30 if motion else 0
        keys = [[start[0] + shift, start[1] + 50] for _ in range(17)]
        keys[9] = [start[0] + shift, start[1]]
        keys[10] = [start[0] + shift - 20, start[1] + 20]
        box = [start[0] + shift - 50, start[1] - 20, start[0] + shift + 50, start[1] + 180]
        person = {
            "role": "player",
            "track_id": 1,
            "bbox": box,
            "pose": {
                "format": "coco17",
                "coordinate_space": "source",
                "keypoints": keys,
                "scores": [1.0] * 17,
                "valid": [True] * 17,
            },
            "resolved_identity": {"team_id": "a", "number": "7"} if identity else None,
        }
        # H_current = H_reference * image_translation_inverse.
        current = [[h[r][0], h[r][1], h[r][2] - h[r][0] * shift] for r in range(3)]
        result.append(
            {
                "frame_index": i,
                "timestamp_s": i / 30,
                "segment_id": 0,
                "included": True,
                "width": 1920,
                "height": 1080,
                "scene_cut": False,
                "persons": [person],
                "court": {"status": "fit", "image_to_court": current},
                "ball": {"status": "ok", "confidence": 0.9, "position_px": [x + shift, y]},
            }
        )
    return result


def detect(frames, manifest=None):
    index = ShotIndex(manifest or {"court_coordinate_system": GEOMETRY}, ROSTER)
    for row in frames:
        index.add_frame(row)
    index.finish()
    return index


class GeometryTests(unittest.TestCase):
    def test_elevated_rim_matches_independent_known_camera(self):
        h, expected = camera()
        anchors = rim_anchors(h, GEOMETRY, 1920, 1080)
        self.assertTrue(any(math.dist(p, expected) < 1e-6 for p in anchors))
        floor = project(inverse(h), (27.051, 7.62))
        self.assertGreater(math.dist(floor, expected), 100)

    def test_invalid_or_unsupported_camera_has_no_invented_rim(self):
        for h in (
            None,
            [[0] * 3] * 3,
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[float("nan"), 0, 1], [1, 0, 1], [0, 1, 0]],
        ):
            self.assertEqual(rim_anchors(h, GEOMETRY, 1920, 1080), [])
        self.assertEqual(rim_anchors(camera()[0], {"league": "unknown"}, 1920, 1080), [])

    def test_temporal_fit_recovers_curvature_with_irregular_sampling(self):
        times = [3, 3.04, 3.1, 3.16, 3.28, 3.5, 3.8]
        values = [2 * (t - 3) ** 2 - 3 * (t - 3) + 5 for t in times]
        a, b, c, error = quadratic(times, values)
        for value, expected in [(a, 2), (b, -3), (c, 5), (error, 0)]:
            self.assertAlmostEqual(value, expected)


class ShotTests(unittest.TestCase):
    def test_clean_release_and_arc_near_rim_make_one_probable_event(self):
        index = detect(rows())
        events = index.events
        self.assertEqual(len(events), 1)
        shot = events[0]
        self.assertEqual(shot["status"], "probable")
        self.assertEqual(shot["outcome"], "unknown")
        self.assertTrue(shot["evidence"]["arc"])
        self.assertTrue(shot["evidence"]["rim_approach"])
        self.assertEqual(shot["shooter"]["subject_id"], "roster:a:7")
        self.assertLess(abs(shot["time_s"] - 5 / 30), 0.2)
        self.assertEqual(index.samples, [])

    def test_camera_pan_and_low_frame_rate_preserve_detection(self):
        a, b = detect(rows()).events, detect(rows(motion=True)).events
        self.assertEqual(len(a), len(b))
        self.assertEqual(b[0]["status"], "probable")
        self.assertAlmostEqual(
            a[0]["evidence"]["rise_body_heights"], b[0]["evidence"]["rise_body_heights"]
        )

        sparse = rows()[::6]
        for index, row in enumerate(sparse):
            row["frame_index"] = index
        self.assertEqual(detect(sparse).events[0]["status"], "probable")

    def test_dribble_held_ball_and_flat_pass_do_not_become_shots(self):
        for kind in ("held", "dribble", "pass"):
            self.assertEqual(detect(rows(kind)).events, [], kind)

    def test_missing_rim_or_camera_cannot_become_probable(self):
        for missing in ("rim", "camera"):
            frames = rows()
            if missing == "camera":
                for row in frames:
                    row["court"] = None
                result = detect(frames)
            else:
                result = detect(frames, {"court_coordinate_system": {}})
            self.assertTrue(result.events)
            self.assertTrue(all(e["status"] == "candidate" for e in result.events))

    def test_lost_release_can_only_be_an_unattributed_candidate(self):
        frames = rows()
        for row in frames:
            row["persons"][0]["pose"] = None
        # Fallback still has player boxes for scale and court/ball evidence.
        result = detect(frames)
        self.assertTrue(result.metadata()["available"])
        events = result.events
        self.assertTrue(events)
        self.assertTrue(
            all(e["status"] == "candidate" and not e["release_observed"] for e in events)
        )
        self.assertTrue(all(e["shooter"]["subject_id"] is None for e in events))

    def test_cut_exclusion_gap_and_dimension_change_never_join_flights(self):
        for boundary in ("cut", "exclusion", "gap", "size"):
            frames = rows()
            # Separate upward and downward observations; the shortened rise alone
            # must not be promoted through this boundary.
            for row in frames[11:]:
                if boundary == "cut":
                    row["segment_id"] = 1
                if boundary == "gap":
                    row["timestamp_s"] += 1
                if boundary == "size":
                    row["width"] = 1280
            if boundary == "cut":
                frames[11]["scene_cut"] = True
            if boundary == "exclusion":
                frames[11]["included"] = False
            events = detect(frames).events
            self.assertFalse(any(event["status"] == "probable" for event in events))
            for event in events:
                ids = [point["frame_index"] for point in event["trajectory"]]
                self.assertTrue(
                    max(ids) < 11 or min(ids) >= (12 if boundary == "exclusion" else 11), boundary
                )
                if min(ids) >= 11:
                    self.assertGreaterEqual(
                        event["release_window_s"][0],
                        (
                            frames[min(ids) - 1]["timestamp_s"]
                            if boundary == "gap"
                            else frames[11]["timestamp_s"]
                        ),
                    )
                    self.assertEqual(event["status"], "candidate")

    def test_low_confidence_and_false_jump_are_not_complete_shots(self):
        frames = rows()
        for row in frames:
            row["ball"]["confidence"] = 0.1
        self.assertFalse(detect(frames).metadata()["available"])
        self.assertEqual(detect(frames).events, [])
        frames = rows()
        for row in frames[11:]:
            row["ball"]["position_px"] = [100, 100]
        self.assertEqual(detect(frames).events, [])

    def test_unknown_and_duplicate_identity_do_not_create_a_named_shooter(self):
        self.assertIsNone(detect(rows(identity=False)).events[0]["shooter"]["subject_id"])
        frame = rows()[0]
        other = copy.deepcopy(frame["persons"][0])
        other["track_id"] = 2
        other["pose"]["keypoints"][9] = [0, 0]
        other["pose"]["keypoints"][10] = [0, 0]
        frame["persons"].append(other)
        self.assertIsNone(hand_contact(frame, frame["ball"]["position_px"])["subject_id"])

    def test_competing_or_referee_hand_never_attributes_a_release(self):
        for role in ("player", "referee"):
            frame = rows()[0]
            p = copy.deepcopy(frame["persons"][0])
            p["track_id"] = 2
            p["role"] = role
            frame["persons"].append(p)
            self.assertTrue(hand_contact(frame, frame["ball"]["position_px"])["ambiguous"])

    def test_short_occlusion_retains_observed_points_without_interpolating(self):
        frames = rows()
        for row in frames[16:19]:
            row["ball"]["position_px"] = None
        events = detect(frames).events
        self.assertEqual(len(events), 1)
        self.assertFalse(any(p["frame_index"] in (16, 17, 18) for p in events[0]["trajectory"]))


class ShotIntegrationTests(unittest.TestCase):
    def test_receiver_before_the_rim_rejects_a_high_pass(self):
        frames = rows()
        for row in frames:
            if row["frame_index"] < 16:
                continue
            receiver = copy.deepcopy(row["persons"][0])
            receiver["track_id"] = 2
            xy = row["ball"]["position_px"]
            receiver["pose"]["keypoints"][9] = xy[:]
            receiver["bbox"] = [xy[0] - 30, xy[1] - 20, xy[0] + 30, xy[1] + 180]
            receiver["resolved_identity"] = {"team_id": "a", "number": "8"}
            row["persons"].append(receiver)
        self.assertEqual(detect(frames).events, [])

    def test_nonfinite_or_offscreen_positions_do_not_fill_a_flight(self):
        for point in ([float("nan"), 1], [-10, 200], [200, 2000]):
            frames = rows()
            for row in frames:
                row["ball"]["position_px"] = point
            self.assertEqual(detect(frames).events, [])

    def test_release_window_includes_association_uncertainty(self):
        event = detect(rows()).events[0]
        a, b = event["release_window_s"]
        self.assertLessEqual(a, 5 / 30)
        self.assertGreaterEqual(b, 5 / 30)

    def test_identity_transition_does_not_attribute_the_shot_to_either_name(self):
        frames = rows()
        for row in frames[6:]:
            row["persons"][0]["resolved_identity"] = {"team_id": "b", "number": "9"}
        event = detect(frames).events[0]
        self.assertIsNone(event["shooter"]["subject_id"])
        self.assertIsNone(event["shooter"]["team_id"])


class MissingContextTests(unittest.TestCase):
    def test_ball_without_people_cannot_invent_release_or_body_scale(self):
        frames = rows()
        for row in frames:
            row["persons"] = []
        result = detect(frames)
        self.assertFalse(result.metadata()["available"])
        self.assertEqual(result.events, [])
