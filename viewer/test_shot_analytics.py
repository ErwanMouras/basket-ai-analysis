"""Shot counts conserve events; ground maps never fabricate missing positions."""

from types import SimpleNamespace
import unittest

from viewer.shot_analytics import assign_directions, ground_origin, summarize_shots
from viewer.test_shots import GEOMETRY, ROSTER, rows, detect
from viewer.shots import ShotIndex


def event(index=1, team="a", subject="roster:a:7", xy=(24, 3), direction=1, status="probable"):
    return {
        "id": f"shot-{index}",
        "status": status,
        "time_s": 1.0,
        "segment_id": 0,
        "shooter": {"team_id": team, "subject_id": subject, "track_id": index},
        "origin": {"position_m": xy},
        "court_direction": direction,
    }


class ShotAnalyticsTests(unittest.TestCase):
    def summarize(self, events, geometry=None, available=True):
        return summarize_shots(events, geometry or GEOMETRY, ROSTER, available)

    def test_maps_include_candidates_while_player_counts_use_probables(self):
        data = self.summarize([event(), event(2, status="candidate")])
        self.assertEqual(data["player_counts"]["roster:a:7"], 1)
        team = data["shot_maps"]["maps"][0]
        self.assertEqual(
            (team["detected_count"], team["located_count"], sum(team["grid"])), (2, 2, 2)
        )
        self.assertEqual((team["probable_count"], team["candidate_count"]), (1, 1))
        self.assertEqual([p["status"] for p in team["points"]], ["probable", "candidate"])
        self.assertEqual(data["shot_maps"]["statuses"], ["probable", "candidate"])

    def test_candidates_with_missing_locations_or_teams_are_reported(self):
        data = self.summarize([event(status="candidate", xy=None), event(2, status="candidate", team=None)])
        team = data["shot_maps"]["maps"][0]
        self.assertEqual((team["candidate_count"], team["unlocated_count"], sum(team["grid"])), (1, 1, 0))
        self.assertEqual(data["unassigned_candidate_count"], 1)
        self.assertEqual(data["unassigned_probable_count"], 0)
        self.assertEqual(data["player_counts"]["roster:a:7"], 0)

    def test_camera_directions_rotate_origins_into_the_same_half_court(self):
        length, width = GEOMETRY["length"], GEOMETRY["width"]
        data = self.summarize(
            [event(xy=(24, 3)), event(2, xy=(length - 24, width - 3), direction=-1)]
        )
        court = data["shot_maps"]["court"]
        team = data["shot_maps"]["maps"][0]
        self.assertEqual(court["length"], length / 2)
        self.assertEqual(team["located_count"], 2)
        self.assertEqual(team["points"][0]["x"], team["points"][1]["x"])
        self.assertAlmostEqual(team["points"][0]["y"], team["points"][1]["y"])
        self.assertEqual(max(team["grid"]), 2)

    def test_missing_position_direction_and_outside_half_remain_counted(self):
        shots = [
            event(xy=None),
            event(2, direction=None),
            event(3, xy=(2, 3)),
            event(4, xy=(24, -1)),
        ]
        data = self.summarize(shots)
        team = data["shot_maps"]["maps"][0]
        self.assertEqual(data["player_counts"]["roster:a:7"], 4)
        self.assertEqual(
            (team["detected_count"], team["located_count"], team["unlocated_count"]), (4, 0, 4)
        )
        self.assertEqual(sum(team["grid"]), 0)

    def test_unknown_player_and_team_are_not_assigned_to_roster_names(self):
        data = self.summarize([event(subject=None), event(2, team=None, subject=None)])
        self.assertEqual(data["player_counts"]["roster:a:7"], 0)
        self.assertEqual(data["shot_maps"]["maps"][0]["detected_count"], 1)
        self.assertEqual(data["unassigned_probable_count"], 1)

    def test_missing_data_is_unavailable_not_a_measured_zero(self):
        self.assertIsNone(self.summarize([], available=False)["player_counts"])
        result = summarize_shots([event()], {}, ROSTER, True)
        self.assertEqual(result["player_counts"]["roster:a:7"], 1)
        self.assertFalse(result["shot_maps"]["available"])
        self.assertEqual(result["shot_maps"]["maps"][0]["unlocated_count"], 1)

    def test_ground_origin_uses_player_positions_before_release_not_the_ball(self):
        frames = rows()
        for row in frames:
            row["persons"][0]["position_m"] = [24, 4]
            row["persons"][0]["ground_point_method"] = "ankles"
        detector = ShotIndex({"court_coordinate_system": GEOMETRY}, ROSTER)
        for row in frames:
            detector.add_frame(row)
        detector.finish()
        shot = detector.events[0]
        self.assertEqual(shot["origin"]["position_m"], [24, 4])
        self.assertLessEqual(shot["origin"]["end_s"], shot["time_s"])
        self.assertGreaterEqual(shot["origin"]["support_frames"], 2)
        self.assertEqual(shot["shooter"]["subject_id"], "roster:a:7")

    def test_unstable_or_missing_ground_positions_do_not_use_airborne_ball(self):
        self.assertIsNone(detect(rows()).events[0]["origin"]["position_m"])
        frames = rows()
        for i, row in enumerate(frames):
            row["persons"][0]["position_m"] = [20 if i % 2 else 27, 4]
        self.assertIsNone(detect(frames).events[0]["origin"]["position_m"])

    def test_ground_origin_never_crosses_a_cut(self):
        contact = {
            "track_id": 1,
            "team_id": "a",
            "ambiguous": False,
            "unique_track": True,
            "position_m": [24, 4],
            "ground_point_method": "ankles",
        }
        samples = [SimpleNamespace(time=i / 30, contact=contact, court=True) for i in range(4)]
        result = ground_origin(
            samples, 3, {"track_id": 1, "team_id": "a"}, GEOMETRY, lambda a, b: b is samples[3]
        )
        self.assertIsNone(result["position_m"])

    def test_quarter_and_camera_evidence_controls_orientation_not_origin_x(self):
        phases = SimpleNamespace(
            teams=("a", "b"),
            orientation_evidence=[
                {"plan_id": 0, "period": 1, "reference_team_id": "a", "direction": -1},
                {"plan_id": 1, "period": 1, "reference_team_id": "a", "direction": 1},
            ],
        )
        plans = [
            {"id": 0, "segment_id": 0, "start_s": 0.0, "end_s": 2.0},
            {"id": 1, "segment_id": 1, "start_s": 2.1, "end_s": 4.0},
        ]
        shots = [event(), event(2, team="b", subject=None)]
        scoreboard = SimpleNamespace(at=lambda time: {"period": 1})
        assign_directions(shots, phases, plans, scoreboard)
        self.assertEqual([e["court_direction"] for e in shots], [-1, 1])
        shots[0].update(segment_id=1, time_s=3.0)
        assign_directions(shots, phases, plans, scoreboard)
        self.assertEqual(shots[0]["court_direction"], 1)
        scoreboard.at = lambda time: {"period": 2}
        assign_directions(shots, phases, plans, scoreboard)
        self.assertIsNone(shots[0]["court_direction"])
