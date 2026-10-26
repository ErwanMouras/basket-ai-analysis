"""Behavioral checks for conservative phase labels and basket-aligned heatmaps."""

import copy
import unittest

from viewer.heatmaps import HeatmapIndex
from viewer.phases import PhaseIndex, contact, team_of

GEOMETRY = {"units": "metres", "length": 30.0, "width": 15.0}
ROSTER = {
    "teams": [
        {
            "team_id": team,
            "name": team.upper(),
            "players": [{"number": str(track), "name": f"{team}-{track}"} for track in tracks],
        }
        for team, tracks in [("a", [1, 2, 3]), ("b", [4, 5, 6])]
    ]
}


def phase_frame(index, time=None, owner="a", mirror=False, **extra):
    people = []
    xs = (19, 18, 17, 24, 25, 26) if owner == "a" else (4, 5, 6, 11, 10, 9)
    for track in range(1, 7):
        team = "a" if track <= 3 else "b"
        carrier = team == owner and track in (1, 4)
        x = xs[track - 1]
        y = 2 + (track % 3) * 3
        if mirror:
            x, y = 30 - x, 15 - y
        keys = [[0.0, 0.0] for _ in range(17)]
        keys[9] = [100.0, 100.0] if carrier else [280.0 + track, 140.0]
        people.append(
            {
                "track_id": track,
                "role": "player",
                "position_m": [x, y],
                "bbox": [80, 60, 120, 160] if carrier else [260 + track, 100, 300 + track, 200],
                "ground_point_method": "scripted",
                "resolved_identity": {"team_id": team, "number": str(track)},
                "pose": {
                    "format": "coco17",
                    "coordinate_space": "source",
                    "keypoints": keys,
                    "scores": [0.0] * 9 + [1.0] + [0.0] * 7,
                },
            }
        )
    return {
        "frame_index": index,
        "timestamp_s": index / 10 if time is None else time,
        "segment_id": 0,
        "included": True,
        "width": 400,
        "height": 300,
        "persons": people,
        "ball": {"status": "ok", "confidence": 0.9, "position_px": [100.0, 100.0]},
        "court": {"status": "fit", "image_to_court": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        **extra,
    }


def phases(rows):
    result = PhaseIndex(("a", "b"), GEOMETRY, use_court_context=False)
    plan, previous = -1, None
    for row in rows:
        if previous is None or row.get("scene_cut") or row["segment_id"] != previous["segment_id"]:
            plan += 1
        result.add_frame(row, plan)
        previous = row
    result.finish()
    return result


def owner_at(index, time):
    return next((span.owner for span in index.spans if span.start <= time < span.end), None)


class PhaseTests(unittest.TestCase):
    def test_team_switch_splits_attack_inside_the_same_camera_plan(self):
        index = phases([phase_frame(i, owner="a" if i < 15 else "b") for i in range(30)])
        self.assertEqual(owner_at(index, 0.65), "a")
        self.assertEqual(owner_at(index, 2.45), "b")
        self.assertIsNone(owner_at(index, 1.45))
        self.assertEqual({span.plan for span in index.spans}, {0})
        meta = index.metadata()
        self.assertAlmostEqual(meta["classified_s"] + meta["unknown_s"], 2.9)
        self.assertTrue(meta["experimental"])

    def test_unknown_players_referees_and_contested_contacts_block_attribution(self):
        for role, team in [("player", None), ("referee", None), ("player", "b")]:
            rows = [phase_frame(i) for i in range(10)]
            for row in rows:
                competitor = copy.deepcopy(row["persons"][0])
                competitor["track_id"] = 99
                competitor["role"] = role
                competitor["resolved_identity"] = (
                    {"team_id": team, "number": "99"} if team else None
                )
                row["persons"].append(competitor)
            self.assertEqual(phases(rows).metadata()["classified_s"], 0)

    def test_missing_poses_allow_only_unambiguous_body_contacts(self):
        row = phase_frame(0)
        for player in row["persons"]:
            player["pose"] = None
        self.assertEqual(contact(row, ("a", "b"), GEOMETRY).team, "a")
        unknown = copy.deepcopy(row["persons"][0])
        unknown["resolved_identity"] = None
        unknown["track_id"] = 99
        row["persons"].append(unknown)
        self.assertIsNone(contact(row, ("a", "b"), GEOMETRY).team)

    def test_ball_quality_and_explicitly_unresolved_teams_are_respected(self):
        for change in (
            {"status": "not_detected"},
            {"confidence": 0.1},
            {"position_px": [float("nan"), 2]},
            {"position_px": [900.0, 900.0]},
        ):
            row = phase_frame(0)
            row["ball"].update(change)
            self.assertIsNone(contact(row, ("a", "b"), GEOMETRY).team)
        person = {"resolved_team_id": None, "jersey": {"team_id": "a", "status": "confirmed"}}
        self.assertIsNone(team_of(person), "stale causal team must not override final resolution")

    def test_long_ball_absences_are_not_filled_by_temporal_continuity(self):
        rows = [phase_frame(i) for i in range(35)]
        for row in rows[10:25]:
            row["ball"]["position_px"] = None
        index = phases(rows)
        self.assertEqual(owner_at(index, 0.55), "a")
        self.assertIsNone(owner_at(index, 1.65))
        self.assertEqual(owner_at(index, 2.85), "a")

    def test_one_low_rate_contact_does_not_count_as_repeated_evidence(self):
        rows = [phase_frame(i, time=i * 0.2) for i in range(5)]
        for row in rows:
            if row["frame_index"] != 2:
                row["ball"]["position_px"] = None
        self.assertEqual(phases(rows).metadata()["classified_s"], 0)

    def test_cuts_exclusions_missing_frames_and_time_gaps_never_share_evidence(self):
        cases = [
            [phase_frame(0), phase_frame(1), phase_frame(2, scene_cut=True), phase_frame(3)],
            [
                phase_frame(0),
                phase_frame(1),
                phase_frame(2, included=False),
                phase_frame(3),
                phase_frame(4),
            ],
            [phase_frame(0), phase_frame(1), phase_frame(4), phase_frame(5)],
            [phase_frame(0, 0), phase_frame(1, 0.1), phase_frame(2, 1), phase_frame(3, 1.1)],
        ]
        for rows in cases:
            self.assertEqual(phases(rows).metadata()["classified_s"], 0)

    def test_frontcourt_contacts_establish_direction_but_inbounds_do_not(self):
        rows = [phase_frame(i) for i in range(10)]
        self.assertEqual(phases(rows).directions, {0: 1})
        for row in rows:
            row["persons"][0]["position_m"][0] = 28.0
        index = phases(rows)
        self.assertGreater(index.metadata()["classified_s"], 0)
        self.assertEqual(index.directions, {})

    def test_conflicting_direction_evidence_stays_unoriented(self):
        rows = [phase_frame(i, mirror=i >= 15) for i in range(30)]
        index = phases(rows)
        self.assertGreater(index.metadata()["classified_s"], 2)
        self.assertEqual(index.directions, {})
        self.assertEqual(index.metadata()["oriented_s"], 0)

    def test_presentation_ranges_do_not_relabel_short_unknown_gaps(self):
        rows = [phase_frame(i) for i in range(20)]
        rows[10]["persons"][0]["resolved_identity"] = None
        index = phases(rows)
        self.assertIsNone(owner_at(index, 0.95))
        self.assertEqual(len(index.episodes), 1)
        passage = index.episodes[0]
        self.assertGreater(passage["duration_s"], passage["classified_s"])


class PhaseHeatmapTests(unittest.TestCase):
    def build(self, rows):
        result = HeatmapIndex({"court_coordinate_system": GEOMETRY}, ROSTER)
        for row in rows:
            result.add_frame(row)
        result.finish()
        return result

    def test_opposite_camera_directions_are_rotated_without_losing_mass(self):
        rows = [phase_frame(i, mirror=i >= 10, segment_id=int(i >= 10)) for i in range(20)]
        index = self.build(rows)
        legacy = index.query("roster:a:1", 0, start=0.1, end=0.9)["maps"][0]
        combined = index.query("roster:a:1", phase="offense")["maps"][0]
        self.assertAlmostEqual(combined["total"], 1.6)
        for a, b in zip(combined["grid"], legacy["grid"]):
            self.assertAlmostEqual(a, 2 * b)
        self.assertEqual(
            index.query("team:a", phase="offense")["scope"]["normalized_basket"], "right"
        )

    def test_defense_is_the_opponents_attack_and_unknown_time_is_separate(self):
        rows = [phase_frame(i, owner="a" if i < 15 else "b") for i in range(30)]
        index = self.build(rows)
        attack = index.query("team:a", phase="offense")["maps"][0]
        defense = index.query("team:b", phase="defense")["maps"][0]
        self.assertAlmostEqual(
            attack["phase_durations"]["offense_s"], defense["phase_durations"]["defense_s"]
        )
        unknown = index.query("team:a", phase="unknown")["maps"][0]
        self.assertEqual(unknown["total"], 0)
        self.assertGreater(unknown["phase_durations"]["unknown_s"], 0)

    def test_a_known_phase_without_direction_is_not_spatially_aggregated(self):
        rows = [phase_frame(i) for i in range(10)]
        for row in rows:
            row["persons"][0]["position_m"][0] = 28.0
        result = self.build(rows).query("team:a", phase="offense")["maps"][0]
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["observed_s"], 0)
        self.assertGreater(result["phase_durations"]["unknown_s"], 0)

    def test_passage_range_and_partial_bounds_conserve_selected_time(self):
        index = self.build([phase_frame(i) for i in range(10)])
        result = index.query("roster:a:1", phase="offense", episode=0, start=0.2, end=0.7)["maps"][
            0
        ]
        self.assertAlmostEqual(result["total"], 0.5)
        with self.assertRaises(ValueError):
            index.query("team:a", phase="offense", episode=99)
        with self.assertRaises(ValueError):
            index.query("team:a", phase="possession")
        with self.assertRaises(ValueError):
            index.query("team:a", phase="offense", start=-1)
