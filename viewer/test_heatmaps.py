"""Analytical heatmap checks: conservation, time support and identity boundaries."""

import unittest

from viewer.heatmaps import HeatmapIndex


ROSTER = {
    "teams": [
        {
            "team_id": "a",
            "name": "Alpha",
            "players": [{"number": "1", "name": "One"}, {"number": "2", "name": "Two"}],
        },
        {"team_id": "b", "name": "Beta", "players": [{"number": "3", "name": "Three"}]},
    ]
}
MANIFEST = {"court_coordinate_system": {"units": "metres", "length": 30.0, "width": 15.0}}


def person(track=1, number="1", team="a", point=(1.0, 2.0)):
    return {
        "role": "player",
        "track_id": track,
        "position_m": point,
        "ground_point_method": "ankles",
        "resolved_identity": {"team_id": team, "number": number},
    }


def frame(index, time, people=None, **extra):
    return {
        "frame_index": index,
        "timestamp_s": time,
        "segment_id": 0,
        "included": True,
        "persons": [person()] if people is None else people,
        "court": {"status": "fit", "image_to_court": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        **extra,
    }


def build(rows, tracks=(), roster=ROSTER, manifest=MANIFEST):
    index = HeatmapIndex(manifest, roster)
    for row in rows:
        index.add_frame(row)
    for row in tracks:
        index.add_track(row)
    index.finish()
    return index


class HeatmapTests(unittest.TestCase):
    def test_time_weighting_is_independent_of_sampling_rate(self):
        for times in ([0, 0.1, 0.2], [0, 0.05, 0.1, 0.15, 0.2], [0, 0.03, 0.2]):
            index = build([frame(i, t) for i, t in enumerate(times)])
            result = index.query("roster:a:1", 0)["maps"][0]
            self.assertAlmostEqual(result["total"], 0.2)
            self.assertAlmostEqual(sum(result["zones"]), 0.2)
            self.assertAlmostEqual(result["coverage"], 1.0)

    def test_exact_max_gap_is_stable_under_floating_point_rounding(self):
        index = build([frame(i, i / 5) for i in range(5)])
        self.assertAlmostEqual(index.query("team:a", 0)["maps"][0]["total"], 0.8)

    def test_partial_interval_clipping_conserves_time(self):
        index = build([frame(0, 0), frame(1, 0.2)])
        result = index.query("roster:a:1", 0, 0.04, 0.13)["maps"][0]
        self.assertAlmostEqual(result["total"], 0.09)
        self.assertAlmostEqual(result["localized_video_s"], 0.09)

    def test_team_player_seconds_are_not_wall_clock_seconds(self):
        people = [person(), person(2, "2", point=(10.0, 3.0))]
        index = build([frame(0, 0, people), frame(1, 0.1, people), frame(2, 0.2, people)])
        result = index.query("team:a", 0)["maps"][0]
        self.assertAlmostEqual(result["total"], 0.4)
        self.assertAlmostEqual(result["localized_video_s"], 0.2)
        self.assertEqual(result["unit"], "player_s")

    def test_unavailable_calibration_reduces_coverage_without_filling_gaps(self):
        index = build([frame(0, 0), frame(1, 0.1), frame(2, 0.2, court=None), frame(3, 0.3)])
        result = index.query("roster:a:1", 0)["maps"][0]
        self.assertAlmostEqual(result["observed_s"], 0.3)
        self.assertAlmostEqual(result["total"], 0.1)
        self.assertAlmostEqual(result["coverage"], 1 / 3)

    def test_cuts_split_plans_even_when_segment_ids_are_unchanged(self):
        index = build([frame(0, 0), frame(1, 0.1), frame(2, 0.2, scene_cut=True), frame(3, 0.3)])
        self.assertEqual(len(index.plans), 2)
        self.assertAlmostEqual(index.query("roster:a:1", 0)["maps"][0]["total"], 0.1)
        self.assertAlmostEqual(index.query("roster:a:1", 1)["maps"][0]["total"], 0.1)
        with self.assertRaises(ValueError):
            index.query("roster:a:1", 0, 0, 0.3)

    def test_exclusions_missing_frames_and_timestamp_gaps_do_not_bridge(self):
        for rows in (
            [frame(0, 0), frame(1, 0.1, included=False), frame(2, 0.2)],
            [frame(0, 0), frame(2, 0.1)],
            [frame(0, 0), frame(1, 0.3)],
            [frame(0, 0), frame(1, 0.1, []), frame(2, 0.2)],
        ):
            result = build(rows).query("roster:a:1", 0)["maps"][0]
            self.assertEqual(result["total"], 0)

    def test_duplicate_identities_and_identity_transitions_are_excluded(self):
        duplicate = [person(), person(2)]
        rows = [
            frame(0, 0, duplicate),
            frame(1, 0.1, duplicate),
            frame(2, 0.2, [person(number="2")]),
        ]
        index = build(rows)
        self.assertEqual(index.query("team:a", 0)["maps"][0]["total"], 0)

    def test_nonfinite_outside_points_method_changes_and_jumps_are_not_localized(self):
        cases = [
            person(point=(float("nan"), 2)),
            person(point=(31, 2)),
            person(point=(14, 2)),
            {**person(), "ground_point_method": "bbox"},
        ]
        for end in cases:
            result = build([frame(0, 0), frame(1, 0.1, [end])]).query("roster:a:1", 0)["maps"][0]
            self.assertEqual(result["total"], 0)
            self.assertAlmostEqual(result["observed_s"], 0.1)

    def test_anonymous_team_assignments_never_create_named_player_time(self):
        anonymous = {
            **person(),
            "resolved_identity": None,
            "resolved_team_id": "a",
            "majority_jersey_number": "1",
        }
        index = build([frame(0, 0, [anonymous]), frame(1, 0.1, [anonymous])])
        self.assertAlmostEqual(index.query("team:a", 0)["maps"][0]["total"], 0.1)
        self.assertEqual(index.query("roster:a:1", 0)["maps"][0]["total"], 0)
        self.assertFalse(build([frame(0, 0), frame(1, 0.1)], roster=None).metadata()["available"])

    def test_movement_uses_stored_valid_contributions_and_clips_distance(self):
        rows = [frame(0, 0), frame(1, 0.1, [person(point=(2.0, 2.0))])]
        track = {
            "frame_index": 1,
            "segment_id": 0,
            "track_id": 1,
            "subject_id": "roster:a:1",
            "interval_start_s": 0.0,
            "measured_duration_s": 0.1,
            "distance_m": 1.0,
            "reason": None,
        }
        result = build(rows, [track]).query("roster:a:1", 0, 0.025, 0.075, "movement")["maps"][0]
        self.assertAlmostEqual(result["total"], 0.5)
        self.assertAlmostEqual(sum(result["zones"]), 0.5)
        self.assertEqual(result["unit"], "m")
        self.assertEqual(build(rows).query("roster:a:1", 0, mode="movement")["maps"][0]["total"], 0)
        self.assertEqual(
            build(rows, [{**track, "reason": "identity_conflict"}]).query(
                "roster:a:1", 0, mode="movement"
            )["maps"][0]["total"],
            0,
        )

    def test_evolution_and_comparison_keep_separate_denominators(self):
        rows = [
            frame(i, t, [person(), person(3, "3", "b", (20.0, 5.0))])
            for i, t in enumerate([0, 0.1, 0.2])
        ]
        index = build(rows)
        maps = index.query("team:a", 0, mode="comparison", compare="team:b")["maps"]
        self.assertEqual(len(maps), 2)
        self.assertAlmostEqual(maps[0]["total"], maps[1]["total"])
        halves = index.query("team:a", 0, mode="evolution")["maps"]
        self.assertAlmostEqual(sum(item["total"] for item in halves), 0.2)
        self.assertEqual(halves[0]["end_s"], halves[1]["start_s"])

    def test_bad_queries_fail_before_aggregation(self):
        index = build([frame(0, 0), frame(1, 0.1)])
        for extra in (
            {"start": float("nan")},
            {"end": float("inf")},
            {"end": 1},
            {"start": 0.1, "end": 0},
            {"mode": "possession"},
            {"mode": "comparison", "compare": "roster:a:1"},
            {"mode": "comparison", "compare": "team:a"},
        ):
            with self.assertRaises(ValueError):
                index.query("roster:a:1", 0, **extra)
        with self.assertRaises(ValueError):
            index.query("unknown", 0)
        self.assertFalse(build([frame(0, 0)], manifest={}).metadata()["available"])


class HalfCourtTests(unittest.TestCase):
    """Metric clipping, without relying on a particular phase classifier fixture."""

    def index(self, points, direction=1, owner="a", team="a", number="1"):
        from viewer.heatmaps import Interval

        index = build([frame(0, 0), frame(1, 0.2)])
        index.intervals = [
            Interval(
                0,
                0.0,
                0.2,
                team,
                f"roster:{team}:{number}",
                first,
                last,
                owner=owner,
                direction=direction,
            )
            for first, last in points
        ]
        return index

    def test_only_displayed_half_contributes_to_mass_not_localization_quality(self):
        index = self.index([((24, 3), (24, 3)), ((6, 3), (6, 3))])
        data = index.query("team:a", phase="offense")
        chart = data["maps"][0]
        self.assertEqual(data["court"]["length"], 15)
        self.assertEqual(data["court"]["full_length"], 30)
        self.assertEqual(data["court"]["extent"], "half")
        self.assertEqual((chart["nx"], chart["ny"]), (15, 15))
        self.assertAlmostEqual(chart["total"], 0.2)
        self.assertAlmostEqual(chart["outside_half_s"], 0.2)
        self.assertAlmostEqual(chart["localized_s"], 0.4)
        self.assertAlmostEqual(chart["coverage"], 1)
        self.assertAlmostEqual(chart["half_court_video_s"], 0.2)
        self.assertAlmostEqual(chart["zones"][1], 0.2)
        self.assertAlmostEqual(sum(chart["zones"]), chart["total"])

    def test_midline_crossings_clip_exact_time_in_either_direction(self):
        for first, last in [((14.5, 3), (15.5, 3)), ((15.5, 3), (14.5, 3))]:
            index = self.index([(first, last)])
            chart = index.query("team:a", phase="offense")["maps"][0]
            self.assertAlmostEqual(chart["total"], 0.1)
            self.assertAlmostEqual(chart["half_court_video_s"], 0.1)
            self.assertAlmostEqual(chart["outside_half_s"], 0.1)
            clipped = index.query("team:a", phase="offense", start=0.05, end=0.15)["maps"][0]
            self.assertAlmostEqual(clipped["total"], 0.05)
            self.assertAlmostEqual(sum(clipped["zones"]), 0.05)

    def test_reversed_cameras_produce_identical_half_court_cells(self):
        normal = self.index([((14.5, 3), (15.5, 3))])
        reversed_view = self.index([((15.5, 12), (14.5, 12))], direction=-1)
        a = normal.query("team:a", phase="offense")["maps"][0]
        b = reversed_view.query("team:a", phase="offense")["maps"][0]
        self.assertEqual(a["grid"], b["grid"])
        self.assertEqual(a["zones"], b["zones"])
        # The same metric normalization applies to the team defending that basket.
        defense = self.index([((15.5, 12), (14.5, 12))], direction=-1, owner="b")
        c = defense.query("team:a", phase="defense")["maps"][0]
        self.assertEqual(a["grid"], c["grid"])

    def test_midline_boundary_and_empty_half_do_not_create_mass(self):
        for point, expected in [((15, 3), 0.2), ((14.999, 3), 0)]:
            chart = self.index([(point, point)]).query("team:a", phase="offense")["maps"][0]
            self.assertAlmostEqual(chart["total"], expected)
            self.assertAlmostEqual(chart["half_court_video_s"], expected)
        touching = self.index([((14.5, 3), (15, 3))]).query("team:a", phase="offense")["maps"][0]
        self.assertEqual(touching["total"], 0)

    def test_evolution_retains_only_time_after_arrival_in_half(self):
        data = self.index([((14.5, 3), (15.5, 3))]).query(
            "team:a", phase="offense", mode="evolution"
        )
        self.assertEqual(data["maps"][0]["total"], 0)
        self.assertAlmostEqual(data["maps"][1]["total"], 0.1)
        self.assertEqual(data["normalization"], "share_of_half_court_mass")

    def test_comparison_automatically_uses_only_the_two_roster_teams(self):
        index = self.index([((24, 3), (24, 3))])
        data = index.query("team:a", phase="offense", mode="comparison")
        self.assertEqual([item["subject"]["id"] for item in data["maps"]], ["team:a", "team:b"])
        with self.assertRaises(ValueError):
            index.query("roster:a:1", phase="offense", mode="comparison")
        with self.assertRaises(ValueError):
            index.query("team:a", phase="offense", mode="comparison", compare="team:a")
