"""Attack/defense completion from learned quarter sides and observed court area."""

import copy
import unittest

from viewer.heatmaps import HeatmapIndex
from viewer.phases import PhaseIndex, occupied_half
from viewer.scoreboard import ScoreboardIndex
from viewer.test_phases import GEOMETRY, ROSTER, phase_frame
from viewer.test_scoreboard import document, sample


def settled_frame(index, half=1, owner=None, **extra):
    row = phase_frame(index, owner=owner or "a", **extra)
    for person, x in zip(row["persons"], (20, 21, 23, 26, 27, 28)):
        person["position_m"][0] = x if half == 1 else 30 - x
    if owner is None:
        row["ball"] = {"status": "unavailable", "position_px": None, "confidence": None}
    return row


def scoreboard(rows, periods=None):
    return ScoreboardIndex(
        document([sample(r["timestamp_s"], p) for r, p in zip(rows, periods or [1] * len(rows))]),
        "source",
        ("a", "b"),
    )


def index_for(rows, periods=None, enabled=True, with_quarter=True):
    result = PhaseIndex(
        ("a", "b"),
        GEOMETRY,
        scoreboard=scoreboard(rows, periods) if with_quarter else None,
        use_court_context=enabled,
    )
    plan, previous = -1, None
    for row in rows:
        if (
            previous is None
            or row.get("scene_cut")
            or previous.get("segment_id") != row.get("segment_id")
        ):
            plan += 1
        result.add_frame(row, plan)
        previous = row
    result.finish()
    return result


def span_at(index, time):
    return next((s for s in index.spans if s.start <= time < s.end), None)


def anchored_rows():
    return [phase_frame(i) for i in range(12)] + [settled_frame(i) for i in range(12, 42)]


class CourtContextTests(unittest.TestCase):
    def test_score_update_guard_does_not_extend_the_previous_attack_under_the_rim(self):
        rows = anchored_rows()
        readings = [
            sample(row["timestamp_s"], scores={"a": 2, "b": 4 if i < 26 else 6})
            for i, row in enumerate(rows)
        ]
        board = ScoreboardIndex(document(readings), "source", ("a", "b"))
        index = PhaseIndex(("a", "b"), GEOMETRY, scoreboard=board)
        for row in rows:
            index.add_frame(row, 0)
        index.finish()
        self.assertIsNone(span_at(index, 2.5).owner)
        self.assertEqual(span_at(index, 3.95).reason, "quarter_court_context")
        # The score guard never overrides a contact-based attribution.
        self.assertEqual(span_at(index, 0.8).owner, "a")

    def test_learned_sides_classify_a_long_ball_absence_from_the_occupied_half(self):
        rows = anchored_rows()
        before, after = index_for(rows, enabled=False), index_for(rows)
        self.assertIsNone(span_at(before, 2.5).owner)
        self.assertEqual(span_at(after, 2.5).owner, "a")
        self.assertEqual(span_at(after, 2.5).reason, "quarter_court_context")
        self.assertGreater(after.metadata()["classification_sources"]["court_context_s"], 1)
        self.assertEqual(before.period_directions, after.period_directions)
        for first, second in zip(before.spans, after.spans):
            if first.owner:
                self.assertEqual(
                    (first.owner, first.method, first.reason),
                    (second.owner, second.method, second.reason),
                )

    def test_same_learned_sides_assign_the_other_team_in_the_other_half(self):
        rows = [phase_frame(i) for i in range(12)] + [settled_frame(i, -1) for i in range(12, 42)]
        self.assertEqual(span_at(index_for(rows), 2.5).owner, "b")

    def test_zone_alone_cannot_establish_its_own_direction_or_cross_quarters(self):
        self.assertIsNone(span_at(index_for([settled_frame(i) for i in range(42)]), 2.5).owner)
        rows = anchored_rows()
        self.assertIsNone(span_at(index_for(rows, with_quarter=False), 2.5).owner)
        self.assertIsNone(span_at(index_for(rows, [1] * 12 + [2] * 30), 2.5).owner)

    def test_a_confirmed_contact_overrides_the_zone_after_a_rebound_or_inbound(self):
        rows = [phase_frame(i) for i in range(12)] + [
            settled_frame(i, owner="b") for i in range(12, 42)
        ]
        index = index_for(rows)
        self.assertEqual(span_at(index, 2.5).owner, "b")
        self.assertNotEqual(span_at(index, 2.5).method, "court")

    def test_even_a_single_opposing_contact_blocks_spatial_guessing_nearby(self):
        rows = anchored_rows()
        rows[25] = settled_frame(25, owner="b")
        index = index_for(rows)
        self.assertIsNone(span_at(index, 2.45).owner)
        self.assertIsNone(span_at(index, 2.85).owner)
        self.assertEqual(span_at(index, 3.95).owner, "a")

    def test_referee_and_unassigned_contacts_remain_unknown(self):
        for kind in ("referee", "unassigned", "contested"):
            rows = anchored_rows()
            for i in range(20, 31):
                row = settled_frame(i, owner="a")
                if kind == "referee":
                    row["persons"][0]["role"] = "referee"
                elif kind == "unassigned":
                    row["persons"][0]["resolved_identity"] = None
                else:
                    rival = copy.deepcopy(row["persons"][0])
                    rival.update(track_id=99, resolved_identity={"team_id": "b", "number": "99"})
                    row["persons"].append(rival)
                rows[i] = row
            self.assertIsNone(span_at(index_for(rows), 2.5).owner, kind)

    def test_midcourt_transitions_and_widely_split_teams_are_not_settled_attacks(self):
        for xs in ((13, 14, 15, 15, 16, 17), (3, 4, 5, 25, 26, 27), (9, 10, 11, 12, 13, 14)):
            rows = anchored_rows()
            for row in rows[12:]:
                for person, x in zip(row["persons"], xs):
                    person["position_m"][0] = x
            self.assertIsNone(span_at(index_for(rows), 2.5).owner)

    def test_bad_calibration_duplicate_tracks_and_unknown_outliers_do_not_support_zone(self):
        for change in ("calibration", "duplicates", "unknown_outliers", "missing_opponents"):
            rows = anchored_rows()
            for row in rows[12:]:
                if change == "calibration":
                    row["court"]["status"] = "unavailable"
                elif change == "duplicates":
                    row["persons"][1]["track_id"] = row["persons"][2]["track_id"]
                elif change == "unknown_outliers":
                    for i in range(8):
                        other = copy.deepcopy(row["persons"][0])
                        other.update(track_id=100 + i, resolved_identity=None, position_m=[3, 5])
                        row["persons"].append(other)
                else:
                    row["persons"] = row["persons"][:4]
            self.assertIsNone(span_at(index_for(rows), 2.5).owner, change)

    def test_short_views_and_exclusions_cannot_share_spatial_votes(self):
        rows = (
            [phase_frame(i) for i in range(12)]
            + [settled_frame(i, segment_id=1) for i in range(12, 15)]
            + [settled_frame(i, segment_id=2) for i in range(15, 18)]
        )
        index = index_for(rows)
        self.assertTrue(all(s.method != "court" for s in index.spans))
        rows = anchored_rows()
        for row in rows[12:]:
            row["included"] = row["frame_index"] % 3 != 0
        self.assertTrue(all(s.method != "court" for s in index_for(rows).spans))

    def test_a_verified_reversed_view_uses_its_own_direction(self):
        rows = (
            [phase_frame(i) for i in range(12)]
            + [phase_frame(i, mirror=True, segment_id=1) for i in range(12, 24)]
            + [settled_frame(i, -1, segment_id=1) for i in range(24, 54)]
        )
        index = index_for(rows)
        self.assertEqual(span_at(index, 3.5).owner, "a")
        self.assertEqual(span_at(index, 3.5).direction, -1)
        self.assertEqual(index.period_directions[0]["direction"], 1)

    def test_heatmaps_and_clipped_source_totals_count_spatial_time_once(self):
        rows = anchored_rows()
        index = HeatmapIndex({"court_coordinate_system": GEOMETRY}, ROSTER, scoreboard(rows))
        for row in rows:
            index.add_frame(row)
        index.finish()
        a = index.query("team:a", phase="offense", start=2, end=3)
        b = index.query("team:b", phase="defense", start=2, end=3)
        for data in (a, b):
            self.assertAlmostEqual(data["phase_analysis"]["court_context_s"], 1.0)
            self.assertAlmostEqual(data["phase_analysis"]["contacts_s"], 0.0)
            self.assertAlmostEqual(data["maps"][0]["total"], 3.0)
            self.assertAlmostEqual(sum(data["maps"][0]["grid"]), 3.0)
        sources = index.phases.metadata()["classification_sources"]
        self.assertAlmostEqual(sum(sources.values()), index.phases.metadata()["classified_s"])

    def test_compact_timeline_preserves_cuts_and_unknown_holes(self):
        rows = anchored_rows()
        rows[20]["included"] = False
        rows[30]["scene_cut"] = True
        index = index_for(rows)
        timeline = index.timeline()
        self.assertTrue(any(s["source"] == "court_context" for s in timeline))
        self.assertTrue(all(not (s["start_s"] < 2.0 < s["end_s"]) for s in timeline))
        self.assertTrue(all(not (s["start_s"] < 3.0 < s["end_s"]) for s in timeline))
        self.assertAlmostEqual(
            sum(s["end_s"] - s["start_s"] for s in timeline),
            sum(s.end - s.start for s in index.spans),
        )


if __name__ == "__main__":
    unittest.main()
