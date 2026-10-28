"""Scorebug validation and quarter-lock behavior, without OCR/network dependencies."""

import copy
import unittest

from viewer.phases import PhaseIndex
from viewer.scoreboard import ScoreboardIndex, confirm_readings
from viewer.scoreboard_extract import parse_clock, parse_period, summarize
from viewer.test_phases import GEOMETRY, phase_frame


def document(samples, sha="source"):
    return {
        "schema_version": 1,
        "source_sha256": sha,
        "sample_s": 0.1,
        "teams": [{"team_id": "b", "label": "BBB"}, {"team_id": "a", "label": "AAA"}],
        "samples": samples,
    }


def sample(time, period=1, scores=None, clock=600):
    return {
        "time_s": time,
        "period": period,
        "clock_s": clock,
        "scores": {"a": 2, "b": 4} if scores is None else scores,
    }


def phase_index(rows, periods):
    scoreboard = ScoreboardIndex(
        document([sample(r["timestamp_s"], p) for r, p in zip(rows, periods)]), "source", ("a", "b")
    )
    index = PhaseIndex(("a", "b"), GEOMETRY, scoreboard=scoreboard)
    for row in rows:
        index.add_frame(row, row.get("segment_id", 0))
    index.finish()
    return index


class ScoreboardTests(unittest.TestCase):
    def test_development_audit_records_abstentions_and_checks_source_binding(self):
        data = document([sample(0), sample(0.1)])
        data["provenance"] = {"method": "scripted_test"}
        checkpoints = {"source_sha256": "source", "readings": [sample(0), sample(1)]}
        result = summarize(data, checkpoints)
        self.assertEqual(result["development_audit"]["scores"]["agree"], 1)
        self.assertEqual(result["development_audit"]["scores"]["unavailable"], 1)
        self.assertFalse(result["independent_accuracy_validation"])
        checkpoints["source_sha256"] = "another-video"
        with self.assertRaises(ValueError):
            summarize(data, checkpoints)

    def test_two_read_confirmation_does_not_turn_a_glitch_or_missing_graphic_into_a_score(self):
        raw = [sample(i * 0.1) for i in range(9)]
        raw[3]["scores"] = {"a": 72, "b": 4}
        raw[5].update(period=None, scores=None, clock_s=None)
        result = confirm_readings(raw, ("a", "b"), 0.1)
        self.assertIsNone(result[3]["scores"])
        self.assertEqual(result[3]["period"], 1)
        self.assertIsNone(result[5]["period"])
        self.assertEqual(result[6]["scores"], {"a": 2, "b": 4})

    def test_period_and_score_remain_readable_when_clock_is_not(self):
        raw = [sample(i * 0.1, clock=None) for i in range(3)]
        result = confirm_readings(raw, ("a", "b"), 0.1)
        self.assertEqual([r["period"] for r in result], [1] * 3)
        self.assertEqual(result[1]["scores"], {"a": 2, "b": 4})
        self.assertIsNone(result[1]["clock_s"])

    def test_scores_and_periods_rewinding_are_not_silently_presented_as_live_updates(self):
        raw = [sample(i * 0.1, 2 if i < 4 else 1) for i in range(8)]
        raw[2]["scores"] = raw[3]["scores"] = {"a": 0, "b": 2}
        result = confirm_readings(raw, ("a", "b"), 0.1)
        self.assertIsNone(result[2]["scores"])
        self.assertIsNone(result[5]["period"])

    def test_two_reads_far_apart_do_not_confirm_each_other(self):
        self.assertTrue(
            all(
                r["period"] is None
                for r in confirm_readings([sample(0), sample(5)], ("a", "b"), 0.5)
            )
        )

    def test_source_binding_team_validation_and_bad_schema_disable_sidecar(self):
        base = document([sample(0), sample(0.1)])
        for mutate in (
            lambda d: d.update(source_sha256="other"),
            lambda d: d.update(schema_version=2),
            lambda d: d["teams"][0].update(team_id="c"),
            lambda d: d["samples"][1].update(time_s=0),
            lambda d: d["samples"][1].update(scores={"a": True, "b": 4}),
        ):
            data = copy.deepcopy(base)
            mutate(data)
            self.assertFalse(ScoreboardIndex(data, "source", ("a", "b")).metadata()["available"])
        self.assertFalse(ScoreboardIndex(base, None, ("a", "b")).metadata()["available"])

    def test_lookup_never_shows_a_future_score_or_holds_across_a_long_gap(self):
        index = ScoreboardIndex(
            document([sample(0), sample(0.1, scores={"a": 4, "b": 4}), sample(2)]),
            "source",
            ("a", "b"),
        )
        self.assertIsNone(index.at(-0.1))
        self.assertEqual(index.at(0.09)["scores"]["a"], 2)
        self.assertEqual(index.at(0.1)["scores"]["a"], 4)
        self.assertIsNone(index.at(1))
        self.assertEqual(len(index.events), 3)
        self.assertEqual(index.metadata()["teams"][0]["team_id"], "b")

    def test_period_and_clock_parsing_keeps_game_time_distinct_from_source_time(self):
        self.assertEqual(parse_clock("11:24"), 684)
        self.assertEqual(parse_clock("8.7"), 8.7)
        self.assertIsNone(parse_clock("11:65"))
        self.assertIsNone(parse_clock("1124"))
        self.assertEqual(
            [parse_period(x) for x in ("1ST", "2ND", "3RD", "4TH", "OT", "2OT")], [1, 2, 3, 4, 5, 6]
        )
        self.assertIsNone(parse_period("HALFTIME"))

    def test_quarter_ranges_stop_at_the_next_period_without_overlap(self):
        index = ScoreboardIndex(
            document([sample(0), sample(0.1), sample(0.2, 3), sample(0.3, 3)]), "source", ("a", "b")
        )
        first, second = index.periods
        self.assertEqual(first["end_s"], second["start_s"])
        self.assertEqual(first["end_s"], 0.2)

    def test_quarter_lock_reuses_direction_in_a_new_plan_without_layout_anchors(self):
        rows = [phase_frame(i, segment_id=0 if i < 15 else 1) for i in range(30)]
        for row in rows[15:]:
            for person in row["persons"][3:]:
                person["position_m"][0] = 14
        index = phase_index(rows, [1] * 30)
        spans = [s for s in index.spans if s.plan == 1 and s.owner]
        self.assertTrue(spans)
        self.assertEqual({s.direction for s in spans}, {1})
        self.assertEqual(index.period_directions[0]["period"], 1)
        self.assertTrue(all(e["orientation_source"] == "quarter_lock" for e in index.episodes))
        inherited = next(e for e in index.orientation_evidence if e['plan_id'] == 1)
        self.assertEqual(inherited['direction'], 1)
        self.assertEqual(inherited['source'], 'quarter_lock')

    def test_small_opposing_layout_vote_cannot_flip_a_locked_quarter(self):
        rows = [
            phase_frame(i, segment_id=0 if i < 15 else 1, mirror=15 <= i < 19) for i in range(30)
        ]
        for row in rows[19:]:
            for person in row["persons"][3:]:
                person["position_m"][0] = 14
        index = phase_index(rows, [1] * 30)
        self.assertEqual({s.direction for s in index.spans if s.plan == 1 and s.owner}, {1})

    def test_verified_reverse_camera_does_not_reverse_the_whole_quarter(self):
        rows = [phase_frame(i, segment_id=0 if i < 15 else 1, mirror=i >= 15) for i in range(30)]
        index = phase_index(rows, [1] * 30)
        self.assertEqual(index.period_directions[0]["direction"], 1)
        self.assertEqual({s.direction for s in index.spans if s.plan == 1 and s.owner}, {-1})
        self.assertEqual(index.episodes[-1]["orientation_source"], "reversed_view")

    def test_quarters_are_independent_and_do_not_alternate_baskets_automatically(self):
        rows = [phase_frame(i, mirror=i >= 30) for i in range(60)]
        index = phase_index(rows, [1] * 15 + [2] * 15 + [3] * 15 + [4] * 15)
        self.assertEqual(
            [(q["period"], q["direction"]) for q in index.period_directions],
            [(1, 1), (2, 1), (3, -1), (4, -1)],
        )
        self.assertTrue(
            all(not (s.start < boundary < s.end) for s in index.spans for boundary in (1.5, 3, 4.5))
        )
        self.assertEqual({e["period"] for e in index.episodes}, {1, 2, 3, 4})

    def test_period_without_reliable_anchor_stays_unoriented(self):
        rows = [phase_frame(i) for i in range(30)]
        for row in rows[15:]:
            for person in row["persons"][3:]:
                person["position_m"][0] = 14
        index = phase_index(rows, [1] * 15 + [2] * 15)
        self.assertEqual({s.direction for s in index.spans if s.period == 2}, {None})


if __name__ == "__main__":
    unittest.main()
