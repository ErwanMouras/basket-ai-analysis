"""Installed attacks alternate through hidden neutral transitions, never possessions alone."""

import unittest

from viewer.heatmaps import HeatmapIndex
from viewer.phases import PhaseIndex
from viewer.scoreboard import ScoreboardIndex
from viewer.test_phases import GEOMETRY, ROSTER, phase_frame
from viewer.test_scoreboard import document, sample


def play_frame(index, half=1, owner="a", **extra):
    row = phase_frame(index, owner=owner or "a", **extra)
    for person, x in zip(row["persons"], (21, 20, 22, 26, 27, 25)):
        person["position_m"][0] = x if half == 1 else 30 - x
    if owner is None:
        row["ball"] = {"status": "unavailable", "confidence": None, "position_px": None}
    return row


def scoreboard(rows):
    return ScoreboardIndex(
        document([sample(row["timestamp_s"]) for row in rows]), "source", ("a", "b")
    )


def index_for(rows, board=None):
    index = PhaseIndex(("a", "b"), GEOMETRY, scoreboard=board or scoreboard(rows))
    plan, previous = -1, None
    for row in rows:
        if previous is None or row.get("scene_cut") or row["segment_id"] != previous["segment_id"]:
            plan += 1
        index.add_frame(row, plan)
        previous = row
    index.finish()
    return index


def span_at(index, time):
    return next((s for s in index.spans if s.start <= time < s.end), None)


def transition_rows():
    return (
        [play_frame(i) for i in range(15)]
        + [play_frame(i, owner="b") for i in range(15, 30)]
        + [play_frame(i, half=-1, owner="b") for i in range(30, 50)]
    )


class SettledTests(unittest.TestCase):
    def test_a_cut_does_not_restart_the_old_attack_after_a_confirmed_turnover(self):
        rows = transition_rows()
        rows[23]["scene_cut"] = True
        for i in range(23, 40):
            rows[i] = play_frame(i, half=1 if i < 30 else -1, owner=None, scene_cut=i == 23)
        index = index_for(rows)
        self.assertIsNone(span_at(index, 2.8).owner)
        self.assertEqual(span_at(index, 3.7).owner, "b")

    def test_turnover_is_neutral_until_the_new_team_reaches_the_other_half(self):
        index = index_for(transition_rows())
        self.assertEqual(span_at(index, 0.8).owner, "a")
        self.assertIsNone(span_at(index, 2.3).owner)
        self.assertEqual(span_at(index, 2.3).method, "neutral")
        self.assertEqual(span_at(index, 3.5).owner, "b")
        self.assertGreater(index.metadata()["neutral_s"], 0)

    def test_recovered_ball_under_own_basket_is_never_an_installed_attack(self):
        index = index_for(
            [play_frame(i) for i in range(15)] + [play_frame(i, owner="b") for i in range(15, 60)]
        )
        self.assertTrue(all(s.owner is None for s in index.spans if s.start >= 2))

    def test_hidden_ball_keeps_an_established_phase_without_fragmenting_it(self):
        rows = [play_frame(i, owner="a" if i < 10 or i >= 40 else None) for i in range(55)]
        index = index_for(rows)
        self.assertEqual(span_at(index, 2.5).owner, "a")
        self.assertEqual(len(index.episodes), 1)

    def test_waiting_for_opposite_half_survives_a_hidden_ball(self):
        rows = transition_rows()
        for i in range(24, 40):
            rows[i] = play_frame(i, half=1 if i < 30 else -1, owner=None)
        index = index_for(rows)
        self.assertIsNone(span_at(index, 2.7).owner)
        self.assertEqual(span_at(index, 3.7).owner, "b")

    def test_a_single_opposing_detection_does_not_confirm_a_turnover(self):
        rows = [play_frame(i) for i in range(50)]
        rows[25] = play_frame(25, owner="b")
        index = index_for(rows)
        self.assertEqual(span_at(index, 2.5).owner, "a")
        self.assertEqual(span_at(index, 3.2).owner, "a")

    def test_a_confirmed_steal_back_can_restart_the_original_attack(self):
        rows = (
            [play_frame(i) for i in range(15)]
            + [play_frame(i, owner="b") for i in range(15, 30)]
            + [play_frame(i) for i in range(30, 50)]
        )
        index = index_for(rows)
        self.assertIsNone(span_at(index, 2.4).owner)
        self.assertEqual(span_at(index, 3.6).owner, "a")
        self.assertEqual(len([e for e in index.episodes if e["attacking_team_id"] == "a"]), 2)

    def test_frontcourt_control_does_not_require_every_off_ball_player(self):
        rows = [play_frame(i) for i in range(25)]
        for row in rows[12:]:
            row["persons"] = row["persons"][:4]
        self.assertEqual(span_at(index_for(rows), 1.5).owner, "a")

    def test_midcourt_control_is_not_an_installed_attack(self):
        rows = [play_frame(i) for i in range(15)] + [
            play_frame(i, owner="b") for i in range(15, 50)
        ]
        for row in rows[15:]:
            for p in row["persons"]:
                p["position_m"][0] = 15
        self.assertIsNone(span_at(index_for(rows), 3.5).owner)

    def test_court_disappearance_hides_labels_without_fabricating_positions(self):
        rows = [play_frame(i) for i in range(40)]
        for row in rows[15:25]:
            row["court"]["status"] = "unavailable"
        index = index_for(rows)
        self.assertIsNone(span_at(index, 2).owner)
        self.assertEqual(span_at(index, 3).owner, "a")

    def test_score_update_ends_the_phase_without_assigning_the_other_team(self):
        rows = [play_frame(i, owner="a" if i < 15 else None) for i in range(50)]
        readings = [
            sample(row["timestamp_s"], scores={"a": 2 if i < 20 else 4, "b": 4})
            for i, row in enumerate(rows)
        ]
        board = ScoreboardIndex(document(readings), "source", ("a", "b"))
        index = index_for(rows, board)
        self.assertEqual(span_at(index, 1).owner, "a")
        self.assertIsNone(span_at(index, 3).owner)

    def test_timeline_and_passages_never_include_neutral_intervals(self):
        index = index_for(transition_rows())
        self.assertTrue(all(p["attacking_team_id"] for p in index.timeline()))
        self.assertTrue(all(not (p["start_s"] <= 2.3 < p["end_s"]) for p in index.timeline()))
        for episode in index.episodes:
            self.assertAlmostEqual(episode["duration_s"], episode["classified_s"])

    def test_heatmaps_count_only_installed_play_with_no_transfer_of_transition_time(self):
        rows = transition_rows()
        index = HeatmapIndex({"court_coordinate_system": GEOMETRY}, ROSTER, scoreboard(rows))
        for row in rows:
            index.add_frame(row)
        index.finish()
        for team, phase in (("a", "offense"), ("b", "defense")):
            data = index.query("team:" + team, phase=phase, start=2, end=2.8)
            self.assertEqual(data["maps"][0]["total"], 0)
        attack = index.query("team:b", phase="offense", start=3.5, end=4.5)
        defense = index.query("team:a", phase="defense", start=3.5, end=4.5)
        for data in (attack, defense):
            self.assertAlmostEqual(data["maps"][0]["total"], 3.0)
            self.assertAlmostEqual(sum(data["maps"][0]["grid"]), 3.0)


if __name__ == "__main__":
    unittest.main()
