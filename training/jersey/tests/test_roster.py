import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from training.common.provenance import ROOT
from training.jersey.config import settings
from training.jersey.roster import Roster
from training.jersey.temporal import JerseyRecognizer
from training.jersey.tests.test_temporal import (
    Clock,
    Reader,
    fixture_roster,
    observation,
)
from training.players.predict import DEFAULTS, predict_video


class RosterTests(unittest.TestCase):
    def test_shared_number_keeps_names_ambiguous_and_zero_strings_distinct(self):
        payload = fixture_roster().payload
        payload["teams"][1]["players"].append({"number": "23", "name": "Other Player"})
        roster = Roster(payload)
        result = roster.identity("23")
        self.assertEqual(result["identity_status"], "ambiguous")
        self.assertIsNone(result["player_name"])
        self.assertIsNone(result["team_id"])
        self.assertEqual(len(result["candidates"]), 2)
        for n in ("0", "00", "01"):
            self.assertTrue(roster.allows(n))
            self.assertEqual(roster.identity(n)["identity_status"], "unique_number")
        self.assertNotEqual(
            roster.identity("0")["player_name"], roster.identity("00")["player_name"]
        )
        result["candidates"][0]["player_name"] = "corrupted"
        self.assertNotEqual(
            roster.identity("23")["candidates"][0]["player_name"], "corrupted"
        )

    def test_disabled_entries_never_authorize_number_or_name(self):
        payload = fixture_roster().payload
        payload["teams"][0]["players"][0]["eligible"] = False
        roster = Roster(payload)
        self.assertFalse(roster.allows("23"))
        self.assertEqual(roster.identity("23")["candidates"], [])
        self.assertEqual(roster.provenance["players"], 5)
        self.assertEqual(roster.provenance["eligible_players"], 4)

    def test_invalid_roster_fails_closed(self):
        original = fixture_roster().payload
        bad = []
        for n in (23, "２３", "123", "23 ", ""):
            p = copy.deepcopy(original)
            p["teams"][0]["players"][0]["number"] = n
            bad.append(p)
        p = copy.deepcopy(original)
        p["teams"][0]["players"].append(p["teams"][0]["players"][0])
        bad.append(p)
        p = copy.deepcopy(original)
        p["teams"][1]["team_id"] = "a"
        bad.append(p)
        p = copy.deepcopy(original)
        p["teams"][0]["players"] = []
        bad.append(p)
        p = copy.deepcopy(original)
        p["teams"][0]["players"][0]["eligible"] = "false"
        bad.append(p)
        p = copy.deepcopy(original)
        p["teams"][0]["players"][0]["name"] = ""
        bad.append(p)
        p = copy.deepcopy(original)
        p["schema_version"] = True
        bad.append(p)
        p = copy.deepcopy(original)
        p["teams"][0]["players"][0]["eligble"] = False
        bad.append(p)
        for payload in bad:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                Roster(payload)

    def test_optional_team_color_must_be_hex(self):
        payload = fixture_roster().payload
        payload["teams"][0]["color"] = "#12ABef"
        self.assertEqual(Roster(payload).payload["teams"][0]["color"], "#12ABef")
        for invalid in ("red", "#123", "#12345678", 123, " #123456"):
            payload["teams"][0]["color"] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "Team color"):
                Roster(payload)

    def test_loading_snapshot_missing_malformed_and_duplicate_json(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "jersey.json"
            with self.assertRaises(FileNotFoundError):
                Roster.load(path)
            path.write_text('{"schema_version":1,"schema_version":1,"teams":[]}')
            with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
                Roster.load(path)
            path.write_text("not json")
            with self.assertRaises(ValueError):
                Roster.load(path)
            path.write_text(json.dumps(fixture_roster().payload))
            roster = Roster.load(path)
            path.write_text("{}")
            self.assertTrue(roster.allows("23"))
            self.assertEqual(roster.identity("23")["player_name"], "Player A")

    def test_missing_roster_fails_before_model_load_and_output_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = {
                **DEFAULTS,
                "video": str(root / "video.mp4"),
                "output": str(root / "out"),
                "tracking": {"enabled": True},
                "jersey": {"enabled": True},
            }
            with (
                patch("training.players.predict.Detector") as detector,
                self.assertRaisesRegex(FileNotFoundError, "jersey.json"),
            ):
                predict_video(cfg)
            detector.assert_not_called()
            self.assertFalse((root / "out").exists())

    def test_high_confidence_unlisted_number_never_votes_or_becomes_identity(self):
        clock = Clock()
        reader = Reader(clock, text="99")
        engine = JerseyRecognizer(
            settings(), roster=fixture_roster(), reader=reader, clock=clock
        )
        image = np.random.default_rng(4).integers(0, 256, (160, 100, 3), dtype=np.uint8)
        readings = []
        for frame in range(40):
            clock.value += 0.1
            detections, rows, _ = engine.update(
                image,
                [observation()],
                frame_index=frame,
                timestamp_seconds=frame / 10,
                segment_id=0,
            )
            readings += rows
            self.assertIsNone(detections[0]["jersey"]["number"])
            self.assertIsNone(detections[0]["jersey"]["player_name"])
        self.assertTrue(readings)
        self.assertTrue(
            all(
                r["number"] is None
                and r["raw_number"] == "99"
                and r["rejection_reason"] == "not_in_roster"
                for r in readings
            )
        )
        self.assertEqual(len(engine.states[1]["votes"]), 0)
        self.assertEqual(engine.stats["roster_rejections"], len(readings))
        self.assertIsNone(engine.finish()[0]["jersey"]["number"])

    def test_real_game_example_has_both_full_squads_and_known_ambiguities(self):
        path = ROOT / "training/jersey/examples/hou_lal_g1_2026/jersey.json"
        roster = Roster.load(path)
        self.assertEqual(roster.provenance["players"], 30)
        self.assertEqual(roster.provenance["eligible_players"], 25)
        self.assertEqual(roster.identity("23")["player_name"], "LeBron James")
        self.assertEqual(roster.identity("28")["identity_status"], "ambiguous")
        self.assertEqual(roster.identity("10")["identity_status"], "ambiguous")
        self.assertFalse(roster.allows("77"))
        self.assertFalse(roster.allows("7"))
        self.assertTrue(roster.allows("0"))
