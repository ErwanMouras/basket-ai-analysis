import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlops.gold import evaluate, prepare
from training.common.files import write_json, write_jsonl


class GoldTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.gold = self.root / "gold.json"
        self.run = self.root / "run"
        self.run.mkdir()
        write_json(self.run / "run.json", {"status": "completed", "source": {"sha256": "video", "width": 100, "height": 100}})
        self.frames = [{"frame_index": i, "segment_id": 0, "timestamp_s": i / 10, "persons": [
            {"person_id": "a", "bbox": [10, 10, 30, 50], "position_m": [i / 10, 0.], "team_id": "a", "number": "00"}]} for i in range(4)]
        self.predictions = [{"frame_index": i, "segment_id": 0, "timestamp_s": i / 10, "persons": [
            {"role": "player", "track_id": 1 if i < 2 else 2, "bbox": [10, 10, 30, 50], "position_m": [i / 10, 0.],
             "jersey": {"identity_status": "unique_number", "status": "confirmed", "team_id": "a", "number": "00"}}]} for i in range(4)]
        self.save()

    def save(self, split="val"):
        write_json(self.gold, {"schema_version": 1, "status": "reviewed", "reviewer": "fixture",
                              "split": split, "source_sha256": "video", "frames": self.frames})
        write_jsonl(self.run / "observations.jsonl", self.predictions)
        write_jsonl(self.run / "tracks.jsonl", [
            {"frame_index": i, "segment_id": 0, "track_id": 1 if i < 2 else 2,
             "distance_m": .1 if i in (1, 3) else None} for i in range(4)])

    def test_known_identity_switch_and_shared_interval_distance(self):
        result = evaluate(self.gold, self.run)
        self.assertEqual(result["tracking"]["idf1"], .5)
        self.assertEqual(result["tracking"]["id_switches"], 1)
        self.assertEqual(result["geometry"]["median_error_m"], 0.)
        self.assertEqual(result["identity"]["accepted_precision"], 1.)
        self.assertAlmostEqual(result["distance"]["interval_coverage"], 2 / 3)
        self.assertAlmostEqual(result["distance"]["trajectories"][0]["absolute_error_m"], 0.)

    def test_final_abstentions_override_stale_causal_identity(self):
        for frame in self.predictions:
            frame["persons"][0]["majority_jersey_number"] = "00"
        self.save()
        result = evaluate(self.gold, self.run)
        self.assertEqual(result["identity"]["accepted_identity_coverage"], 0.)
        for frame in self.predictions:
            person = frame["persons"][0]
            person["resolved_identity"] = {"team_id": "a", "number": "00"}
            person["jersey_number_suppressed"] = True
        self.save()
        self.assertEqual(evaluate(self.gold, self.run)["identity"]["accepted_identity_coverage"], 0.)

    def test_gold_changes_during_evaluation_are_rejected(self):
        from mlops.gold import box_iou

        def change_gold(a, b):
            document = json.loads(self.gold.read_text())
            document["reviewer"] = "changed during evaluation"
            write_json(self.gold, document)
            return box_iou(a, b)

        with patch("mlops.gold.box_iou", side_effect=change_gold):
            with self.assertRaisesRegex(ValueError, "changed during evaluation"):
                evaluate(self.gold, self.run)

    def test_numbers_preserve_zeros_and_false_person_identity_is_penalized(self):
        for frame in self.predictions:
            frame["persons"][0]["jersey"]["number"] = "0"
        self.save()
        result = evaluate(self.gold, self.run)
        self.assertEqual(result["identity"]["accepted_precision"], 0.)
        self.predictions[0]["persons"].append({**self.predictions[0]["persons"][0], "track_id": 3, "bbox": [60, 10, 80, 50]})
        self.save()
        self.assertEqual(evaluate(self.gold, self.run)["counts"]["unmatched_identity_acceptances"], 1)

    def test_reviewed_status_source_binding_and_test_freeze_are_required(self):
        self.save("test")
        with self.assertRaisesRegex(ValueError, "frozen"):
            evaluate(self.gold, self.run)
        _, _, protocol = prepare(self.gold, self.run, .5)
        freeze = self.root / "freeze.json"
        write_json(freeze, protocol)
        self.assertEqual(evaluate(self.gold, self.run, frozen_recipe=freeze)["split"], "test")
        self.frames[0]["persons"][0]["position_m"] = [1., 0.]
        self.save("test")
        with self.assertRaisesRegex(ValueError, "frozen"):
            evaluate(self.gold, self.run, frozen_recipe=freeze)

    def test_empty_predictions_produce_misses_not_false_perfect_scores(self):
        for frame in self.predictions:
            frame["persons"] = []
        self.save()
        result = evaluate(self.gold, self.run)
        self.assertEqual(result["tracking"]["idf1"], 0.)
        self.assertIsNone(result["identity"]["accepted_precision"])
        self.assertEqual(result["geometry"]["coverage"], 0.)

    def test_independent_cut_labels_and_holdout_landmarks(self):
        self.frames[2]["scene_cut"] = True
        self.frames[2]["segment_id"] = self.frames[3]["segment_id"] = 1
        self.frames[0]["landmarks"] = [{"source_px": [10., 10.], "court_m": [10., 10.], "holdout": True}]
        self.predictions[0]["court"] = {"image_to_court": [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]}
        self.save()
        result = evaluate(self.gold, self.run)
        self.assertEqual(result["cuts"]["recall"], 0.)
        self.assertEqual(result["counts"]["cut_fn"], 1)
        self.assertEqual(result["geometry"]["holdout_median_error_m"], 0.)
