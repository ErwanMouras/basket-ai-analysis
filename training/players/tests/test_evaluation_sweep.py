"""Sweep planning, preflight and dashboard without invoking model inference."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from training.players.evaluation import sweep


class SweepTests(unittest.TestCase):
    def test_default_matrix_contains_each_architecture_once(self):
        plan = sweep.load_plan("training/players/configs/evaluate_all.yaml")
        variants = {(m["recipe"]["model"], m["recipe"]["variant"]) for m in plan["models"]}
        self.assertEqual(variants, sweep.EXPECTED)
        self.assertEqual(len(plan["models"]), 10)
        self.assertTrue(all(m["recipe"]["split"] == "val" for m in plan["models"]))
        self.assertTrue(all(m["recipe"]["max_images"] is None for m in plan["models"]))
        self.assertTrue(all(m["recipe"]["device"] == "cuda:0" for m in plan["models"]))
        self.assertEqual(plan["models"][-1]["recipe"]["referee_source_class"], 3)
        self.assertIsNone(plan["models"][-1]["recipe"]["referee_weights"])

    def test_duplicate_variant_is_rejected_before_any_run(self):
        source = Path("training/players/configs/evaluate_all.yaml")
        payload = yaml.safe_load(source.read_text())
        payload["models"][1]["variant"] = "yolo26n"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.yaml"
            path.write_text(yaml.safe_dump(payload))
            with self.assertRaisesRegex(ValueError, "repeated model"):
                sweep.load_plan(path)

    def test_missing_weights_block_every_model_before_output_creation(self):
        plan = sweep.load_plan("training/players/configs/evaluate_all.yaml")
        with tempfile.TemporaryDirectory() as temp:
            plan["output"] = str(Path(temp) / "new")
            with patch.object(sweep, "readiness", return_value={"ready": False, "dataset": "missing",
                "validation_images": 0, "annotations": {"player": 0, "referee": 0},
                "models": [], "problems": ["missing weights"]}), \
                 patch.object(sweep, "evaluate_model") as evaluate:
                with self.assertRaisesRegex(ValueError, "aucune évaluation"):
                    sweep.run_sweep(plan)
                evaluate.assert_not_called()
            self.assertFalse(Path(plan["output"]).exists())

    def test_dashboard_shows_both_roles_and_report_links(self):
        result = {"run_id": "abc", "metrics": {
            "global": {"ap50_95": .5},
            "groups": [{"group": "class", "value": "player", "ap50_95": .6},
                       {"group": "class", "value": "referee", "ap50_95": .4}]},
            "performance": {"adapter": {"p95_ms": 12.5}}}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "evaluations/yolov8_ebard/abc"
            sweep.dashboard(root, [{"name": "yolov8_ebard", "status": "FINISHED",
                                    "output": str(output), "result": result},
                                   {"name": "rfdetr_large", "status": "FAILED", "error": "bad weights"}],
                            {"abc": 1})
            page = (root / "overview.html").read_text()
            for text in ("Joueurs", "Arbitres", "0.600", "0.400", "12.5 ms/image",
                         "evaluations/yolov8_ebard/abc/report.html", "bad weights"):
                self.assertIn(text, page)

    def test_run_coordinates_ten_models_and_writes_overview_without_inference(self):
        plan = sweep.load_plan("training/players/configs/evaluate_all.yaml")
        audit = {"ready": True, "dataset": plan["dataset"], "validation_images": 2,
                 "annotations": {"player": 1, "referee": 1}, "models": [], "problems": []}
        calls = []

        def fake_evaluation(recipe):
            calls.append(recipe)
            output = Path(recipe["output"]) / f"run-{len(calls)}"
            output.mkdir(parents=True)
            result = {"run_id": f"run-{len(calls)}", "metrics": {
                "global": {"ap50_95": .5},
                "groups": [{"group": "class", "value": role, "ap50_95": .5}
                           for role in ("player", "referee")]},
                "performance": {"adapter": {"p95_ms": 10.0}}}
            (output / "result.json").write_text(json.dumps(result))
            return output

        with tempfile.TemporaryDirectory() as temp:
            plan["output"] = str(Path(temp) / "sweeps")
            with patch.object(sweep, "readiness", return_value=audit), \
                 patch.object(sweep, "evaluate_model", side_effect=fake_evaluation), \
                 patch.object(sweep, "compare", return_value={"rows": []}) as comparison:
                output, complete = sweep.run_sweep(plan)
            self.assertTrue(complete)
            self.assertEqual(len(calls), 10)
            self.assertTrue((output / "overview.html").is_file())
            self.assertEqual(len(comparison.call_args.args[0]), 10)


if __name__ == "__main__":
    unittest.main()
