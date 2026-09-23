"""Small analytical fixtures; no training, network, DVC or real-data writes."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from training.common.files import write_json
from training.common.provenance import object_hash
from training.players.evaluation.config import DEFAULTS, frozen_payload, require_frozen, validate_config
from training.players.evaluation.metrics import (AREA_RANGES, coco_metrics, evaluate, match_frame,
                                                normalize_predictions, operating_metrics)
from training.players.export.config import ExportConfig
from training.players.export.geometry import Geometry
from training.players.models import Detector

HAS_COCO = importlib.util.find_spec("pycocotools") is not None
PROTOCOL = {"score_floor": .001, "score_threshold": .5, "iou_threshold": .5, "max_detections": 100}


def detection(box=(10, 10, 30, 30), score=.9):
    return {"class_id": 0, "bbox": list(box), "confidence": score}


def record(name="a", boxes=((10, 10, 30, 30),)):
    return {"frame_id": name, "source_id": "clip", "match_id": "match", "venue_id": None,
            "frame_index": 0, "image": name + ".png", "width": 100, "height": 80,
            "transform": Geometry.build(100, 80, ExportConfig()).to_dict(),
            "boxes": [{"source_bbox": list(box), "bbox": list(box), "occluded": False} for box in boxes]}


class MatchingTests(unittest.TestCase):
    def test_cli_freezes_test_without_loading_weights_or_writing_dataset(self):
        from training.players.tests.training_fixtures import make_dataset
        from training.players.export.dataset import export_dataset
        from training.players.export.verify import verify_export

        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            make_dataset(base)
            export_dataset(base / "source", base / "test-export", ExportConfig(splits=("test",)))
            root = (base / "test-export/coco").resolve()
            before = verify_export(root)
            weights = base / "unloadable.pt"
            weights.write_bytes(b"freeze hashes weights but does not deserialize them")
            config = {**DEFAULTS, "weights": str(weights), "dataset": str(root), "split": "test",
                      "output": str(base / "runs")}
            recipe = base / "recipe.json"
            frozen = base / "frozen.json"
            write_json(recipe, config)
            command = [sys.executable, "-m", "training.players.evaluation", "evaluate", "--config", str(recipe)]
            result = subprocess.run(command + ["--freeze-recipe", str(frozen)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            config["frozen_recipe"] = str(frozen)
            require_frozen(config, json.loads(frozen.read_text()))
            self.assertEqual(verify_export(root), before)
            self.assertFalse((base / "runs").exists())
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Test is reserved for a frozen recipe", result.stderr)

    def test_tp_fp_fn_and_duplicate(self):
        rows = [record(boxes=((10, 10, 30, 30), (50, 50, 70, 70))), record("empty", ())]
        predictions = {"a": [detection(), detection(score=.8)], "empty": [detection()]}
        result, events = operating_metrics(rows, predictions, score_threshold=.5, iou_threshold=.5)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 2, 1))
        self.assertAlmostEqual(result["precision"], 1/3)
        self.assertEqual(result["recall"], .5)
        self.assertAlmostEqual(result["f1"], .4)
        self.assertEqual(len(events), 4)

    def test_empty_and_missed_images(self):
        result, _ = operating_metrics([record("empty", ())], {"empty": []}, score_threshold=.5, iou_threshold=.5)
        self.assertIsNone(result["recall"])
        self.assertIsNone(result["precision"])
        self.assertIsNone(result["f1"])
        result, _ = operating_metrics([record()], {"a": []}, score_threshold=.5, iou_threshold=.5)
        self.assertEqual(result["fn"], 1)
        self.assertEqual(result["f1"], 0)

    def test_threshold_inclusive_and_greedy_best_match(self):
        boxes = [[0, 0, 20, 10], [10, 0, 20, 10]]
        events = match_frame(boxes, [detection((10, 0, 20, 10), .5), detection((0, 0, 10, 10), .4)],
                             score_threshold=.5, iou_threshold=.5)
        self.assertEqual(events[0]["gt_index"], 1)
        self.assertEqual(events[1]["kind"], "FN")
        events = match_frame(boxes[:1], [detection((0, 0, 10, 10), .5)], score_threshold=.5, iou_threshold=.5)
        self.assertEqual(events[0]["kind"], "TP")

    def test_size_ignores_other_size_matches_and_background(self):
        result = match_frame([[0, 0, 100, 80]], [detection((0, 0, 100, 80))],
                             score_threshold=.5, iou_threshold=.5, area_range=AREA_RANGES["small"])
        self.assertEqual(result, [])

    def test_coordinate_roundtrip_stretch_letterbox_and_padding(self):
        for mode in ("stretch", "letterbox"):
            row = record()
            geometry = Geometry.build(101, 57, ExportConfig(resize_width=160, resize_height=160, resize_mode=mode))
            row["transform"] = geometry.to_dict()
            source = [1.5, 2.7, 99.1, 55.9]
            result = normalize_predictions(row, [detection(geometry.bbox(source))])
            np.testing.assert_allclose(result[0]["bbox"], source)
        row["transform"] = Geometry.build(100, 50, ExportConfig(resize_width=100, resize_height=100)).to_dict()
        self.assertEqual(normalize_predictions(row, [detection((0, 0, 10, 20))]), [])
        self.assertEqual(normalize_predictions(row, [detection((0, 20, 10, 40))])[0]["bbox"], [0, 0, 10, 15])

    def test_score_floor_cap_and_invalid_predictions(self):
        result = normalize_predictions(record(), [detection(score=.1), detection(score=.9), detection(score=.001)], max_detections=1)
        self.assertEqual(result[0]["confidence"], .9)
        for bad in (detection(score=float("nan")), detection((20, 20, 10, 10)), {**detection(), "class_id": 1}):
            with self.assertRaises(ValueError):
                normalize_predictions(record(), [bad])

    def test_recipe_validation(self):
        for change in ({"precision": "amp"}, {"warmup": 0}, {"score_floor": .25},
                       {"score_threshold": 0}, {"max_detections": True}, {"split": "train"},
                       {"resolution": 65}, {"iou_threshold": float("nan")}, {"unknown": 1},
                       {"split": "test", "purpose": "smoke"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_config({**DEFAULTS, "weights": "local.pt", **change})

    def test_final_test_requires_matching_frozen_recipe(self):
        config = {**DEFAULTS, "weights": "local.pt", "split": "test"}
        expected = frozen_payload(config, checkpoint_sha256="w", dataset_id="d", selection_id="s", evaluator_id="e")
        with self.assertRaises(ValueError):
            require_frozen(config, expected)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "frozen.json"
            write_json(path, expected)
            config["frozen_recipe"] = str(path)
            require_frozen(config, expected)
            for key in ("checkpoint_sha256", "dataset_id", "selection_id", "evaluator_id"):
                altered = {**expected, key: "changed"}
                with self.assertRaises(ValueError):
                    require_frozen(config, altered)
            altered = frozen_payload({**config, "score_threshold": .7}, checkpoint_sha256="w", dataset_id="d", selection_id="s", evaluator_id="e")
            with self.assertRaises(ValueError):
                require_frozen(config, altered)

    def test_training_checkpoint_links_parent_and_preserves_smoke(self):
        from training.players.evaluation.run import checkpoint_reference
        from training.common.provenance import file_hash

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "checkpoint.pt"
            path.write_bytes(b"synthetic mock checkpoint")
            config = {**DEFAULTS, "weights": str(path)}
            state = {"family": "yolo", "run_id": "parent", "contract": {"purpose": "smoke", "epochs": 2}}
            with patch.dict("sys.modules", {"torch": SimpleNamespace(load=lambda *a, **kw: {"players_training": state})}):
                reference = checkpoint_reference(config, file_hash(path))
                self.assertTrue(reference["smoke"])
                self.assertEqual(reference["training_run_id"], "parent")
                with self.assertRaises(ValueError):
                    checkpoint_reference({**config, "model": "rfdetr"}, file_hash(path))
            with patch.dict("sys.modules", {"torch": SimpleNamespace(load=lambda *a, **kw: {})}):
                with self.assertRaises(ValueError):
                    checkpoint_reference(config, file_hash(path))


@unittest.skipUnless(HAS_COCO, "Install the dedicated players environment for COCO tests")
class CocoTests(unittest.TestCase):
    def test_occlusion_cohorts_keep_other_players_and_negative_fp(self):
        row = record(boxes=((10, 10, 30, 30), (50, 50, 70, 70)))
        row["boxes"][0]["occluded"] = True
        result, _, _ = evaluate([row, record("negative", ())],
                                 {"a": [detection()], "negative": [detection()]}, PROTOCOL)
        cohorts = [g for g in result["groups"] if g["group"] == "occlusion_image_cohort"]
        self.assertEqual([g["support"] for g in cohorts], [2, 2, 0])
        self.assertEqual(cohorts[0]["fn"], 1)
        self.assertEqual(result["global"]["fp"], 1)

    def test_perfect_predictions_and_ap_not_truncated_at_operating_score(self):
        rows = [record(), record("negative", ())]
        predictions = {"a": [detection(score=.2)], "negative": []}
        metrics, curves, _ = evaluate(rows, predictions, PROTOCOL)
        self.assertAlmostEqual(metrics["global"]["ap50_95"], 1)
        self.assertAlmostEqual(metrics["global"]["ap50"], 1)
        self.assertEqual(metrics["global"]["recall"], 0)
        self.assertEqual(len(curves["coco_pr"]), 1010)
        unsupported = [g for g in metrics["groups"] if not g["support"]]
        self.assertTrue(unsupported)
        self.assertTrue(all(g["ap50"] is None for g in unsupported))

    def test_empty_detections_no_gt_and_duplicate_ap(self):
        metrics, _ = coco_metrics([record()], {"a": []})
        self.assertEqual(metrics["ap50_95"], 0)
        metrics, _ = coco_metrics([record("empty", ())], {"empty": [detection()]})
        self.assertIsNone(metrics["ap50_95"])
        rows = [record(boxes=((10, 10, 30, 30), (50, 50, 70, 70)))]
        metrics, _ = coco_metrics(rows, {"a": [detection(), detection(score=.8), detection((50, 50, 70, 70), .7)]})
        self.assertAlmostEqual(metrics["ap50"], (51 + 50 * 2/3) / 101)

    def test_missing_predictions_rejected(self):
        with self.assertRaises(ValueError):
            evaluate([record()], {}, PROTOCOL)

    def test_identical_native_outputs_from_both_adapters_have_equal_metrics(self):
        class Tensor:
            def __init__(self, value): self.value = np.asarray(value)
            def cpu(self): return self
            def numpy(self): return self.value
        boxes = [[10, 10, 30, 30], [10, 10, 30, 30]]
        scores = [.9, .8]
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        outputs = []
        for family, cls in (("yolo", 0), ("rfdetr", 1)):
            detector = Detector.__new__(Detector)
            detector.family, detector.source_class, detector.device, detector.resolution = family, cls, "cpu", 640
            if family == "yolo":
                native = [SimpleNamespace(boxes=SimpleNamespace(xyxy=Tensor(boxes), conf=Tensor(scores)))]
            else:
                native = SimpleNamespace(xyxy=np.array(boxes), confidence=np.array(scores), class_id=np.array([1, 1]))
            detector.model = SimpleNamespace(predict=lambda *args, **kwargs: native)
            with patch.dict("sys.modules", {"torch": SimpleNamespace(device=lambda value: value)}):
                predictions = normalize_predictions(record(), detector.predict(image, confidence=.001, max_detections=100, square=True))
            outputs.append(evaluate([record()], {"a": predictions}, PROTOCOL))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0][0]["global"]["fp"], 1)


class ComparisonTests(unittest.TestCase):
    def test_protocol_grouping_smoke_failed_unsupported_and_ties(self):
        from training.players.evaluation.compare import compare
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            paths = []
            for index in range(8):
                protocol = {"resolution": 640 if index != 1 else 384, "precision": "fp32"}
                comparison = {"dataset_id": "same" if index != 2 else "other", "protocol": protocol}
                perf = {k: {"median_ms": 1, "p95_ms": 2, "images_per_second": 1000} for k in ("forward", "adapter", "image_pipeline")}
                perf.update(gpu_peak_allocated_bytes=None, gpu_peak_reserved_bytes=None, cpu_rss_sampled_peak_bytes=100)
                result = {"run_id": str(index), "status": "FINISHED", "purpose": "smoke" if index == 3 else "evaluation",
                          "reference": {"smoke": index == 4}, "comparison": comparison, "comparison_id": object_hash(comparison),
                          "model": {"family": "yolo", "variant": "yolo26n", "resolution": protocol["resolution"], "source_class": 0, "checkpoint_sha256": "w"},
                          "metrics": {"global": {"ap50": .6, "ap50_95": None if index == 6 else .5}}, "performance": perf}
                path = base / str(index)
                write_json(path / "result.json", result)
                write_json(path / "summary.json", {"run_id": str(index), "status": "FAILED" if index == 5 else "FINISHED"})
                paths.append(path)
            (base / "5/result.json").unlink()  # A failed real run may not have a result yet.
            report = compare(paths, base / "comparison")
            self.assertEqual(len(report["rows"]), 4)
            self.assertEqual(len(report["excluded"]), 4)
            self.assertTrue(all(r["rank_in_protocol"] == 1 for r in report["rows"]))
            report = compare(paths, base / "with-smoke", include_smoke=True)
            self.assertEqual(len(report["rows"]), 6)
            self.assertTrue((base / "comparison/comparison.html").is_file())


@unittest.skipUnless(os.environ.get("PLAYERS_EVALUATION_TESTS"), "Opt-in real model evaluation")
class RealEvaluationTests(unittest.TestCase):
    def test_pretrained_evaluation_and_mlflow(self):
        from training.players.evaluation.run import run
        from training.players.tests.training_fixtures import make_dataset
        from mlflow import MlflowClient

        family = os.environ["PLAYERS_EVALUATION_TESTS"]
        with tempfile.TemporaryDirectory() as temp:
            base = Path(os.environ.get("PLAYERS_TEST_OUTPUT", temp))
            base.mkdir(parents=True, exist_ok=True)
            exports = make_dataset(base)
            config = {**DEFAULTS, "model": family, "variant": "yolo26n" if family == "yolo" else "rfdetr_nano",
                      "source_class": 0 if family == "yolo" else 1,
                      "weights": os.environ["PLAYERS_TEST_WEIGHTS"], "reference": "local pretrained technical fixture",
                      "dataset": str(exports / "coco"), "output": str(base / "runs"),
                      "mlflow_uri": f"sqlite:///{base / 'mlflow.db'}", "device": os.environ.get("PLAYERS_TEST_DEVICE", "cpu"),
                      "resolution": 128, "warmup": 1, "repeats": 2, "purpose": "smoke"}
            validate_config(config)
            with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": config["mlflow_uri"]}):
                output = run(config)
            result = json.loads((output / "result.json").read_text())
            self.assertEqual(result["purpose"], "smoke")
            self.assertEqual(result["reference"]["kind"], "imported")
            self.assertEqual(result["performance"]["adapter"]["samples"], 4)
            self.assertEqual(result["performance"]["forward"]["samples"], 4)
            for artifact in ("predictions.jsonl", "report.html", "review.csv", "curves.svg", "metrics.csv"):
                self.assertTrue((output / artifact).is_file())
            client = MlflowClient(tracking_uri=config["mlflow_uri"])
            tracked = client.get_run(result["run_id"])
            self.assertEqual(tracked.info.status, "FINISHED")
            self.assertEqual(client.get_experiment(tracked.info.experiment_id).name, "players-evaluation")
            self.assertIn("adapter.p95_ms", tracked.data.metrics)
