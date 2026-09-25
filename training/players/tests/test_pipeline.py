"""Small local fixtures for stage reuse, streaming video and registry operations."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from training.common.files import write_json
from training.common.provenance import file_hash, object_hash
from training.players.orchestration.state import input_files, inputs_match, inventory, lock, receipt_valid
from training.players.predict import DEFAULTS as VIDEO_DEFAULTS, predict_video
from training.players.inference import predict_image


class FakeDetector:
    family = "yolo"
    provenance = {"family": "yolo", "variant": "yolo26n", "resolution": 128,
                  "source_class": 0, "checkpoint_sha256": "fixture"}
    def predict(self, image, **kwargs):
        return [{"class_id": 0, "bbox": [5., 6., 25., 40.], "confidence": .8}]


def video_fixture(path, frames=4):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 48))
    if not writer.isOpened():
        raise RuntimeError("mp4v unavailable")
    for index in range(frames):
        writer.write(np.full((48, 64, 3), index * 30, dtype=np.uint8))
    writer.release()


class StateTests(unittest.TestCase):
    def test_parent_interrupt_stops_its_child_process(self):
        from training.players.orchestration.pipeline import Pipeline
        with tempfile.TemporaryDirectory() as temp:
            pipeline = Pipeline({"output": temp})
            pipeline.progress = {"steps": {}}
            real_popen, real_sleep = subprocess.Popen, time.sleep
            children = []
            calls = 0
            def launch(*args, **kwargs):
                child = real_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
                children.append(child)
                return child
            def sleep(seconds):
                nonlocal calls
                calls += 1
                real_sleep(min(seconds, .1))
                if calls == 1:
                    raise KeyboardInterrupt("parent interrupt fixture")
            with patch("training.players.orchestration.pipeline.subprocess.Popen", side_effect=launch), \
                 patch("training.players.orchestration.pipeline.time.sleep", side_effect=sleep):
                with self.assertRaises(KeyboardInterrupt):
                    pipeline.stage("validate", "validate", {}, {})
            self.assertEqual(len(children), 1)
            self.assertIsNotNone(children[0].poll())
            self.assertEqual(pipeline.progress["steps"]["validate"]["status"], "KILLED")

    def test_failed_resume_preflight_does_not_hide_older_interrupted_run(self):
        from training.players.orchestration.pipeline import Pipeline
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pipeline = Pipeline({"output": str(root)})
            config = {"model": "yolo"}
            fingerprint = object_hash({"config": config, "inputs": input_files(pipeline.code),
                                       "environment": pipeline.environment, "kind": "train"})
            older, latest = root / "older", root / "latest"
            write_json(older / "live.json", {"run_id": "interrupted-run", "checkpoints": {}})
            write_json(latest / "live.json", {"status": "FAILED"})
            pipeline.progress = {"steps": {"train": {"attempts": [
                {"directory": str(path), "fingerprint": fingerprint} for path in (older, latest)]}}}
            with patch("training.players.orchestration.pipeline.subprocess.Popen") as launch:
                with self.assertRaisesRegex(ValueError, "no verified resumable checkpoint"):
                    pipeline.stage("train", "train", config, {})
            launch.assert_not_called()

    def test_completed_receipt_survives_lost_parent_update_without_new_process(self):
        from training.players.orchestration.pipeline import Pipeline
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pipeline = Pipeline({"output": str(root)})
            config = {"fixture": "configuration"}
            inputs = input_files(pipeline.code)
            fingerprint = object_hash({"config": config, "inputs": inputs,
                                       "environment": pipeline.environment, "kind": "validate"})
            attempt = root / "attempt"
            artifact = root / "artifact.json"
            artifact.write_text("finished")
            write_json(attempt / "receipt.json", {"fingerprint": fingerprint,
                "result": {"output": str(artifact)}, "artifact_roots": [str(artifact)],
                "artifacts": inventory([artifact])})
            pipeline.progress = {"steps": {"validate": {"status": "RUNNING", "attempts": [
                {"directory": str(attempt), "fingerprint": fingerprint}]}}}
            with patch("training.players.orchestration.pipeline.subprocess.Popen") as launch:
                result = pipeline.stage("validate", "validate", config, {})
            launch.assert_not_called()
            self.assertEqual(result["output"], str(artifact))
            self.assertEqual(pipeline.progress["steps"]["validate"]["status"], "REUSED")

    def test_receipt_requires_matching_config_inputs_and_all_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "artifact.json"
            path.write_text("one")
            digest = object_hash({"config": "one", "inputs": input_files([path])})
            receipt = {"fingerprint": digest, "artifact_roots": [str(root)], "artifacts": inventory([root])}
            self.assertTrue(receipt_valid(receipt, digest))
            self.assertFalse(receipt_valid(receipt, object_hash({"config": "two"})))
            (root / "unexpected").write_text("extra")
            self.assertFalse(receipt_valid(receipt, digest))
            (root / "unexpected").unlink()
            path.write_text("two")
            self.assertFalse(receipt_valid(receipt, digest))
            path.unlink()
            self.assertFalse(receipt_valid(receipt, digest))

    def test_missing_input_creation_and_concurrent_lock_are_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = input_files([root / "absent"])
            self.assertTrue(inputs_match(inputs))
            (root / "absent").write_text("new")
            self.assertFalse(inputs_match(inputs))
            with lock(root / "lock"):
                with self.assertRaises(ValueError), lock(root / "lock"):
                    pass

    def test_pipeline_rejects_foreign_output_and_test_selection(self):
        from training.players.orchestration.config import load_config
        from training.players.orchestration.pipeline import Pipeline
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            export = root / "export.yaml"
            export.write_text("splits: [test]\n")
            recipe = root / "pipeline.json"
            write_json(recipe, {"schema_version": 1, "source": str(root / "source"),
                "output": str(root / "output"), "export_config": str(export),
                "trials": [{"name": "a", "evaluate": "missing"}]})
            with self.assertRaisesRegex(ValueError, "final test"):
                load_config(recipe)
            out = root / "output"
            out.mkdir()
            (out / "foreign.txt").write_text("keep")
            with self.assertRaisesRegex(ValueError, "foreign"):
                Pipeline({"output": str(out)}).run()
            self.assertEqual((out / "foreign.txt").read_text(), "keep")


class VideoTests(unittest.TestCase):
    def test_streamed_video_matches_common_predictions_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video_fixture(root / "input.mp4")
            before = file_hash(root / "input.mp4")
            config = {**VIDEO_DEFAULTS, "video": str(root / "input.mp4"), "output": str(root / "output")}
            detector = FakeDetector()
            with patch("training.players.predict.runtime", return_value={}), patch("training.players.predict.predict_image", wraps=predict_image) as calls:
                result = predict_video(config, detector=detector)
            self.assertEqual(calls.call_count, 4)
            rows = [json.loads(line) for line in (root / "output/predictions.jsonl").read_text().splitlines()]
            self.assertEqual([r["frame_index"] for r in rows], [0, 1, 2, 3])
            self.assertEqual(rows[0]["detections"], predict_image(detector, np.zeros((48, 64, 3), dtype=np.uint8), config))
            self.assertEqual(result["frames"], 4)
            capture = cv2.VideoCapture(str(root / "output/annotated.mp4"))
            count = 0
            while capture.read()[0]:
                count += 1
            capture.release()
            self.assertEqual(count, 4)
            self.assertEqual(file_hash(root / "input.mp4"), before)
            self.assertFalse(list(root.rglob("*.playersann.json")))

    def test_interruption_keeps_explicit_partial_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video_fixture(root / "input.mp4")
            config = {**VIDEO_DEFAULTS, "video": str(root / "input.mp4"), "output": str(root / "output")}
            with patch("training.players.predict.runtime", return_value={}), self.assertRaises(KeyboardInterrupt):
                predict_video(config, detector=FakeDetector(), stop_after_frame=2)
            progress = json.loads((root / "output/progress.json").read_text())
            self.assertEqual(progress["status"], "KILLED")
            self.assertEqual(progress["frames"], 2)
            self.assertEqual(len((root / "output/predictions.partial.jsonl").read_text().splitlines()), 2)
            self.assertFalse((root / "output/result.json").exists())
            self.assertFalse((root / "output/predictions.jsonl").exists())
            with self.assertRaises(FileExistsError):
                predict_video(config, detector=FakeDetector())


@unittest.skipUnless(importlib.util.find_spec("mlflow"), "Dedicated players environment required for local registry")
class RegistryAliasTests(unittest.TestCase):
    def test_explicit_promotion_rollback_stale_alias_and_smoke_guard(self):
        from mlflow import MlflowClient
        from training.players.registry import NAME, promote
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            uri = f"sqlite:///{root / 'mlflow.db'}"
            client = MlflowClient(tracking_uri=uri, registry_uri=uri)
            exp = client.create_experiment("players-evaluation", artifact_location=str(root / "artifacts"))
            evaluation = client.create_run(exp)
            client.set_terminated(evaluation.info.run_id)
            client.create_registered_model(NAME)
            versions = []
            for smoke in (False, False, True):
                run = client.create_run(exp)
                metadata = {"smoke": smoke, "evaluation_run_id": evaluation.info.run_id}
                metadata["candidate_id"] = object_hash(metadata)
                write_json(root / "candidate.json", metadata)
                client.log_artifact(run.info.run_id, str(root / "candidate.json"), "model/data/bundle")
                client.set_terminated(run.info.run_id)
                version = client.create_model_version(NAME, source=f"runs:/{run.info.run_id}/model", run_id=run.info.run_id,
                    tags={"players.smoke": "false", "players.candidate_id": metadata["candidate_id"]})
                versions.append(str(version.version))
            with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": uri}):
                one = promote("champion", versions[0], expected_previous=None, reason="fixture validation", mlflow_uri=uri)
                two = promote("champion", versions[1], expected_previous=versions[0], reason="fixture upgrade", mlflow_uri=uri)
                back = promote("champion", versions[0], expected_previous=versions[1], reason="fixture rollback", mlflow_uri=uri)
                self.assertEqual(back["previous"], versions[1])
                self.assertEqual(str(client.get_model_version_by_alias(NAME, "champion").version), versions[0])
                self.assertNotEqual(one["audit_run_id"], two["audit_run_id"])
                repeated = promote("champion", versions[0], expected_previous=versions[0], reason="same target", mlflow_uri=uri)
                self.assertTrue(repeated["reused"])
                with self.assertRaisesRegex(ValueError, "Alias changed"):
                    promote("champion", versions[1], expected_previous=None, reason="stale", mlflow_uri=uri)
                with self.assertRaisesRegex(ValueError, "Smoke"):
                    promote("champion", versions[2], expected_previous=versions[0], reason="smoke forbidden", mlflow_uri=uri)
            history = client.get_experiment_by_name("players-registry-history")
            self.assertEqual(len(client.search_runs([history.experiment_id])), 3)


@unittest.skipUnless(os.environ.get("PLAYERS_PIPELINE_TESTS"), "Opt-in short real pipeline")
class RealPipelineTests(unittest.TestCase):
    def test_interrupt_resume_reuse_register_reload_and_video(self):
        import mlflow
        from mlflow import MlflowClient
        from training.players.tests.training_fixtures import make_dataset
        from training.players.learning.config import load_config as train_config
        from training.players.evaluation.config import DEFAULTS as EVALUATION
        from training.players.orchestration.config import load_config
        from training.players.orchestration.pipeline import Pipeline
        from training.players.registry import register_candidate, load_candidate, promote

        family = os.environ["PLAYERS_PIPELINE_TESTS"]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(os.environ.get("PLAYERS_TEST_OUTPUT", temp)).resolve()
            root.mkdir(parents=True, exist_ok=True)
            make_dataset(root)
            source_before = inventory([root / "source"])
            weights = str(Path(os.environ["PLAYERS_TEST_WEIGHTS"]).resolve())
            uri = f"sqlite:///{root / 'mlflow.db'}"
            training = train_config(family, Path(__file__).parents[1] / f"configs/train_{family}_smoke.yaml")
            training.update(weights=weights, epochs=2, batch_size=2, resolution=128,
                device=os.environ.get("PLAYERS_TEST_DEVICE", "cpu"), cpu_threads=2, purpose="smoke",
                max_train_images=4, max_val_images=2, mlflow_uri=uri)
            evaluation = {**EVALUATION, "model": family, "variant": training["variant"], "weights": weights,
                "source_class": 0, "resolution": 128, "warmup": 1, "repeats": 1, "cpu_threads": 2,
                "device": training["device"], "mlflow_uri": uri}
            write_json(root / "train.json", training)
            write_json(root / "evaluate.json", evaluation)
            write_json(root / "export.json", {"splits": ["train", "val"]})
            pipeline_path = root / "pipeline.json"
            write_json(pipeline_path, {"schema_version": 1, "source": str(root / "source"),
                "output": str(root / "pipeline"), "export_config": str(root / "export.json"),
                "trials": [{"name": family, "train": str(root / "train.json"), "evaluate": str(root / "evaluate.json")}]})
            with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": uri}):
                with self.assertRaises(KeyboardInterrupt):
                    Pipeline(load_config(pipeline_path), stop_after_epoch=1).run()
                interrupted = json.loads((root / "pipeline/progress.json").read_text())
                self.assertEqual(interrupted["status"], "KILLED")
                parent = interrupted["run_id"]
                progress = Pipeline(load_config(pipeline_path)).run()
                self.assertEqual(progress["status"], "FINISHED")
                trained = progress["steps"]["train-" + family]["result"]
                client = MlflowClient(tracking_uri=uri)
                self.assertEqual(client.get_run(parent).info.status, "KILLED")
                self.assertEqual(client.get_run(trained["run_id"]).data.tags["resumed_from_run"], parent)
                attempts = {k: len(v["attempts"]) for k, v in progress["steps"].items()}
                reused = Pipeline(load_config(pipeline_path)).run()
                self.assertEqual({k: len(v["attempts"]) for k, v in reused["steps"].items()}, attempts)
                self.assertTrue(all(v["status"] == "REUSED" for v in reused["steps"].values()))
                # Corrupt only the comparison artifact: inference and training remain reusable.
                report = Path(reused["steps"]["compare"]["result"]["output"])
                (report / "comparison.csv").write_text("corrupt")
                repaired = Pipeline(load_config(pipeline_path)).run()
                self.assertEqual(len(repaired["steps"]["compare"]["attempts"]), attempts["compare"] + 1)
                self.assertEqual(len(repaired["steps"]["train-" + family]["attempts"]), 2)
                evaluation_path = Path(repaired["steps"]["evaluate-" + family]["result"]["output"])
                registered = register_candidate(evaluation_path, trained["best"], mlflow_uri=uri)
                duplicate = register_candidate(evaluation_path, trained["best"], mlflow_uri=uri)
                self.assertEqual(registered["version"], duplicate["version"])
                self.assertTrue(duplicate["reused"])
                with self.assertRaises(ValueError):
                    promote("champion", registered["version"], expected_previous=None, reason="smoke cannot promote", mlflow_uri=uri)
                detector, metadata = load_candidate(registered["model_uri"], uri)
                self.assertEqual(metadata["training_run_id"], trained["run_id"])
                mlflow.set_tracking_uri(uri)
                mlflow.set_registry_uri(uri)
                reloaded = mlflow.pyfunc.load_model(registered["model_uri"])
                image = np.zeros((48, 64, 3), dtype=np.uint8)
                self.assertEqual(reloaded.predict(image), [predict_image(detector, image, evaluation)])
                video_fixture(root / "input.mp4")
                video = {**VIDEO_DEFAULTS, "registry_uri": registered["model_uri"], "mlflow_uri": uri,
                         "video": str(root / "input.mp4"), "output": str(root / "video")}
                predicted = predict_video(video)
                self.assertEqual(predicted["frames"], 4)
                self.assertEqual(inventory([root / "source"]), source_before)
                write_json(root / "validation.json", {"family": family, "parent_run_id": parent,
                    "training": trained, "evaluation": str(evaluation_path), "registered": registered,
                    "video": predicted, "status": "passed"})
