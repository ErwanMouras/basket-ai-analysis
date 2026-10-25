import json
import tempfile
import unittest
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from training.ball.evaluation.config import frozen_payload, require_frozen, load_config, validate_checkpoint_for_split
from training.ball.evaluation.frames import causal_windows, cut_safe_samples
from training.ball.learning.data import frame_index


class PolicyTests(unittest.TestCase):
    def test_prepare_rejects_overlapping_output_and_empty_selection(self):
        from training.ball.evaluation.__main__ import prepare

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "dataset"
            root.mkdir()
            config = {"dataset": str(root), "output": str(root / "results"), "split": "val"}
            with self.assertRaisesRegex(ValueError, "disjoint"):
                prepare(config)
            (root / "manifest.json").write_text(json.dumps({
                "format": "tracknet-totnet", "parameters": {"tracknet_layouts": ["sdk"]}}))
            config["output"] = str(Path(temp) / "results")
            with patch("training.ball.evaluation.__main__.verify_dataset"), patch(
                    "training.ball.evaluation.__main__.frame_index", return_value={}):
                with self.assertRaisesRegex(ValueError, "selection is empty"):
                    prepare(config)

    def test_evaluation_rejects_an_export_replaced_during_inference(self):
        from training.ball.evaluation.__main__ import evaluate

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "dataset"
            root.mkdir()
            output = Path(temp) / "results"
            output.mkdir()
            weights = Path(temp) / "weights.pt"
            weights.write_bytes(b"fixture weights")
            manifest = {"format": "tracknet-totnet", "parameters": {
                "tracknet_layouts": ["sdk"], "unknown_position": "exclude", "occluded_position": "include"},
                "sources": [{"clip_id": "a", "split": "val"}], "dataset_id": "original", "export_id": "original"}
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            record = {"clip_id": "a", "frame_index": 0, "image": "a.jpg"}
            config = {"checkpoint": str(weights), "dataset": str(root), "output": str(output),
                      "split": "val", "threshold": .5, "temporal_policy": "causal_last", "cut_frames": {}}

            class Predictor:
                geometry = {}
                provenance = {"training_run_id": "fixture"}

                def windows(self, dataset):
                    manifest_path.write_text(json.dumps({**manifest, "export_id": "replacement"}))
                    yield from ()

            class Dataset:
                samples = []

                def __len__(self):
                    return 0

            inference = SimpleNamespace(V5Predictor=lambda *args: Predictor(),
                                        SDKFrames=lambda *args: Dataset())
            with patch.dict(sys.modules, {"training.ball.evaluation.inference": inference}), patch(
                    "training.ball.evaluation.__main__.verify_dataset"), patch(
                    "training.ball.evaluation.__main__.frame_index", return_value={"a.jpg": record}), patch(
                    "training.ball.evaluation.__main__.tracked_run", return_value=nullcontext((MagicMock(), "run", output))), patch(
                    "training.ball.evaluation.__main__.frame_result", return_value={}):
                with self.assertRaisesRegex(ValueError, "dataset.*changed"):
                    evaluate(config)
            self.assertFalse((output / "metrics.json").exists())

    def test_cut_resets_causal_context_and_smoke_is_forbidden_on_test(self):
        frames = [{"frame_index": i, "image": str(i), "clip_id": "a"} for i in range(6)]
        samples = [({}, frames[i:i + 3]) for i in range(4)]
        selected = cut_safe_samples(samples, {"a": [3]})
        self.assertEqual([f[-1]["frame_index"] for _, f in selected], [2, 5])
        for trained in ({"max_train_batches": 1}, {"max_val_batches": 5}, {"purpose": "smoke"}):
            with self.assertRaisesRegex(ValueError, "final test"):
                validate_checkpoint_for_split(trained, "test")
        validate_checkpoint_for_split({"max_train_batches": None, "max_val_batches": None}, "test")

    def test_causal_uses_last_head_only_and_leaves_startup_uncovered(self):
        frames = [{"frame_index": i, "image": str(i), "clip_id": "a"} for i in range(4)]
        heatmaps = np.zeros((3, 4, 4))
        heatmaps[0, 0, 0] = 1
        heatmaps[2, 2, 3] = 1
        samples = [({}, frames[:3]), ({}, frames[1:])]
        result = dict(causal_windows(samples, [(frames[:3], heatmaps), (frames[1:], heatmaps)], .5))
        self.assertEqual(set(result), {"2", "3"})
        self.assertEqual(result["2"]["position_input"], [3, 2])
        self.assertEqual(result["2"]["future_context_frames"], 0)
        self.assertEqual(result["2"]["window_count"], 1)

    def test_causal_rejects_gaps_and_duplicate_targets(self):
        frames = [{"frame_index": i, "image": str(i)} for i in (0, 2, 3)]
        maps = np.zeros((3, 4, 4))
        with self.assertRaisesRegex(ValueError, "gap"):
            list(causal_windows([({}, frames)], [(frames, maps)], .5))
        frames[1]["frame_index"], frames[2]["frame_index"] = 1, 2
        with self.assertRaisesRegex(ValueError, "Duplicated"):
            list(causal_windows([({}, frames)] * 2, [(frames, maps)] * 2, .5))

    def test_final_test_freeze_binds_weights_policy_and_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            weights = root / "weights.pt"
            weights.write_bytes(b"weights")
            frozen = root / "frozen.json"
            config_path = root / "recipe.yaml"
            config_path.write_text(f"checkpoint: {weights}\nsplit: test\nfrozen_recipe: {frozen}\ntemporal_policy: causal_last\n")
            config = load_config(config_path)
            payload = frozen_payload(config, {"dataset_id": "d", "export_id": "e"}, [{"image": "a"}])
            frozen.write_text(json.dumps(payload))
            require_frozen(config, payload)
            weights.write_bytes(b"different weights")
            with self.assertRaisesRegex(ValueError, "no longer matches"):
                require_frozen(config, frozen_payload(config, {"dataset_id": "d", "export_id": "e"}, [{"image": "a"}]))
            config["temporal_policy"] = "offline_mean"
            self.assertNotEqual(frozen_payload(config, {"dataset_id": "d", "export_id": "e"}, [])['recipe_id'], payload['recipe_id'])

    def test_training_readers_still_refuse_test(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "frames.jsonl").write_text(json.dumps({"split": "test", "image": "test.png"}) + "\n")
            with self.assertRaises(ValueError):
                frame_index(root, "test")
            self.assertEqual(list(frame_index(root, "test", allow_test=True)), ["test.png"])
