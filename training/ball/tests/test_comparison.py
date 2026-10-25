"""Guard causal fairness and the shared ball-localization contract."""

import copy
import unittest
from unittest.mock import patch

import numpy as np

from training.ball.comparison.protocol import identity, validate, windows
from training.ball.evaluation.frames import frame_result, summarize
from training.common.provenance import object_hash


def frame(clip, index):
    return {"clip_id": clip, "frame_index": index, "split": "val"}


class ComparisonProtocolTests(unittest.TestCase):
    def test_context_never_crosses_gaps_clips_or_cuts(self):
        frames = [frame("a", i) for i in [0, 1, 2, 4, 5, 6, 7, 8]]
        frames += [frame("b", i) for i in range(4)]
        actual = list(windows(list(reversed(frames)), 3, {"a": [7]}))
        self.assertEqual([identity(w[-1]) for w in actual], [("a", 2), ("a", 6), ("b", 2), ("b", 3)])
        for window in actual:
            self.assertLessEqual(max(f["frame_index"] for f in window), window[-1]["frame_index"])

    def test_longest_context_defines_identical_available_targets(self):
        frames = [frame("a", i) for i in list(range(12)) + list(range(14, 25))]
        common = {identity(w[-1]) for w in windows(frames, 8)}
        self.assertEqual(len(common), 9)
        for length in [1, 3, 8]:
            targets = {identity(w[-1]) for w in windows(frames, length)}
            self.assertEqual(targets & common, common)
        self.assertNotIn(("a", 14), common)

    def test_protocol_rejects_tampering_code_drift_and_test_split(self):
        with patch("training.ball.comparison.protocol.evaluator_hashes", return_value={"evaluator": "abc"}):
            body = {"split": "val", "evaluator_sha256": {"evaluator": "abc"}}
            valid = {**body, "protocol_id": object_hash(body)}
            validate(valid)
            tampered = {**valid, "split": "test"}
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                validate(tampered)
            for changed in [{**body, "split": "test"}, {**body, "evaluator_sha256": {"evaluator": "def"}}]:
                with self.assertRaises(ValueError):
                    validate({**changed, "protocol_id": object_hash(changed)})

    def test_source_tolerance_abstention_unknown_and_displacement(self):
        f = {**frame("a", 8), "image": "x.jpg", "match_id": "match", "timestamp_seconds": 0.2,
             "width": 200, "height": 100, "source_annotation": {"cx": 100, "cy": 50},
             "has_position": True, "position_excluded": False, "status": "visible", "occluded": False,
             "position": [60, 30]}
        source = {"geometry": {"left": 10, "top": 5, "scale_x": 0.5, "scale_y": 0.5},
                  "video_metadata": {"width": 400, "height": 200}, "video": "v", "sidecar": "a"}
        config = {"tolerance_space": "source", "tolerance_px": 15, "unknown_position": "ignore"}
        geometry = {"input_width": 100, "input_height": 50}
        prediction = {"position_input": [35, 15], "window_count": 1, "confidence": 0.9, "future_context_frames": 0}
        row = frame_result(f, source, prediction, geometry, config)
        self.assertEqual(row["error_export_px"], 10)
        self.assertEqual(row["error_source_px"], 20)
        self.assertEqual(row["outcome"], "displaced")
        result = summarize([row])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 1, 1))
        unknown = copy.deepcopy(f)
        unknown.update(status="unknown_position", has_position=False, position=None)
        self.assertEqual(frame_result(unknown, source, prediction, geometry, config)["outcome"], "ignored")
        abstention = {**prediction, "position_input": None}
        self.assertEqual(frame_result(f, source, abstention, geometry, config)["outcome"], "fn")


class CausalDecoderTests(unittest.TestCase):
    def test_yolo_selects_one_highest_confidence_center_and_preserves_rgb(self):
        from types import SimpleNamespace
        import torch
        from training.ball.comparison.worker import Predictor

        class Boxes:
            conf = torch.tensor([0.4, 0.8])
            xywh = torch.tensor([[1., 2., 3., 4.], [10., 12., 4., 6.]])

            def __len__(self):
                return 2

        class DetectionAdapter:
            def predict(self, image, **kwargs):
                self.image, self.kwargs = image, kwargs
                return [SimpleNamespace(boxes=Boxes())]

        adapter = DetectionAdapter()
        predictor = Predictor.__new__(Predictor)
        predictor.length, predictor.yolo, predictor.device = 1, True, torch.device("cpu")
        predictor.geometry = {"input_width": 32, "input_height": 16}
        predictor.protocol = {"thresholds": {"yolo": 0.25}, "yolo_iou": 0.7}
        predictor.model = adapter
        image = np.full((32, 64, 3), [10, 20, 30], np.uint8)
        result = predictor.predict([image])
        self.assertEqual(result["position_input"], [10, 12])
        self.assertEqual(adapter.image[0, 0].tolist(), [30, 20, 10])
        self.assertEqual(adapter.image.shape, (16, 32, 3))
        self.assertEqual(adapter.kwargs["imgsz"], [16, 32])
        self.assertEqual(adapter.kwargs["conf"], 0.25)
        self.assertEqual(adapter.kwargs["classes"], [0])

    def test_v3_decodes_only_last_head_and_preserves_rgb_time_order(self):
        import torch
        from training.ball.comparison.worker import Predictor

        class Network:
            def __call__(self, tensor):
                self.tensor = tensor
                result = torch.zeros((1, 8, 16, 16))
                result[0, :-1, 1, 1] = 1  # earlier heads must not influence target
                result[0, -1, 8, 10] = 1
                return result

        predictor = Predictor.__new__(Predictor)
        predictor.length, predictor.yolo, predictor.device = 8, False, torch.device("cpu")
        predictor.geometry = {"input_width": 16, "input_height": 16}
        predictor.candidate = {"model": "tracknet_v3"}
        predictor.protocol = {"thresholds": {"heatmap": 0.5}}
        predictor.model = Network()
        images = [np.full((16, 16, 3), i * 20, np.uint8) for i in range(8)]
        result = predictor.predict(images)
        self.assertEqual(result["position_input"], [10, 8])
        self.assertEqual(result["future_context_frames"], 0)
        self.assertEqual(tuple(predictor.model.tensor.shape), (1, 24, 16, 16))
        self.assertAlmostEqual(float(predictor.model.tensor[0, -1, 0, 0]), 140 / 255)


if __name__ == "__main__":
    unittest.main()
