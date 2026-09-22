"""Annotation safety and interchange contracts, without model dependencies."""

import json
import unittest
from copy import deepcopy
from pathlib import Path

from training.players.contracts import (
    validate_annotations,
    validate_manifest,
    validate_predictions,
    verified_frames,
)

EXAMPLES = Path(__file__).parents[1] / "configs"


def example(name):
    return json.loads((EXAMPLES / f"{name}.example.json").read_text())


class ContractsTests(unittest.TestCase):
    def test_examples_round_trip_and_return_independent_documents(self):
        for name, validate in (
            ("annotations", validate_annotations),
            ("predictions", validate_predictions),
            ("manifest", validate_manifest),
        ):
            with self.subTest(name=name):
                original = example(name)
                result = validate(original)
                self.assertEqual(json.loads(json.dumps(result)), original)
                result["schema_version"] = 99
                self.assertEqual(original["schema_version"], 1)
                validate(original)

    def test_only_verified_frames_are_exportable_including_empty_frames(self):
        document = example("annotations")
        for index, status in enumerate(("proposed", "in_progress"), start=3):
            frame = deepcopy(document["frames"][0])
            frame.update(frame_index=index, review_status=status)
            document["frames"].append(frame)
        frames = verified_frames(document)
        self.assertEqual([frame["frame_index"] for frame in frames], [0, 1])
        self.assertEqual(frames[1]["boxes"], [])
        frames[0]["boxes"].clear()
        self.assertTrue(document["frames"][0]["boxes"])

    def test_multiple_players_and_source_edges_are_allowed(self):
        document = example("annotations")
        second = deepcopy(document["frames"][0]["boxes"][0])
        second.update(object_id="box-2", bbox=[0, 0, 1280, 720])
        document["frames"][0]["boxes"].append(second)
        self.assertEqual(len(validate_annotations(document)["frames"][0]["boxes"]), 2)

    def test_invalid_boxes_are_rejected(self):
        cases = [
            {"bbox": [-1, 0, 100, 100]},
            {"bbox": [0, 0, 1281, 720]},
            {"bbox": [1, 1, 1, 2]},
            {"bbox": [2, 2, 1, 1]},
            {"bbox": [0, 0, float("nan"), 3]},
            {"bbox": [False, 0, 1, 2]},
            {"bbox": [0, 1, 2]},
            {"class_id": 1},
            {"class_id": False},
            {"occluded": 1},
            {"truncated": "unknown"},
            {"confidence": 1.1},
            {"confidence": float("inf")},
            {"object_id": ""},
            {"jersey_number": 23},
        ]
        for changes in cases:
            with self.subTest(changes=changes):
                document = example("annotations")
                document["frames"][0]["boxes"][0].update(changes)
                with self.assertRaises(ValueError):
                    validate_annotations(document)

    def test_ambiguous_annotation_frames_are_rejected(self):
        for changes in (
            {"frame_index": -1},
            {"frame_index": 100},
            {"frame_index": True},
            {"review_status": "unannotated"},
            {"review_status": "done"},
        ):
            with self.subTest(changes=changes):
                document = example("annotations")
                document["frames"][0].update(changes)
                with self.assertRaises(ValueError):
                    validate_annotations(document)
        document = example("annotations")
        document["frames"].append(deepcopy(document["frames"][0]))
        with self.assertRaises(ValueError):
            validate_annotations(document)
        document = example("annotations")
        document["frames"][0]["boxes"] *= 2
        with self.assertRaises(ValueError):
            validate_annotations(document)

    def test_wrong_versions_types_and_unknown_fields_fail(self):
        for name, validate in (
            ("annotations", validate_annotations),
            ("predictions", validate_predictions),
            ("manifest", validate_manifest),
        ):
            for changes in (
                {"schema_version": 2},
                {"schema_version": True},
                {"artifact_type": "ball_annotations"},
                {"surprise": 1},
                {1: "non-string key", "extra": True},
            ):
                with self.subTest(name=name, changes=changes):
                    document = example(name)
                    document.update(changes)
                    with self.assertRaises(ValueError):
                        validate(document)

    def test_source_identity_and_geometry_are_strict(self):
        for changes in (
            {"path": "../escape.mp4"},
            {"path": "/clip.mp4"},
            {"path": "train/./clip.mp4"},
            {"path": "C:\\clip.mp4"},
            {"sha256": "bad"},
            {"match_id": " "},
            {"split": "validation"},
            {"width": True},
            {"height": 0},
            {"fps": float("nan")},
            {"frame_count": 0},
            {"venue_id": " room"},
        ):
            with self.subTest(changes=changes):
                document = example("annotations")
                document["source"].update(changes)
                with self.assertRaises(ValueError):
                    validate_annotations(document)

    def test_image_source_and_inference_without_dataset_split(self):
        document = example("predictions")
        document["source"].update(
            kind="image", path="image.jpg", fps=None, frame_count=1, split=None
        )
        document["frames"] = document["frames"][:1]
        self.assertEqual(validate_predictions(document), document)
        document["source"]["fps"] = 25
        with self.assertRaises(ValueError):
            validate_predictions(document)

    def test_prediction_scores_and_model_provenance(self):
        for score in (-0.1, 1.1, True, float("nan")):
            document = example("predictions")
            document["frames"][0]["detections"][0]["confidence"] = score
            with self.subTest(score=score), self.assertRaises(ValueError):
                validate_predictions(document)
        document = example("predictions")
        document["model"]["checkpoint_sha256"] = "unknown"
        with self.assertRaises(ValueError):
            validate_predictions(document)
        document = example("predictions")
        document["frames"].append(document["frames"][0])
        with self.assertRaises(ValueError):
            validate_predictions(document)

    def test_model_proposals_require_a_weight_fingerprint(self):
        document = example("annotations")
        document["provenance"] = [{"kind": "model", "reference": "yolo"}]
        with self.assertRaises(ValueError):
            validate_annotations(document)
        document["provenance"][0]["sha256"] = "a" * 64
        validate_annotations(document)

    def test_manifest_rejects_missing_inventory_and_unsafe_paths(self):
        for inventory in (
            {},
            {"../image.jpg": "a" * 64},
            {"manifest.json": "a" * 64},
            {"image.jpg": "invalid"},
        ):
            with self.subTest(inventory=inventory):
                document = example("manifest")
                document["artifacts"] = inventory
                with self.assertRaises(ValueError):
                    validate_manifest(document)

    def test_manifest_rejects_source_and_match_leakage(self):
        for collision in ("source_id", "path", "match_id", "sha256"):
            document = example("manifest")
            second = deepcopy(document["sources"][0])
            second["source"].update(
                source_id="other",
                path="val/other.mp4",
                match_id="other",
                sha256="b" * 64,
                split="val",
            )
            second["source"][collision] = document["sources"][0]["source"][collision]
            document["splits"].append("val")
            document["sources"].append(second)
            with self.subTest(collision=collision), self.assertRaises(ValueError):
                validate_manifest(document)

    def test_manifest_format_classes_splits_and_parameters(self):
        for changes in (
            {"format": "tracknet"},
            {"classes": {"0": "ball"}},
            {"splits": ["train", "train"]},
            {"splits": ["val"]},
            {"parameters": {"nested": {1: "not a JSON key"}}},
            {"parameters": {"threshold": float("inf")}},
        ):
            with self.subTest(changes=changes):
                document = example("manifest")
                document.update(changes)
                with self.assertRaises(ValueError):
                    validate_manifest(document)


if __name__ == "__main__":
    unittest.main()
