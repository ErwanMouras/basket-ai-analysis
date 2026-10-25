import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pipeline.adapters import People
from pipeline.config import load_config


class RegistryAdapterTests(unittest.TestCase):
    def test_registry_roles_keep_the_tracker_person_class_and_source_classes(self):
        detector = SimpleNamespace(family="yolo", source_class=2, referee_source_class=3,
                                   model=SimpleNamespace(names={2: "player", 3: "referee"}), provenance={"checkpoint_sha256": "x"},
                                   predict=lambda *a, **k: [
                                       {"class_id": 0, "bbox": [1, 1, 10, 20], "confidence": .9},
                                       {"class_id": 1, "bbox": [20, 1, 30, 20], "confidence": .8}])
        metadata = {"model_uri": "models:/players-detector/2", "version": "2", "candidate_id": "candidate"}
        config = load_config()["people"]
        config.update(adapter="registry", registry_uri=metadata["model_uri"])
        with patch("training.players.registry.load_candidate", return_value=(detector, metadata)):
            people = People(config)
            result = people.predict(np.zeros((40, 40, 3), np.uint8))
            self.assertEqual([d["class_id"] for d in result], [0, 0])
            self.assertEqual([d["source_class_id"] for d in result], [2, 3])
            self.assertEqual([d["role"] for d in result], ["player", "referee"])
            detector.model.names[2] = "person"
            with self.assertRaisesRegex(ValueError, "COCO person"):
                People(config)
