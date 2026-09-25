"""Delivery contracts: corrupt bundles and validation isolation."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.common.files import write_json
from training.common.provenance import ROOT, file_hash, object_hash
from training.players.registry_loader import detector_from_bundle


class ValidationContracts(unittest.TestCase):
    def test_corrupt_registry_weights_and_metadata_fail_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "weights.pt"
            weights.write_bytes(b"fixture")
            metadata = {"weights_file": weights.name, "model": {"family": "yolo",
                "variant": "yolo26n", "checkpoint_sha256": file_hash(weights),
                "resolution": 128, "source_class": 0}}
            metadata["candidate_id"] = object_hash(metadata)
            write_json(root / "candidate.json", metadata)
            with patch("training.players.registry_loader.Detector") as detector:
                detector_from_bundle(root)
                detector.assert_called_once()
                detector.reset_mock()
                weights.write_bytes(b"corrupt")
                with self.assertRaisesRegex(ValueError, "checkpoint integrity"):
                    detector_from_bundle(root)
                metadata["smoke"] = False
                write_json(root / "candidate.json", metadata)
                with self.assertRaisesRegex(ValueError, "metadata integrity"):
                    detector_from_bundle(root)
                detector.assert_not_called()

    def test_offline_guard_is_inherited_by_python_children(self):
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).parent / "offline")}
        code = "import socket; socket.create_connection(('example.invalid', 443))"
        result = subprocess.run([sys.executable, "-c", code], env=env,
                                capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Network disabled during synthetic players validation", result.stderr)

    def test_demo_requires_explicit_gpu_opt_in_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "untouched"
            result = subprocess.run([sys.executable, "-m", "training.players.tests.validate",
                "--output", str(output), "--device", "cuda:0"], cwd=ROOT,
                capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(output.exists())
