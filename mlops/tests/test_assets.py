import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlops.assets import create, verify
from training.common.files import write_json
from training.common.provenance import file_hash, object_hash


class AssetsTests(unittest.TestCase):
    def test_bundle_hash_and_artifact_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "weights.pt").write_bytes(b"fixture weights")
            manifest = {"schema_version": 1, "artifacts": [{"path": "weights.pt", "sha256": file_hash(root / "weights.pt")}]}
            manifest["bundle_id"] = object_hash(manifest)
            write_json(root / "manifest.json", manifest)
            self.assertEqual(verify(root), manifest)
            (root / "weights.pt").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "integrity"):
                verify(root)

    def test_ball_bundle_requires_its_reference_checkout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "ball.pt").write_bytes(b"fixture weights")
            manifest = {"schema_version": 1, "artifacts": [
                {"role": "ball", "path": "ball.pt", "sha256": file_hash(root / "ball.pt")}],
                "tracknet_reference": {"revision": "missing"}}
            manifest["bundle_id"] = object_hash(manifest)
            write_json(root / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "missing.*TrackNet"):
                verify(root)

    def test_cpu_bundle_rejects_unsupported_people_backends_before_copying(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "bundle"
            for adapter in ("rfdetr", "registry"):
                with self.subTest(adapter=adapter), patch("mlops.assets.load_config", return_value={
                        "people": {"adapter": adapter}}):
                    with self.assertRaisesRegex(ValueError, "local YOLO"):
                        create("config.yaml", output)
                    self.assertFalse(output.exists())
