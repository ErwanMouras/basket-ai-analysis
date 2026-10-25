import tempfile
import unittest
from pathlib import Path

from mlops.assets import verify
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
