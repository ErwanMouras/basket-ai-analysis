"""Shared I/O, configuration, provenance and source audits."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.common.config import merge_settings, positive, read_yaml
from training.common.files import write_bytes, write_json, write_jsonl
from training.common.provenance import file_hash, object_hash, source_fingerprints
from training.players.export.metadata import audit_sources, source_videos


class SharedTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_yaml_and_nested_recipe_validation(self):
        config = self.root / "config.yaml"
        config.write_text("model:\n  width: 640\n")
        defaults = read_yaml(config)
        result = merge_settings(defaults, {"model": {"width": 960}})
        self.assertEqual(result["model"]["width"], 960)
        self.assertEqual(defaults["model"]["width"], 640)
        with self.assertRaises(ValueError):
            merge_settings(defaults, {"model": {"widht": 960}})
        with self.assertRaises(TypeError):
            merge_settings(defaults, {"model": 960})
        config.write_text("- not a mapping\n")
        with self.assertRaises(ValueError):
            read_yaml(config)
        for value in (True, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                positive({"value": value}, "value")

    def test_atomic_failure_preserves_original_and_removes_temporary(self):
        path = self.root / "nested/data.json"
        write_json(path, {"value": "première version"})
        original = path.read_bytes()
        with patch(
            "training.common.files.os.replace", side_effect=OSError("disk failure")
        ):
            with self.assertRaises(OSError):
                write_json(path, {"value": "new"})
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(path.parent.glob("*.tmp")), [])
        with self.assertRaises(ValueError):
            write_json(path, {"value": float("nan")})
        self.assertEqual(path.read_bytes(), original)

    def test_jsonl_failure_does_not_publish_partial_rows(self):
        path = self.root / "rows.jsonl"
        write_jsonl(path, [{"value": 0}])
        original = path.read_bytes()

        def broken_rows():
            yield {"value": 1}
            raise RuntimeError("interrupted producer")

        with self.assertRaises(RuntimeError):
            write_jsonl(path, broken_rows())
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.root.glob("*.tmp")), [])
        write_jsonl(path, [{"value": 2}, {"value": 3}])
        self.assertEqual(
            [json.loads(line)["value"] for line in path.read_text().splitlines()],
            [2, 3],
        )

    def test_bytes_and_content_fingerprints(self):
        path = self.root / "model.py"
        write_bytes(path, b"one")
        first = file_hash(path)
        self.assertEqual(source_fingerprints(self.root, [path]), {"model.py": first})
        write_bytes(path, b"two")
        self.assertNotEqual(file_hash(path), first)
        self.assertEqual(object_hash({"b": 1, "a": 2}), object_hash({"a": 2, "b": 1}))

    def _video(self, split, name, content=b"synthetic media identity", match="game"):
        video = self.root / split / name / "clip.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(content)
        video.with_name(video.name + ".meta.yaml").write_text(f"match_id: {match}\n")
        return video

    def test_players_sidecars_discover_missing_videos(self):
        video = self._video("train", "game")
        video.with_name(video.name + ".playersann.json").write_text("{}")
        video.unlink()
        self.assertEqual(source_videos(self.root), [video])
        with self.assertRaises(FileNotFoundError):
            audit_sources(self.root)

    def test_source_audit_preserves_identity_and_rejects_leakage(self):
        video = self._video("train", "game")
        sources, checks = audit_sources(self.root)
        self.assertEqual(sources[video]["match_id"], "game")
        self.assertEqual(sources[video]["split"], "train")
        self.assertIsNone(sources[video]["venue_id"])
        self.assertEqual(checks[video], file_hash(video))
        duplicate = self._video("val", "other", match="other")
        with self.assertRaisesRegex(ValueError, "Video content"):
            audit_sources(self.root)
        duplicate.write_bytes(b"different content")
        duplicate.with_name(duplicate.name + ".meta.yaml").write_text(
            "match_id: game\n"
        )
        with self.assertRaisesRegex(ValueError, "Match occurs"):
            audit_sources(self.root)

    def test_ball_public_helpers_remain_compatible(self):
        from training.ball.export.config import read_yaml as ball_yaml
        from training.ball.export.files import write_json as ball_json
        from training.ball.export.sources import object_hash as ball_hash
        from training.ball.learning.config import merge_settings as ball_merge

        self.assertIs(ball_yaml, read_yaml)
        self.assertIs(ball_json, write_json)
        self.assertIs(ball_hash, object_hash)
        self.assertIs(ball_merge, merge_settings)


if __name__ == "__main__":
    unittest.main()
