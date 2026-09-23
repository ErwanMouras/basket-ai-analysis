"""Editing, cache import and detector proposal safety on synthetic local media."""

import json
import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from training.players.annotator.media import MediaReader, selected_frames
from training.players.annotator.model import Store
from training.players.annotator.preannotate import preannotate_cache, preannotate_model
from training.players.contracts import verified_frames


class MediaFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.media = self.root / "train/game/clip.avi"
        self.media.parent.mkdir(parents=True)
        writer = cv2.VideoWriter(
            str(self.media), cv2.VideoWriter_fourcc(*"MJPG"), 25, (100, 80)
        )
        self.assertTrue(writer.isOpened())
        for i in range(4):
            writer.write(np.full((80, 100, 3), i * 40, dtype=np.uint8))
        writer.release()
        self.reader = MediaReader(self.media)
        self.addCleanup(self.reader.close)
        self.store = Store(self.reader, self.root, match_id="game")


class AnnotationTests(MediaFixture):
    def test_edit_verify_move_resize_delete_and_reopen(self):
        first = self.store.add_box(0, [1, 2, 20, 40])
        second = self.store.add_box(0, [30, 0, 100, 80])
        self.store.verify(0)
        self.assertEqual(len(verified_frames(self.store.document)), 1)
        self.store.edit_box(0, first, bbox=[10, 5, 35, 60], occluded=True)
        self.assertEqual(self.store.frame(0)["review_status"], "in_progress")
        self.store.delete_box(0, second)
        self.store.verify(1)  # An explicit empty negative.
        restored = Store(self.reader, self.root)
        self.assertEqual(restored.document, self.store.document)
        self.assertEqual(restored.frame(0)["boxes"][0]["bbox"], [10, 5, 35, 60])
        self.assertTrue(restored.frame(0)["boxes"][0]["occluded"])
        self.assertEqual(
            [f["frame_index"] for f in verified_frames(restored.document)], [1]
        )
        self.assertEqual(restored.frame(2)["review_status"], "unannotated")
        restored.reopen(1)
        self.assertEqual(verified_frames(restored.document), [])

    def test_failed_save_keeps_disk_memory_and_verification(self):
        box = self.store.add_box(0, [0, 0, 100, 80])
        self.store.verify(0)
        before, content = self.store.document, self.store.path.read_bytes()
        with patch(
            "training.common.files.os.replace", side_effect=OSError("disk full")
        ):
            with self.assertRaises(OSError):
                self.store.edit_box(0, box, bbox=[1, 1, 99, 79])
        self.assertEqual(self.store.document, before)
        self.assertEqual(self.store.path.read_bytes(), content)
        self.assertEqual(list(self.media.parent.glob("*.tmp")), [])
        with self.assertRaises(ValueError):
            self.store.edit_box(0, box, bbox=[-1, 0, 10, 20])
        self.assertEqual(self.store.document, before)

    def test_two_editors_cannot_overwrite_each_other(self):
        other = Store(self.reader, self.root, match_id="game")
        self.store.verify(0)
        with self.assertRaisesRegex(ValueError, "another process"):
            other.verify(1)
        self.assertEqual(other.document["frames"], [])
        self.assertEqual(
            Store(self.reader, self.root).frame(0)["review_status"], "verified"
        )

    def test_changed_media_and_conflicting_identity_fail(self):
        self.store.verify(0)
        with self.assertRaisesRegex(ValueError, "does not match"):
            Store(self.reader, self.root, match_id="other")
        with self.media.open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.store.verify(1)
        with self.assertRaisesRegex(ValueError, "changed"):
            Store(self.reader, self.root)

    def test_sampling_and_decoder_do_not_relabel_failed_frames(self):
        self.assertEqual(list(selected_frames(4, step=2)), [0, 2])
        for args in ({"step": 0}, {"stop": 5}, {"start": -1}):
            with self.assertRaises(ValueError):
                selected_frames(4, **args)
        self.assertGreater(self.reader.read(3).mean(), self.reader.read(0).mean())
        with self.assertRaises(ValueError):
            self.reader.read(4)
        self.assertLessEqual(len(self.reader.cache), 4)

    def cache(self, embedded=True):
        source = self.store.document["source"]
        meta = {
            "version": 1,
            "source": "legacy/clip.avi",
            "players_model": "old.pt",
            "source_info": {"width": 100, "height": 80, "fps": 25, "total_frames": 4},
            "frames_cached": 3,
        }
        if embedded:
            meta["source_sha256"] = source["sha256"]
        payload = {
            "meta": meta,
            "frames": [
                {"frame_idx": i, "detections": [[1, 2, 20, 40, 0.9, 0.0]]}
                for i in range(3)
            ],
        }
        path = self.root / "cache.json"
        path.write_text(json.dumps(payload))
        return path, payload

    def test_cache_import_preserves_manual_work_and_reports_uncovered_frames(self):
        self.store.verify(0)
        self.store.add_box(1, [0, 0, 10, 20])
        path, _ = self.cache()
        report = preannotate_cache(self.store, path, range(4))
        self.assertEqual(report, {"added": 1, "selected": 4, "not_cached": 1})
        self.assertEqual(self.store.frame(0)["boxes"], [])
        self.assertEqual(self.store.frame(1)["review_status"], "in_progress")
        self.assertEqual(self.store.frame(2)["review_status"], "proposed")
        self.assertEqual(self.store.frame(3)["review_status"], "unannotated")
        self.assertEqual(preannotate_cache(self.store, path, range(4))["added"], 0)
        self.assertEqual(
            [f["frame_index"] for f in verified_frames(self.store.document)], [0]
        )

    def test_legacy_cache_requires_explicit_content_binding(self):
        path, _ = self.cache(embedded=False)
        with self.assertRaisesRegex(ValueError, "explicitly bind"):
            preannotate_cache(self.store, path, [0])
        self.assertFalse(self.store.path.exists())
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            preannotate_cache(self.store, path, [0], cache_video_sha256="b" * 64)
        preannotate_cache(
            self.store,
            path,
            [0],
            cache_video_sha256=self.store.document["source"]["sha256"],
        )
        self.assertIn(
            "operator_asserted", self.store.document["provenance"][-1]["reference"]
        )

    def test_malformed_cache_is_rejected_before_any_save(self):
        path, original = self.cache()
        variants = []
        for key, value in (
            ("frames_cached", 99),
            ("source_sha256", "f" * 64),
            ("source", "other.avi"),
        ):
            bad = deepcopy(original)
            bad["meta"][key] = value
            variants.append(bad)
        bad = deepcopy(original)
        bad["frames"][2]["frame_idx"] = 1
        variants.append(bad)
        bad = deepcopy(original)
        bad["frames"][2]["detections"][0][0] = -10
        variants.append(bad)
        for document in variants:
            path.write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                preannotate_cache(self.store, path, [0, 1])
            self.assertFalse(self.store.path.exists())

    def test_model_prediction_arguments_provenance_and_partial_failure(self):
        weights = self.root / "weights.pt"
        weights.write_bytes(b"mock checkpoint")
        seen = []

        class FakeModel:
            task = "detect"
            names = {0: "player"}

            def predict(inner, image, **kwargs):
                seen.append(kwargs)
                if len(seen) == 2:
                    raise RuntimeError("interrupted inference")
                data = SimpleNamespace(
                    cpu=lambda: SimpleNamespace(
                        numpy=lambda: np.array([[1, 2, 20, 40, 0.8, 0]])
                    )
                )
                return [SimpleNamespace(boxes=SimpleNamespace(data=data))]

        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            preannotate_model(
                self.store,
                self.reader,
                weights,
                [0, 1],
                model_factory=lambda _: FakeModel(),
            )
        self.assertEqual(self.store.frame(0)["review_status"], "proposed")
        self.assertEqual(self.store.frame(1)["review_status"], "unannotated")
        self.assertEqual(seen[0]["device"], "cpu")
        self.assertEqual(seen[0]["classes"], [0])
        self.assertEqual(self.store.document["provenance"][-1]["kind"], "model")
        self.assertEqual(verified_frames(self.store.document), [])


@unittest.skipUnless(
    os.environ.get("PLAYERS_GUI_TESTS") == "1",
    "Set PLAYERS_GUI_TESTS=1 with a Tk display",
)
class GuiTests(MediaFixture):
    def test_actual_canvas_creation_move_resize_verify_delete_and_navigation(self):
        import tkinter as tk

        from training.players.annotator.app import App

        root = tk.Tk()
        app = None
        try:
            app = App(root, [self.media], self.root, step=1, match_id="game")
            root.update()
            root.withdraw()
            app.redraw()

            def point(x, y):
                coords = app.transform.display([x, y, x, y])
                return coords[:2]

            app.press(*point(10, 10))
            app.motion(*point(30, 40))
            app.release(*point(30, 40))
            self.assertEqual(len(app.store.frame(0)["boxes"]), 1)
            app.verify()
            app.press(*point(20, 25))
            app.release(*point(30, 30))
            box = app.store.frame(0)["boxes"][0]
            np.testing.assert_allclose(
                box["bbox"], [20, 15, 40, 45], atol=2 / app.transform.scale
            )
            self.assertEqual(app.store.frame(0)["review_status"], "in_progress")
            app.press(*point(40, 45))
            app.release(*point(60, 60))
            np.testing.assert_allclose(
                app.store.frame(0)["boxes"][0]["bbox"],
                [20, 15, 60, 60],
                atol=2 / app.transform.scale,
            )
            app.toggle("occluded")
            self.assertTrue(app.store.frame(0)["boxes"][0]["occluded"])
            app.verify()
            app.delete()
            self.assertEqual(app.store.frame(0)["review_status"], "in_progress")
            app.verify()
            app.navigate(1)
            self.assertEqual(app.index, 1)
            with patch.object(
                app.reader, "read", side_effect=ValueError("decode failure")
            ):
                with self.assertRaises(ValueError):
                    app.goto(2)
            self.assertEqual(app.index, 1)
            self.assertEqual(
                Store(self.reader, self.root).frame(0)["review_status"], "verified"
            )
        finally:
            if app:
                app.reader.close()
            root.destroy()


@unittest.skipUnless(
    os.environ.get("PLAYERS_MODEL_TESTS") == "1",
    "Set PLAYERS_MODEL_TESTS=1 and PLAYERS_CHECKPOINT for CPU inference",
)
class RealModelTests(MediaFixture):
    def test_local_checkpoint_cpu_inference(self):
        result = preannotate_model(
            self.store,
            self.reader,
            Path(os.environ["PLAYERS_CHECKPOINT"]),
            [0],
            imgsz=64,
            device="cpu",
        )
        self.assertEqual(result["added"], 1)
        self.assertEqual(self.store.frame(0)["review_status"], "proposed")
        self.assertEqual(verified_frames(self.store.document), [])
