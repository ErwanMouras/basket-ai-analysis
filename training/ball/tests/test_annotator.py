"""Data integrity, frame identity and real Tk interaction tests on synthetic video."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from training.ball.annotator.model import Annotation, Store, VideoMeta, Viewport
from training.ball.annotator.video import VideoReader


def write_video(path, width=320, height=180, count=6):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("MJPG writer unavailable for synthetic test video")
    try:
        for index in range(count):
            frame = np.full((height, width, 3), index * 35, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.meta = VideoMeta(str(Path(self.tmp.name) / "clip.mp4"), 1920, 1080, 25, 8)
        self.store = Store(self.meta)

    def test_roundtrip_three_states_and_unannotated_frame(self):
        self.assertFalse(self.store.path.exists())
        self.store.update(0, Annotation(120, 200, 14))
        self.store.update(1, Annotation(122, 201, 14, True))
        self.store.update(2, Annotation(occluded=True))
        other = Store(self.meta)
        self.assertEqual(other.annotations, self.store.annotations)
        self.assertNotIn(3, other.annotations)
        payload = json.loads(other.path.read_text())
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(
            payload["annotations"]["2"],
            {"cx": None, "cy": None, "radius": None, "occluded": True},
        )
        other.update(1, None)
        self.assertNotIn("1", json.loads(other.path.read_text())["annotations"])

    def test_legacy_document_can_move_with_video_and_preserves_extra_metadata(self):
        legacy = {
            "schema_version": 1,
            "video": {
                "path": "/old/NBA/clip.mp4",
                "width": 1920,
                "height": 1080,
                "fps": 25.0,
                "frame_count": 8,
            },
            "default_radius": 14,
            "annotations": {
                "0": {"cx": 200.0, "cy": 100.0, "radius": 14.0, "occluded": False}
            },
            "provenance": {"review": "manual"},
        }
        self.store.path.write_text(json.dumps(legacy))
        other = Store(self.meta)
        other.update(1, Annotation(occluded=True))
        self.assertEqual(
            json.loads(other.path.read_text())["provenance"], legacy["provenance"]
        )
        self.assertEqual(other.annotations[0], Annotation(200, 100, 14))

    def test_invalid_annotations_never_write(self):
        for ann in [
            Annotation(),
            Annotation(1, None, 3, True),
            Annotation(float("nan"), 3, 4),
            Annotation(1920, 3, 4),
            Annotation(1, 1080, 4),
            Annotation(-1, 3, 4),
            Annotation(1, 3, 0),
            Annotation(True, 3, 4),
            Annotation(1, 3, 4, "false"),
        ]:
            with self.subTest(ann=ann), self.assertRaises(ValueError):
                self.store.update(0, ann)
        self.assertFalse(self.store.path.exists())
        self.assertEqual(self.store.annotations, {})

    def test_rejects_wrong_video_metadata_and_frame_index(self):
        self.store.update(0, Annotation(2, 3, 4))
        for changes in [
            {"width": 960},
            {"height": 540},
            {"frame_count": 7},
            {"fps": 30},
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                Store(replace(self.meta, **changes))
        for frame_idx in [-1, 8, True]:
            with self.assertRaises(ValueError):
                self.store.update(frame_idx, Annotation(2, 3, 4))

    def test_rejects_invalid_json_without_overwriting(self):
        self.store.path.write_text('{"broken":')
        with self.assertRaises(ValueError):
            Store(self.meta)
        self.assertEqual(self.store.path.read_text(), '{"broken":')

    def test_radius_keeps_unknown_position_all_null(self):
        self.store.update(0, Annotation(occluded=True))
        self.store.adjust_radius(0, 1)
        self.assertEqual(Store(self.meta).annotations[0], Annotation(occluded=True))
        self.assertEqual(self.store.default_radius, 15)
        self.store.update(1, Annotation(2, 3, 8))
        self.store.adjust_radius(1, 1)
        self.assertEqual(self.store.annotations[1].radius, 9)

    def test_failed_save_preserves_disk_and_memory(self):
        self.store.update(0, Annotation(2, 3, 4))
        before = self.store.path.read_bytes()
        with patch(
            "training.ball.annotator.model.os.replace",
            side_effect=OSError("disk failure"),
        ):
            with self.assertRaises(OSError):
                self.store.update(1, Annotation(occluded=True))
        self.assertEqual(self.store.path.read_bytes(), before)
        self.assertNotIn(1, self.store.annotations)
        self.assertEqual(list(Path(self.tmp.name).glob("*.tmp")), [])

    def test_detects_concurrent_change(self):
        second = Store(self.meta)
        self.store.update(0, Annotation(2, 3, 4))
        before = self.store.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "another tool"):
            second.update(1, Annotation(occluded=True))
        self.assertEqual(self.store.path.read_bytes(), before)

    def test_display_mapping_preserves_source_coordinates(self):
        for width, height in [(960, 540), (800, 800), (853, 479), (2400, 1400)]:
            vp = Viewport.fit(1920, 1080, width, height)
            with self.subTest(size=(width, height)):
                self.assertLessEqual(vp.width, 1920)
                self.assertLessEqual(vp.height, 1080)
                for x, y in [(0, 0), (1919, 1079), (123.25, 450.75)]:
                    result = vp.to_source(*vp.to_display(x, y))
                    self.assertAlmostEqual(result[0], x)
                    self.assertAlmostEqual(result[1], y)
                self.assertIsNone(vp.to_source(vp.left - 1, vp.top))
                self.assertIsNone(vp.to_source(vp.left + vp.width, vp.top))
                self.assertIsNone(vp.to_source(vp.left, vp.top + vp.height))


class VideoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "video with spaces.avi"
        write_video(self.path)
        self.reader = VideoReader(self.path)
        self.addCleanup(self.reader.close)

    def test_random_seeks_keep_frame_identity_and_source_size(self):
        self.assertEqual(self.reader.meta.frame_count, 6)
        for i in [0, 5, 2, 3, 0, 4, 1]:
            frame = self.reader.read(i)
            self.assertEqual(frame.shape, (180, 320, 3))
            self.assertAlmostEqual(float(frame.mean()), i * 35, delta=3)

    def test_failed_tail_does_not_return_previous_frame(self):
        self.reader.meta = replace(self.reader.meta, frame_count=7)
        with self.assertRaisesRegex(ValueError, "not decodable"):
            self.reader.read(6)
        self.assertNotIn(6, self.reader._cache)
        self.assertAlmostEqual(float(self.reader.read(2).mean()), 70, delta=3)


class GuiTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk

        from training.ball.annotator.app import App

        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"No Tk display: {exc}")
        self.addCleanup(self.root.destroy)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        path = Path(self.tmp.name) / "ui.avi"
        write_video(path, width=1920, height=1080)
        self.reader = VideoReader(path)
        self.addCleanup(self.reader.close)
        self.store = Store(self.reader.meta)
        self.app = App(self.root, self.reader, self.store)
        self.root.geometry("1000x700")
        self.root.update()
        self.root.focus_force()
        self.root.update()

    def press(self, key):
        self.root.event_generate(f"<KeyPress-{key}>")
        self.root.update()

    def test_click_resize_keyboard_states_and_resume(self):
        app = self.app
        vp = app.viewport
        self.assertLess(vp.width, self.reader.meta.width)
        x, y = map(round, vp.to_display(960, 540))
        app.canvas.event_generate("<Button-1>", x=x, y=y)
        self.root.update()
        ann = Store(self.reader.meta).annotations[0]
        self.assertAlmostEqual(ann.cx, 960, delta=3)
        self.assertAlmostEqual(ann.cy, 540, delta=3)
        self.assertEqual(ann.radius, 14)
        before = self.store.path.read_bytes()
        self.root.geometry("820x550")
        self.root.update()
        self.assertEqual(self.store.path.read_bytes(), before)
        self.press("v")
        self.assertTrue(self.store.annotations[0].occluded)
        self.press("Right")
        self.assertEqual(app.frame_idx, 1)
        self.press("c")
        self.assertEqual(self.store.annotations[1], self.store.annotations[0])
        self.press("h")
        self.press("Up")
        self.assertEqual(
            Store(self.reader.meta).annotations[1], Annotation(occluded=True)
        )
        self.press("x")
        self.assertNotIn(1, self.store.annotations)
        self.press("h")
        self.assertEqual(app.frame_idx, 1)
        self.assertEqual(self.store.annotations[1], Annotation(occluded=True))
        self.assertIn("position unknown", app.info.get())
        # An unreadable target must not change the active annotation index.
        with patch.object(
            self.reader, "read", side_effect=ValueError("decode failure")
        ):
            with self.assertRaises(ValueError):
                app.goto(5)
        self.assertEqual(app.frame_idx, 1)
        self.assertNotIn(5, self.store.annotations)

    def test_click_in_letterbox_does_not_annotate(self):
        vp = self.app.viewport
        self.app.place(vp.left - 1, vp.top)
        self.assertFalse(self.store.path.exists())

    def test_last_frame_shortcut(self):
        self.press("G")
        self.assertEqual(self.app.frame_idx, self.reader.playable_frame_count - 1)


if __name__ == "__main__":
    unittest.main()
