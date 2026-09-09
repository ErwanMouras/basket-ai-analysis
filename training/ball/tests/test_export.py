"""Exercise real image/annotation conversion, temporal contracts and safe reexports."""

import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from training.ball.annotator.model import Annotation, VideoMeta
from training.ball.export.config import ExportConfig
from training.ball.export.dataset import export_dataset, output_lock, verify_dataset
from training.ball.export.sources import Geometry
from training.ball.export.temporal import sdk_heatmap


def create_clip(root, relative="train/game/video.avi", seed=0, annotations=None):
    video = root / relative
    video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 48))
    if not writer.isOpened():
        raise RuntimeError("MJPG encoder unavailable")
    for index in range(12):
        frame = np.full((48, 64, 3), seed + index * 10, dtype=np.uint8)
        frame[15:25, 20:30] = (25, 90, 170)
        writer.write(frame)
    writer.release()
    if annotations is None:
        annotations = {index: Annotation(25, 20, 4) for index in range(10)}
        annotations[3] = Annotation(occluded=True)
        annotations[4] = Annotation(25, 20, 4, True)
    sidecar = video.with_name(video.name + ".ballann.json")
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "video": asdict(VideoMeta(str(video), 64, 48, 25, 12)),
                "annotations": {
                    str(index): asdict(ann) for index, ann in annotations.items()
                },
            }
        )
    )
    video.with_name(video.name + ".meta.yaml").write_text(
        yaml.safe_dump(
            {
                "match_id": "/".join(Path(relative).parts[1:-1]) or video.stem,
                "split": Path(relative).parts[0],
            }
        )
    )
    return video, sidecar


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "datas"
        self.output = self.root / "exports"
        self.video, self.sidecar = create_clip(self.source)
        create_clip(self.source, "val/other/video.avi", seed=5)
        self.config = ExportConfig()

    def export(
        self, config=None, formats=("yolo", "coco", "tracknet-totnet"), **kwargs
    ):
        return export_dataset(
            self.source,
            self.output,
            config or self.config,
            formats,
            progress=lambda message: None,
            **kwargs,
        )

    def manifest(self, name):
        return json.loads((self.output / name / "manifest.json").read_text())

    def test_global_export_has_shared_identity_portable_paths_and_single_decode(self):
        from training.ball.annotator.video import VideoReader

        calls = []
        original = VideoReader.read

        def read(reader, index):
            calls.append((reader.meta.path, index))
            return original(reader, index)

        with patch.object(VideoReader, "read", read):
            report = self.export()
        self.assertEqual(report["frames"], 18)
        self.assertEqual(
            len({self.manifest(name)["dataset_id"] for name in report["formats"]}), 1
        )
        self.assertEqual(self.manifest("yolo")["frame_counts"], {"train": 9, "val": 9})
        # Validation and extraction probe the last frame; interior frames decode once.
        self.assertEqual(sum(index == 5 for _, index in calls), 2)
        for name in report["formats"]:
            self.assertGreater(verify_dataset(self.output / name)["files"], 18)
            self.assertNotIn(
                str(self.root), (self.output / name / "manifest.json").read_text()
            )
        yolo_image = next((self.output / "yolo/images/train").rglob("*.jpg"))
        coco_image = self.output / "coco" / yolo_image.relative_to(self.output / "yolo")
        self.assertEqual(yolo_image.stat().st_ino, coco_image.stat().st_ino)
        metadata = yaml.safe_load((self.output / "yolo/data.yaml").read_text())
        self.assertEqual(metadata["names"], {0: "ball"})

    def test_sdk_triplets_never_cross_gaps_or_clips_and_keep_occlusion_status(self):
        self.export()
        rows = read_csv(self.output / "tracknet-totnet/labels_context_train.csv")
        self.assertEqual(len(rows), 5)  # 0..2 and 4..9, excluding unknown frame 3.
        for row in rows:
            paths = [Path(row[key]) for key in ("path_prev", "path", "path_next")]
            indices = [int(path.stem.split("_")[-1]) for path in paths]
            self.assertEqual(indices, list(range(indices[0], indices[0] + 3)))
            self.assertEqual(len({path.parent for path in paths}), 1)
            self.assertNotIn(3, indices)
            for key in (
                "path_prev",
                "path",
                "path_next",
                "gt_path_prev",
                "gt_path",
                "gt_path_next",
            ):
                self.assertTrue((self.output / "tracknet-totnet" / row[key]).is_file())
        row = next(
            row for row in rows if row["status_prev"] == "occluded_with_position"
        )
        self.assertEqual(row["visibility_prev"], "1")
        target = cv2.imread(
            str(self.output / "tracknet-totnet" / row["gt_path_prev"]), 0
        )
        self.assertEqual(set(np.unique(target)), {0, 255})
        self.assertEqual(target[20, 25], 255)

    def test_empty_policy_keeps_explicit_unknown_but_not_unannotated_frames(self):
        self.export(replace(self.config, unknown_position="empty"))
        labels = list((self.output / "yolo/labels/train").rglob("*.txt"))
        self.assertEqual(len(labels), 10)
        self.assertEqual(sum(not path.read_text() for path in labels), 1)
        target = next(
            (self.output / "tracknet-totnet/gts/images/train").rglob("frame_000003.png")
        )
        self.assertFalse(cv2.imread(str(target), 0).any())
        coco = json.loads((self.output / "coco/annotations/train.json").read_text())
        self.assertEqual((len(coco["images"]), len(coco["annotations"])), (10, 9))

    def test_excluding_occluded_positions_keeps_frames_and_temporal_context(self):
        # Put the occlusion inside a continuous segment so dropping that frame
        # would break the triplets on both sides, not just shorten a segment.
        sidecars = sorted(self.source.rglob("*.ballann.json"))
        for sidecar in sidecars:
            payload = json.loads(sidecar.read_text())
            payload["annotations"]["4"]["occluded"] = False
            payload["annotations"]["6"]["occluded"] = True
            sidecar.write_text(json.dumps(payload))
        source_bytes = {path: path.read_bytes() for path in sidecars}
        training_config = self.root / "training.yaml"
        training_config.write_text(
            yaml.safe_dump(
                {
                    "profiles": {
                        "tracknet_v4": {
                            "version": 4,
                            "input_width": 32,
                            "input_height": 24,
                            "sequence_length": 3,
                            "sequence_stride": 1,
                            "target_radius": 2.5,
                        }
                    }
                }
            )
        )

        for unknown_policy, frame_count, triplet_count in (
            ("exclude", 9, 5),
            ("empty", 10, 8),
        ):
            with self.subTest(unknown_position=unknown_policy):
                config = replace(
                    self.config,
                    occluded_position="exclude",
                    unknown_position=unknown_policy,
                    tracknet_layouts=("sdk", "v3", "v4"),
                )
                result = self.export(config, training_config=training_config)
                self.assertEqual(result["frames"], 2 * frame_count)
                for name in ("yolo", "coco", "tracknet-totnet"):
                    manifest = self.manifest(name)
                    self.assertEqual(
                        manifest["frame_counts"],
                        {"train": frame_count, "val": frame_count},
                    )
                    self.assertEqual(
                        manifest["status_counts"]["occluded_with_position"], 2
                    )
                    self.assertEqual(
                        manifest["sources"][0]["excluded_annotated_frames"],
                        10 - frame_count,
                    )
                    verify_dataset(self.output / name)
                self.assertEqual(self.manifest("yolo")["report"]["boxes"], 16)
                label = next(
                    (self.output / "yolo/labels/train").rglob("frame_000006.txt")
                )
                self.assertEqual(label.read_text(), "")
                coco = json.loads(
                    (self.output / "coco/annotations/train.json").read_text()
                )
                image = next(
                    image
                    for image in coco["images"]
                    if image["file_name"].endswith("frame_000006.jpg")
                )
                self.assertTrue((self.output / "coco" / image["file_name"]).is_file())
                self.assertEqual(len(coco["annotations"]), 8)
                self.assertNotIn(
                    image["id"], [ann["image_id"] for ann in coco["annotations"]]
                )

                temporal = self.output / "tracknet-totnet"
                records = [
                    json.loads(line)
                    for line in (temporal / "frames.jsonl").read_text().splitlines()
                ]
                record = next(
                    r
                    for r in records
                    if r["split"] == "train" and r["frame_index"] == 6
                )
                self.assertEqual(
                    record["source_annotation"], asdict(Annotation(25, 20, 4, True))
                )
                self.assertEqual(record["status"], "occluded_with_position")
                self.assertTrue(record["position_excluded"])
                self.assertFalse(record["has_position"])
                self.assertIsNone(record["position"])
                self.assertIsNone(record["bbox"])
                self.assertFalse(any(r["frame_index"] >= 10 for r in records))

                rows = read_csv(temporal / "labels_context_train.csv")
                self.assertEqual(len(rows), triplet_count)
                row = next(
                    row for row in rows if row["path"].endswith("frame_000006.jpg")
                )
                self.assertEqual(row["visibility_current"], "0")
                self.assertEqual((row["x_current"], row["y_current"]), ("", ""))
                self.assertTrue(row["path_prev"].endswith("frame_000005.jpg"))
                self.assertTrue(row["path_next"].endswith("frame_000007.jpg"))
                self.assertFalse(cv2.imread(str(temporal / row["gt_path"]), 0).any())
                self.assertTrue(
                    cv2.imread(str(temporal / row["gt_path_prev"]), 0).any()
                )
                self.assertEqual(
                    self.manifest("tracknet-totnet")["report"]["sdk_triplets"],
                    {"train": triplet_count, "val": triplet_count},
                )

                v3_rows = [
                    row
                    for path in (temporal / "v3/train").rglob("*.csv")
                    for row in read_csv(path)
                ]
                v3_row = next(row for row in v3_rows if row["SourceFrame"] == "6")
                self.assertEqual(
                    (v3_row["Visibility"], v3_row["X"], v3_row["Y"]), ("0", "0", "0")
                )
                samples = [
                    json.loads(line)
                    for line in (temporal / "v4/samples.jsonl").read_text().splitlines()
                ]
                sample = next(s for s in samples if s["source_frames"] == [5, 6, 7])
                target_path = temporal / sample["file"].replace("x_data_", "y_data_")
                targets = np.load(target_path)[sample["sample"]]
                self.assertFalse(targets[1].any())
                self.assertTrue(targets[0].any())
                self.assertTrue(targets[2].any())
        for path, content in source_bytes.items():
            self.assertEqual(path.read_bytes(), content)

    def test_resizing_scales_boxes_and_centers_in_both_formats(self):
        payload = json.loads(self.sidecar.read_text())
        payload["annotations"]["0"] = asdict(Annotation(1, 2, 4))
        self.sidecar.write_text(json.dumps(payload))
        config = replace(
            self.config, resize_width=128, resize_height=128, resize_mode="letterbox"
        )
        self.export(config)
        # Content 128x96 with 16px top padding. Source box [0,0,5,6].
        coco = json.loads((self.output / "coco/annotations/train.json").read_text())
        box = coco["annotations"][0]
        self.assertEqual(box["bbox"], [0, 16, 10, 12])
        self.assertEqual(box["keypoints"], [2, 20, 2])
        label = next((self.output / "yolo/labels/train").rglob("frame_000000.txt"))
        values = [float(x) for x in label.read_text().split()[1:]]
        np.testing.assert_allclose(values, [5 / 128, 22 / 128, 10 / 128, 12 / 128])
        image = next((self.output / "coco/images/train").rglob("frame_000000.jpg"))
        self.assertEqual(cv2.imread(str(image)).shape[:2], (128, 128))
        self.assertEqual(coco["annotations"][3]["keypoints"][-1], 1)

    def test_test_split_is_opt_in_and_conflicting_metadata_is_rejected(self):
        create_clip(self.source, "test/held_out/video.avi", seed=8)
        self.export(formats=("yolo",))
        self.assertNotIn("test", self.manifest("yolo")["frame_counts"])
        self.assertEqual(len(self.manifest("yolo")["excluded_sources"]), 1)
        self.export(
            replace(self.config, splits=("train", "val", "test")), formats=("yolo",)
        )
        self.assertEqual(self.manifest("yolo")["frame_counts"]["test"], 9)
        self.video.with_name(self.video.name + ".meta.yaml").write_text("split: val\n")
        with self.assertRaisesRegex(ValueError, "Split conflict"):
            self.export()

    def test_duplicate_video_across_splits_is_rejected(self):
        other = self.source / "val/other/video.avi"
        shutil.copyfile(self.video, other)
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            self.export()

    def test_match_identity_is_inherited_and_preserved_in_manifest_and_frames(self):
        self.video.with_name(self.video.name + ".meta.yaml").unlink()
        match_path = self.video.parent / "match.yaml"
        match_path.write_text("match_id: match-001\nvenue_id: arena-a\n")
        self.export(formats=("yolo",))
        source = self.manifest("yolo")["sources"][0]
        self.assertEqual(
            (source["match_id"], source["venue_id"]), ("match-001", "arena-a")
        )
        self.assertEqual(
            source["identity_metadata"][0]["path"], "train/game/match.yaml"
        )
        self.assertEqual(len(source["identity_metadata"][0]["sha256"]), 64)
        self.assertIsNone(self.manifest("yolo")["sources"][1]["venue_id"])
        frame = json.loads(
            (self.output / "yolo/frames.jsonl").read_text().splitlines()[0]
        )
        self.assertEqual(
            (frame["match_id"], frame["venue_id"]), ("match-001", "arena-a")
        )

    def test_different_clips_of_one_match_cannot_cross_any_split(self):
        for split, annotated in (("val", True), ("test", True), ("test", False)):
            with self.subTest(split=split, annotated=annotated):
                video, sidecar = create_clip(
                    self.source, f"{split}/renamed/later.avi", seed=9
                )
                video.with_name(video.name + ".meta.yaml").write_text(
                    "match_id: game\n"
                )
                if not annotated:
                    sidecar.unlink()
                try:
                    with self.assertRaisesRegex(
                        ValueError, "Match occurs in multiple splits"
                    ):
                        self.export()
                    self.assertFalse((self.output / "yolo").exists())
                finally:
                    shutil.rmtree(video.parent)

    def test_duplicate_content_in_unselected_test_without_annotations_is_rejected(self):
        video, sidecar = create_clip(self.source, "test/held_out/video.avi", seed=8)
        sidecar.unlink()
        shutil.copyfile(self.video, video)
        with self.assertRaisesRegex(
            ValueError, "Video content occurs in multiple splits"
        ):
            self.export()

    def test_multiple_clips_of_one_match_within_train_are_allowed(self):
        video, _ = create_clip(self.source, "train/renamed/second.avi", seed=9)
        video.with_name(video.name + ".meta.yaml").write_text("match_id: game\n")
        self.export(formats=("yolo",))
        sources = self.manifest("yolo")["sources"]
        self.assertEqual(sum(source["match_id"] == "game" for source in sources), 2)
        self.assertEqual(self.manifest("yolo")["frame_counts"]["train"], 18)

    def test_missing_or_conflicting_identity_is_rejected(self):
        metadata = self.video.with_name(self.video.name + ".meta.yaml")
        metadata.write_text("split: train\n")
        with self.assertRaisesRegex(ValueError, "Missing match_id"):
            self.export()
        match_path = self.video.parent / "match.yaml"
        match_path.write_text("match_id: game\nvenue_id: arena-a\n")
        for text, error in (
            ("match_id: different\n", "Conflicting match_id"),
            ("match_id: game\nvenue_id: arena-b\n", "Conflicting venue_id"),
            ("match_id: 42\n", "match_id must be"),
        ):
            with self.subTest(text=text):
                metadata.write_text(text)
                with self.assertRaisesRegex(ValueError, error):
                    self.export()

    def test_holdout_identity_edits_during_export_abort_publication(self):
        from training.ball.export.dataset import write_yolo

        video, _ = create_clip(self.source, "test/held_out/video.avi", seed=8)
        metadata = video.with_name(video.name + ".meta.yaml")

        def edit_metadata(*args):
            metadata.write_text("match_id: game\n")
            return write_yolo(*args)

        with patch(
            "training.ball.export.dataset.write_yolo", side_effect=edit_metadata
        ):
            with self.assertRaisesRegex(ValueError, "Source changed"):
                self.export()
        self.assertFalse((self.output / "yolo").exists())

    def test_resolved_yaml_captures_defaults_and_cli_overrides_and_is_verified(self):
        config_path = self.root / "minimal.yaml"
        config_path.write_text("jpeg_quality: 90\n")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "training.ball.export",
                "--source",
                str(self.source),
                "--output",
                str(self.output),
                "--config",
                str(config_path),
                "--tracknet-layouts",
                "v3",
                "--format",
                "coco",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        root = self.output / "coco"
        resolved_path = root / "export_config.resolved.yaml"
        resolved = ExportConfig.from_file(resolved_path)
        self.assertEqual(resolved.jpeg_quality, 90)
        self.assertEqual(resolved.tracknet_layouts, ["v3"])
        self.assertEqual(resolved.unknown_position, "exclude")
        self.assertEqual(
            len(yaml.safe_load(resolved_path.read_text())), len(self.config.to_dict())
        )
        audit = json.loads((root / "split_audit.json").read_text())
        self.assertEqual(audit["splits_checked"], ["train", "val", "test"])
        self.assertEqual(len(audit["sources"]), 2)
        verify_dataset(root)
        resolved_path.write_text(resolved_path.read_text().replace("90", "91"))
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            verify_dataset(root)

    def test_missing_split_and_bad_metadata_fail_before_writing(self):
        create_clip(self.source, "unsplit/video.avi", seed=20)
        with self.assertRaisesRegex(ValueError, "Place annotations"):
            self.export()
        shutil.rmtree(self.source / "unsplit")
        payload = json.loads(self.sidecar.read_text())
        payload["video"]["width"] = 100
        self.sidecar.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "width"):
            self.export()
        self.assertFalse((self.output / "yolo").exists())

    def test_repeat_is_deterministic_and_single_format_preserves_other_exports(self):
        self.export()
        old_yolo = self.manifest("yolo")
        old_coco = self.manifest("coco")
        self.export(formats=("yolo",))
        self.assertEqual(self.manifest("yolo"), old_yolo)
        self.assertEqual(self.manifest("coco"), old_coco)
        payload = json.loads(self.sidecar.read_text())
        del payload["annotations"]["5"]
        self.sidecar.write_text(json.dumps(payload))
        self.export(formats=("yolo",))
        self.assertNotEqual(self.manifest("yolo")["dataset_id"], old_yolo["dataset_id"])
        self.assertFalse(
            list((self.output / "yolo/images/train").rglob("frame_000005.jpg"))
        )
        self.assertEqual(self.manifest("coco"), old_coco)

    def test_writer_failure_keeps_previous_exports_and_cleans_staging(self):
        self.export()
        before = self.manifest("yolo")
        with patch(
            "training.ball.export.dataset.write_sdk", side_effect=OSError("disk full")
        ):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.export()
        self.assertEqual(self.manifest("yolo"), before)
        self.assertEqual(list(self.output.glob(".ball-export-*")), [])
        verify_dataset(self.output / "tracknet-totnet")

    def test_publish_failure_rolls_back_every_selected_format(self):
        self.export()
        before = {
            name: self.manifest(name) for name in ("yolo", "coco", "tracknet-totnet")
        }
        original = os.replace
        failed = False

        def fail_once(source, destination):
            nonlocal failed
            if Path(destination) == self.output / "coco" and not failed:
                failed = True
                raise OSError("rename failure")
            return original(source, destination)

        with patch("training.ball.export.dataset.os.replace", side_effect=fail_once):
            with self.assertRaisesRegex(OSError, "rename failure"):
                self.export()
        for name, manifest in before.items():
            self.assertEqual(self.manifest(name), manifest)
            verify_dataset(self.output / name)

    def test_source_edit_during_export_does_not_publish(self):
        from training.ball.export.dataset import write_yolo

        def edit_source(*args):
            self.sidecar.write_text(self.sidecar.read_text() + "\n")
            return write_yolo(*args)

        with patch("training.ball.export.dataset.write_yolo", side_effect=edit_source):
            with self.assertRaisesRegex(ValueError, "Source changed"):
                self.export()
        self.assertFalse((self.output / "yolo").exists())

    def test_failed_rollback_retains_the_old_exports_for_recovery(self):
        self.export()
        original = os.replace

        def fail_publication_and_restoration(source, destination):
            if Path(destination) == self.output / "yolo":
                raise OSError("filesystem unavailable")
            return original(source, destination)

        with patch(
            "training.ball.export.dataset.os.replace",
            side_effect=fail_publication_and_restoration,
        ):
            with self.assertRaisesRegex(OSError, "Rollback failed"):
                self.export()
        staging = next(self.output.glob(".ball-export-*"))
        verify_dataset(staging / ".previous/yolo")
        verify_dataset(self.output / "coco")

    def test_unowned_output_source_overlap_and_concurrent_export_are_rejected(self):
        (self.output / "yolo").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "not an owned"):
            self.export()
        shutil.rmtree(self.output / "yolo")
        with output_lock(self.output):
            with self.assertRaisesRegex(ValueError, "Another export"):
                self.export()
        with self.assertRaisesRegex(ValueError, "overlap"):
            export_dataset(self.source, self.source / "exports", self.config)

    def test_verifier_detects_corrupted_labels_and_works_after_moving(self):
        self.export(formats=("yolo",))
        moved = self.root / "moved"
        shutil.move(self.output / "yolo", moved)
        verify_dataset(moved)
        label = next((moved / "labels").rglob("*.txt"))
        label.write_text("corrupted")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            verify_dataset(moved)

    def test_v3_layout_keeps_contiguous_rallies_and_original_frame_mapping(self):
        self.export(
            replace(self.config, tracknet_layouts=("v3",)), formats=("tracknet-totnet",)
        )
        root = self.output / "tracknet-totnet/v3"
        segments = json.loads((root / "segments.json").read_text())
        self.assertEqual(len(segments), 4)
        self.assertEqual([row["first_source_frame"] for row in segments], [0, 4, 0, 4])
        for segment in segments:
            match = self.output / "tracknet-totnet" / segment["match"]
            frames = match / "frame" / segment["rally"]
            self.assertTrue((frames / "0.png").is_file())
            labels = read_csv(match / "csv" / (segment["rally"] + "_ball.csv"))
            self.assertEqual(
                [int(row["Frame"]) for row in labels],
                list(range(segment["frame_count"])),
            )
            self.assertEqual(
                int(labels[0]["SourceFrame"]), segment["first_source_frame"]
            )

    def test_v4_prepared_arrays_have_native_layout_targets_and_training_fingerprint(
        self,
    ):
        config_path = self.root / "training.yaml"
        profile = {
            "version": 4,
            "input_width": 32,
            "input_height": 24,
            "sequence_length": 3,
            "sequence_stride": 1,
            "target_radius": 2.5,
        }
        config_path.write_text(yaml.safe_dump({"profiles": {"tracknet_v4": profile}}))
        self.export(
            replace(self.config, tracknet_layouts=("sdk", "v4")),
            formats=("tracknet-totnet",),
            training_config=config_path,
        )
        root = self.output / "tracknet-totnet/v4/processed_data/train"
        xs = sorted(root.glob("x_data_*.npy"))
        self.assertEqual(sum(np.load(path).shape[0] for path in xs), 5)
        x = np.load(xs[0])
        y = np.load(xs[0].with_name(xs[0].name.replace("x_data", "y_data")))
        self.assertEqual(x.shape[1:], (9, 24, 32))
        self.assertEqual(y.shape[1:], (3, 24, 32))
        self.assertTrue(0 <= x.min() <= x.max() <= 1)
        self.assertEqual(set(np.unique(y)), {0, 1})
        self.assertEqual(y[0, 0, 10, 12], 1)
        self.assertEqual(
            self.manifest("tracknet-totnet")["training_preparation"], profile
        )
        verify_dataset(self.output / "tracknet-totnet")

    def test_sdk_data_config_uses_training_size_without_reexport_and_rejects_bad_input(
        self,
    ):
        self.export(formats=("tracknet-totnet",))
        path = self.output / "tracknet-totnet/sdk_dataset.py"
        spec = importlib.util.spec_from_file_location("exported_sdk_config", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for width, height in ((512, 288), (864, 480)):
            config = module.build_data(width, height)
            self.assertEqual(config["input_size"], (height, width))
            self.assertEqual(config["original_size"], (48, 64))
            self.assertTrue(Path(config["data"]["train"]["csv_path"]).is_file())
            config["data"]["train"]["pipeline"].insert(0, {"type": "OcclusionAugment"})
            self.assertEqual(
                config["data"]["val"]["pipeline"][0]["type"], "LoadMultiImagesFromPaths"
            )
        with self.assertRaisesRegex(ValueError, "divisible"):
            module.build_data(850, 480)
        verify_dataset(self.output / "tracknet-totnet")


class GeometryAndConfigTests(unittest.TestCase):
    def test_stretch_uses_separate_scales_and_border_heatmap_keeps_known_position(self):
        geo = Geometry.for_video(
            VideoMeta("a", 1920, 1080, 30, 10),
            ExportConfig(resize_width=864, resize_height=480, resize_mode="stretch"),
        )
        self.assertNotEqual(geo.scale_x, geo.scale_y)
        target = sdk_heatmap(
            {"width": 20, "height": 10, "position": [19.9, 9.9]}, 40, 10
        )
        self.assertEqual(target[-1, -1], 255)

    def test_invalid_settings_fail_explicitly(self):
        for options in (
            {"resize_width": 100},
            {"resize_mode": "unknown"},
            {"heatmap_variance": float("nan")},
            {"jpeg_quality": True},
            {"tracknet_layouts": ["v8"]},
            {"splits": "train"},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ExportConfig(**options)


if __name__ == "__main__":
    unittest.main()
