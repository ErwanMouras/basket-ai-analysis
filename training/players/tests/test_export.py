"""CPU fixtures for reviewed-only exports, leakage, geometry and publication."""

import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from training.common.files import write_json
from training.common.provenance import file_hash
from training.players.annotator.media import MediaReader, source_record
from training.players.annotator.model import sidecar_path
from training.players.export.config import ExportConfig
from training.players.export.dataset import export_dataset
from training.players.export.geometry import Geometry
from training.players.export.publication import verify_bundle
from training.players.export.sources import check_unchanged
from training.players.export.verify import (
    artifact_inventory,
    dataset_identity,
    export_identity,
    read_json,
    verify_export,
)


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = self.base / "source"
        self.output = self.base / "exports"
        self.train = self.fixture("train/game-a/positive.png", color=20)
        self.negative = self.fixture("train/game-a/negative.png", color=30, boxes=[])
        self.val = self.fixture("val/game-b/positive.png", color=60)

    def fixture(
        self,
        relative,
        *,
        color=100,
        boxes=None,
        status="verified",
        match=None,
        venue="hall",
        video=False,
    ):
        path = self.source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if video:
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (100, 80)
            )
            self.assertTrue(writer.isOpened())
            for index in range(6):
                writer.write(np.full((80, 100, 3), color + index * 10, dtype=np.uint8))
            writer.release()
        else:
            self.assertTrue(
                cv2.imwrite(str(path), np.full((81, 101, 3), color, dtype=np.uint8))
            )
        reader = MediaReader(path)
        try:
            source = source_record(
                reader,
                self.source,
                match_id=match or Path(relative).parts[1],
                venue_id=venue,
                split=Path(relative).parts[0],
            )
        finally:
            reader.close()
        if boxes is None:
            boxes = [
                {
                    "object_id": "b",
                    "class_id": 0,
                    "bbox": [10.25, 2.5, 90, 75],
                    "occluded": True,
                    "truncated": None,
                },
                {
                    "object_id": "a",
                    "class_id": 0,
                    "bbox": [0, 0, 100, 80],
                    "occluded": False,
                    "truncated": True,
                },
            ]
        write_json(
            sidecar_path(path),
            {
                "schema_version": 1,
                "artifact_type": "players_annotations",
                "source": source,
                "provenance": [
                    {"kind": "manual", "reference": "synthetic-test-fixture"}
                ],
                "frames": [{"frame_index": 0, "review_status": status, "boxes": boxes}],
            },
        )
        return path

    def edit(self, path, change):
        sidecar = sidecar_path(path)
        document = read_json(sidecar)
        change(document)
        write_json(sidecar, document)

    def export(self, **config):
        return export_dataset(self.source, self.output, ExportConfig(**config))

    def records(self, fmt="yolo"):
        return [
            json.loads(line)
            for line in (self.output / fmt / "frames.jsonl").read_text().splitlines()
        ]

    def test_formats_share_verified_images_and_boxes_including_negatives(self):
        self.fixture("train/game-a/proposed.png", color=31, status="proposed")
        self.fixture("train/game-a/incomplete.png", color=32, status="in_progress")
        self.fixture(
            "train/game-a/unannotated.png", color=33, status="unannotated", boxes=[]
        )
        self.export()
        self.assertEqual(self.records(), self.records("coco"))
        self.assertEqual(len(self.records()), 3)
        self.assertEqual(
            read_json(self.output / "yolo/stats.json")["total"],
            {"images": 3, "boxes": 4, "negatives": 1},
        )
        for split in ("train", "val"):
            coco = read_json(self.output / "coco/annotations" / f"{split}.json")
            self.assertEqual(coco["categories"], [{"id": 1, "name": "player"}])
            for img in coco["images"]:
                name = img["file_name"]
                self.assertEqual(
                    (self.output / "yolo" / name).read_bytes(),
                    (self.output / "coco" / name).read_bytes(),
                )
                label = (
                    self.output / "yolo" / name.replace("images/", "labels/")
                ).with_suffix(".txt")
                boxes = [a for a in coco["annotations"] if a["image_id"] == img["id"]]
                lines = label.read_text().splitlines()
                self.assertEqual(len(boxes), len(lines))
                for line, ann in zip(lines, boxes):
                    cls, cx, cy, w, h = map(float, line.split())
                    self.assertEqual(cls, 0)
                    x, y, bw, bh = ann["bbox"]
                    np.testing.assert_allclose(
                        [
                            (cx - w / 2) * img["width"],
                            (cy - h / 2) * img["height"],
                            w * img["width"],
                            h * img["height"],
                        ],
                        [x, y, bw, bh],
                        atol=1e-12,
                    )
                if not boxes:
                    self.assertEqual(label.read_bytes(), b"")
        manifests = verify_bundle(self.output)
        self.assertEqual(
            manifests["yolo"]["dataset_id"], manifests["coco"]["dataset_id"]
        )
        self.assertNotEqual(
            manifests["yolo"]["export_id"], manifests["coco"]["export_id"]
        )

    def test_geometry_roundtrip_letterbox_and_stretch_and_jpeg(self):
        for mode in ("letterbox", "stretch"):
            with self.subTest(mode=mode):
                self.export(
                    resize_width=64,
                    resize_height=48,
                    resize_mode=mode,
                    image_format="jpg",
                )
                for record in self.records():
                    transform = Geometry(**record["transform"])
                    for box in record["boxes"]:
                        np.testing.assert_allclose(
                            transform.bbox(box["bbox"], inverse=True),
                            box["source_bbox"],
                            atol=1e-12,
                        )
                    image = cv2.imread(str(self.output / "coco" / record["image"]))
                    self.assertEqual(image.shape[:2], (48, 64))
                verify_bundle(self.output)
        geometry = Geometry.build(
            101, 81, ExportConfig(resize_width=64, resize_height=48)
        )
        self.assertEqual(
            (
                geometry.content_width,
                geometry.content_height,
                geometry.left,
                geometry.top,
            ),
            (60, 48, 2, 0),
        )
        np.testing.assert_allclose(geometry.bbox([0, 0, 101, 81]), [2, 0, 62, 48])

    def test_reproducible_after_relocation_and_idempotent_publication(self):
        first = self.export()
        before = verify_bundle(self.output)
        self.assertEqual(first, self.export())
        other_source, other_output = self.base / "moved", self.base / "other-output"
        shutil.copytree(self.source, other_source)
        export_dataset(other_source, other_output)
        self.assertEqual(before, verify_bundle(other_output))
        self.assertEqual(len(list((self.output / ".generations").iterdir())), 1)

    def test_changed_annotation_changes_ids_and_preserves_old_generation(self):
        old = self.export()
        self.edit(
            self.train, lambda d: d["frames"][0]["boxes"][0].update(occluded=False)
        )
        new = self.export()
        self.assertNotEqual(old["dataset_id"], new["dataset_id"])
        self.assertTrue(Path(old["generation"]).is_dir())
        verify_export(Path(old["generation"]) / "yolo")

    def test_video_sampling_keeps_only_explicit_reviewed_frames(self):
        video = self.fixture("train/game-c/clip.avi", color=100, video=True)

        def frames(document):
            boxes = document["frames"][0]["boxes"]
            document["frames"] = [
                {"frame_index": 0, "review_status": "verified", "boxes": boxes},
                {"frame_index": 1, "review_status": "proposed", "boxes": []},
                {"frame_index": 2, "review_status": "in_progress", "boxes": boxes},
                {"frame_index": 4, "review_status": "verified", "boxes": []},
            ]

        self.edit(video, frames)
        self.export()
        source_id = read_json(sidecar_path(video))["source"]["source_id"]
        frames = [r for r in self.records() if r["source_id"] == source_id]
        self.assertEqual([r["frame_index"] for r in frames], [0, 4])
        values = [
            cv2.imread(str(self.output / "yolo" / r["image"])).mean() for r in frames
        ]
        self.assertGreater(values[1], values[0] + 30)
        audit = read_json(self.output / "yolo/split_audit.json")
        row = next(s for s in audit["sources"] if s["path"].endswith("clip.avi"))
        self.assertEqual(
            row["status_counts"],
            {"absent": 2, "in_progress": 1, "proposed": 1, "verified": 2},
        )

    def test_no_verified_frames_fails_without_publishing(self):
        for media in (self.train, self.negative, self.val):
            self.edit(media, lambda d: d["frames"][0].update(review_status="proposed"))
        with self.assertRaisesRegex(ValueError, "No verified"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_missing_sidecar_is_excluded_and_not_an_implicit_negative(self):
        sidecar_path(self.negative).unlink()
        self.negative.with_name(self.negative.name + ".meta.yaml").write_text(
            "match_id: game-a\n"
        )
        self.export()
        self.assertEqual(len(self.records()), 2)
        self.assertEqual(
            read_json(self.output / "yolo/stats.json")["total"]["negatives"], 0
        )

    def test_malformed_even_unreviewed_annotations_fail(self):
        self.edit(
            self.train,
            lambda d: d["frames"][0].update(
                review_status="in_progress", boxes=[{"bbox": [0, 0, 5, 5]}]
            ),
        )
        with self.assertRaisesRegex(ValueError, "fields"):
            self.export()

    def test_changed_source_content_or_geometry_fails(self):
        with self.train.open("ab") as handle:
            handle.write(b"modified")
        with self.assertRaisesRegex(ValueError, "content mismatch"):
            self.export()
        self.edit(
            self.train,
            lambda d: d["source"].update(sha256=file_hash(self.train), width=102),
        )
        with self.assertRaisesRegex(ValueError, "width mismatch"):
            self.export()

    def test_cross_split_match_in_excluded_test_is_rejected(self):
        self.fixture("test/held-out/clip.png", color=180, match="game-a")
        with self.assertRaisesRegex(ValueError, "match occurs"):
            self.export()

    def test_identical_media_in_different_splits_is_rejected(self):
        duplicate = self.fixture("test/held-out/copy.png", color=20)
        self.assertEqual(file_hash(self.train), file_hash(duplicate))
        with self.assertRaisesRegex(ValueError, "content occurs"):
            self.export()

    def test_identical_decoded_pixels_with_different_encodings_are_rejected(self):
        self.fixture("val/game-b/copy.bmp", color=20)
        with self.assertRaisesRegex(ValueError, "decoded frame"):
            self.export()

    def test_venue_separation_is_explicit_and_requires_known_venues(self):
        with self.assertRaisesRegex(ValueError, "venue occurs"):
            self.export(split_by_venue=True)
        self.edit(self.val, lambda d: d["source"].update(venue_id=None))
        with self.assertRaisesRegex(ValueError, "requires venue_id"):
            self.export(split_by_venue=True)
        self.edit(self.val, lambda d: d["source"].update(venue_id="other-hall"))
        self.export(split_by_venue=True)

    def test_folder_split_and_metadata_conflicts_are_rejected(self):
        self.edit(self.train, lambda d: d["source"].update(split="val"))
        with self.assertRaisesRegex(ValueError, "Conflicting split"):
            self.export()
        self.edit(self.train, lambda d: d["source"].update(split=None))
        with self.assertRaisesRegex(ValueError, "split mismatch"):
            self.export()

    def test_test_split_is_opt_in_and_audited_when_excluded(self):
        self.fixture("test/game-c/held.png", color=180)
        self.export()
        self.assertFalse((self.output / "coco/annotations/test.json").exists())
        audit = read_json(self.output / "yolo/split_audit.json")
        self.assertEqual(audit["splits_checked"], ["train", "val", "test"])
        self.assertEqual(
            next(s for s in audit["sources"] if s["split"] == "test")["reason"],
            "split_not_selected",
        )
        self.export(splits=["test"])
        self.assertEqual({r["split"] for r in self.records()}, {"test"})

    def test_artifact_tampering_missing_extra_and_symlinks_fail(self):
        self.export()
        root = (self.output / "yolo").resolve()
        label = next(root.glob("labels/train/*.txt"))
        original = label.read_bytes()
        for action in ("modify", "delete", "extra", "symlink"):
            with self.subTest(action=action):
                if action == "modify":
                    label.write_text("0 .5 .5 1 1\n")
                elif action == "delete":
                    label.unlink()
                elif action == "extra":
                    (root / "foreign.txt").write_text("oops")
                else:
                    (root / "foreign.txt").symlink_to(self.train)
                with self.assertRaises(ValueError):
                    verify_bundle(self.output)
                (root / "foreign.txt").unlink(missing_ok=True)
                label.write_bytes(original)
        verify_bundle(self.output)

    def test_semantic_mismatch_rejected_even_if_manifest_hashes_are_recomputed(self):
        self.export()
        root = (self.output / "coco").resolve()
        path = root / "annotations/train.json"
        coco = read_json(path)
        coco["annotations"][0]["bbox"][0] += 1
        write_json(path, coco)
        manifest = read_json(root / "manifest.json")
        manifest["artifacts"] = artifact_inventory(root)
        manifest["dataset_id"] = dataset_identity(manifest)
        manifest["export_id"] = export_identity(manifest)
        write_json(root / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "COCO annotations disagree"):
            verify_export(root)

    def test_manifest_path_traversal_and_identity_tampering_fail(self):
        self.export()
        path = self.output / "yolo/manifest.json"
        original = read_json(path)
        altered = {**original, "dataset_id": "0" * 64}
        write_json(path, altered)
        with self.assertRaisesRegex(ValueError, "Invalid dataset_id"):
            verify_export(path.parent)
        original["artifacts"]["../outside"] = "0" * 64
        write_json(path, original)
        with self.assertRaisesRegex(ValueError, "traversal"):
            verify_export(path.parent)

    def test_failed_encoding_preserves_previous_complete_pair(self):
        old = self.export()
        self.edit(
            self.train, lambda d: d["frames"][0]["boxes"][0].update(occluded=False)
        )
        with patch(
            "training.players.export.dataset.cv2.imencode", return_value=(False, None)
        ):
            with self.assertRaisesRegex(ValueError, "encode"):
                self.export()
        self.assertEqual(
            verify_bundle(self.output)["yolo"]["dataset_id"], old["dataset_id"]
        )
        self.assertFalse(list((self.output / ".generations").glob(".tmp-*")))

    def test_failed_atomic_switch_preserves_previous_complete_pair(self):
        old = self.export()
        self.edit(
            self.train, lambda d: d["frames"][0]["boxes"][0].update(occluded=False)
        )
        replace = os.replace

        def fail_switch(src, dst):
            if Path(dst) == self.output / "current":
                # While a new generation is ready, readers still see the full old pair.
                self.assertEqual(
                    verify_bundle(self.output)["yolo"]["dataset_id"], old["dataset_id"]
                )
                raise OSError("simulated publish failure")
            return replace(src, dst)

        with patch(
            "training.players.export.publication.os.replace", side_effect=fail_switch
        ):
            with self.assertRaisesRegex(OSError, "publish failure"):
                self.export()
        self.assertEqual(
            verify_bundle(self.output)["yolo"]["dataset_id"], old["dataset_id"]
        )
        new = self.export()
        self.assertNotEqual(new["dataset_id"], old["dataset_id"])

    def test_first_failure_does_not_publish_one_format(self):
        with patch(
            "training.players.export.dataset.coco_document",
            side_effect=ValueError("COCO failure"),
        ):
            with self.assertRaisesRegex(ValueError, "COCO failure"):
                self.export()
        self.assertFalse((self.output / "current").exists())
        self.assertFalse((self.output / "yolo").exists())
        self.assertFalse((self.output / "coco").exists())
        self.export()

    def test_source_changes_during_export_prevent_publication(self):
        old = self.export()

        def changed(root, checks, inventory):
            self.edit(
                self.train, lambda d: d["frames"][0].update(review_status="in_progress")
            )
            check_unchanged(root, checks, inventory)

        with patch(
            "training.players.export.dataset.check_unchanged", side_effect=changed
        ):
            with self.assertRaisesRegex(ValueError, "changed during export"):
                self.export()
        self.assertEqual(
            verify_bundle(self.output)["yolo"]["dataset_id"], old["dataset_id"]
        )

    def test_new_metadata_or_media_during_export_prevent_publication(self):
        for kind in ("metadata", "media"):
            with self.subTest(kind=kind):
                added = self.train.with_name(
                    "match.yaml" if kind == "metadata" else "new.png"
                )

                def changed(root, checks, inventory):
                    added.write_text("match_id: game-a\n")
                    check_unchanged(root, checks, inventory)

                with patch(
                    "training.players.export.dataset.check_unchanged",
                    side_effect=changed,
                ):
                    with self.assertRaisesRegex(ValueError, "changed during export"):
                        self.export()
                added.unlink()

    def test_foreign_output_is_untouched_and_nested_output_refused(self):
        self.output.mkdir()
        foreign = self.output / "valuable.txt"
        foreign.write_text("keep me")
        with self.assertRaisesRegex(ValueError, "foreign"):
            self.export()
        self.assertEqual(list(self.output.iterdir()), [foreign])
        self.assertEqual(foreign.read_text(), "keep me")
        with self.assertRaisesRegex(ValueError, "disjoint"):
            export_dataset(self.source, self.source / "exports")

    def test_concurrent_publication_is_refused(self):
        self.export()
        with (self.output / ".export.lock").open("r+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, "Another player export"):
                self.export()
        verify_bundle(self.output)

    def test_corrupt_existing_generation_is_not_silently_reused(self):
        self.export()
        (self.output / "coco/stats.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "integrity mismatch"):
            self.export()

    def test_cli_exports_and_verifies_without_source_data(self):
        command = [
            sys.executable,
            "-m",
            "training.players.export",
            "--output",
            str(self.output),
        ]
        result = subprocess.run(
            [*command, "--source", str(self.source)], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        dataset_id = json.loads(result.stdout)["dataset_id"]
        shutil.rmtree(self.source)
        result = subprocess.run([*command, "--verify"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["dataset_id"], dataset_id)

    def test_invalid_configurations_fail(self):
        for settings in (
            {"splits": []},
            {"splits": ["train", "train"]},
            {"resize_width": 64},
            {"resize_width": True, "resize_height": 64},
            {"split_by_venue": "false"},
            {"image_format": "tiff"},
            {"jpeg_quality": 0},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                ExportConfig(**settings)

    def test_annotation_venue_cannot_disagree_with_resolved_metadata(self):
        self.edit(self.val, lambda d: d["source"].update(venue_id=None))
        (self.val.parent / "match.yaml").write_text("venue_id: other-hall\n")
        with self.assertRaisesRegex(ValueError, "venue mismatch"):
            self.export(split_by_venue=True)

    def test_missing_empty_split_directory_and_format_alias_fail_verification(self):
        sidecar_path(self.val).unlink()
        self.val.unlink()
        self.export()
        empty = self.output / "yolo/images/val"
        empty.rmdir()
        with self.assertRaisesRegex(ValueError, "Missing split directory"):
            verify_bundle(self.output)
        empty.mkdir()
        (self.output / "yolo").unlink()
        with self.assertRaisesRegex(ValueError, "publication layout"):
            verify_bundle(self.output)

    def test_missing_media_and_duplicate_source_identity_fail(self):
        source_id = read_json(sidecar_path(self.train))["source"]["source_id"]
        self.edit(self.val, lambda d: d["source"].update(source_id=source_id))
        with self.assertRaisesRegex(ValueError, "Duplicate source_id"):
            self.export()
        self.edit(self.val, lambda d: d["source"].update(source_id="different-source"))
        self.val.unlink()
        with self.assertRaisesRegex(ValueError, "Missing media"):
            self.export()


if __name__ == "__main__":
    unittest.main()
