"""Transactional sidecar editing; edits always invalidate frame verification."""

import fcntl
import json
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from training.common.files import write_bytes
from training.players.contracts import validate_annotations

from .media import resolve_identity, source_record


def sidecar_path(media):
    path = Path(media)
    return path.with_name(path.name + ".playersann.json")


class Store:
    def __init__(self, reader, root, **identity):
        self.media = reader.path
        self.path = sidecar_path(self.media)
        if self.path.is_symlink():
            raise ValueError("Annotation sidecars cannot be symlinks")
        self.snapshot = self.path.read_bytes() if self.path.exists() else None
        previous = (
            validate_annotations(json.loads(self.snapshot))
            if self.snapshot is not None
            else None
        )
        supplied = dict(identity)
        if previous:
            for key in ("match_id", "venue_id", "split"):
                if supplied.get(key) is None:
                    supplied[key] = previous["source"][key]
        before = self._media_stat()
        if before != reader.signature:
            raise ValueError("Source changed since the decoder was opened")
        source = source_record(
            reader, root, **resolve_identity(self.media, root, **supplied)
        )
        if self._media_stat() != before:
            raise ValueError("Source changed while opening annotations")
        self.media_stat = before
        if previous and previous["source"] != source:
            raise ValueError(
                "Sidecar identity, content or geometry does not match this source"
            )
        self._document = previous or {
            "schema_version": 1,
            "artifact_type": "players_annotations",
            "source": source,
            "provenance": [{"kind": "manual", "reference": "players-annotator"}],
            "frames": [],
        }
        validate_annotations(self._document)

    def _media_stat(self):
        stat = self.media.stat()
        return stat.st_ino, stat.st_size, stat.st_mtime_ns

    @property
    def document(self):
        return deepcopy(self._document)

    def frame(self, index):
        if (
            type(index) is not int
            or not 0 <= index < self._document["source"]["frame_count"]
        ):
            raise ValueError("Frame index outside source")
        return deepcopy(
            next(
                (f for f in self._document["frames"] if f["frame_index"] == index),
                {"frame_index": index, "review_status": "unannotated", "boxes": []},
            )
        )

    def _commit(self, document):
        validated = validate_annotations(document)
        content = (
            json.dumps(validated, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        ).encode()
        lock_path = self.path.with_name(self.path.name + ".lock")
        if lock_path.is_symlink():
            raise ValueError("Annotation lock cannot be a symlink")
        # Keep the lock inode stable: deleting it would allow simultaneous writers.
        with lock_path.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("Another process is saving these annotations") from exc
            if self.path.is_symlink() or self._media_stat() != self.media_stat:
                raise ValueError("Source or sidecar changed; reopen before editing")
            current = self.path.read_bytes() if self.path.exists() else None
            if current != self.snapshot:
                raise ValueError(
                    "Annotations changed in another process; reopen before editing"
                )
            write_bytes(self.path, content)
        self._document, self.snapshot = validated, content

    def _set_frame(self, frame):
        document = self.document
        document["frames"] = sorted(
            [f for f in document["frames"] if f["frame_index"] != frame["frame_index"]]
            + [frame],
            key=lambda f: f["frame_index"],
        )
        manual = {"kind": "manual", "reference": "players-annotator"}
        if manual not in document["provenance"]:
            document["provenance"].append(manual)
        self._commit(document)

    def add_box(self, index, bbox):
        frame = self.frame(index)
        box_id = uuid4().hex
        frame["boxes"].append(
            {
                "object_id": box_id,
                "class_id": 0,
                "bbox": list(bbox),
                "occluded": None,
                "truncated": None,
            }
        )
        frame["review_status"] = "in_progress"
        self._set_frame(frame)
        return box_id

    def edit_box(self, index, box_id, **changes):
        if set(changes) - {"bbox", "occluded", "truncated"}:
            raise ValueError("Only geometry and visibility can be edited")
        frame = self.frame(index)
        box = next((b for b in frame["boxes"] if b["object_id"] == box_id), None)
        if box is None:
            raise ValueError("Unknown box")
        box.update(deepcopy(changes))
        frame["review_status"] = "in_progress"
        self._set_frame(frame)

    def delete_box(self, index, box_id):
        frame = self.frame(index)
        if not any(b["object_id"] == box_id for b in frame["boxes"]):
            raise ValueError("Unknown box")
        frame["boxes"] = [b for b in frame["boxes"] if b["object_id"] != box_id]
        frame["review_status"] = "in_progress"
        self._set_frame(frame)

    def verify(self, index):
        frame = self.frame(index)
        frame["review_status"] = "verified"
        self._set_frame(frame)

    def reopen(self, index):
        frame = self.frame(index)
        frame["review_status"] = "in_progress"
        self._set_frame(frame)

    def add_proposals(self, frames, provenance):
        """Never replace prior proposals, manual edits or verified empty frames."""
        candidate = self.document
        occupied = {
            f["frame_index"]
            for f in candidate["frames"]
            if f["review_status"] != "unannotated"
        }
        additions = []
        seen = set()
        for frame in frames:
            if frame["review_status"] != "proposed" or frame["frame_index"] in seen:
                raise ValueError("Expected distinct proposed frames")
            seen.add(frame["frame_index"])
            if frame["frame_index"] not in occupied:
                additions.append(deepcopy(frame))
        if not additions:
            return 0
        replaced = {f["frame_index"] for f in additions}
        candidate["frames"] = sorted(
            [f for f in candidate["frames"] if f["frame_index"] not in replaced]
            + additions,
            key=lambda f: f["frame_index"],
        )
        for entry in provenance:
            if entry not in candidate["provenance"]:
                candidate["provenance"].append(deepcopy(entry))
        self._commit(candidate)
        return len(additions)
