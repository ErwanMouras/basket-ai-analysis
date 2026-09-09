"""Lightweight indexes that preserve export splits, clips and frame adjacency."""

import csv
import json
from pathlib import Path

import numpy as np


def rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def frame_index(root, split):
    if split not in ("train", "val"):
        raise ValueError("Training readers accept only train and val")
    result = {}
    with (root / "frames.jsonl").open() as handle:
        for line in handle:
            frame = json.loads(line)
            if frame["split"] == split:
                result[frame["image"]] = frame
    if not result:
        raise ValueError(f"Empty {split} split")
    return result


def consecutive(frames):
    if len({(frame["split"], frame["clip_id"]) for frame in frames}) != 1:
        raise ValueError("A sequence crosses a clip or split")
    indices = [frame["frame_index"] for frame in frames]
    if indices != list(range(indices[0], indices[0] + len(indices))):
        raise ValueError("A sequence contains a frame gap")


def checked_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing file or path outside export: {relative}")
    return path


def sdk_samples(root, split, stride):
    records = frame_index(root, split)
    samples, offsets = [], {}
    for row in rows(root / f"labels_context_{split}.csv"):
        paths = [row[key] for key in ("path_prev", "path", "path_next")]
        if any(path not in records for path in paths):
            raise ValueError("SDK row references another split or an unknown image")
        frames = [records[path] for path in paths]
        consecutive(frames)
        for time, frame in zip(("prev", "current", "next"), frames):
            if (
                int(row[f"visibility_{time}"]) != int(frame["has_position"])
                or row[f"status_{time}"] != frame["status"]
            ):
                raise ValueError("SDK annotation policy differs from frames.jsonl")
            if (
                frame["has_position"]
                and [float(row[f"x_{time}"]), float(row[f"y_{time}"])]
                != frame["position"]
            ):
                raise ValueError("SDK coordinates differ from frames.jsonl")
        clip = frames[0]["clip_id"]
        first = frames[0]["frame_index"]
        previous, offset = offsets.get(clip, (first - 1, -1))
        offset = offset + 1 if first == previous + 1 else 0
        offsets[clip] = (first, offset)
        if offset % stride == 0:
            samples.append((row, frames))
    if not samples:
        raise ValueError(f"No consecutive SDK sequences in {split}")
    return samples


def v3_samples(root, split, profile):
    records = frame_index(root, split)
    by_source = {
        (frame["clip_id"], frame["frame_index"]): frame for frame in records.values()
    }
    samples = []
    length, stride = profile["sequence_length"], profile["sequence_stride"]
    for segment in json.loads((root / "v3/segments.json").read_text()):
        if Path(segment["match"]).parts[1] != split:
            continue
        folder = root / segment["match"]
        labels = rows(folder / "csv" / f"{segment['rally']}_ball.csv")
        frames = []
        for index, row in enumerate(labels):
            if int(row["Frame"]) != index:
                raise ValueError("V3 local frames must be consecutive from zero")
            frame = by_source[(segment["clip_id"], int(row["SourceFrame"]))]
            if int(row["Visibility"]) != int(frame["has_position"]):
                raise ValueError("V3 visibility differs from exported target policy")
            point = frame["position"] or [0, 0]
            if [float(row["X"]), float(row["Y"])] != point or row["Status"] != frame[
                "status"
            ]:
                raise ValueError("V3 label differs from frames.jsonl")
            if frame["has_position"]:
                x = int(point[0] * profile["input_width"] / frame["width"])
                y = int(point[1] * profile["input_height"] / frame["height"])
                if x == y == 0:
                    raise ValueError(
                        "V3 known position becomes the (0, 0) absence sentinel after resize"
                    )
            image = folder / "frame" / segment["rally"] / f"{index}.png"
            checked_path(root, image.relative_to(root))
            frames.append((image, frame))
        if frames:
            consecutive([frame for _, frame in frames])
        for start in range(0, len(frames) - length + 1, stride):
            samples.append(frames[start : start + length])
    if not samples:
        raise ValueError(f"No V3 segments long enough for {length} frames in {split}")
    return samples


class V4Arrays:
    """Memory-map one array pair per batch; never concatenate an entire split."""

    def __init__(self, config, split):
        self.root = Path(config["dataset"])
        self.split = split
        self.batch_size = config["batch_size"]
        self.seed = config["seed"]
        self.profile = config["geometry"]
        prepared = json.loads((self.root / "v4/preparation.json").read_text())[
            "profile"
        ]
        if prepared != self.profile:
            raise ValueError("V4 preparation profile changed; regenerate the v4 export")
        records = frame_index(self.root, split)
        known = {(row["clip_id"], row["frame_index"]): row for row in records.values()}
        mappings = {}
        with (self.root / "v4/samples.jsonl").open() as handle:
            for line in handle:
                sample = json.loads(line)
                if Path(sample["file"]).parts[2] != split:
                    continue
                frames = [
                    known[(sample["clip_id"], index)]
                    for index in sample["source_frames"]
                ]
                consecutive(frames)
                mappings[(sample["file"], sample["sample"])] = frames
        self.files = []
        total = 0
        for x_path in sorted(
            (self.root / "v4/processed_data" / split).glob("x_data_*.npy")
        ):
            y_path = x_path.with_name(x_path.name.replace("x_data_", "y_data_"))
            x, y = np.load(x_path, mmap_mode="r"), np.load(y_path, mmap_mode="r")
            h, w = self.profile["input_height"], self.profile["input_width"]
            if (
                x.dtype != np.float32
                or y.dtype != np.float32
                or x.shape[1:] != (9, h, w)
                or y.shape != (len(x), 3, h, w)
            ):
                raise ValueError(f"Invalid V4 array shape/dtype: {x_path}")
            for index in range(len(x)):
                if (x_path.relative_to(self.root).as_posix(), index) not in mappings:
                    raise ValueError("V4 array has no corresponding sample metadata")
            self.files.append((x_path, y_path, len(x)))
            total += len(x)
        if total == 0 or total != len(mappings):
            raise ValueError("Empty or inconsistent V4 split")

    def batches(self, epoch):
        rng = np.random.default_rng(self.seed + epoch)
        order = np.arange(len(self.files))
        if self.split == "train":
            rng.shuffle(order)
        for file_index in order:
            x_path, y_path, count = self.files[file_index]
            x, y = np.load(x_path, mmap_mode="r"), np.load(y_path, mmap_mode="r")
            indices = np.arange(count)
            if self.split == "train":
                rng.shuffle(indices)
            for start in range(0, count, self.batch_size):
                selection = indices[start : start + self.batch_size]
                yield np.array(x[selection]), np.array(y[selection])
