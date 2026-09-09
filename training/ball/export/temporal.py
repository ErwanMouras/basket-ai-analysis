"""Adapters for the V4/V5 SDK and the reference V3 and TensorFlow V4 loaders."""

from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .config import ExportConfig
from .files import link_or_copy, write_csv, write_image, write_json, write_jsonl
from .sources import continuous_segments


def sdk_heatmap(record: dict, radius: int, variance: float):
    """NBA's quantized Gaussian support (binary 0/255), with safe border rounding."""
    width, height = record["width"], record["height"]
    target = np.zeros((height, width), dtype=np.uint8)
    if record["position"] is None:
        return target
    cx = min(width - 1, max(0, round(record["position"][0])))
    cy = min(height - 1, max(0, round(record["position"][1])))
    x1, x2 = max(0, cx - radius), min(width, cx + radius + 1)
    y1, y2 = max(0, cy - radius), min(height, cy + radius + 1)
    yy, xx = np.ogrid[y1:y2, x1:x2]
    distance = ((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)
    gaussian = (np.exp(-distance / (2 * variance)) * 255).astype(np.uint8)
    target[y1:y2, x1:x2] = np.where(gaussian > 0, 255, 0)
    return target


def triplets(clip_records):
    for records in clip_records:
        for segment in continuous_segments(records):
            for index in range(len(segment) - 2):
                yield segment[index : index + 3]


def write_sdk(root: Path, clip_records: list[list[dict]], config: ExportConfig):
    counts = Counter()
    columns = [
        "path_prev",
        "path",
        "path_next",
        "gt_path_prev",
        "gt_path",
        "gt_path_next",
    ]
    columns += [
        f"{axis}_{time}" for time in ("prev", "current", "next") for axis in ("x", "y")
    ]
    columns += [f"visibility_{time}" for time in ("prev", "current", "next")]
    columns += [f"status_{time}" for time in ("prev", "current", "next")]
    contexts = {split: [] for split in config.splits}
    for records in clip_records:
        labels = []
        for record in records:
            image = Path(record["image"])
            target = Path("gts") / image.with_suffix(".png")
            write_image(
                root / target,
                sdk_heatmap(record, config.heatmap_radius, config.heatmap_variance),
            )
            labels.append(
                {
                    "file name": image.name,
                    "frame_idx": record["frame_index"],
                    "visibility": int(record["has_position"]),
                    "x-coordinate": record["position"][0]
                    if record["has_position"]
                    else "",
                    "y-coordinate": record["position"][1]
                    if record["has_position"]
                    else "",
                    "status": record["status"],
                }
            )
        if labels:
            write_csv(
                root / Path(records[0]["image"]).parent / "Label.csv",
                list(labels[0]),
                labels,
            )
    for frames in triplets(clip_records):
        row = {}
        for record, time, path_key in zip(
            frames, ("prev", "current", "next"), ("path_prev", "path", "path_next")
        ):
            row[path_key] = record["image"]
            row["gt_" + path_key] = (
                Path("gts") / Path(record["image"]).with_suffix(".png")
            ).as_posix()
            row[f"x_{time}"] = record["position"][0] if record["has_position"] else ""
            row[f"y_{time}"] = record["position"][1] if record["has_position"] else ""
            row[f"visibility_{time}"] = int(record["has_position"])
            row[f"status_{time}"] = record["status"]
        contexts[frames[0]["split"]].append(row)
        counts[frames[0]["split"]] += 1
    if not sum(counts.values()):
        raise ValueError(
            "SDK export requires at least three consecutive selected frames"
        )
    for split, rows in contexts.items():
        write_csv(root / f"labels_context_{split}.csv", columns, rows)
    write_json(
        root / "sdk.json",
        {
            "compatible_models": [
                "TrackNetV4 (V5 SDK)",
                "TrackNetV5",
                "TrackNetV5 + TOTNet ideas",
            ],
            "sequence_length": 3,
            "sequence_stride": 1,
            "visibility_meaning": (
                "target_position_available; status describes source occlusion"
            ),
            "target": {
                "encoding": "binary_uint8_0_255",
                "radius": config.heatmap_radius,
                "variance": config.heatmap_variance,
                "coordinate_space": "exported_image",
            },
            "csv": {split: f"labels_context_{split}.csv" for split in config.splits},
        },
    )
    (root / "sdk_dataset.py").write_bytes(
        Path(__file__).with_name("sdk_config.py").read_bytes()
    )
    return dict(counts)


def write_v3(root: Path, clip_records: list[list[dict]]):
    """Give each continuous segment its own rally; never stitch annotation gaps."""
    mappings = []
    counts = Counter()
    for match_number, records in enumerate(clip_records, 1):
        for segment_number, segment in enumerate(continuous_segments(records), 1):
            split = segment[0]["split"]
            match = Path("v3") / split / f"match{match_number}"
            rally = f"segment_{segment_number:04d}"
            rows = []
            for local_index, record in enumerate(segment):
                source = root / record["image"]
                target = root / match / "frame" / rally / f"{local_index}.png"
                if source.suffix == ".png":
                    link_or_copy(source, target)
                else:
                    image = cv2.imread(str(source))
                    if image is None:
                        raise ValueError(f"Cannot read exported frame: {source}")
                    write_image(target, image)
                x, y = record["position"] or (0, 0)
                if record["has_position"] and x == y == 0:
                    raise ValueError(
                        "The reference V3 loader treats (0, 0) as absent; "
                        "use a custom loader for this frame"
                    )
                rows.append(
                    {
                        "Frame": local_index,
                        "Visibility": int(record["has_position"]),
                        "X": x,
                        "Y": y,
                        "Status": record["status"],
                        "SourceFrame": record["frame_index"],
                    }
                )
            folder = "corrected_csv" if split == "test" else "csv"
            write_csv(root / match / folder / f"{rally}_ball.csv", list(rows[0]), rows)
            mappings.append(
                {
                    "match": match.as_posix(),
                    "rally": rally,
                    "clip_id": segment[0]["clip_id"],
                    "first_source_frame": segment[0]["frame_index"],
                    "frame_count": len(segment),
                }
            )
            counts[split] += 1
    write_json(root / "v3" / "segments.json", mappings)
    return dict(counts)


def write_v4(root: Path, clip_records: list[list[dict]], profile: dict):
    """Write bounded chunks in the reference TensorFlow loader's (N, C, H, W) layout."""
    width, height = profile["input_width"], profile["input_height"]
    radius, stride = profile.get("target_radius", 2.5), profile["sequence_stride"]
    counts, files = Counter(), Counter()
    samples = []
    yy, xx = np.ogrid[:height, :width]
    # Limit a chunk to roughly 32 MiB, even when the model input is large.
    chunk_size = max(1, (32 * 1024**2) // (12 * width * height * 4))
    for records in clip_records:
        for segment in continuous_segments(records):
            starts = range(0, len(segment) - 2, stride)
            for offset in range(0, len(starts), chunk_size):
                chunk = starts[offset : offset + chunk_size]
                split = segment[0]["split"]
                files[split] += 1
                folder = root / "v4" / "processed_data" / split
                folder.mkdir(parents=True, exist_ok=True)
                x_path = folder / f"x_data_{files[split]}.npy"
                y_path = folder / f"y_data_{files[split]}.npy"
                x_data = np.lib.format.open_memmap(
                    x_path,
                    mode="w+",
                    dtype="float32",
                    shape=(len(chunk), 9, height, width),
                )
                y_data = np.lib.format.open_memmap(
                    y_path,
                    mode="w+",
                    dtype="float32",
                    shape=(len(chunk), 3, height, width),
                )
                for sample_index, start in enumerate(chunk):
                    frames = segment[start : start + 3]
                    for time, record in enumerate(frames):
                        image = cv2.imread(str(root / record["image"]))
                        if image is None:
                            raise ValueError(
                                f"Cannot read exported frame: {record['image']}"
                            )
                        image = cv2.resize(
                            image, (width, height), interpolation=cv2.INTER_NEAREST
                        )
                        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        x_data[sample_index, time * 3 : (time + 1) * 3] = (
                            rgb.transpose(2, 0, 1) / 255.0
                        )
                        target = np.zeros((height, width), dtype=np.float32)
                        if record["has_position"]:
                            cx = min(
                                width - 1,
                                int(record["position"][0] * width / record["width"]),
                            )
                            cy = min(
                                height - 1,
                                int(record["position"][1] * height / record["height"]),
                            )
                            target = (
                                (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
                            ).astype(np.float32)
                        y_data[sample_index, time] = target
                    samples.append(
                        {
                            "file": x_path.relative_to(root).as_posix(),
                            "sample": sample_index,
                            "clip_id": frames[0]["clip_id"],
                            "source_frames": [
                                record["frame_index"] for record in frames
                            ],
                        }
                    )
                    counts[split] += 1
                x_data.flush()
                y_data.flush()
                del x_data, y_data
    if not counts:
        raise ValueError(
            "V4 preparation requires at least three consecutive selected frames"
        )
    write_json(
        root / "v4" / "preparation.json", {"profile": profile, "samples": dict(counts)}
    )
    write_jsonl(root / "v4" / "samples.jsonl", samples)
    return dict(counts)
