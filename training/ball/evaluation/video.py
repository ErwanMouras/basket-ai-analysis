"""Review exported frames at clip FPS, with explicit source frame numbers and gaps."""

import csv
from collections import defaultdict
from pathlib import Path

import cv2

from training.ball.export.files import write_csv
from training.ball.learning.data import checked_path


def draw_overlay(image, row, source, max_width, gap=0):
    h, w = image.shape[:2]
    scale = min(1, max_width / w)
    width, height = max(2, int(w * scale) // 2 * 2), max(2, int(h * scale) // 2 * 2)
    canvas = cv2.resize(image, (width, height))
    sx, sy = width / w, height / h
    annotation = row["source_annotation"]
    truth = row["target_export"]
    if truth is None and annotation["cx"] is not None:
        geo = source["geometry"]
        truth = [
            annotation["cx"] * geo["scale_x"] + geo["left"],
            annotation["cy"] * geo["scale_y"] + geo["top"],
        ]
    for point, color, marker in (
        (truth, (0, 255, 0), cv2.MARKER_CROSS),
        (row["position_export"], (255, 0, 255), cv2.MARKER_TILTED_CROSS),
    ):
        if point is not None:
            xy = (round(point[0] * sx), round(point[1] * sy))
            cv2.drawMarker(canvas, xy, color, marker, 18, 2)
            cv2.circle(canvas, xy, 12, color, 1)
    error = row["error_source_px"]
    lines = [
        f"frame {row['frame_index']} | {row['timestamp_seconds']:.3f}s | {row['status']}",
        f"{row['outcome']} | source error: {error:.1f}px"
        if error is not None
        else row["outcome"],
        f"GT green / prediction magenta | windows {row['window_count']} | offline raw",
    ]
    if gap:
        lines.append(f"GAP: {gap} source frames omitted")
    for index, line in enumerate(lines):
        position = (8, 22 + index * 22)
        cv2.putText(canvas, line, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(
            canvas, line, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
    return canvas


def write_videos(root, output, rows, sources, max_width):
    groups = defaultdict(list)
    for row in rows:
        groups[row["clip_id"]].append(row)
    folder = output / "videos"
    folder.mkdir()
    columns = [
        "video",
        "video_frame",
        "clip_id",
        "frame_index",
        "timestamp_seconds",
        "omitted_source_frames",
    ]
    with (output / "video_frames.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        timeline = csv.DictWriter(handle, fieldnames=columns)
        timeline.writeheader()
        for clip, frames in sorted(groups.items()):
            source = sources[clip]
            writer, previous = None, None
            # Export clip IDs are hashes; do not derive filenames from match names.
            path = folder / f"{clip}.mp4"
            try:
                for index, row in enumerate(frames):
                    image = cv2.imread(str(checked_path(Path(root), row["image"])))
                    if image is None:
                        raise ValueError(f"Cannot read review frame: {row['image']}")
                    gap = (
                        row["frame_index"] - previous - 1 if previous is not None else 0
                    )
                    canvas = draw_overlay(image, row, source, max_width, gap)
                    if writer is None:
                        writer = cv2.VideoWriter(
                            str(path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            source["video_metadata"]["fps"],
                            (canvas.shape[1], canvas.shape[0]),
                        )
                        if not writer.isOpened():
                            raise OSError(f"Cannot create review video: {path}")
                    writer.write(canvas)
                    timeline.writerow(
                        dict(
                            zip(
                                columns,
                                (
                                    path.relative_to(output).as_posix(),
                                    index,
                                    clip,
                                    row["frame_index"],
                                    row["timestamp_seconds"],
                                    gap,
                                ),
                            )
                        )
                    )
                    previous = row["frame_index"]
            finally:
                if writer is not None:
                    writer.release()


def write_review(output, rows):
    columns = [
        "clip_id",
        "match_id",
        "source_video",
        "source_sidecar",
        "frame_index",
        "status",
        "outcome",
        "confidence",
        "error_source_px",
        "error_export_px",
        "review_decision",
        "review_notes",
    ]
    candidates = [row for row in rows if row["outcome"] not in ("tp", "tn")]
    candidates.sort(
        key=lambda row: (
            row["outcome"],
            -(row["error_source_px"] or 0),
            row["clip_id"],
            row["frame_index"],
        )
    )
    write_csv(
        output / "review.csv",
        columns,
        ({key: row.get(key, "") for key in columns} for row in candidates),
    )
