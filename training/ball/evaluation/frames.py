"""Merge overlapping windows, restore coordinates and score each frame once."""

from collections import Counter, defaultdict

import numpy as np

from training.ball.learning.metrics import center


def merge_windows(samples, predictions, threshold):
    """Average heatmaps before decoding; release a frame after its last window.

    Samples are ordered by clip and start frame by the inference reader. Only
    overlapping heatmaps remain in memory, rather than the entire validation set.
    """
    last_use = {
        frame["image"]: index
        for index, (_, frames) in enumerate(samples)
        for frame in frames
    }
    pending = {}
    for index, ((_, expected), (frames, heatmaps)) in enumerate(
        zip(samples, predictions, strict=True)
    ):
        if [f["image"] for f in frames] != [f["image"] for f in expected]:
            raise ValueError("Predictions do not follow the declared window order")
        if len(heatmaps) != len(frames) or not np.isfinite(heatmaps).all():
            raise ValueError("Invalid window heatmaps")
        for frame, heatmap in zip(frames, heatmaps):
            key = frame["image"]
            future = frames[-1]["frame_index"] - frame["frame_index"]
            if key not in pending:
                pending[key] = [heatmap.astype(np.float32, copy=True), 1, future]
            else:
                pending[key][0] += heatmap
                pending[key][1] += 1
                pending[key][2] = max(pending[key][2], future)
            if last_use[key] == index:
                total, count, future = pending.pop(key)
                total /= count
                point = center(total > threshold)
                yield (
                    key,
                    {
                        "position_input": point.tolist() if point is not None else None,
                        "confidence": float(total.max()),
                        "window_count": count,
                        "future_context_frames": future,
                    },
                )


def frame_result(frame, source, prediction, geometry, config):
    prediction = prediction or {
        "position_input": None,
        "confidence": None,
        "window_count": 0,
        "future_context_frames": None,
    }
    point = prediction["position_input"]
    exported, original = None, None
    if point is not None:
        exported = [
            point[0] * frame["width"] / geometry["input_width"],
            point[1] * frame["height"] / geometry["input_height"],
        ]
        transform = source["geometry"]
        original = [
            (exported[0] - transform["left"]) / transform["scale_x"],
            (exported[1] - transform["top"]) / transform["scale_y"],
        ]
    annotation = frame["source_annotation"]
    truth_source = (
        [annotation["cx"], annotation["cy"]] if frame["has_position"] else None
    )
    error_export = error_source = None
    if exported is not None and truth_source is not None:
        error_export = float(np.linalg.norm(np.array(exported) - frame["position"]))
        error_source = float(np.linalg.norm(np.array(original) - truth_source))
    error = error_source if config["tolerance_space"] == "source" else error_export
    if not prediction["window_count"]:
        outcome = "uncovered"
    elif frame["position_excluded"] or (
        frame["status"] == "unknown_position" and config["unknown_position"] == "ignore"
    ):
        outcome = "ignored"
    elif truth_source is None:
        outcome = "tn" if point is None else "fp"
    elif point is None:
        outcome = "fn"
    else:
        outcome = "tp" if error <= config["tolerance_px"] else "displaced"
    meta = source["video_metadata"]
    in_source = (
        0 <= original[0] < meta["width"] and 0 <= original[1] < meta["height"]
        if original is not None
        else None
    )
    return {
        **{
            key: frame[key]
            for key in (
                "clip_id",
                "match_id",
                "split",
                "frame_index",
                "timestamp_seconds",
                "image",
                "status",
                "occluded",
                "position_excluded",
                "has_position",
            )
        },
        "source_video": source["video"],
        "source_sidecar": source["sidecar"],
        "source_annotation": annotation,
        "target_export": frame["position"],
        "target_source": truth_source,
        **prediction,
        "position_export": exported,
        "position_source": original,
        "prediction_in_source": in_source,
        "error_export_px": error_export,
        "error_source_px": error_source,
        "outcome": outcome,
    }


def summarize(rows):
    counts = Counter(row["outcome"] for row in rows)
    tp, fp, fn = (
        counts["tp"],
        counts["fp"] + counts["displaced"],
        counts["fn"] + counts["displaced"],
    )
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    covered = len(rows) - counts["uncovered"]
    known = sum(row["has_position"] for row in rows)
    known_covered = sum(row["has_position"] and row["window_count"] > 0 for row in rows)
    result = {
        "frames": len(rows),
        "covered_frames": covered,
        "uncovered_frames": counts["uncovered"],
        "ignored_frames": counts["ignored"],
        "scored_frames": covered - counts["ignored"],
        "coverage": covered / len(rows) if rows else None,
        "known_frames": known,
        "known_covered_frames": known_covered,
        "known_coverage": known_covered / known if known else None,
        "negative_frames": counts["fp"] + counts["tn"],
        "tp": tp,
        "tn": counts["tn"],
        "fp": fp,
        "fn": fn,
        "displaced": counts["displaced"],
        "precision": precision,
        "recall": recall,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }
    for space in ("export", "source"):
        errors = [
            row[f"error_{space}_px"]
            for row in rows
            if row["outcome"] in ("tp", "displaced")
        ]
        result["localized_pairs"] = len(errors)
        for name, value in (
            ("mean", np.mean(errors) if errors else None),
            ("median", np.median(errors) if errors else None),
            ("p95", np.percentile(errors, 95) if errors else None),
        ):
            result[f"error_{space}_{name}_px"] = (
                float(value) if value is not None else None
            )
    return result


def report(rows):
    result = {"overall": summarize(rows)}
    for key in ("clip_id", "match_id", "status"):
        groups = defaultdict(list)
        if key == "status":
            for status in ("visible", "occluded_with_position", "unknown_position"):
                groups[status] = []
        for row in rows:
            groups[row[key]].append(row)
        result[f"by_{key}"] = {
            key: summarize(group) for key, group in sorted(groups.items())
        }
    return result
