"""Paired court replay on local clips; predictions are shared, never annotations.

Run from the repository root with the players environment. Real footage, full
records and comparison videos stay in the ignored output directory.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from training.common.provenance import file_hash
from training.court.calibration import CourtCalibrator
from training.court.config import settings
from training.court.drawing import draw_court
from training.court.model import CourtKeypoints
from training.court.geometry import metadata
from pipeline.contracts import validate_frame, validate_statistics


class SharedPredictions:
    def __init__(self, cached, config):
        self.cached = cached
        self.config = config
        self.model = None
        self.index = 0
        self.new_predictions = 0
        self.provenance = {"prediction_source": "shared_cached_or_local_checkpoint"}

    def predict(self, image):
        if self.index not in self.cached:
            if self.model is None:
                self.model = CourtKeypoints(self.config)
            self.cached[self.index] = self.model.predict(image).tolist()
            self.new_predictions += 1
        return np.array(self.cached[self.index], dtype=float)


def distribution(values):
    return {"count": len(values), "median": float(np.median(values)) if values else None,
            "p95": float(np.percentile(values, 95)) if values else None}


def summarize(rows):
    reports = [r["accepted_fit"] for r in rows if r["status"] == "fit"]
    def numbers(key):
        return [r[key] for r in reports if r.get(key) is not None]
    return {"frames": len(rows), "statuses": dict(Counter(r["status"] for r in rows)),
            "coverage": sum(r["image_to_court"] is not None for r in rows) / len(rows),
            "cuts": [r["frame_index"] for r in rows if r["scene_cut"]],
            "smoothed_fits": sum(r.get("smoothing_applied", False) for r in reports),
            "correction_px": distribution(numbers("output_correction_median_px")),
            "raw_reprojection_px": distribution(numbers("reprojection_rmse_px")),
            "smoothed_reprojection_px": distribution(numbers("stabilized_reprojection_rmse_px"))}


def export_viewer_runs(path, output):
    """Court-only previews: no copied player positions or stale statistics."""
    source = json.loads((path / "result.json").read_text())
    for variant in ("before", "after"):
        target = output / "viewer-runs" / f"{path.name}-{variant}"
        target.mkdir(parents=True, exist_ok=True)
        count = 0
        with (output / path.name / f"{variant}.jsonl").open() as records, (target / "observations.jsonl").open("w") as observations:
            for line in records:
                record = json.loads(line)
                row = {"schema_version": 1, "artifact_type": "video_observation", "run_id": target.name,
                       "frame_index": record["frame_index"], "timestamp_s": record["timestamp_seconds"],
                       "segment_id": record["segment_id"], "scene_cut": record["scene_cut"],
                       "included": True, "width": source["width"], "height": source["height"],
                       "coordinate_space": "source_pixels", "distance_unit": "m",
                       "persons": [], "ball": {"position_px": None, "confidence": None, "status": "disabled"}, "court": record,
                       "stages": {"court": {"status": "ok" if record["image_to_court"] is not None else "unavailable"}}}
                observations.write(json.dumps(validate_frame(row), allow_nan=False) + "\n")
                count += 1
        manifest = {"schema_version": 1, "status": "completed", "run_id": target.name, "frames": count,
                    "source": {"path": source["source"], "sha256": source["source_sha256"],
                               "nominal_fps": source["fps"], "width": source["width"], "height": source["height"]},
                    "court_coordinate_system": metadata(), "capabilities": {"court": "ok"},
                    "warnings": ["Court-only paired review; player positions and statistics not computed."]}
        (target / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
        stats = {"schema_version": 1, "metric": "observed_distance", "distance_unit": "m",
                 "players": [], "limitations": manifest["warnings"]}
        (target / "statistics.json").write_text(json.dumps(validate_statistics(stats)) + "\n")


def compare(path, output, device, max_frames):
    meta = json.loads((path / "court.json").read_text())
    result = json.loads((path / "result.json").read_text())
    source = Path(result["source"])
    if file_hash(source) != result["source_sha256"]:
        raise ValueError(f"Source hash mismatch: {source}")
    historical = [json.loads(line) for line in (path / "calibrations.jsonl").open()]
    predictions = {r["frame_index"]: r["keypoints"] for r in historical if r["keypoints"] is not None}
    out = output / path.name
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "shared-keypoints.json"
    if cache.exists():
        predictions.update({int(k): v for k, v in json.loads(cache.read_text()).items()})
    cfg = settings({**meta["config"], "device": device})
    shared = SharedPredictions(predictions, cfg)
    configs = [{**cfg, "smoothing_seconds": 0}, cfg]
    calibrators = [CourtCalibrator(c, fps=result["fps"], model=shared) for c in configs]
    rows = [[], []]
    cap = cv2.VideoCapture(str(source))
    writer = cv2.VideoWriter(str(out / "comparison.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             result["fps"], (1280, 360))
    if not cap.isOpened() or not writer.isOpened():
        raise RuntimeError("Cannot open source or comparison writer")
    count = min(len(historical), max_frames) if max_frames else len(historical)
    try:
        with ExitStack() as stack:
            detections = stack.enter_context((path / "predictions.jsonl").open())
            files = [stack.enter_context((out / f"{name}.jsonl").open("w")) for name in ("before", "after")]
            for index in range(count):
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError(f"Early video EOF: {index}/{count}")
                people = json.loads(next(detections))["detections"]
                shared.index = index
                panels = []
                for variant, calibrator in enumerate(calibrators):
                    cv2.setRNGSeed(index)
                    record = calibrator.update(image, people, frame_index=index,
                                               timestamp_seconds=historical[index]["timestamp_seconds"],
                                               force_cut=historical[index]["cut_reason"] == "explicit")
                    rows[variant].append(record)
                    files[variant].write(json.dumps(record, allow_nan=False) + "\n")
                    panel = image.copy()
                    draw_court(panel, record)  # identical rendering on both sides
                    panel = cv2.resize(panel, (640, 360))
                    cv2.rectangle(panel, (0, 0), (640, 30), (15, 20, 25), -1)
                    title = "BEFORE - direct recalibration" if variant == 0 else "AFTER - motion-compensated smoothing"
                    cv2.putText(panel, f"{title} | {index / result['fps']:.2f}s", (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, .45, (245, 245, 245), 1, cv2.LINE_AA)
                    panels.append(panel)
                writer.write(np.hstack(panels))
                if index % 300 == 0:
                    print(f"{path.name}: {index}/{count}", flush=True)
    finally:
        cap.release()
        writer.release()
        cache.write_text(json.dumps(shared.cached))
    before, after = map(summarize, rows)
    paired = [(a["accepted_fit"], b["accepted_fit"]) for a, b in zip(*rows)
              if a["status"] == b["status"] == "fit"
              and a["accepted_fit"].get("output_correction_median_px") is not None
              and b["accepted_fit"].get("output_correction_median_px") is not None]
    return {"clip": path.name, "source_sha256": result["source_sha256"],
            "historical_predictions": path.as_posix(), "new_shared_predictions": shared.new_predictions,
            "config_before": configs[0], "config_after": configs[1],
            "before": before, "after": after,
            "paired_fits": len(paired),
            "paired_correction_before_px": distribution([a["output_correction_median_px"] for a, _ in paired]),
            "paired_correction_after_px": distribution([b["output_correction_median_px"] for _, b in paired]),
            "availability_changes": sum((a["image_to_court"] is None) != (b["image_to_court"] is None) for a, b in zip(*rows))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("clips", nargs="+", type=Path)
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    cv2.setNumThreads(4)
    results = []
    for clip in args.clips:
        results.append(compare(clip, args.output, args.device, args.max_frames))
        export_viewer_runs(clip, args.output)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"scope": "paired replay, predicted landmarks, no ground truth",
                                          "results": results}, indent=2) + "\n")
        print(json.dumps({"clip": clip.name, "before": results[-1]["before"], "after": results[-1]["after"]}), flush=True)


if __name__ == "__main__":
    main()
