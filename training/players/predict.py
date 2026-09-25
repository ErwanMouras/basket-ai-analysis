"""Bounded-memory video inference. Predictions never become reviewed annotations."""

import argparse
import json
import math
import os
import time
from pathlib import Path

import cv2

from training.common.config import merge_settings, read_yaml
from training.common.files import write_json
from training.common.provenance import ROOT, file_hash
from training.players.evaluation.config import DEFAULTS as EVAL_DEFAULTS, validate_config
from training.players.evaluation.benchmark import runtime
from training.players.inference import predict_image
from training.players.models import Detector

DEFAULTS = {k: EVAL_DEFAULTS[k] for k in (
    "model", "variant", "weights", "source_class", "device", "resolution", "precision",
    "cpu_threads", "score_floor", "score_threshold", "max_detections")}
DEFAULTS.update(video=None, output="runs/players/video", max_frames=None, codec="mp4v",
                registry_uri=None, mlflow_uri="sqlite:///mlflow.db")


def load_config(path, *, video=None, checkpoint=None):
    config = merge_settings(DEFAULTS, read_yaml(Path(path)) if path else {})
    if video:
        config["video"] = str(video)
    if checkpoint:
        config["weights"] = str(checkpoint)
    if bool(config["weights"]) == bool(config["registry_uri"]):
        raise ValueError("Choose exactly one local checkpoint or models:/players-detector reference")
    if not isinstance(config["video"], str) or not config["video"]:
        raise ValueError("A source video is required")
    if not isinstance(config["output"], str) or not config["output"]:
        raise ValueError("An output directory is required")
    if config["codec"] not in ("mp4v",):
        raise ValueError("This video path supports the mp4v codec only")
    if config["max_frames"] is not None and (type(config["max_frames"]) is not int or config["max_frames"] < 1):
        raise ValueError("max_frames must be null or a positive integer")
    validation = {**EVAL_DEFAULTS, **{k: v for k, v in config.items() if k in EVAL_DEFAULTS}}
    validation["weights"] = config["weights"] or "registry"
    validate_config(validation)
    for key in ("video", "output", "weights"):
        if config[key] is not None:
            config[key] = str((ROOT / Path(config[key]).expanduser()).resolve())
    return config


def predict_video(config, *, detector=None, stop_after_frame=None):
    config = dict(config)
    video, output = Path(config["video"]), Path(config["output"])
    digest = file_hash(video)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    progress = {"step": "video", "status": "RUNNING", "run_id": None, "frames": 0,
                "elapsed_seconds": 0, "eta_seconds": None, "outputs": {}, "error": None}
    capture, writer = None, None
    try:
        if detector is None:
            if config["registry_uri"]:
                from training.players.registry import load_candidate
                detector, registered = load_candidate(config["registry_uri"], config["mlflow_uri"], device=config["device"])
                config.update({k: detector.provenance[k] for k in ("variant", "resolution", "source_class")})
                config["model"] = detector.family
                write_json(output / "registry.json", registered)
            else:
                detector = Detector(config["model"], config["variant"], config["weights"],
                                    source_class=config["source_class"], device=config["device"], resolution=config["resolution"])
        environment = runtime(config)
        write_json(output / "config.resolved.json", config)
        write_json(output / "model.json", detector.provenance)
        write_json(output / "environment.json", environment)
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise ValueError("Cannot open source video")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no valid frame rate")
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        limit = config["max_frames"]
        expected = min(total, limit) if total > 0 and limit else total
        progress.update(frames_total=expected if expected > 0 else None)
        write_json(output / "progress.json", progress)
        with (output / "predictions.partial.jsonl").open("w", encoding="utf-8") as handle:
            while limit is None or progress["frames"] < limit:
                ok, image = capture.read()
                if not ok:
                    break
                height, width = image.shape[:2]
                if writer is None:
                    if width % 2 or height % 2:
                        raise ValueError("mp4v requires even source dimensions; refusing silent crop")
                    dimensions = (width, height)
                    writer = cv2.VideoWriter(str(output / "annotated.partial.mp4"),
                                             cv2.VideoWriter_fourcc(*config["codec"]), fps, dimensions)
                    if not writer.isOpened():
                        raise ValueError("mp4v encoder is unavailable")
                elif dimensions != (width, height):
                    raise ValueError("Variable video dimensions are unsupported")
                detections = predict_image(detector, image, config)
                index = progress["frames"]
                handle.write(json.dumps({"schema_version": 1, "artifact_type": "players_video_predictions",
                    "coordinate_space": "source", "source_sha256": digest, "frame_index": index,
                    "timestamp_seconds": index / fps, "detections": detections}, allow_nan=False) + "\n")
                handle.flush()
                for detection in detections:
                    if detection["confidence"] < config["score_threshold"]:
                        continue
                    x1, y1, x2, y2 = [round(v) for v in detection["bbox"]]
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 2)
                    cv2.putText(image, f"player {detection['confidence']:.2f}", (x1, max(15, y1)),
                                cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 220, 0), 1)
                writer.write(image)
                progress["frames"] += 1
                elapsed = time.monotonic() - started
                progress.update(batch=progress["frames"], elapsed_seconds=elapsed,
                    eta_seconds=elapsed / progress["frames"] * max(0, expected - progress["frames"]) if expected > 0 else None)
                write_json(output / "progress.json", progress)
                if stop_after_frame is not None and progress["frames"] >= stop_after_frame:
                    raise KeyboardInterrupt("Requested video interruption")
        if not progress["frames"]:
            raise ValueError("Video contains no decoded frame")
        if expected > 0 and progress["frames"] != expected:
            raise ValueError("Video ended before its declared frame count")
        writer.release()
        writer = None
        if file_hash(video) != digest:
            raise ValueError("Source video changed during inference")
        os.replace(output / "predictions.partial.jsonl", output / "predictions.jsonl")
        os.replace(output / "annotated.partial.mp4", output / "annotated.mp4")
        elapsed = time.monotonic() - started
        result = {"source": str(video), "source_sha256": digest, "frames": progress["frames"],
                  "fps": fps, "width": width, "height": height, "audio": False,
                  "elapsed_seconds": elapsed, "end_to_end_fps": progress["frames"] / elapsed,
                  "timing_scope": "model load, decode, inference, draw, encode, JSON and progress writes; warmup absent",
                  "artifacts": {name: file_hash(output / name) for name in ("predictions.jsonl", "annotated.mp4")}}
        write_json(output / "result.json", result)
        progress.update(status="FINISHED", eta_seconds=0, elapsed_seconds=elapsed,
                        outputs={name: str(output / name) for name in result["artifacts"]})
        return result
    except BaseException as exc:
        progress.update(status="KILLED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                        elapsed_seconds=time.monotonic() - started,
                        error={"type": type(exc).__name__, "message": str(exc)})
        write_json(output / "error.json", progress["error"])
        raise
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        write_json(output / "progress.json", progress)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    print(json.dumps(predict_video(load_config(args.config, video=args.video, checkpoint=args.checkpoint)), indent=2))


if __name__ == "__main__":
    main()
