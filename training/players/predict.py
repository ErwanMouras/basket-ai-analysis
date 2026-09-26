"""Bounded-memory video inference. Predictions never become reviewed annotations."""

import argparse
import json
import math
import os
import time
from contextlib import ExitStack
from pathlib import Path
from uuid import uuid4

import cv2

from training.common.config import merge_settings, read_yaml
from training.common.files import write_json
from training.common.provenance import ROOT, file_hash
from training.players.evaluation.benchmark import runtime
from training.players.evaluation.config import DEFAULTS as EVAL_DEFAULTS
from training.players.evaluation.config import validate_config
from training.players.inference import predict_image
from training.players.models import Detector
from training.players.pose import DEFAULTS as POSE_DEFAULTS
from training.players.pose import PlayerPose, draw_pose
from training.players.pose import settings as pose_settings
from training.players.tracking import DEFAULTS as TRACKING_DEFAULTS
from training.players.tracking import PlayerTracker
from training.players.tracking import settings as tracking_settings

DEFAULTS = {k: EVAL_DEFAULTS[k] for k in (
    "model", "variant", "weights", "source_class", "device", "resolution", "precision",
    "cpu_threads", "score_floor", "score_threshold", "max_detections")}
DEFAULTS.update(video=None, output="runs/players/video", max_frames=None, codec="mp4v",
                registry_uri=None, mlflow_uri="sqlite:///mlflow.db", tracking=TRACKING_DEFAULTS, pose=POSE_DEFAULTS)


def load_config(path, *, video=None, checkpoint=None, tracker=None, output=None, max_frames=None,
                pose=None, pose_weights=None, pose_device=None):
    config = merge_settings(DEFAULTS, read_yaml(Path(path)) if path else {})
    if video:
        config["video"] = str(video)
    if checkpoint:
        config["weights"] = str(checkpoint)
    if output is not None:
        config["output"] = str(output)
    if max_frames is not None:
        config["max_frames"] = max_frames
    if tracker is not None:
        config["tracking"] = {**config["tracking"], "enabled": tracker != "none"}
        if tracker != "none":
            config["tracking"]["tracker"] = tracker
    config["pose"] = dict(config["pose"])
    if pose is not None:
        config["pose"]["enabled"] = pose
    if pose_weights is not None:
        config["pose"]["weights"] = str(pose_weights)
    if pose_device is not None:
        config["pose"]["device"] = pose_device
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
    config["tracking"] = tracking_settings(config["tracking"], score_floor=config["score_floor"])
    config["pose"] = pose_settings(config["pose"], score_floor=config["score_floor"])
    for key in ("video", "output", "weights"):
        if config[key] is not None:
            config[key] = str((ROOT / Path(config[key]).expanduser()).resolve())
    return config


def predict_video(config, *, detector=None, pose_estimator=None, stop_after_frame=None):
    config = dict(config)
    config["tracking"] = tracking_settings(config.get("tracking"), score_floor=config["score_floor"])
    config["pose"] = pose_settings(config.get("pose"), score_floor=config["score_floor"])
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
        tracker = PlayerTracker(config["tracking"], fps=fps) if config["tracking"]["enabled"] else None
        tracking_run_id = uuid4().hex if tracker is not None else None
        tracked_observations = 0
        pose_model = (pose_estimator if pose_estimator is not None else PlayerPose(config["pose"])) if config["pose"]["enabled"] else None
        pose_run_id = uuid4().hex if pose_model is not None else None
        posed_observations = valid_keypoints = 0
        if pose_model is not None:
            write_json(output / "pose.json", {
                **pose_model.provenance, "pose_run_id": pose_run_id,
                "tracking_run_id": tracking_run_id, "source_sha256": digest,
                "detector": detector.provenance,
                "adapter_sha256": file_hash(Path(__file__).with_name("pose.py")),
            })
        if tracker is not None:
            write_json(output / "tracking.json", {
                **tracker.provenance, "tracking_run_id": tracking_run_id,
                "source_sha256": digest, "detector": detector.provenance,
                "adapter_sha256": file_hash(Path(__file__).with_name("tracking.py")),
            })
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        limit = config["max_frames"]
        expected = min(total, limit) if total > 0 and limit else total
        progress.update(frames_total=expected if expected > 0 else None)
        write_json(output / "progress.json", progress)
        with ExitStack() as stack:
            handle = stack.enter_context((output / "predictions.partial.jsonl").open("w", encoding="utf-8"))
            tracks_handle = (stack.enter_context((output / "tracks.partial.jsonl").open("w", encoding="utf-8"))
                             if tracker is not None else None)
            poses_handle = (stack.enter_context((output / "poses.partial.jsonl").open("w", encoding="utf-8"))
                            if pose_model is not None else None)
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
                displayed = detections
                if tracker is not None:
                    displayed = tracker.update(detections, image, frame_index=index)
                    tracks_handle.write(json.dumps({"schema_version": 1,
                        "artifact_type": "players_video_tracks", "coordinate_space": "source",
                        "source_sha256": digest, "tracking_run_id": tracking_run_id,
                        "segment_id": tracker.segment_id, "frame_index": index,
                        "timestamp_seconds": index / fps, "detections": displayed}, allow_nan=False) + "\n")
                    tracks_handle.flush()
                    tracked_observations += sum(d["track_id"] is not None for d in displayed)
                if pose_model is not None:
                    displayed = pose_model.predict(image, displayed)
                    poses_handle.write(json.dumps({"schema_version": 1,
                        "artifact_type": "players_video_poses", "coordinate_space": "source",
                        "source_sha256": digest, "pose_run_id": pose_run_id,
                        "tracking_run_id": tracking_run_id,
                        "segment_id": tracker.segment_id if tracker is not None else 0,
                        "frame_index": index, "timestamp_seconds": index / fps,
                        "detections": displayed}, allow_nan=False) + "\n")
                    poses_handle.flush()
                    posed_observations += sum(d["pose"] is not None for d in displayed)
                    valid_keypoints += sum(d["pose"]["valid_keypoints"] for d in displayed if d["pose"] is not None)
                for detection in displayed:
                    if detection["confidence"] < config["score_threshold"]:
                        continue
                    x1, y1, x2, y2 = [round(v) for v in detection["bbox"]]
                    track_id = detection.get("track_id")
                    color = ((60 + track_id * 67 % 196, 60 + track_id * 131 % 196,
                              60 + track_id * 43 % 196) if track_id is not None else (0, 220, 0))
                    label = f"#{track_id}" if track_id is not None else ("player ?" if tracker else "player")
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(image, f"{label} {detection['confidence']:.2f}", (x1, max(15, y1)),
                                cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
                    if pose_model is not None:
                        draw_pose(image, detection["pose"], color)
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
        artifacts = ["predictions.jsonl", "annotated.mp4"]
        if tracker is not None:
            os.replace(output / "tracks.partial.jsonl", output / "tracks.jsonl")
            artifacts.extend(["tracks.jsonl", "tracking.json"])
        if pose_model is not None:
            os.replace(output / "poses.partial.jsonl", output / "poses.jsonl")
            artifacts.extend(["poses.jsonl", "pose.json"])
        elapsed = time.monotonic() - started
        result = {"source": str(video), "source_sha256": digest, "frames": progress["frames"],
                  "fps": fps, "width": width, "height": height, "audio": False,
                  "elapsed_seconds": elapsed, "end_to_end_fps": progress["frames"] / elapsed,
                  "timing_scope": "model load, decode, detection, optional tracking and pose, draw, encode, JSON and progress writes; warmup absent",
                  "tracking": {"enabled": tracker is not None, "tracking_run_id": tracking_run_id,
                               "tracked_observations": tracked_observations},
                  "pose": {"enabled": pose_model is not None, "pose_run_id": pose_run_id,
                           "posed_observations": posed_observations, "valid_keypoints": valid_keypoints},
                  "artifacts": {name: file_hash(output / name) for name in artifacts}}
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
    parser.add_argument("--tracker", choices=("botsort", "bytetrack", "none"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--pose", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pose-weights", type=Path)
    parser.add_argument("--pose-device")
    args = parser.parse_args()
    print(json.dumps(predict_video(load_config(args.config, video=args.video, checkpoint=args.checkpoint,
        tracker=args.tracker, output=args.output, max_frames=args.max_frames,
        pose=args.pose, pose_weights=args.pose_weights, pose_device=args.pose_device)), indent=2))


if __name__ == "__main__":
    main()
