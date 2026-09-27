"""Streaming analysis graph and atomic artifact publication."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os
import resource
import time
from uuid import uuid4

from pipeline.adapters import Ball, People, RoleTracker
from pipeline.config import included
from pipeline.contracts import validate_frame, validate_statistics
from pipeline.distance import Distance
from pipeline.runtime import Runtime
from pipeline.video import Video
from training.common.files import write_json
from training.common.provenance import file_hash
from training.court.calibration import CourtCalibrator
from training.court.geometry import metadata as court_metadata
from training.court.model import CourtKeypoints
from training.jersey.reader import ParseqReader
from training.jersey.roster import Roster
from training.jersey.temporal import JerseyRecognizer
from training.players.pose import PlayerPose


class Models:
    def __init__(self, config, fps, roster, runtime):
        import cv2
        import torch
        torch.set_num_threads(config["cpu_threads"])
        torch.manual_seed(0)
        cv2.setNumThreads(config["cpu_threads"])
        cv2.setRNGSeed(0)
        self.capabilities = {}
        self.provenance = {}
        self.errors = {}
        self.clock = 0.0
        self.ball = None
        for device in {config[name]["device"] for name in ("people", "court", "pose", "jersey")}:
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.init()
                torch.cuda.reset_peak_memory_stats(int(device.split(":")[1]))
        # Check all required files before constructing expensive models.
        for name in config["required"]:
            section = "people" if name in ("players", "referees") else name
            c = config[section]
            if not c.get("enabled", True) or not c.get("weights") or not Path(c["weights"]).is_file():
                raise ValueError(f"Required capability {name} has no enabled local checkpoint")
        self.people = People(config["people"])
        self.provenance["people"] = self.people.provenance
        self.capabilities["players"] = "ok" if "player" in self.people.roles.values() else "unavailable"
        self.capabilities["referees"] = "ok" if "referee" in self.people.roles.values() else "unavailable"
        self.tracker = RoleTracker(config["tracking"], fps=fps, roles=self.people.roles.values())
        self.provenance["tracking"] = self.tracker.provenance
        self.court_points = self.court = None

        def initialize(name, factory):
            if not config[name]["enabled"]:
                self.capabilities[name] = "disabled"
                return None
            try:
                value = factory()
            except Exception as exc:
                self.capabilities[name] = "error"
                self.errors[name] = f"{type(exc).__name__}: {exc}"
                if name in config["required"]:
                    raise
                return None
            self.capabilities[name] = "ok"
            self.provenance[name] = value.provenance
            return value

        try:
            self.court_points = initialize("court", lambda: CourtKeypoints(config["court"]))
            if self.court_points:
                points = self.court_points

                class GuardedPoints:
                    provenance = points.provenance

                    def predict(self, image):
                        return runtime.call("court", config["court"]["device"], points.predict, image).result()

                self.court = CourtCalibrator(config["court"], fps=fps, model=GuardedPoints())
            self.pose = initialize("pose", lambda: PlayerPose(config["pose"]))
            self.jersey = initialize("jersey", lambda: JerseyRecognizer(
                config["jersey"], roster=roster,
                reader=ParseqReader(config["jersey"], manage_threads=False), clock=lambda: self.clock))
            self.ball = initialize("ball", lambda: Ball(config["ball"], config["cpu_threads"]))
            for name in config["required"]:
                if self.capabilities[name] != "ok":
                    raise ValueError(f"Required capability unavailable: {name}")
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.ball:
            self.ball.close()

    def resources(self):
        import torch
        result = {"main_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  "cuda_scope": "PyTorch allocations only; excludes ONNX Runtime and CUDA contexts"}
        if torch.cuda.is_available():
            result["main_peak_torch_cuda_allocated_mb"] = {
                str(i): torch.cuda.max_memory_allocated(i) / 1024**2 for i in range(torch.cuda.device_count())}
        if self.ball:
            result["ball_worker"] = self.ball.resources()
        return result


def _line(handle, row):
    handle.write(json.dumps(row, allow_nan=False, ensure_ascii=False) + "\n")


def _next(iterator):
    return next(iterator, None)


def analyze(video_path, output, config, roster_path=None, *, models_factory=Models):
    """Run one video. Injected models are useful for deterministic integration tests."""
    # An explicitly invalid roster fails before output creation or any model load.
    roster = Roster.load(roster_path) if roster_path is not None else None
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    run_id = uuid4().hex
    started = time.perf_counter()
    runtime = Runtime(config)
    decoder = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-decode")
    video = models = None
    manifest = {"schema_version": 1, "run_id": run_id, "status": "running",
                "config": config, "roster_sha256": roster.sha256 if roster else None,
                "warnings": [], "artifacts": {}}
    root = Path(__file__).resolve().parents[1]
    manifest["code_sha256"] = {str(path.relative_to(root)): file_hash(path)
                               for folder in ("pipeline", "training/players", "training/court", "training/jersey", "training/ball", "training/common")
                               for path in sorted((root / folder).rglob("*.py")) if "tests" not in path.parts}
    write_json(output / "run.json", manifest)
    if roster:
        write_json(output / "roster.json", roster.payload)
    frames_done = 0
    try:
        video = Video(video_path, config["timestamp_policy"])
        models = models_factory(config, video.fps, roster, runtime)
        loaded = time.perf_counter()
        manifest.update(source=video.source, capabilities=models.capabilities, models=models.provenance,
                        errors=models.errors, court_coordinate_system=court_metadata(),
                        ocr_scheduling="source_time_reproducible",
                        tracking_time_basis="nominal_fps_buffer_and_source_timestamps_for_metrics")
        write_json(output / "run.json", manifest)
        metric = Distance(config["distance"], run_id)
        pending = deque()
        history = deque(maxlen=3)
        iterator = iter(video.frames(config["max_frames"]))
        next_frame = decoder.submit(_next, iterator)
        exhausted = False
        segment = 0
        previous_time = None
        boundaries = [x for key in ("include_intervals", "exclude_intervals") for span in config[key] for x in span]
        resets = set(config["tracking"]["reset_frames"])
        stage_failures = set()

        def optional(name, future):
            try:
                return future.result()
            except Exception as exc:
                models.errors[name] = f"{type(exc).__name__}: {exc}"
                models.capabilities[name] = "error"
                stage_failures.add(name)
                if name in config["required"]:
                    raise
                return None

        with (output / "observations.partial.jsonl").open("w") as observations, \
             (output / "tracks.partial.jsonl").open("w") as tracks, \
             (output / "jersey_reads.partial.jsonl").open("w") as reads:
            while pending or not exhausted:
                # At most prefetch source images + one decoder image, plus the ball triplet.
                while not exhausted and len(pending) < config["prefetch"]:
                    frame = next_frame.result()
                    if frame is None:
                        exhausted = True
                        break
                    next_frame = decoder.submit(_next, iterator)
                    people = runtime.call("people", config["people"]["device"], models.people.predict, frame.image)
                    court = None
                    if models.court_points and "court" not in stage_failures and frame.index % config["court"]["detect_every"] == 0:
                        court = runtime.call("court", config["court"]["device"], models.court_points.predict, frame.image)
                    pending.append((frame, people, court))
                if not pending:
                    continue
                frame, people_future, court_future = pending.popleft()
                detections = people_future.result()
                # Pose depends on source boxes, not tracker IDs: overlap it with CPU
                # calibration/tracking, then join by the unchanged detection index.
                pose_future = None
                if models.pose and "pose" not in stage_failures:
                    pose_future = runtime.call("pose", config["pose"]["device"], models.pose.predict, frame.image, detections)
                force_cut = frame.index in resets or (previous_time is not None and any(previous_time < b <= frame.timestamp for b in boundaries))
                court_record = None
                if models.court and "court" not in stage_failures:
                    points = optional("court", court_future) if court_future else None
                    if "court" not in stage_failures:
                        try:
                            court_record = models.court.update(frame.image, detections, frame_index=frame.index,
                                                               timestamp_seconds=frame.timestamp, force_cut=force_cut,
                                                               keypoints_prediction=points)
                        except Exception as exc:
                            if "court" in config["required"]:
                                raise
                            models.errors["court"] = f"{type(exc).__name__}: {exc}"
                            models.capabilities["court"] = "error"
                            stage_failures.add("court")
                cut = force_cut or bool(court_record and court_record["scene_cut"])
                if cut and frame.index > 0:
                    segment += 1
                    history.clear()
                if court_record:
                    court_record["segment_id"] = segment
                tracked = models.tracker.update(detections, frame.image, frame_index=frame.index, scene_cut=cut)
                history.append(frame)
                ball_future = None
                if models.ball and "ball" not in stage_failures and len(history) == 3:
                    ball_future = runtime.call("ball", config["ball"]["device"], models.ball.predict, list(history))
                stages = {name: {"status": status} for name, status in models.capabilities.items()}
                if pose_future:
                    posed = optional("pose", pose_future)
                    if posed is not None:
                        if len(posed) != len(tracked) or any(p["detection_index"] != i or p["bbox"] != tracked[i]["bbox"] for i, p in enumerate(posed)):
                            raise ValueError("Pose observations do not match their source detections")
                        tracked = [{**d, **{k: p[k] for k in ("pose", "pose_status", "detection_index")}}
                                   for d, p in zip(tracked, posed, strict=True)]
                if models.jersey and "jersey" not in stage_failures:
                    models.clock = frame.timestamp
                    enriched = optional("jersey", runtime.call("jersey", config["jersey"]["device"], models.jersey.update,
                                         frame.image, [p for p in tracked if p["role"] == "player"],
                                         frame_index=frame.index, timestamp_seconds=frame.timestamp, segment_id=segment))
                    if enriched is not None:
                        players, readings, _ = enriched
                        by_track = {p["track_id"]: p for p in players if p["track_id"] is not None}
                        tracked = [by_track.get(p["track_id"], p) if p["role"] == "player" else p for p in tracked]
                        for reading in readings:
                            _line(reads, {"schema_version": 1, "run_id": run_id, **reading})
                positions = models.court.project_players(tracked, court_record, frame.image.shape) if court_record else [
                    {"position_m": None, "status": "calibration_unavailable", "ground_point_method": None,
                     "ground_point_px": None} for _ in tracked]
                persons = []
                for n, (p, position) in enumerate(zip(tracked, positions, strict=True)):
                    persons.append({**p, "observation_id": f"{run_id}:{frame.index}:{n}",
                                    "position_m": position["position_m"], "position_status": position["status"],
                                    "ground_point_px": position["ground_point_px"],
                                    "ground_point_method": position["ground_point_method"]})
                ball = optional("ball", ball_future) if ball_future else None
                if ball is None:
                    ball = {"position_px": None, "confidence": None,
                            "status": models.capabilities["ball"] if models.capabilities["ball"] != "ok" else "unavailable",
                            "reason": "insufficient_segment_context" if models.capabilities["ball"] == "ok" else "capability_unavailable"}
                for name in stage_failures:
                    stages[name] = {"status": "error", "reason": models.errors[name]}
                stages["ball"] = {"status": ball["status"]}
                if court_record:
                    stages["court"] = {"status": "ok" if court_record["image_to_court"] is not None else "unavailable"}
                for role, name in (("player", "players"), ("referee", "referees")):
                    if stages[name]["status"] == "ok" and not any(p["role"] == role for p in persons):
                        stages[name]["status"] = "not_detected"
                row = {"schema_version": 1, "artifact_type": "video_observation", "run_id": run_id,
                       "frame_index": frame.index, "timestamp_s": frame.timestamp, "segment_id": segment,
                       "scene_cut": cut, "included": included(config, frame.timestamp),
                       "width": frame.image.shape[1], "height": frame.image.shape[0],
                       "coordinate_space": "source_pixels", "distance_unit": "m",
                       "persons": persons, "ball": ball, "court": court_record, "stages": stages}
                validate_frame(row)
                _line(observations, row)
                for contribution in metric.update(row):
                    _line(tracks, contribution)
                frames_done += 1
                previous_time = frame.timestamp
                if frames_done == 1 or frames_done % 30 == 0:
                    write_json(output / "progress.json", {"run_id": run_id, "status": "running", "frames": frames_done,
                                                          "elapsed_s": time.perf_counter() - started})
        if models.jersey:
            models.jersey.finish()
        manifest["source"] = dict(video.source)
        if config["max_frames"] is None and frames_done != video.source["declared_frames"]:
            manifest["warnings"].append("declared_decoded_frame_count_mismatch")
        manifest["warnings"].extend(f"capability_{name}_{status}" for name, status in models.capabilities.items() if status != "ok")
        runtime.close()
        finished = time.perf_counter()
        partial = any(status != "ok" for status in models.capabilities.values())
        stats = {**metric.result(), "status": "partial" if partial else "completed", "source": dict(video.source),
                 "capabilities": models.capabilities, "artifacts": {"observations": "observations.jsonl", "tracks": "tracks.jsonl"}}
        validate_statistics(stats)
        write_json(output / "statistics.partial.json", stats)
        for name in ("observations.jsonl", "tracks.jsonl", "jersey_reads.jsonl", "statistics.json"):
            stem, suffix = name.rsplit(".", 1)
            os.replace(output / f"{stem}.partial.{suffix}", output / name)
            manifest["artifacts"][name] = {"sha256": file_hash(output / name)}
        manifest.update(status=stats["status"], frames=frames_done, segments=segment + 1,
                        timings={"load_s": loaded - started, "processing_s": finished - loaded,
                                 "total_s": finished - started, "fps": frames_done / (finished - loaded),
                                 "stages": dict(runtime.timings)},
                        peak_main_process_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                        memory_scope="main process high-water RSS; excludes isolated ball worker")
        if hasattr(models, "resources") and "ball" not in stage_failures:
            manifest["resources"] = models.resources()
        manifest["timings"]["finalization_s"] = time.perf_counter() - finished
        manifest["timings"]["total_s"] = time.perf_counter() - started
        write_json(output / "run.json", manifest)
        write_json(output / "progress.json", {"run_id": run_id, "status": stats["status"], "frames": frames_done})
        return stats
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        frames=frames_done, error=f"{type(exc).__name__}: {exc}")
        write_json(output / "run.json", manifest)
        write_json(output / "progress.json", {"run_id": run_id, "status": manifest["status"], "frames": frames_done})
        raise
    finally:
        decoder.shutdown(wait=True, cancel_futures=True)
        runtime.close()
        if models:
            models.close()
        if video:
            video.close()


def recompute(run_path, output_path, distance_config=None):
    """Recalculate metrics without model inference, only from a committed artifact."""
    run_path = Path(run_path)
    manifest = json.loads((run_path / "run.json").read_text())
    if manifest["status"] not in ("completed", "partial"):
        raise ValueError("Recompute requires a completed analysis artifact")
    source = run_path / "observations.jsonl"
    if file_hash(source) != manifest["artifacts"]["observations.jsonl"]["sha256"]:
        raise ValueError("Observation artifact hash mismatch")
    if Path(output_path).exists():
        raise FileExistsError(output_path)
    metric = Distance(distance_config or manifest["config"]["distance"], manifest["run_id"])
    with source.open() as handle:
        for line in handle:
            row = validate_frame(json.loads(line))
            if row["run_id"] != manifest["run_id"]:
                raise ValueError("Observation belongs to another run")
            metric.update(row)
    result = {**metric.result(), "status": manifest["status"], "source": manifest["source"],
              "capabilities": manifest["capabilities"]}
    validate_statistics(result)
    write_json(Path(output_path), result)
    return result
