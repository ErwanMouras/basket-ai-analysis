"""Full video analysis with NVDEC and CUDA computation through final statistics."""

import faulthandler
import hashlib
import importlib.metadata
import json
import os
import resource
import signal
import threading
import time
from collections import defaultdict
from itertools import islice
from pathlib import Path
from uuid import uuid4

import torch

from pipeline.cuda.core import CudaOnly, Video, export, tensor
from pipeline.cuda.distance import CudaDistance
from pipeline.cuda.geometry import Court
from pipeline.cuda.identity import replay, resolve
from pipeline.cuda.integrity import geometry_bytes, observation_metadata, timestamps
from pipeline.cuda.models import Models, encode_mask
from pipeline.cuda.resume import copy_prefix, inspect_prefix
from pipeline.cuda.streams import ModelsInFlight
from pipeline.cuda.tracking import Tracker
from pipeline.cuda.vision import Evidence
from pipeline.numeric import CudaCounter, cuda_numbers
from pipeline.progress import Progress
from training.common.files import write_json
from training.common.provenance import file_hash
from training.court.geometry import metadata as court_metadata
from training.jersey.display import display_colors
from training.jersey.roster import Roster


def health(output, manifest, device, progress, *, expected_timestamps=None, expected_geometry_digest=None):
    geometry_digest = hashlib.sha256() if expected_geometry_digest else None
    counts = CudaCounter(device=device)
    stages = CudaCounter(device=device)
    exclusions = CudaCounter(device=device)
    tracks = set()
    frames = manifest["frames"]
    index = 0
    previous_time = None
    if expected_timestamps is not None and len(expected_timestamps) != frames:
        raise ValueError("Health target differs from source timestamp count")
    progress.update("health", 0, frames)
    frame_keys = (
        "frames",
        "included_frames",
        "calibrated_frames",
        "cuts",
        "ball_localized_frames",
    )
    person_keys = (
        "player_observations",
        "projected_player_observations",
        "named_player_observations",
        "suppressed_identity_observations",
    )
    with (output / "observations.jsonl").open() as stream:
        iterator = (json.loads(line) for line in stream)
        while batch := list(islice(iterator, 512)):
            stop = index + len(batch)
            previous_time = timestamps(
                [row["timestamp_s"] for row in batch],
                device,
                previous=previous_time,
                expected=expected_timestamps[index:stop]
                if expected_timestamps is not None
                else None,
            )
            frame_flags, person_flags, tags = [], [], []
            for row in batch:
                if geometry_digest is not None:
                    geometry_digest.update(geometry_bytes(row))
                observation_metadata(row, manifest["run_id"], index)
                index += 1
                frame_flags.append(
                    [
                        True,
                        bool(row["included"]),
                        (row.get("court") or {}).get("image_to_court") is not None,
                        bool(row["scene_cut"]),
                        row["ball"]["position_px"] is not None,
                    ]
                )
                tags.extend(
                    f"{name}:{stage['status']}" for name, stage in row["stages"].items()
                )
                for person in row["persons"]:
                    if person["role"] != "player":
                        continue
                    person_flags.append(
                        [
                            True,
                            person["position_m"] is not None,
                            bool(person.get("resolved_identity")),
                            bool(person.get("jersey_number_suppressed")),
                        ]
                    )
                    if person["track_id"] is not None:
                        tracks.add((row["segment_id"], person["track_id"]))
            totals = (
                tensor(frame_flags, device=device, dtype=torch.bool).sum(0).double()
            )
            counts.update(dict(zip(frame_keys, totals.unbind())))
            if person_flags:
                totals = (
                    tensor(person_flags, device=device, dtype=torch.bool)
                    .sum(0)
                    .double()
                )
                counts.update(dict(zip(person_keys, totals.unbind())))
            stages.update(tags)
            progress.update("health", batch[-1]["frame_index"] + 1, frames)
    if not bool(counts["frames"] == frames):
        raise ValueError("CUDA health frame count mismatch")
    if geometry_digest is not None and geometry_digest.hexdigest() != expected_geometry_digest:
        raise ValueError("Color-only refinement changed source geometry or detections")
    statistics = json.loads((output / "statistics.json").read_text())
    for player in statistics["players"]:
        exclusions.update(player["excluded_intervals"])

    def ratio(a, b):
        return counts[a] / counts[b] if bool(counts[b] > 0) else None

    for name, artifact in manifest["artifacts"].items():
        if file_hash(output / name) != artifact["sha256"]:
            raise ValueError(f"Artifact integrity mismatch: {name}")
    return {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "status": manifest["status"],
        "backend": "native_CUDA",
        "device": device,
        "counts": dict(counts),
        "stage_status_counts": dict(stages),
        "player_track_fragments": tensor(len(tracks), device=device),
        "exclusion_counts": dict(exclusions),
        "integrity": {
            "expected_frames": frames,
            "verified_frames": index,
            "contiguous_frame_indices": True,
            "unique_observation_and_track_ids": True,
            "unique_resolved_identities_per_frame": True,
            "ordered_finite_timestamps": True,
            "timestamps_match_source": expected_timestamps is not None,
            **({"source_geometry_and_detections_unchanged": True,
                "immutable_geometry_sha256": geometry_digest.hexdigest()} if geometry_digest is not None else {}),
        },
        "ratios": {
            "calibration_availability": ratio("calibrated_frames", "frames"),
            "ball_localization_availability": ratio("ball_localized_frames", "frames"),
            "player_projection_availability": ratio(
                "projected_player_observations", "player_observations"
            ),
            "named_identity_availability": ratio(
                "named_player_observations", "player_observations"
            ),
        },
        "interpretation": "Availability and integrity only; not accuracy or full-match ground-truth statistics.",
    }


def _included(config, now):
    included = not config["include_intervals"] or any(
        bool((now >= a) & (now < b)) for a, b in config["include_intervals"]
    )
    return included and not any(
        bool((now >= a) & (now < b)) for a, b in config["exclude_intervals"]
    )


def analyze(video_path, output, config, roster_path, *, resume_from=None):
    from pipeline.execution_policy import enforce_execution_policy

    plan = enforce_execution_policy(config, roster=roster_path is not None)
    roster = Roster.load(roster_path)
    checkpoint = (
        inspect_prefix(resume_from, config, roster.sha256) if resume_from else None
    )
    start_frame = checkpoint["start"] if checkpoint else 0
    device = config["people"]["device"]
    if not torch.cuda.is_available():
        raise RuntimeError("Native CUDA requires a GPU; CPU fallback forbidden")
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)  # numerical work is CUDA; host handles control/I/O only
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    run_id = checkpoint["manifest"]["run_id"] if checkpoint else uuid4().hex
    started = time.perf_counter()
    progress = Progress(
        output, run_id, on_stall=lambda: faulthandler.dump_traceback(all_threads=True)
    )
    guard = CudaOnly()
    video = None
    in_flight = None
    frames = 0
    timings = defaultdict(float)
    palette = display_colors(roster)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "config": config,
        "execution_policy": plan,
        "pid": os.getpid(),
        "backend": "native_cuda",
        "statistics_device": device,
        "roster_sha256": roster.sha256,
        "display_colors": palette,
        "artifacts": {},
        "warnings": [],
        "host_scope": "compressed file I/O, demux metadata, orchestration, IDs, JSON serialization, artifact hashes and progress UI; no CPU tensor arithmetic",
        "execution_devices": {
            name: device
            for name in (
                "image_preprocessing",
                "people",
                "segmentation",
                "ball",
                "pose",
                "jersey",
                "tracking",
                "court_calibration",
                "identity_votes",
                "spatial_handoffs",
                "distance_duration_coverage",
                "health",
            )
        },
        "decoder_device": "NVIDIA_NVDEC",
        "court_coordinate_system": court_metadata(),
        "capabilities": {
            name: "ok"
            for name in ("players", "referees", "court", "ball", "pose", "jersey")
        },
    }
    root = Path(__file__).resolve().parents[2]
    manifest["code_sha256"] = {
        str(path.relative_to(root)): file_hash(path)
        for folder in ("pipeline", "training")
        for path in sorted((root / folder).rglob("*.py"))
        if "tests" not in path.parts
    }
    manifest["dependencies"] = {
        name: importlib.metadata.version(name)
        for name in ("torch", "triton", "onnx", "pynvvideocodec", "ultralytics")
    }
    write_json(output / "run.json", manifest)
    write_json(output / "roster.json", roster.payload)
    previous_term = None
    if threading.current_thread() is threading.main_thread():
        previous_term = signal.getsignal(signal.SIGTERM)

        def terminate(_signal, _frame):
            raise KeyboardInterrupt("Termination requested")

        signal.signal(signal.SIGTERM, terminate)
    try:
        video = Video(video_path, device, batch_size=config["prefetch"])
        if checkpoint:
            if checkpoint["manifest"]["source"]["sha256"] != video.source["sha256"]:
                raise ValueError("Resume video differs from the original run")
            if start_frame:
                timestamps(
                    checkpoint["timestamps"],
                    device,
                    expected=video.timestamps[:start_frame],
                )
            copy_prefix(checkpoint, output)
            frames = start_frame
            manifest["resume"] = {
                "source_run": str(checkpoint["source"]),
                "source_manifest_sha256": checkpoint["manifest_sha256"],
                "retained_frames": start_frame,
                "replayed_frames": checkpoint["available_frames"] - start_frame,
                "boundary": "last_scene_cut",
                "prefix_code_sha256": checkpoint["manifest"]["code_sha256"],
                "prefix_cuda_guard": checkpoint["manifest"].get("cuda_guard"),
                "prefix_sha256": checkpoint["prefix_sha256"],
                "prefix_verified": True,
            }
        total = (
            min(video.source["presentation_frames"], config["max_frames"])
            if config["max_frames"]
            else video.source["presentation_frames"]
        )
        progress.update("loading", total_frames=total)
        models = Models(config)
        # The dispatch guard also checks tensor factories in third-party code.
        # The device scope fixes defaults such as YOLO26's torch.arange(batch).
        with torch.inference_mode(), cuda_numbers(device), guard:
            for model in (
                models.people.model,
                models.ball.model,
                models.court.model,
                models.segmentation.predictor.model,
                models.jersey.model,
            ):
                if any(not parameter.is_cuda for parameter in model.parameters()):
                    raise RuntimeError("A model parameter remained on CPU")
            tracker = Tracker(config["tracking"], video.fps, device)
            with torch.device(device):
                models.prepare_graphs((video.source["height"], video.source["width"]))
            court = Court(config["court"], models.court, device)
            if checkpoint and start_frame:
                tracker.counter = checkpoint["track_counter"]
                court.segment = checkpoint["segment"] - 1
            evidence = Evidence(config["jersey"], roster, models.jersey, device)
            manifest.update(
                source=video.source,
                models={
                    **models.provenance,
                    "tracking": tracker.provenance,
                    "court_calibration": court.provenance,
                },
            )
            manifest["timings"] = {"load_s": time.perf_counter() - started}
            write_json(output / "run.json", export(manifest))
            in_flight = (
                ModelsInFlight(device, config["gpu_concurrency"])
                if config["mode"] == "parallel"
                else None
            )
            manifest["gpu_scheduling"] = (
                "independent_CUDA_streams" if in_flight else "sequential"
            )
            progress.update("perception", start_frame, total, frames=start_frame)
            progress.metrics(retained_frames=start_frame)
            throughput_started = time.perf_counter()
            window_started, window_frames = throughput_started, start_frame
            previous = None
            boundaries = [
                value
                for name in ("include_intervals", "exclude_intervals")
                for span in config[name]
                for value in span
            ]
            resets = set(config["tracking"]["reset_frames"])
            with (
                (output / "observations.partial.jsonl").open(
                    "a" if checkpoint else "w"
                ) as observations,
                (output / "jersey_reads.partial.jsonl").open(
                    "a" if checkpoint else "w"
                ) as reads,
            ):
                for index, now, image in video.frames(
                    config["max_frames"], start=start_frame
                ):
                    if index != frames:
                        raise ValueError(
                            f"Inference skipped/repeated a frame: expected {frames}, got {index}"
                        )
                    tick = time.perf_counter()
                    rows = models.people.predict(
                        image, classes=[int(k) for k in config["people"]["class_roles"]]
                    )
                    if not bool(torch.isfinite(rows).all()):
                        raise ValueError("Nonfinite people predictions")
                    timings["people_s"] += time.perf_counter() - tick
                    tick = time.perf_counter()
                    pose_job = segmentation_job = ball_job = None
                    if in_flight:
                        pose_job = in_flight.submit(
                            "pose", models.pose.predict, image, rows
                        )
                        segmentation_job = in_flight.submit(
                            "segmentation",
                            models.segmentation.predict,
                            image,
                            rows[:, :4],
                        )
                        ball_job = in_flight.submit(
                            "ball",
                            models.ball.predict,
                            image,
                            classes=[0],
                            stretch=(
                                config["ball"]["input_height"],
                                config["ball"]["input_width"],
                            ),
                        )
                    else:
                        points, scores, valid = models.pose.predict(image, rows)
                        timings["pose_s"] += time.perf_counter() - tick
                    tick = time.perf_counter()
                    force = (
                        (
                            checkpoint is not None
                            and index == start_frame
                            and start_frame > 0
                        )
                        or index in resets
                        or (
                            previous is not None
                            and any(
                                bool((previous < b) & (now >= b)) for b in boundaries
                            )
                        )
                    )
                    record = court.update(image, rows, index, now, force_cut=force)
                    tracks = tracker.update(rows, index, cut=record["scene_cut"])
                    if pose_job is not None:
                        points, scores, valid = in_flight.result(pose_job)
                    positions = court.positions(
                        rows, points, valid, record, image.shape[-2:]
                    )
                    timings["court_tracking_s"] += time.perf_counter() - tick
                    tick = time.perf_counter()
                    masks, quality = (
                        in_flight.result(segmentation_job)
                        if segmentation_job is not None
                        else models.segmentation.predict(image, rows[:, :4])
                    )
                    timings["segmentation_s"] += time.perf_counter() - tick
                    tick = time.perf_counter()
                    jerseys, readings = evidence.update(
                        image,
                        rows,
                        tracks,
                        court.segment,
                        index,
                        now,
                        points,
                        valid,
                        masks,
                    )
                    for read in readings:
                        reads.write(
                            json.dumps(
                                export({"run_id": run_id, **read}),
                                allow_nan=False,
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    timings["jersey_s"] += time.perf_counter() - tick
                    tick = time.perf_counter()
                    balls = (
                        in_flight.result(ball_job)
                        if ball_job is not None
                        else models.ball.predict(
                            image,
                            classes=[0],
                            stretch=(
                                config["ball"]["input_height"],
                                config["ball"]["input_width"],
                            ),
                        )
                    )
                    if len(balls):
                        best = balls[balls[:, 4].argmax()]
                        location = (best[:2] + best[2:4]) / 2
                        ball = {
                            "position_px": location,
                            "confidence": best[4],
                            "status": "ok",
                            "context_frames": 1,
                            "future_context_frames": 0,
                            "source_frames": [index],
                        }
                    else:
                        ball = {
                            "position_px": None,
                            "confidence": None,
                            "status": "not_detected",
                            "context_frames": 1,
                            "future_context_frames": 0,
                            "source_frames": [index],
                        }
                    timings["ball_s"] += time.perf_counter() - tick
                    persons = []
                    positive_scores = (scores > 0).tolist()
                    pose_usable = valid.any(1).tolist()
                    class_ids = rows[:, 5].long().tolist()
                    for n, row in enumerate(rows):
                        role = config["people"]["class_roles"][str(class_ids[n])]
                        segmentation, nonempty = encode_mask(masks[n], row[:4])
                        pose = {
                            "format": "coco17",
                            "coordinate_space": "source",
                            "keypoints": [
                                points[n, k] if positive_scores[n][k] else None
                                for k in range(17)
                            ],
                            "scores": scores[n],
                            "valid": valid[n],
                            "valid_keypoints": valid[n].sum(),
                            "score": scores[n].mean(),
                        }
                        person = {
                            "bbox": row[:4],
                            "confidence": row[4],
                            "class_id": 0,
                            "source_class_id": class_ids[n],
                            "role": role,
                            "track_id": tracks[n],
                            "detection_index": n,
                            "observation_id": f"{run_id}:{index}:{n}",
                            "segmentation": {
                                "status": "ok" if nonempty else "empty",
                                "quality_score": quality[n],
                                "mask": segmentation,
                            },
                            "pose": pose,
                            "pose_status": "estimated"
                            if pose_usable[n]
                            else "low_keypoint_confidence",
                            "jersey": {
                                "number": None,
                                "status": "unknown",
                                "identity_status": "unresolved",
                            },
                            "cuda_evidence": jerseys[n],
                            "display_color": palette["referee"]
                            if role == "referee"
                            else palette["unassigned"],
                            **positions[n],
                        }
                        if jerseys[n] and jerseys[n]["team_id"]:
                            person["display_color"] = palette["teams"][
                                jerseys[n]["team_id"]
                            ]
                        persons.append(person)
                    stages = {
                        name: {"status": "ok"} for name in manifest["capabilities"]
                    }
                    stages["ball"] = {"status": ball["status"]}
                    stages["court"] = {"status": "ok" if court.state else "unavailable"}
                    for role, name in (("player", "players"), ("referee", "referees")):
                        if not any(person["role"] == role for person in persons):
                            stages[name] = {"status": "not_detected"}
                    observation = {
                        "schema_version": 1,
                        "artifact_type": "video_observation",
                        "run_id": run_id,
                        "frame_index": index,
                        "timestamp_s": now,
                        "segment_id": court.segment,
                        "scene_cut": record["scene_cut"],
                        "included": _included(config, now),
                        "width": image.shape[2],
                        "height": image.shape[1],
                        "coordinate_space": "source_pixels",
                        "distance_unit": "m",
                        "persons": persons,
                        "ball": ball,
                        "court": record,
                        "stages": stages,
                    }
                    observation_metadata(observation, run_id, index)
                    observations.write(
                        json.dumps(
                            export(observation), allow_nan=False, ensure_ascii=False
                        )
                        + "\n"
                    )
                    frames = index + 1
                    previous = now
                    if frames == 1 or frames % 30 == 0:
                        observations.flush()
                        reads.flush()
                        progress.update("perception", frames, total, frames=frames)
                        clock = time.perf_counter()
                        recent_fps = (frames - window_frames) / (clock - window_started)
                        progress.metrics(
                            throughput_fps=recent_fps,
                            throughput_basis="last_30_frames"
                            if frames - window_frames == 30
                            else "since_resume",
                            perception_remaining_s=(total - frames) / recent_fps,
                            processed_since_resume=frames - start_frame,
                        )
                        window_started, window_frames = clock, frames
                    if frames % 300 == 0:
                        manifest.update(
                            frames=frames,
                            source=video.source,
                            cuda_guard=dict(
                                guard.calls + in_flight.calls
                                if in_flight
                                else guard.calls
                            ),
                        )
                        manifest["timings"].update(
                            **timings,
                            perception_s=time.perf_counter()
                            - started
                            - manifest["timings"]["load_s"],
                        )
                        write_json(output / "run.json", export(manifest))
            if in_flight:
                in_flight.close()
                guard.calls.update(in_flight.calls)
                in_flight = None
            torch.cuda.synchronize(device)
            if frames != total:
                raise ValueError(f"Incomplete perception: {frames}/{total} frames")
            manifest["frame_integrity"] = {
                "target_frames": total,
                "perception_complete": True,
                "health_verified": False,
            }
            perception_finished = time.perf_counter()
            manifest["timings"].update(
                perception_s=perception_finished
                - started
                - manifest["timings"]["load_s"],
                **timings,
            )
            tick = time.perf_counter()
            identities = resolve(output, roster, device, progress, frames)
            manifest["timings"]["identity_s"] = time.perf_counter() - tick
            tick = time.perf_counter()
            metric = replay(
                output, roster, config, run_id, progress, frames, identities, palette
            )
            manifest["timings"]["statistics_s"] = time.perf_counter() - tick
            stats = {
                **metric.result(),
                "status": "completed",
                "source": dict(video.source),
                "capabilities": manifest["capabilities"],
                "artifacts": {
                    "observations": "observations.jsonl",
                    "tracks": "tracks.jsonl",
                },
            }
            # Validate the numeric metric invariant on CUDA before JSON export.
            for player in stats["players"]:
                measured = tensor(player["measured_duration_s"], device=device)
                tracked = tensor(player["tracked_duration_s"], device=device)
                if not bool((measured >= 0) & (measured <= tracked + 1e-9)):
                    raise ValueError("CUDA metric duration invariant failed")
            write_json(output / "statistics.partial.json", export(stats))
            artifacts = [
                "observations.jsonl",
                "jersey_reads.jsonl",
                "tracks.jsonl",
                "statistics.json",
                "team_votes.json",
            ]
            for n, name in enumerate(artifacts):
                progress.update("publication", n, len(artifacts))
                stem, suffix = name.rsplit(".", 1)
                (output / f"{stem}.partial.{suffix}").replace(output / name)
                manifest["artifacts"][name] = {"sha256": file_hash(output / name)}
            manifest.update(
                status="validating",
                frames=frames,
                segments=court.segment + 1,
                source=video.source,
                spatial_handoffs=metric.spatial_handoffs,
                resolved_player_tracks=tensor(len(identities[1]), device=device),
                peak_main_process_rss_mb=resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                / 1024,
                resources={
                    "main_peak_torch_cuda_allocated_mb": torch.cuda.max_memory_allocated(
                        device
                    )
                    / 1024**2
                },
            )
            write_json(output / "run.json", export(manifest))
            tick = time.perf_counter()
            checked = health(
                output,
                manifest,
                device,
                progress,
                expected_timestamps=video.timestamps[:total],
            )
            checked["status"] = "completed"
            write_json(output / "health.json", export(checked))
            manifest["status"] = "completed"
            manifest["frame_integrity"]["health_verified"] = True
            manifest["timings"]["health_s"] = time.perf_counter() - tick
            manifest["artifacts"]["health.json"] = {
                "sha256": file_hash(output / "health.json")
            }
            manifest["timings"]["total_s"] = time.perf_counter() - started
            manifest["cuda_guard"] = {
                **dict(guard.calls),
                "cpu_tensor_calculations": 0,
                "fallback_events": [],
            }
            write_json(output / "run.json", export(manifest))
            progress.update("health", frames, frames, status="completed")
            return export(stats)
    except BaseException as error:
        manifest.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            frames=frames,
            error=f"{type(error).__name__}: {error}",
            cuda_guard=dict(guard.calls),
        )
        write_json(output / "run.json", export(manifest))
        progress.update(
            progress.state["stage"],
            progress.state["stage_frames"],
            progress.state["stage_total_frames"],
            frames=frames,
            status=manifest["status"],
            error=manifest["error"],
        )
        raise
    finally:
        if previous_term is not None:
            signal.signal(signal.SIGTERM, previous_term)
        progress.close()
        if in_flight:
            in_flight.close()
        if video:
            video.close()


def recompute(run_path, output_path, distance_config=None):
    run_path = Path(run_path)
    manifest = json.loads((run_path / "run.json").read_text())
    config = distance_config or manifest["config"]["distance"]
    if manifest["status"] not in ("completed", "partial"):
        raise ValueError("Recompute requires a finalized run")
    if Path(output_path).exists():
        raise FileExistsError(output_path)
    source = run_path / "observations.jsonl"
    if file_hash(source) != manifest["artifacts"]["observations.jsonl"]["sha256"]:
        raise ValueError("Observation artifact hash mismatch")
    device = config["device"]
    with torch.inference_mode(), torch.device(device), cuda_numbers(device), CudaOnly():
        metric = CudaDistance(config, manifest["run_id"])
        with source.open() as stream:
            for line in stream:
                row = json.loads(line)
                row["timestamp_s"] = tensor(row["timestamp_s"], device=device)
                metric.update(row)
        result = {
            **metric.result(),
            "status": manifest["status"],
            "source": manifest["source"],
            "capabilities": manifest["capabilities"],
        }
        write_json(Path(output_path), export(result))
        return export(result)
