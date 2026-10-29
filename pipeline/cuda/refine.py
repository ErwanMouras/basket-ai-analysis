"""Refine a complete saved CUDA match, then rebuild all identities and metrics.

No detector/pose/segmentation/court/ball weights or saved coordinates are changed.
The original run stays immutable; publication requires complete frame/PTS checks.
"""

import argparse
import copy
import faulthandler
import fcntl
import json
import os
import shutil
import time
import traceback
from itertools import groupby, islice
from pathlib import Path
from uuid import uuid4

import torch

from pipeline.cuda.core import CudaOnly, Video, export, tensor
from pipeline.cuda.identity import collision_frames, replay, resolve
from pipeline.cuda.recovery import deduplicate_persons, recover, smooth_referee_roles
from pipeline.cuda.refined_identity import (
    complete_verified_lineups,
    official_lineups,
    propagate_motion,
)
from pipeline.cuda.run import health
from pipeline.numeric import CudaCounter, cuda_numbers
from pipeline.progress import Progress
from training.common.files import write_json
from training.common.provenance import file_hash
from training.jersey.roster import Roster

PHASES = [
    ("loading", 1),
    ("recovery", 55),
    ("refinement", 12),
    ("identity_colors", 4),
    ("identity_ocr", 2),
    ("identity_resolution", 4),
    ("identity_handoffs", 3),
    ("identity_collisions", 1),
    ("identity_motion", 8),
    ("statistics", 7),
    ("publication", 1),
    ("health", 2),
]

COLOR_PHASES = [
    ("loading", 1),
    ("color_features", 35),
    ("color_calibration", 2),
    ("refinement", 9),
    ("identity_colors", 5),
    ("identity_ocr", 2),
    ("identity_resolution", 7),
    ("identity_handoffs", 5),
    ("identity_collisions", 1),
    ("identity_motion", 15),
    ("statistics", 14),
    ("publication", 1),
    ("health", 3),
]


def frame_colors(path):
    with Path(path).open() as stream:
        for frame, items in groupby(
            (json.loads(line) for line in stream), key=lambda r: r["frame_index"]
        ):
            yield frame, {r["track_id"]: r for r in items}


def normalize(source, recovery, output, run_id, frames, device, progress):
    """Stream all observations, retaining original scores, geometry and weak bodies."""
    colors = iter(frame_colors(recovery / "team_reads.jsonl"))
    color_frame, votes = next(colors, (None, {}))
    counts = CudaCounter(device=device)
    times = []
    progress.update("refinement", 0, frames)
    with (
        (source / "observations.jsonl").open() as original,
        (output / "observations.partial.jsonl").open("w") as cleaned,
        (output / "removed_detections.jsonl").open("w") as removed,
    ):
        rows = (json.loads(line) for line in islice(original, frames))
        before, row, after = None, next(rows, None), next(rows, None)
        for index in range(frames):
            if row is None or row["frame_index"] != index:
                raise ValueError(
                    "Missing/repeated source observation during refinement"
                )
            times.append(row["timestamp_s"])
            while color_frame is not None and color_frame < index:
                color_frame, votes = next(colors, (None, {}))
            local_votes = votes if color_frame == index else {}
            refined = smooth_referee_roles(row, before, after, device)
            refined, rejected = deduplicate_persons(refined, device)
            counts["source_players"] += tensor(
                [p["role"] == "player" for p in row["persons"]],
                device=device,
                dtype=torch.bool,
            ).sum()
            counts["source_named_players"] += tensor(
                [
                    p["role"] == "player" and bool(p.get("resolved_identity"))
                    for p in row["persons"]
                ],
                device=device,
                dtype=torch.bool,
            ).sum()
            counts["retained_players"] += tensor(
                [p["role"] == "player" for p in refined["persons"]],
                device=device,
                dtype=torch.bool,
            ).sum()
            counts["baseline_named_on_retained_players"] += tensor(
                [
                    p["role"] == "player" and bool(p.get("resolved_identity"))
                    for p in refined["persons"]
                ],
                device=device,
                dtype=torch.bool,
            ).sum()
            refined["source_run_id"] = refined["run_id"]
            refined["run_id"] = run_id
            for n, p in enumerate(refined["persons"]):
                p["source_observation_id"] = p.get("observation_id")
                p["source_detection_index"] = p["detection_index"]
                p["detection_index"] = n
                p["observation_id"] = f"{run_id}:{index}:{n}"
                vote = local_votes.get(p["track_id"])
                evidence = p.get("cuda_evidence") or {}
                if (
                    p["role"] == "player"
                    and not evidence.get("team_id")
                    and vote
                    and vote["team_id"]
                ):
                    p["cuda_evidence"] = {
                        **evidence,
                        "team_id": vote["team_id"],
                        "color_refinement": "CUDA_two_segmented_torso_ROIs",
                    }
                    counts["additional_color_frames"] += 1
            for p in rejected:
                removed.write(
                    json.dumps(
                        {
                            "frame_index": index,
                            "source_observation_id": p.get("observation_id"),
                            "source_role": p["role"],
                            "reason": p["rejection_reason"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                counts["removed_duplicate_observations"] += 1
            cleaned.write(
                json.dumps(refined, ensure_ascii=False, allow_nan=False) + "\n"
            )
            if index % 150 == 0:
                cleaned.flush()
                progress.update("refinement", index + 1, frames)
            before, row, after = row, after, next(rows, None)
    progress.update("refinement", frames, frames)
    return export(dict(counts)), tensor(times, device=device)


def run(
    source,
    output,
    *,
    step=2,
    max_frames=None,
    recovery_path=None,
    clock_review=None,
    colors_only=False,
    color_features=None,
):
    if (colors_only and recovery_path is not None) or (
        color_features and not colors_only
    ):
        raise ValueError(
            "Color-only refinement reuses source OCR; --recovery is incompatible"
        )
    source = Path(source).resolve(strict=True)
    original = json.loads((source / "run.json").read_text())
    if original["status"] != "completed" or original.get("backend") != "native_cuda":
        raise ValueError("A complete native CUDA source run is required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    device = original["config"]["jersey"]["device"]
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    frames = (
        original["frames"]
        if max_frames is None
        else min(original["frames"], max_frames)
    )
    if frames < 1:
        raise ValueError("Empty refinement interval")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Cross-process lock prevents two refinement jobs competing on this GPU.
    lock = (source.parent / ".identity-refinement.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError(
            "Another match identity refinement is already running"
        ) from None
    run_id = uuid4().hex
    manifest = copy.deepcopy(original)
    manifest.update(
        run_id=run_id,
        status="running",
        frames=frames,
        artifacts={},
        timings={},
        pid=os.getpid(),
    )
    manifest.pop("resume", None)
    manifest["frame_integrity"] = {
        "target_frames": frames,
        "perception_complete": True,
        "health_verified": False,
    }
    manifest["gpu_scheduling"] = "CUDA_saved_perception_whole_match_refinement"
    root = Path(__file__).resolve().parents[2]
    manifest["code_sha256"] = {
        str(p.relative_to(root)): file_hash(p)
        for folder in ("pipeline", "training")
        for p in sorted((root / folder).rglob("*.py"))
        if "tests" not in p.parts
    }
    manifest["identity_resolution"] = {
        "method": "CUDA_multicrop_rechecks_color_recovery_bidirectional_continuity",
        "source_run": str(source),
        "source_manifest_sha256": file_hash(source / "run.json"),
        "perception_models_reused_from_source": True,
        "perception_source_code_sha256": original.get("identity_resolution", {}).get(
            "perception_source_code_sha256", original["code_sha256"]
        ),
        "reviewed_initialization": False,
        "manual_detection_decisions_applied": 0,
        "ocr_frame_step": step,
        "all_source_frames_replayed": frames == original["frames"],
        "weak_detections_retained": True,
    }
    if colors_only:
        manifest["identity_resolution"].update(
            method="CUDA_match_calibrated_colors_existing_OCR_and_geometry",
            ocr_recomputed=False,
            perception_recomputed=False,
            detections_removed=0,
            source_ocr_sha256=original["artifacts"]["jersey_reads.jsonl"]["sha256"],
        )
    write_json(output / "run.json", manifest)
    shutil.copyfile(source / "roster.json", output / "roster.json")
    roster = Roster.load(output / "roster.json")
    progress = Progress(
        output,
        run_id,
        phases=COLOR_PHASES if colors_only else PHASES,
        on_stall=lambda: faulthandler.dump_traceback(all_threads=True),
    )
    progress.metrics(total_frames=frames, reused_perception_frames=frames)
    guard = CudaOnly()
    started = time.perf_counter()
    try:
        # Verify immutable evidence and all source model checkpoints before reuse.
        for name in ("observations.jsonl", "jersey_reads.jsonl"):
            if file_hash(source / name) != original["artifacts"][name]["sha256"]:
                raise ValueError(f"Source artifact changed: {name}")
        for name in ("people", "ball", "court", "pose", "segmentation", "jersey"):
            model = original["models"][name]
            path = original["config"][name].get("weights")
            if (
                path
                and model.get("checkpoint_sha256")
                and file_hash(Path(path)) != model["checkpoint_sha256"]
            ):
                raise ValueError(f"Model checkpoint changed: {name}")
        tick = time.perf_counter()
        if colors_only:
            from pipeline.cuda.colors import extract

            if color_features is None:
                color_features = output / "colors"
                with (
                    torch.inference_mode(),
                    torch.device(device),
                    cuda_numbers(device),
                    guard,
                ):
                    extract(source, color_features, frames, device, progress)
            else:
                color_features = Path(color_features).resolve(strict=True)
                progress.update("color_features", frames, frames, frames=frames)
            recovery_path = source
            audit = json.loads((source / "recovery-audit.json").read_text())
        elif recovery_path is None:
            recovery_path = output / "recovery"
            audit = recover(
                source,
                recovery_path,
                0,
                frames,
                step=step,
                progress=progress,
                context=False,
                candidates=False,
            )
        else:
            recovery_path = Path(recovery_path).resolve(strict=True)
            audit = json.loads((recovery_path / "audit.json").read_text())
            if (
                audit["source_manifest_sha256"] != file_hash(source / "run.json")
                or audit["start_frame"] != 0
                or audit["stop_frame"] != frames
                or audit["step"] != step
            ):
                raise ValueError(
                    "Recovery evidence belongs to a different run/interval/cadence"
                )
            progress.update("recovery", frames, frames, frames=frames)
        if (
            audit["model"]["checkpoint_sha256"]
            != original["models"]["jersey"]["checkpoint_sha256"]
        ):
            raise ValueError("OCR checkpoint differs from source")
        manifest["timings"]["color_features_s" if colors_only else "recovery_s"] = (
            time.perf_counter() - tick
        )
        shutil.copyfile(
            recovery_path / ("recovery-audit.json" if colors_only else "audit.json"),
            output / "recovery-audit.json",
        )
        with (
            (recovery_path / "jersey_reads.jsonl").open() as reads,
            (output / "jersey_reads.partial.jsonl").open("w") as merged,
        ):
            for line in reads:
                read = json.loads(line)
                if read["frame_index"] >= frames:
                    continue
                read["source_run_id"] = read.get("run_id")
                read["run_id"] = run_id
                merged.write(json.dumps(read, ensure_ascii=False) + "\n")
        with torch.inference_mode(), torch.device(device), cuda_numbers(device), guard:
            tick = time.perf_counter()
            if colors_only:
                from pipeline.cuda.colors import apply_colors

                baseline, source_times = apply_colors(
                    source, color_features, output, run_id, frames, device, progress
                )
            else:
                baseline, source_times = normalize(
                    source, recovery_path, output, run_id, frames, device, progress
                )
            manifest["timings"]["normalization_s"] = time.perf_counter() - tick
            tick = time.perf_counter()
            identities, rows = resolve(
                output,
                roster,
                device,
                progress,
                frames,
                refinement=True,
                keep_rows=True,
            )
            lineup_count = 0
            if colors_only and not clock_review:
                # Reuse only the already verified interval; no new video-clock
                # alignment or extrapolation from elapsed video time.
                constraint = original.get("identity_resolution", {}).get(
                    "official_lineup_constraint"
                )
                if constraint:
                    official_path = (
                        root / "datas/test/SAC_SAS/sources/espn-summary-401909879.json"
                    )
                    if file_hash(official_path) != constraint["official_sha256"]:
                        raise ValueError("Official lineup source changed")
                    interval = constraint["source_frame_interval"]
                    catalogue = official_lineups(
                        json.loads(official_path.read_text()), roster
                    )
                    lineup_count = complete_verified_lineups(
                        rows, identities, catalogue, roster, interval
                    )
                    manifest["identity_resolution"]["official_lineup_constraint"] = {
                        **constraint,
                        "completed_observations": lineup_count,
                    }
            elif clock_review:
                review_path = Path(clock_review).resolve(strict=True)
                review = json.loads(review_path.read_text())
                official_path = Path(review["official_source"]).resolve(strict=True)
                official = json.loads(official_path.read_text())
                if review.get("source_run") != source.name or review["period"] != 1:
                    raise ValueError(
                        "Clock synchronization does not match this source/Q1"
                    )
                # Synchronization is verified only over this five-second interval.
                # No initial player identities or reviewed detection decisions used.
                from pipeline.cuda.demo import verified_lineup

                verified_lineup(review, roster, official, device)
                start = review["seed_frame_index"]
                duration = (
                    (
                        tensor(5.0, device=device)
                        / (rows[1]["timestamp_s"] - rows[0]["timestamp_s"])
                    )
                    .round()
                    .long()
                )
                interval = (start, min(start + int(duration), frames))
                catalogue = official_lineups(official, roster)
                lineup_count = complete_verified_lineups(
                    rows, identities, catalogue, roster, interval
                )
                manifest["identity_resolution"]["official_lineup_constraint"] = {
                    "review_sha256": file_hash(review_path),
                    "official_sha256": file_hash(official_path),
                    "source_frame_interval": list(interval),
                    "outside_verified_interval": "disabled",
                    "completed_observations": lineup_count,
                }
            proposals = propagate_motion(rows, identities, device, progress)
            direct, assigned, teams, handoffs, _ = identities
            collisions = collision_frames(rows, direct, assigned, handoffs, device)
            identities = direct, assigned, teams, handoffs, collisions
            manifest["timings"]["identity_s"] = time.perf_counter() - tick
            del rows
            tick = time.perf_counter()
            metric = replay(
                output,
                roster,
                manifest["config"],
                run_id,
                progress,
                frames,
                identities,
                manifest["display_colors"],
            )
            stats = {
                **metric.result(),
                "status": "completed",
                "source": manifest["source"],
                "capabilities": manifest["capabilities"],
                "artifacts": {
                    "observations": "observations.jsonl",
                    "tracks": "tracks.jsonl",
                },
            }
            for player in stats["players"]:
                measured, tracked = tensor(
                    [player["measured_duration_s"], player["tracked_duration_s"]],
                    device=device,
                )
                if not bool((measured >= 0) & (measured <= tracked + 1e-9)):
                    raise ValueError("Invalid measured/tracked duration")
            write_json(output / "statistics.partial.json", export(stats))
            manifest["timings"]["statistics_s"] = time.perf_counter() - tick
            manifest.update(
                status="validating",
                spatial_handoffs=metric.spatial_handoffs,
                resolved_player_tracks=len(assigned),
            )
            for n, name in enumerate(
                (
                    "observations.jsonl",
                    "jersey_reads.jsonl",
                    "tracks.jsonl",
                    "statistics.json",
                    "team_votes.json",
                )
            ):
                progress.update("publication", n, 5)
                stem, suffix = name.rsplit(".", 1)
                (output / f"{stem}.partial.{suffix}").replace(output / name)
                manifest["artifacts"][name] = {"sha256": file_hash(output / name)}
            video = Video(manifest["source"]["path"], device)
            if video.source["sha256"] != manifest["source"]["sha256"] or not bool(
                (source_times == video.timestamps[:frames]).all()
            ):
                raise ValueError(
                    "Refined observations differ from source presentation timestamps"
                )
            expected = video.timestamps[:frames].clone()
            video.close()
            checked = health(
                output,
                manifest,
                device,
                progress,
                expected_timestamps=expected,
                expected_geometry_digest=baseline.get("immutable_geometry_sha256")
                if colors_only
                else None,
            )
            checked["status"] = "completed"
            write_json(output / "health.json", export(checked))
            counts = checked["counts"]
            quality = export(
                {
                    "scope": "named availability, not measured identification precision",
                    "baseline": baseline,
                    "final": counts,
                    "baseline_coverage_on_retained_detections": tensor(
                        baseline["baseline_named_on_retained_players"], device=device
                    )
                    / tensor(baseline["retained_players"], device=device).clamp_min(1),
                    "final_coverage": checked["ratios"]["named_identity_availability"],
                    "motion_proposed_observations": proposals,
                    "official_lineup_completed_observations": lineup_count,
                    "reviewed_initialization": False,
                    "manual_detection_decisions_applied": 0,
                    "cpu_tensor_calculations": 0,
                    "fallback_events": [],
                }
            )
            write_json(output / "identity-quality.json", quality)
            for name in (
                "health.json",
                "identity-quality.json",
                "recovery-audit.json",
                "removed_detections.jsonl",
            ):
                manifest["artifacts"][name] = {"sha256": file_hash(output / name)}
            if colors_only:
                # Detection filtering already happened in the source run.
                # This stage changes no geometry and removes no observations.
                for name in (
                    "color-audit.json",
                    "color-calibration.json",
                    "color-readings.jsonl",
                ):
                    manifest["artifacts"][name] = {"sha256": file_hash(output / name)}
            manifest["status"] = "completed"
            manifest["frame_integrity"]["health_verified"] = True
            manifest["cuda_guard"] = dict(guard.calls)
            manifest["identity_resolution"]["recovery_cuda_guard"] = audit["cuda_guard"]
            manifest["identity_resolution"].update(
                cpu_tensor_calculations=0, fallback_events=[]
            )
            manifest["timings"]["total_s"] = time.perf_counter() - started
            write_json(output / "run.json", export(manifest))
            progress.metrics(identity_quality=quality)
            progress.update("health", frames, frames, frames=frames, status="completed")
            print(json.dumps(quality, ensure_ascii=False, indent=2), flush=True)
            return output
    except BaseException as exc:
        manifest.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        write_json(output / "run.json", export(manifest))
        (output / "failure.txt").write_text(traceback.format_exc())
        progress.update(
            progress.state["stage"],
            progress.state["stage_frames"],
            progress.state["stage_total_frames"],
            status=manifest["status"],
            error=manifest["error"],
        )
        raise
    finally:
        progress.close()
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--step", default=2, type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--recovery", type=Path)
    parser.add_argument("--clock-review", type=Path)
    parser.add_argument("--colors-only", action="store_true")
    parser.add_argument("--color-features", type=Path)
    args = parser.parse_args()
    run(
        args.source,
        args.output,
        step=args.step,
        max_frames=args.max_frames,
        recovery_path=args.recovery,
        clock_review=args.clock_review,
        colors_only=args.colors_only,
        color_features=args.color_features,
    )


if __name__ == "__main__":
    main()
