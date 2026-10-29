"""Build a separate, auditable demo from reviewed seeds and CUDA tracking.

The reviewed seed is deliberately explicit: this is not a claim that the whole
match can be named automatically. Perception remains the saved model output.
"""

import argparse
import copy
import json
import shutil
import time
from pathlib import Path
from uuid import uuid4

import torch

from pipeline.cuda.core import CudaOnly, Video, export, tensor
from pipeline.cuda.distance import CudaDistance
from pipeline.cuda.lineup import LineupTracker
from pipeline.cuda.recovery import deduplicate_persons, smooth_referee_roles
from pipeline.cuda.run import health
from pipeline.numeric import CudaCounter, cuda_numbers
from pipeline.progress import Progress
from training.common.files import write_json
from training.common.provenance import file_hash
from training.jersey.roster import Roster


def clock_seconds(value, device):
    parts = value.split(":")
    values = [float(p) for p in parts] if len(parts) == 2 else [0, float(parts[0])]
    return (tensor(values, device=device) * tensor([60, 1], device=device)).sum()


def reviewed_detections(row, decisions):
    persons, dropped = [], []
    for p in row["persons"]:
        decision = decisions.get(p.get("observation_id"))
        if decision:
            action = decision["action"]
            if action == "duplicate":
                dropped.append({**p, "rejection_reason": "reviewed_duplicate_body"})
                continue
            if action == "role_referee":
                p = {
                    **p,
                    "source_role": p["role"],
                    "role": "referee",
                    "role_refinement": "reviewed_visible_referee",
                }
            elif action == "keep":
                p = {**p, "reviewed_keep": True}
            else:
                raise ValueError("Unsupported reviewed detection decision")
        persons.append(p)
    return {**row, "persons": persons}, dropped


def verified_lineup(review, roster, official, device):
    if review["period"] != 1:
        raise ValueError("Starting-lineup verification is restricted to period one")
    start, stop = (
        clock_seconds(review[name], device) for name in ("clock_start", "clock_end")
    )
    if not bool(start >= stop):
        raise ValueError("Invalid countdown-clock interval")
    substitutions = [
        p
        for p in official["plays"]
        if p["period"]["number"] == 1
        and (
            "enters the game" in p["text"].lower()
            or "substitution" in p["type"]["text"].lower()
        )
    ]
    if any(
        bool(clock_seconds(p["clock"]["displayValue"], device) >= stop)
        for p in substitutions
    ):
        raise ValueError(
            "A substitution precedes this demo; starting lineup is no longer sufficient"
        )
    names = {
        a["athlete"]["displayName"]
        for team in official["boxscore"]["players"]
        for group in team["statistics"]
        for a in group["athletes"]
        if a.get("starter")
    }
    players = [
        {
            "team_id": t["team_id"],
            "team_name": t["name"],
            "number": p["number"],
            "player_name": p["name"],
        }
        for t in roster.payload["teams"]
        for p in t["players"]
        if p["name"] in names and p.get("eligible", True)
    ]
    if len(players) != 10 or len(review["seeds"]) != 10:
        raise ValueError("A complete ten-player lineup and reviewed seed are required")
    lookup = {(p["team_id"], p["number"]): j for j, p in enumerate(players)}
    seeds = {
        r["source_track_id"]: lookup[(r["team_id"], r["number"])]
        for r in review["seeds"]
    }
    if len(seeds) != 10 or len(set(seeds.values())) != 10:
        raise ValueError("Reviewed seeds repeat an identity or a source track")
    return players, seeds


def build(source, recovery, review_path, video_path, output):
    source, recovery, review_path, video_path = (
        Path(p).resolve(strict=True)
        for p in (source, recovery, review_path, video_path)
    )
    original = json.loads((source / "run.json").read_text())
    if original["status"] != "completed":
        raise ValueError("Source run is incomplete")
    review = json.loads(review_path.read_text())
    audit = json.loads((recovery / "audit.json").read_text())
    if audit["source_manifest_sha256"] != file_hash(source / "run.json"):
        raise ValueError("Recovery evidence belongs to a different source manifest")
    if audit["start_frame"] != review["seed_frame_index"]:
        raise ValueError("Reviewed seed and recovery interval differ")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; CPU fallback forbidden")
    device = original["config"]["people"]["device"]
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    roster = Roster.load(source / "roster.json")
    official_path = Path(review["official_source"]).resolve(strict=True)
    official = json.loads(official_path.read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = copy.deepcopy(original)
    run_id = uuid4().hex
    manifest.update(
        run_id=run_id,
        status="running",
        artifacts={},
        timings={},
        pid=__import__("os").getpid(),
    )
    manifest.pop("resume", None)
    manifest.pop("frame_integrity", None)
    manifest["spatial_handoffs"] = []
    manifest["resolved_player_tracks"] = 10
    manifest["gpu_scheduling"] = "CUDA_saved_perception_refinement"
    root = Path(__file__).resolve().parents[2]
    manifest["code_sha256"] = {
        str(p.relative_to(root)): file_hash(p)
        for folder in ("pipeline", "training")
        for p in sorted((root / folder).rglob("*.py"))
        if "tests" not in p.parts
    }
    write_json(output / "run.json", manifest)
    shutil.copyfile(source / "roster.json", output / "roster.json")
    shutil.copyfile(review_path, output / "reviewed-seeds.json")
    shutil.copyfile(recovery / "audit.json", output / "recovery-audit.json")
    context = [json.loads(l) for l in (recovery / "observations.context.jsonl").open()]
    decisions = {d["observation_id"]: d for d in review.get("detection_reviews", [])}
    observations = {p["observation_id"] for r in context for p in r["persons"]}
    if not set(decisions) <= observations:
        raise ValueError("Reviewed detection is missing from source context")
    raw_votes = json.loads((source / "team_votes.json").read_text())["tracks"]
    source_teams = {
        int(k.split(":")[1]): v["team_id"]
        for k, v in raw_votes.items()
        if int(k.split(":")[0]) == context[0]["segment_id"]
    }
    progress = Progress(output, run_id)
    guard = CudaOnly()
    started = time.perf_counter()
    try:
        with torch.inference_mode(), torch.device(device), cuda_numbers(device), guard:
            players, seeds = verified_lineup(review, roster, official, device)
            video = Video(video_path, device)
            frames = video.source["presentation_frames"]
            if len(context) < frames:
                raise ValueError("Not enough source observations for demo video")
            decoded = 0
            for index, _, _ in video.frames():
                decoded = index + 1
            if decoded != frames:
                raise ValueError("Demo NVDEC coverage mismatch")
            manifest.update(
                source=video.source,
                frames=frames,
                segments=1,
                roster_sha256=roster.sha256,
            )
            manifest["identity_resolution"] = {
                "method": "reviewed_seed_CUDA_motion_and_official_lineup",
                "reviewed_initialization": True,
                "automatic_full_match_claim": False,
                "source_run": str(source),
                "source_manifest_sha256": file_hash(source / "run.json"),
                "source_video_sha256": original["source"]["sha256"],
                "source_start_frame": audit["start_frame"],
                "source_stop_frame": audit["start_frame"] + frames,
                "official_source_sha256": file_hash(official_path),
                "review_sha256": file_hash(review_path),
                "recovery_model": audit["model"],
                "perception_models_reused_from_source": True,
                "perception_source_code_sha256": original["code_sha256"],
            }
            minimum = manifest["config"]["jersey"]["min_detection_confidence"]
            seed_row, _ = deduplicate_persons(context[0], device, minimum)
            tracker = LineupTracker(seed_row["persons"], players, seeds, device)
            metric = CudaDistance(manifest["config"]["distance"], run_id)
            palette = manifest["display_colors"]
            stable_frames = {j: [] for j in range(10)}
            source_map = {}
            baseline = []
            assigned_counts = []
            totals = []
            duplicate_log = []
            progress.update(
                "perception", frames, frames, frames=frames, total_frames=frames
            )
            with (
                (output / "observations.jsonl").open("w") as observations,
                (output / "tracks.jsonl").open("w") as tracks,
            ):
                for i, row in enumerate(context[:frames]):
                    row, reviewed_drops = reviewed_detections(row, decisions)
                    row = smooth_referee_roles(
                        row,
                        context[i - 1] if i else None,
                        context[i + 1] if i + 1 < len(context) else None,
                        device,
                    )
                    row, duplicates = deduplicate_persons(row, device, minimum)
                    duplicates.extend(reviewed_drops)
                    duplicate_log.extend(
                        {
                            "source_frame_index": row["frame_index"],
                            "source_detection_index": p["detection_index"],
                            "role": p["role"],
                            "reason": p["rejection_reason"],
                        }
                        for p in duplicates
                    )
                    identities, evidence = tracker.update(
                        row["persons"], i, source_teams
                    )
                    baseline.append(
                        [
                            p.get("resolved_identity") is not None
                            for p in row["persons"]
                            if p["role"] == "player"
                        ]
                    )
                    totals.append(
                        len([p for p in row["persons"] if p["role"] == "player"])
                    )
                    assigned_counts.append(len(identities))
                    original_index = row["frame_index"]
                    row.update(
                        source_frame_index=original_index,
                        source_timestamp_s=row["timestamp_s"],
                        frame_index=i,
                        timestamp_s=video.timestamps[i],
                        run_id=run_id,
                        segment_id=0,
                        scene_cut=i == 0,
                    )
                    for n, p in enumerate(row["persons"]):
                        old = p["track_id"]
                        p["source_track_id"] = old
                        p["source_observation_id"] = p.get("observation_id")
                        p["observation_id"] = f"{run_id}:{i}:{n}"
                        p["detection_index"] = n
                        for name in (
                            "resolved_identity",
                            "jersey_number_suppressed",
                            "spatial_handoff",
                            "resolved_team_id",
                        ):
                            p.pop(name, None)
                        p["majority_jersey_number"] = None
                        if p["position_m"] is not None:
                            p["position_m"] = tensor(p["position_m"], device=device)
                        if p["role"] == "player":
                            p["display_color"] = palette["unassigned"]
                            p["track_id"] = None
                            if n in identities:
                                j = identities[n]
                                player = players[j]
                                p["track_id"] = j + 1
                                p["resolved_identity"] = {
                                    **player,
                                    "resolution_method": "reviewed_seed_CUDA_motion_and_official_lineup",
                                }
                                p["majority_jersey_number"] = player["number"]
                                p["resolved_team_id"] = player["team_id"]
                                p["display_color"] = palette["teams"][player["team_id"]]
                                p["identity_tracking"] = evidence[n]
                                stable_frames[j].append(i)
                                source_map[(original_index, old)] = j
                        else:
                            p["track_id"] = None if old is None else old + 1000000
                    for item in export(metric.update(row)):
                        tracks.write(json.dumps(item, ensure_ascii=False) + "\n")
                    observations.write(
                        json.dumps(export(row), ensure_ascii=False, allow_nan=False)
                        + "\n"
                    )
                    if i % 30 == 0:
                        progress.update("statistics", i + 1, frames)
            reads = []
            numbers = {j: CudaCounter(device=device) for j in range(10)}
            for line in (recovery / "jersey_reads.jsonl").open():
                r = json.loads(line)
                j = source_map.get((r["frame_index"], r["track_id"]))
                if j is None:
                    continue
                r.update(
                    source_frame_index=r["frame_index"],
                    source_track_id=r["track_id"],
                    frame_index=r["frame_index"] - audit["start_frame"],
                    track_id=j + 1,
                    segment_id=0,
                    run_id=run_id,
                )
                r["processed_frame"] = r["frame_index"]
                r["timestamp_seconds"] = video.timestamps[r["frame_index"]]
                if r["status"] == "vote":
                    numbers[j][r["number"]] = numbers[j][r["number"]] + 1
                reads.append(r)
            with (output / "jersey_reads.jsonl").open("w") as stream:
                for r in export(reads):
                    stream.write(json.dumps(r, ensure_ascii=False) + "\n")
            votes = {
                "schema_version": 1,
                "evidence_version": 6,
                "roster_sha256": roster.sha256,
                "tracks": {
                    f"0:{j + 1}": {
                        "frames": ff,
                        "team_id": players[j]["team_id"],
                        "number_majority": players[j]["number"],
                        "number_votes": dict(numbers[j]),
                        "episodes": [],
                        "reviewed_seed": True,
                    }
                    for j, ff in stable_frames.items()
                },
            }
            write_json(output / "team_votes.json", export(votes))
            stats = {
                **metric.result(),
                "status": "completed",
                "source": video.source,
                "capabilities": manifest["capabilities"],
                "artifacts": {
                    "observations": "observations.jsonl",
                    "tracks": "tracks.jsonl",
                },
            }
            write_json(output / "statistics.json", export(stats))
            count = tensor(totals, device=device)
            named = tensor(assigned_counts, device=device)
            base = tensor(
                [sum_ for flags in baseline for sum_ in flags],
                device=device,
                dtype=torch.bool,
            ).sum()
            coverage = export(
                {
                    "frames": frames,
                    "player_observations": count.sum(),
                    "named_player_observations": named.sum(),
                    "baseline_named_player_observations": base,
                    "named_fraction": named.sum() / count.sum(),
                    "baseline_named_fraction": base / count.sum(),
                    "frames_all_detections_named": (named == count).sum(),
                    "per_frame_players": count,
                    "per_frame_named": named,
                    "reviewed_initialization": True,
                    "duplicate_detections_removed": len(duplicate_log),
                    "automatic_full_match_claim": False,
                }
            )
            write_json(output / "demo-quality.json", coverage)
            write_json(output / "duplicate-detections.json", duplicate_log)
            for name in (
                "observations.jsonl",
                "tracks.jsonl",
                "jersey_reads.jsonl",
                "statistics.json",
                "team_votes.json",
                "reviewed-seeds.json",
                "recovery-audit.json",
                "demo-quality.json",
                "duplicate-detections.json",
            ):
                manifest["artifacts"][name] = {"sha256": file_hash(output / name)}
            manifest["status"] = "validating"
            write_json(output / "run.json", export(manifest))
            checked = health(
                output, manifest, device, progress, expected_timestamps=video.timestamps
            )
            checked["status"] = "completed"
            write_json(output / "health.json", export(checked))
            manifest["artifacts"]["health.json"] = {
                "sha256": file_hash(output / "health.json")
            }
            manifest.update(
                status="completed",
                frame_integrity={
                    "target_frames": frames,
                    "perception_complete": True,
                    "health_verified": True,
                },
                cuda_guard={
                    **dict(guard.calls),
                    "cpu_tensor_calculations": 0,
                    "fallback_events": [],
                },
                timings={"total_s": time.perf_counter() - started},
            )
            write_json(output / "run.json", export(manifest))
            progress.update("health", frames, frames, status="completed")
            video.close()
            return coverage
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "run.json", export(manifest))
        progress.update(
            progress.state["stage"], status="failed", error=manifest["error"]
        )
        raise
    finally:
        progress.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("source", "recovery", "review", "video", "output"):
        p.add_argument("--" + flag, required=True, type=Path)
    a = p.parse_args()
    print(
        json.dumps(build(a.source, a.recovery, a.review, a.video, a.output), indent=2)
    )


if __name__ == "__main__":
    main()
