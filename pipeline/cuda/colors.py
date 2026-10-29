"""Match-specific CUDA uniform calibration from independent jersey-number anchors.

Only missing torso features need NVDEC reads. Existing OCR, person masks and
geometry are immutable. Validation groups entire source tracks, never frames.
"""

import argparse
import fcntl
import hashlib
import json
from collections import defaultdict
from itertools import islice
from pathlib import Path

import torch

from pipeline.cuda.core import CudaOnly, Video, export, tensor
from pipeline.cuda.identity import split_episodes
from pipeline.cuda.integrity import geometry_bytes
from pipeline.cuda.refine import frame_colors
from pipeline.cuda.vision import bounds, torso
from pipeline.numeric import CudaCounter, cuda_numbers
from pipeline.progress import Progress
from training.common.files import write_json
from training.common.provenance import file_hash
from training.jersey.roster import Roster


def torso_features(image, person):
    """Same segmented neutral-pixel quantiles as the cache, without brightness gates."""
    device = str(image.device)
    payload = (person.get("segmentation") or {}).get("mask")
    if not payload or payload.get("encoding") != "roi_rle_c":
        return None, "insufficient_foreground"
    a, b, c, d = payload["roi"]
    if not (0 <= a < c <= image.shape[-1] and 0 <= b < d <= image.shape[-2]):
        raise ValueError("Saved mask escapes source image")
    counts = tensor(payload["counts"], device=device, dtype=torch.long)
    if not bool((counts >= 0).all() & (counts.sum() == (c - a) * (d - b))):
        raise ValueError("Invalid saved mask RLE lengths")
    mask = torch.repeat_interleave(
        torch.arange(len(counts), device=device).remainder(2).bool(),
        counts,
        output_size=(c - a) * (d - b),
    ).reshape(d - b, c - a)
    box = tensor(person["bbox"], device=device, dtype=torch.float32)
    pose = person["pose"]
    points = tensor(
        [p if p is not None else [0, 0] for p in pose["keypoints"]],
        device=device,
        dtype=torch.float32,
    )
    valid = tensor(pose["valid"], device=device, dtype=torch.bool)
    wh = box[2:] - box[:2]
    regions = [
        torso(box, points, valid),
        torch.cat(
            (
                box[:2] + wh * tensor([0.28, 0.23], device=device),
                box[:2] + wh * tensor([0.72, 0.49], device=device),
            )
        ),
    ]
    features = []
    for region in regions:
        _, (x1, y1, x2, y2) = bounds(region, image.shape[-2:])
        if x2 <= x1 or y2 <= y1:
            return None, "insufficient_foreground"
        foreground = torch.zeros((y2 - y1, x2 - x1), device=device, dtype=torch.bool)
        lo_x, lo_y, hi_x, hi_y = max(a, x1), max(b, y1), min(c, x2), min(d, y2)
        if hi_x > lo_x and hi_y > lo_y:
            foreground[lo_y - y1 : hi_y - y1, lo_x - x1 : hi_x - x1] = mask[
                lo_y - b : hi_y - b, lo_x - a : hi_x - a
            ]
        if not bool((foreground.sum() >= 24) & (foreground.float().mean() >= 0.6)):
            return None, "insufficient_foreground"
        rgb = image[:, y1:y2, x1:x2].float()[:, foreground]
        value = rgb.amax(0)
        neutral = (value - rgb.amin(0)) <= 55
        if not bool((neutral.sum() >= 24) & (neutral.float().mean() >= 0.7)):
            return None, "colored_or_skin_dominated"
        features.append(
            torch.quantile(
                value[neutral],
                tensor([0.25, 0.5, 0.75], device=device, dtype=torch.float32),
            )
        )
    return torch.stack(features), "usable"


def extract(source, output, frames, device, progress):
    """Reuse cached successful ROIs; compute only previously missing features."""
    source, output = Path(source).resolve(strict=True), Path(output)
    manifest = json.loads((source / "run.json").read_text())
    if manifest["status"] != "completed" or manifest.get("backend") != "native_cuda":
        raise ValueError("A completed native CUDA source is required")
    output.mkdir(parents=True, exist_ok=False)
    cached_path = source / "recovery/team_reads.jsonl"
    cache = iter(frame_colors(cached_path)) if cached_path.is_file() else iter(())
    cache_index, cached = next(cache, (None, {}))
    counts = CudaCounter(device=device)
    video = Video(manifest["source"]["path"], device)
    if video.source["sha256"] != manifest["source"]["sha256"]:
        raise ValueError("Video source changed")
    progress.update("color_features", 0, frames, total_frames=frames)
    seen = 0
    try:
        with (
            (source / "observations.jsonl").open() as original,
            (output / "features.jsonl").open("w") as features,
        ):
            for index, now, image in video.frames(frames):
                row = json.loads(next(original))
                if (
                    row["frame_index"] != index
                    or index != seen
                    or not bool(now == tensor(row["timestamp_s"], device=device))
                ):
                    raise ValueError("Color extraction source sequence/PTS mismatch")
                seen = index + 1
                if index % 2 == 0:
                    while cache_index is not None and cache_index < index:
                        cache_index, cached = next(cache, (None, {}))
                    local = cached if cache_index == index else {}
                    batch, kinds = [], []
                    for p in row["persons"]:
                        if p["role"] != "player" or p["track_id"] is None:
                            continue
                        old = local.get(p["track_id"], {}).get("audit", {})
                        if old.get("status") in (
                            "two_roi_agreement",
                            "roi_disagreement",
                        ):
                            values, status = old["value_quantiles"], "usable"
                            kinds.append("cached_features")
                        elif old.get("status") in (
                            "insufficient_foreground",
                            "colored_or_skin_dominated",
                        ):
                            values, status = None, old["status"]
                            kinds.append("cached_rejection")
                        else:
                            values, status = torso_features(image, p)
                            kinds.append("extracted_features")
                        kinds.append(status)
                        batch.append(
                            {
                                "frame_index": index,
                                "segment_id": row["segment_id"],
                                "track_id": p["track_id"],
                                "features": values,
                                "status": status,
                            }
                        )
                    counts.update(kinds)
                    for item in export(batch):
                        features.write(json.dumps(item, ensure_ascii=False) + "\n")
                if index % 150 == 0:
                    features.flush()
                    progress.update("color_features", seen, frames, frames=seen)
    finally:
        video.close()
    if seen != frames:
        raise ValueError("Incomplete NVDEC feature extraction")
    audit = export(
        {
            "source_run": str(source),
            "source_manifest_sha256": file_hash(source / "run.json"),
            "source_observations_sha256": file_hash(source / "observations.jsonl"),
            "source_video_sha256": manifest["source"]["sha256"],
            "frames": frames,
            "frame_step": 2,
            "counts": dict(counts),
            "cache_sha256": file_hash(cached_path) if cached_path.is_file() else None,
            "features_sha256": file_hash(output / "features.jsonl"),
            "numerical_device": device,
            "models_executed": [],
            "cpu_tensor_calculations": 0,
            "fallback_events": [],
        }
    )
    write_json(output / "features-audit.json", audit)
    progress.update("color_features", frames, frames, frames=frames)
    return audit


def anchor_labels(source, roster, device):
    """Independent team anchors: unique roster number and strong OCR agreement."""
    by_track = defaultdict(list)
    with (source / "jersey_reads.jsonl").open() as stream:
        for line in stream:
            read = json.loads(line)
            if read.get("number"):
                by_track[(read["segment_id"], read["track_id"])].append(read)
    labels = {}
    evidence = json.loads((source / "team_votes.json").read_text())["tracks"]
    for tag, track in evidence.items():
        segment, identity = (int(v) for v in tag.split(":"))
        reads = by_track.get((segment, identity), [])
        for episode in track["episodes"]:
            number = episode.get("number_majority")
            # A cropped "11" -> "1" can confidently claim the opposite team.
            # Two-digit unique labels give independent color supervision without
            # trusting those single-digit partial-number hypotheses.
            if not number or len(number) != 2:
                continue
            candidates = roster.identity(number)["candidates"]
            if len(candidates) != 1:
                continue
            local = [
                r
                for r in reads
                if episode["start_frame"] <= r["frame_index"] <= episode["end_frame"]
            ]
            if not local:
                continue
            confidence = tensor([r["confidence"] for r in local], device=device)
            same = tensor(
                [r["number"] == number for r in local], device=device, dtype=torch.bool
            )
            strong = (confidence >= 0.95) & same
            wide = tensor(
                [r.get("region") in ("bbox_upper", "bbox_center") for r in local],
                device=device,
                dtype=torch.bool,
            )
            accepted = confidence >= 0.8
            indices = tensor(
                [r["frame_index"] for r in local], device=device, dtype=torch.long
            )
            if not bool(
                (strong.sum() >= 3)
                & (strong & wide).any()
                & ((same & accepted).sum() >= 0.9 * accepted.sum())
            ):
                continue
            supporting = indices[strong]
            if not bool(supporting.amax() - supporting.amin() >= 4):
                continue
            # Labels belong to one episode; neither old team votes nor display
            # identities participate in the supervision.
            for frame in track["frames"]:
                if episode["start_frame"] <= frame <= episode["end_frame"]:
                    labels[(segment, identity, frame)] = candidates[0]["team_id"]
    return labels


class UniformCalibration:
    """Robust per-ROI class prototypes; no neural checkpoint is modified."""

    def __init__(self, centers, scales, margin, max_distance, teams):
        self.centers, self.scales = centers, scales
        self.margin, self.max_distance, self.teams = margin, max_distance, teams

    def predict(self, features):
        distance = (
            ((features[:, None] - self.centers[None]) / self.scales[None])
            .square()
            .mean(-1)
        )
        winner = distance.argmin(1)
        nearest = distance.gather(1, winner[:, None]).squeeze(1)
        separation = distance.gather(1, (1 - winner)[:, None]).squeeze(1) - nearest
        agree = winner[:, 0] == winner[:, 1]
        accepted = (
            agree
            & (nearest <= self.max_distance).all(1)
            & (separation >= self.margin).all(1)
        )
        return torch.where(accepted, winner[:, 0], -1), separation.amin(1)

    @classmethod
    def fit(cls, values, labels, groups, teams):
        """Hold out whole tracks and select an agreement-constrained threshold."""
        device = str(values.device)
        training = groups.remainder(5) != 0
        heldout = ~training
        centers, scales = [], []
        for j in range(2):
            selected = training & (labels == j)
            validation = heldout & (labels == j)
            if not bool((selected.sum() >= 50) & (validation.sum() >= 20)):
                raise ValueError(
                    "Insufficient independent OCR anchors for color calibration"
                )
            sample = values[selected]
            center = sample.median(0).values
            scale = ((sample - center).abs().median(0).values * 1.4826).clamp(8, 55)
            centers.append(center)
            scales.append(scale)
        centers, scales = torch.stack(centers), torch.stack(scales)
        if not bool((centers[1, :, 1] - centers[0, :, 1] >= 40).all()):
            raise ValueError(
                "Uniform classes are not separated in independent OCR anchors"
            )
        best, best_count, best_audit = None, tensor(-1, device=device), None
        for margin in (1.0, 2.0, 3.0, 4.0, 6.0):
            for limit in (4.0, 6.0, 9.0):
                model = cls(centers, scales, margin, limit, teams)
                predicted, _ = model.predict(values[heldout])
                truth = labels[heldout]
                used = predicted >= 0
                correct = used & (predicted == truth)
                by_team = torch.stack(
                    [
                        torch.stack(
                            (
                                (used & (predicted == j)).sum(),
                                (correct & (predicted == j)).sum(),
                            )
                        )
                        for j in range(2)
                    ]
                )
                good = (by_team[:, 0] >= 20).all() & (
                    by_team[:, 1] >= 0.98 * by_team[:, 0]
                ).all()
                if bool(good & (used.sum() > best_count)):
                    best, best_count = model, used.sum()
                    best_audit = {
                        "heldout_samples": heldout.sum(),
                        "heldout_accepted": used.sum(),
                        "heldout_correct_against_OCR": correct.sum(),
                        "heldout_by_predicted_team_accepted_correct": by_team,
                    }
        if best is None:
            raise ValueError("Color calibration fails the held-out OCR agreement gate")
        return best, {
            **best_audit,
            "training_samples": training.sum(),
            "centers": centers,
            "scales": scales,
            "margin": best.margin,
            "max_distance": best.max_distance,
            "teams": teams,
            "interpretation": "Agreement with independent unique-number OCR anchors; not human-labeled accuracy",
        }


def calibrate(source, feature_dir, roster, device, progress):
    progress.update("color_calibration")
    # Ordering is independently supplied by the reviewed uniform metadata;
    # learned brightness centers must confirm dark < light.
    teams = [
        t["team_id"]
        for t in sorted(
            roster.payload["teams"], key=lambda t: t["uniform_color"].lower()
        )
    ]
    anchors = anchor_labels(source, roster, device)
    samples, labels, groups, quotas = [], [], [], defaultdict(int)
    with (feature_dir / "features.jsonl").open() as stream:
        for line in stream:
            item = json.loads(line)
            key = (item["segment_id"], item["track_id"], item["frame_index"])
            label = anchors.get(key)
            group = (item["segment_id"], item["track_id"])
            if item["features"] is None or label is None or quotas[group] >= 12:
                continue
            quotas[group] += 1
            samples.append(item["features"])
            labels.append(teams.index(label))
            groups.append(item["track_id"])
    values = tensor(samples, device=device, dtype=torch.float32).reshape(-1, 2, 3)
    model, audit = UniformCalibration.fit(
        values,
        tensor(labels, device=device, dtype=torch.long),
        tensor(groups, device=device, dtype=torch.long),
        teams,
    )
    audit.update(
        independent_track_groups=len(quotas),
        features_audit_sha256=file_hash(feature_dir / "features-audit.json"),
        source_ocr_sha256=file_hash(source / "jersey_reads.jsonl"),
        anchor_policy="unique_two_digit_number_3_strong_reads_90pct_agreement_and_wide_bbox_support",
    )
    progress.update("color_calibration", 1, 1)
    return model, export(audit)


def color_consensus(frames, predictions, device):
    """Propagate only 3+ independent sampled frames, never across gaps/switches."""
    result = {}
    for group in split_episodes(frames, predictions, device):
        known = [f for f in group if f in predictions]
        if not known:
            continue
        values = tensor(
            [predictions[f] for f in known], device=device, dtype=torch.long
        )
        counts = torch.bincount(values, minlength=2)
        winner = int(counts.argmax())
        stable = bool((counts[winner] >= 3) & (counts[winner] >= 0.9 * counts.sum()))
        span = tensor(known, device=device, dtype=torch.long)
        stable = stable and bool(span.amax() - span.amin() >= 4)
        if stable:
            for frame in group:
                if frame not in predictions or predictions[frame] == winner:
                    result[frame] = winner
        else:
            for frame in known:
                result[frame] = predictions[frame]
    return result


def apply_colors(source, feature_dir, output, run_id, frames, device, progress):
    """Change only team evidence and run IDs; retain all source detections/OCR."""
    roster = Roster.load(source / "roster.json")
    (output / "removed_detections.jsonl").write_text("")
    feature_audit = json.loads((feature_dir / "features-audit.json").read_text())
    if (
        feature_audit["source_manifest_sha256"] != file_hash(source / "run.json")
        or feature_audit["source_observations_sha256"]
        != file_hash(source / "observations.jsonl")
        or feature_audit["frames"] < frames
        or feature_audit["features_sha256"] != file_hash(feature_dir / "features.jsonl")
    ):
        raise ValueError("Uniform features do not match the immutable source run")
    model, calibration = calibrate(source, feature_dir, roster, device, progress)
    write_json(output / "color-calibration.json", calibration)
    raw = defaultdict(dict)
    with (
        (feature_dir / "features.jsonl").open() as stream,
        (output / "color-readings.jsonl").open("w") as readings,
    ):
        iterator = (json.loads(line) for line in stream)
        while batch := list(islice(iterator, 2048)):
            selected = [
                r
                for r in batch
                if r["features"] is not None and r["frame_index"] < frames
            ]
            if not selected:
                continue
            predicted, margins = model.predict(
                tensor(
                    [r["features"] for r in selected],
                    device=device,
                    dtype=torch.float32,
                )
            )
            codes = predicted.tolist()
            for r, code, margin in zip(selected, codes, margins.tolist(), strict=True):
                if code >= 0:
                    raw[(r["segment_id"], r["track_id"])][r["frame_index"]] = code
                readings.write(
                    json.dumps(
                        {
                            "frame_index": r["frame_index"],
                            "segment_id": r["segment_id"],
                            "track_id": r["track_id"],
                            "team_id": model.teams[code] if code >= 0 else None,
                            "separation": margin,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    tracks = json.loads((source / "team_votes.json").read_text())["tracks"]
    smoothed = {}
    progress.update("refinement", 0, len(tracks) + frames)
    for n, (key, track) in enumerate(tracks.items()):
        tag = tuple(int(v) for v in key.split(":"))
        local_frames = [f for f in track["frames"] if f < frames]
        smoothed[tag] = color_consensus(local_frames, raw.get(tag, {}), device)
        if n % 100 == 0:
            progress.update("refinement", n, len(tracks) + frames)
    times, batch_flags = [], []
    names = (
        "source_players",
        "source_named_players",
        "retained_players",
        "baseline_named_on_retained_players",
        "previous_color_frames",
        "new_color_frames",
        "added_color_frames",
        "changed_color_frames",
        "cleared_color_frames",
        "sampled_color_frames",
        "temporally_supported_color_frames",
    )
    counts = CudaCounter(dict.fromkeys(names, 0), device=device)
    geometry_digest = hashlib.sha256()
    with (
        (source / "observations.jsonl").open() as original,
        (output / "observations.partial.jsonl").open("w") as target,
    ):
        for index, line in enumerate(islice(original, frames)):
            row = json.loads(line)
            if row["frame_index"] != index:
                raise ValueError("Source sequence mismatch during color replacement")
            geometry_digest.update(geometry_bytes(row))
            times.append(row["timestamp_s"])
            row["source_run_id"] = row["run_id"]
            row["run_id"] = run_id
            for n, p in enumerate(row["persons"]):
                p["parent_observation_id"] = p.get("observation_id")
                p["observation_id"] = f"{run_id}:{index}:{n}"
                if p["role"] != "player":
                    continue
                tag = (row["segment_id"], p["track_id"])
                code = smoothed.get(tag, {}).get(index)
                team = model.teams[code] if code is not None else None
                old = p.get("cuda_evidence") or {}
                previous = old.get("team_id")
                named = bool(p.get("resolved_identity"))
                sampled = index in raw.get(tag, {})
                batch_flags.append(
                    [
                        True,
                        named,
                        True,
                        named,
                        bool(previous),
                        team is not None,
                        not previous and team is not None,
                        bool(previous and team and previous != team),
                        bool(previous and team is None),
                        sampled,
                        team is not None and not sampled,
                    ]
                )
                p["cuda_evidence"] = {
                    **old,
                    "previous_team_id": previous,
                    "team_id": team,
                    "color_refinement": "CUDA_match_calibrated_temporal_v1",
                    "color_sampled": sampled,
                }
            target.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            if index % 256 == 255:
                if batch_flags:
                    totals = tensor(batch_flags, device=device, dtype=torch.bool).sum(0)
                    counts.update(dict(zip(names, totals.unbind())))
                batch_flags = []
                target.flush()
                progress.update(
                    "refinement", len(tracks) + index + 1, len(tracks) + frames
                )
    if len(times) != frames:
        raise ValueError("Incomplete source color replay")
    if batch_flags:
        totals = tensor(batch_flags, device=device, dtype=torch.bool).sum(0)
        counts.update(dict(zip(names, totals.unbind())))
    audit = {
        "method": "match_calibrated_segmented_torso_quantiles_and_temporal_consensus",
        "source_manifest_sha256": file_hash(source / "run.json"),
        "feature_audit": feature_audit,
        "calibration": calibration,
        "counts": export(dict(counts)),
        "ocr_recomputed": False,
        "perception_recomputed": False,
        "detections_removed": 0,
        "cpu_tensor_calculations": 0,
        "fallback_events": [],
    }
    write_json(output / "color-audit.json", audit)
    progress.update("refinement", len(tracks) + frames, len(tracks) + frames)
    return {
        **export(dict(counts)),
        "immutable_geometry_sha256": geometry_digest.hexdigest(),
    }, tensor(times, device=device)


def main():
    parser = argparse.ArgumentParser(
        description="Extract only missing uniform features on CUDA"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    manifest = json.loads((args.source / "run.json").read_text())
    frames = (
        manifest["frames"]
        if args.max_frames is None
        else min(manifest["frames"], args.max_frames)
    )
    device = manifest["config"]["jersey"]["device"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = (args.source.parent / ".identity-refinement.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("Another identity refinement is already running") from None
    progress = Progress(
        args.output.parent,
        "uniform-feature-extraction",
        phases=[("loading", 1), ("color_features", 99)],
    )
    guard = CudaOnly()
    try:
        with torch.inference_mode(), torch.device(device), cuda_numbers(device), guard:
            audit = extract(args.source, args.output, frames, device, progress)
            audit["cuda_guard"] = dict(guard.calls)
            write_json(args.output / "features-audit.json", audit)
        progress.update("color_features", frames, frames, status="completed")
    except BaseException as exc:
        progress.update("color_features", status="failed", error=str(exc))
        raise
    finally:
        progress.close()
        lock.close()


if __name__ == "__main__":
    main()
