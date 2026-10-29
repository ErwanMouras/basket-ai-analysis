"""Re-read saved detections on NVDEC/CUDA without rerunning detector models.

Every source frame contributes at most one OCR vote, regardless of crop count.
Sampling continues through the interval instead of stopping at the first vote.
"""

import argparse
import json
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import torch

from pipeline.cuda.core import CudaOnly, Video, export, tensor
from pipeline.cuda.graphs import CapturedCall
from pipeline.cuda.models import Jersey
from pipeline.cuda.vision import bounds, candidate, dominant_color, torso
from pipeline.numeric import cuda_numbers
from training.common.files import write_json
from training.common.provenance import file_hash
from training.jersey.roster import Roster


def saved_tensors(row, device):
    """Decode compact instance RLE and saved pose directly into CUDA tensors."""
    persons = row["persons"]
    boxes = tensor(
        [
            p["bbox"] + [p["confidence"], 2 if p["role"] == "player" else 3]
            for p in persons
        ],
        device=device,
        dtype=torch.float32,
    ).reshape(-1, 6)
    points = tensor(
        [
            [point if point is not None else [0, 0] for point in p["pose"]["keypoints"]]
            for p in persons
        ],
        device=device,
        dtype=torch.float32,
    ).reshape(-1, 17, 2)
    valid = tensor(
        [p["pose"]["valid"] for p in persons], device=device, dtype=torch.bool
    ).reshape(-1, 17)
    masks = torch.zeros(
        (len(persons), row["height"], row["width"]), device=device, dtype=torch.bool
    )
    for i, p in enumerate(persons):
        payload = (p.get("segmentation") or {}).get("mask")
        if not payload:
            continue
        if payload["encoding"] != "roi_rle_c":
            raise ValueError("Unsupported saved mask encoding")
        a, b, c, d = payload["roi"]
        if not (0 <= a < c <= row["width"] and 0 <= b < d <= row["height"]):
            raise ValueError("Saved mask escapes source frame")
        counts = tensor(payload["counts"], device=device, dtype=torch.long)
        if not bool((counts >= 0).all() & (counts.sum() == (c - a) * (d - b))):
            raise ValueError("Invalid saved mask RLE lengths")
        runs = torch.arange(len(counts), device=device).remainder(2).bool()
        masks[i, b:d, a:c] = torch.repeat_interleave(
            runs, counts, output_size=(c - a) * (d - b)
        ).reshape(d - b, c - a)
    return boxes, points, valid, masks


def decode_number(tokens):
    """Token-to-label conversion is categorical metadata, never OCR arithmetic."""
    alphabet = {i: str(i - 1) for i in range(1, 11)}
    return (
        alphabet.get(tokens[0], "")
        if tokens[1] == 0
        else (alphabet.get(tokens[0], "") + alphabet.get(tokens[1], ""))
    )


def read_rank(read, device):
    """An out-of-roster prediction cannot erase a valid source vote."""
    confidence = read["confidence"]
    if not isinstance(confidence, torch.Tensor):
        confidence = tensor(confidence, device=device)
    valid = read.get("number") is not None
    return confidence + (2 if valid else 0)


def uniform_team(image, box, points, valid, mask, tags):
    """Additional achromatic-uniform evidence, with two agreeing torso ROIs.

    Used only to fill missing source color evidence. Colored/skin-dominated,
    mixed, unsegmented and middle-grey torsos remain unknown.
    """
    if not {"dark", "light"} <= set(tags):
        return None, {"status": "unsupported_uniform_palette"}
    wh = box[2:] - box[:2]
    device = str(image.device)
    regions = [
        torso(box, points, valid),
        torch.cat(
            (
                box[:2] + wh * tensor([0.28, 0.23], device=device),
                box[:2] + wh * tensor([0.72, 0.49], device=device),
            )
        ),
    ]
    votes, summaries = [], []
    for region in regions:
        _, (a, b, c, d) = bounds(region, image.shape[-2:])
        foreground = mask[b:d, a:c]
        if (
            c <= a
            or d <= b
            or not bool((foreground.sum() >= 24) & (foreground.float().mean() >= 0.6))
        ):
            return None, {"status": "insufficient_foreground"}
        rgb = image[:, b:d, a:c].float()[:, foreground]
        value = rgb.amax(0)
        chroma = value - rgb.amin(0)
        neutral = chroma <= 55
        if not bool((neutral.sum() >= 24) & (neutral.float().mean() >= 0.7)):
            return None, {"status": "colored_or_skin_dominated"}
        quantiles = torch.quantile(
            value[neutral],
            tensor([0.25, 0.5, 0.75], device=device, dtype=torch.float32),
        )
        if bool((quantiles[1] <= 90) & (quantiles[2] <= 125)):
            votes.append("dark")
        elif bool((quantiles[1] >= 180) & (quantiles[0] >= 150)):
            votes.append("light")
        else:
            return None, {"status": "ambiguous_brightness"}
        summaries.append(quantiles)
    agreed = len(set(votes)) == 1
    return (tags[votes[0]] if agreed else None), {
        "status": "two_roi_agreement" if agreed else "roi_disagreement",
        "value_quantiles": summaries,
    }


def deduplicate_persons(row, device, min_confidence=None):
    """Remove repeated segmentations of the same visible body, across roles.

    Distinct overlapping people have distinct visible masks. A repeated mask
    plus an overlapping box is stronger duplicate evidence than box IoU alone.
    """
    from pipeline.cuda.core import box_iou

    rejected = []
    if min_confidence is not None:
        confidence = tensor([p["confidence"] for p in row["persons"]], device=device)
        accepted = (confidence >= min_confidence).tolist()
        accepted = [
            keep or bool(p.get("reviewed_keep"))
            for p, keep in zip(row["persons"], accepted, strict=True)
        ]
        rejected = [
            {**p, "rejection_reason": "below_configured_detection_confidence"}
            for p, keep in zip(row["persons"], accepted, strict=True)
            if not keep
        ]
        row = {
            **row,
            "persons": [
                p for p, keep in zip(row["persons"], accepted, strict=True) if keep
            ],
        }
    boxes, points, valid, masks = saved_tensors(row, device)
    if len(boxes) < 2:
        return row, rejected
    flat = masks.flatten(1).float()
    area = flat.sum(1)
    intersection = flat @ flat.T
    mask_iou = intersection / (area[:, None] + area[None] - intersection).clamp_min(1)
    box_overlap = box_iou(boxes[:, :4], boxes[:, :4])
    common = valid[:, None] & valid[None]
    pose_error = (
        torch.linalg.vector_norm(points[:, None] - points[None], dim=-1) * common
    ).sum(2) / common.sum(2).clamp_min(1)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp_min(1)
    pose_error /= torch.maximum(heights[:, None], heights[None])
    same_pose = (pose_error <= 0.09) & (common.sum(2) >= 8)
    containment = intersection / torch.minimum(area[:, None], area[None]).clamp_min(1)
    duplicated = ((mask_iou >= 0.8) & (box_overlap >= 0.4)) | (
        same_pose
        & (
            ((mask_iou >= 0.5) & (box_overlap >= 0.5))
            | ((containment >= 0.9) & (box_overlap >= 0.3))
        )
    )
    keep = torch.ones(len(boxes), device=device, dtype=torch.bool)
    for i in boxes[:, 4].argsort(descending=True).tolist():
        if bool(keep[i]):
            keep &= ~duplicated[i]
            keep[i] = True
    chosen = keep.tolist()
    chosen = [
        accepted or bool(p.get("reviewed_keep"))
        for p, accepted in zip(row["persons"], chosen, strict=True)
    ]
    dropped = [
        {**p, "rejection_reason": "same_visible_mask_and_overlapping_box"}
        for p, accepted in zip(row["persons"], chosen, strict=True)
        if not accepted
    ]
    return {
        **row,
        "persons": [
            p for p, accepted in zip(row["persons"], chosen, strict=True) if accepted
        ],
    }, rejected + dropped


def smooth_referee_roles(row, before, after, device):
    """Correct an isolated role flip only with matching referees on both sides."""
    from pipeline.cuda.core import box_iou

    if before is None or after is None:
        return row
    current = [p for p in row["persons"] if p["role"] == "player"]
    if not current:
        return row
    boxes = tensor([p["bbox"] for p in current], device=device)
    supported = torch.ones(len(current), device=device, dtype=torch.bool)
    for neighbor in (before, after):
        referees = [p for p in neighbor["persons"] if p["role"] == "referee"]
        if not referees:
            return row
        matches = box_iou(boxes, tensor([p["bbox"] for p in referees], device=device))
        confidence = tensor([p["confidence"] for p in referees], device=device)
        supported &= ((matches >= 0.6) & (confidence[None] >= 0.6)).any(1)
    flags = iter(supported.tolist())
    persons = []
    for p in row["persons"]:
        if p["role"] == "player" and next(flags):
            p = {
                **p,
                "source_role": "player",
                "role": "referee",
                "role_refinement": "matching_referee_before_and_after",
            }
        persons.append(p)
    return {**row, "persons": persons}


def recover(
    run, output, start, stop, *, step=3, progress=None, context=True, candidates=True
):
    run, output = Path(run).resolve(strict=True), Path(output)
    manifest = json.loads((run / "run.json").read_text())
    if manifest["status"] != "completed" or manifest.get("backend") != "native_cuda":
        raise ValueError("Recovery requires a completed native CUDA run")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; CPU fallback forbidden")
    if not 0 <= start < stop <= manifest["frames"] or step < 1:
        raise ValueError("Invalid recovery interval")
    device = manifest["config"]["jersey"]["device"]
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=False)
    roster = Roster.load(run / "roster.json")
    previous = defaultdict(dict)
    with (run / "jersey_reads.jsonl").open() as stream:
        for line in stream:
            read = json.loads(line)
            if start <= read["frame_index"] < stop:
                frame, track = read["frame_index"], read["track_id"]
                if track in previous[frame]:
                    raise ValueError("Duplicate source OCR vote")
                previous[frame][track] = read
    guard = CudaOnly()
    started = time.perf_counter()
    config = manifest["config"]["jersey"]
    crops_count = 0
    # Checkpoint deserialization uses host staging buffers for file I/O, just
    # as Models.__init__ does in the native runner. Parameter math defaults CUDA.
    with torch.inference_mode(), torch.device(device), cuda_numbers(device):
        reader = Jersey(config)
    if any(not parameter.is_cuda for parameter in reader.model.parameters()):
        raise RuntimeError("PARSeq parameter remained on CPU")
    with torch.inference_mode(), torch.device(device), cuda_numbers(device), guard:
        reader.captured = CapturedCall(
            reader._read_eager,
            torch.zeros((config["batch_size"], 3, 32, 128), device=device),
        )
        video = Video(manifest["source"]["path"], device)
        if video.source["sha256"] != manifest["source"]["sha256"]:
            raise ValueError("Source video changed")
        accepted = torch.zeros((), device=device, dtype=torch.long)
        vote_count = 0
        seen = start
        if progress:
            progress.update(
                "recovery", 0, stop - start, frames=0, total_frames=stop - start
            )
        color_tags = {}
        for team in roster.payload["teams"]:
            rgb = tensor(
                [int(team["uniform_color"][i : i + 2], 16) for i in (1, 3, 5)],
                device=device,
                dtype=torch.float32,
            )
            signature = dominant_color(rgb[:, None, None].expand(3, 32, 128))
            if signature and signature[0] in ("dark", "light"):
                color_tags[signature[0]] = team["team_id"]
        with ExitStack() as stack:
            source_rows = stack.enter_context((run / "observations.jsonl").open())
            for _ in range(start):
                next(source_rows)
            debug = (
                stack.enter_context((output / "candidates.jsonl").open("w"))
                if candidates
                else None
            )
            context_stream = (
                stack.enter_context((output / "observations.context.jsonl").open("w"))
                if context
                else None
            )
            merged = stack.enter_context((output / "jersey_reads.jsonl").open("w"))
            colors = stack.enter_context((output / "team_reads.jsonl").open("w"))
            for index, now, image in video.frames(stop, start=start):
                row = json.loads(next(source_rows))
                if index != seen or row["frame_index"] != index:
                    raise ValueError("Source observation/decoder sequence mismatch")
                if not bool(now == tensor(row["timestamp_s"], device=device)):
                    raise ValueError("Source observation/decoder timestamp mismatch")
                seen = index + 1
                if context_stream:
                    context_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                recovered = {}
                sampled = not (index - start) % step
                if sampled:
                    boxes, points, valid, masks = saved_tensors(row, device)
                crops, targets = [], []
                for n, person in enumerate(row["persons"] if sampled else []):
                    if person["role"] != "player" or person["track_id"] is None:
                        continue
                    team, audit = uniform_team(
                        image, boxes[n, :4], points[n], valid[n], masks[n], color_tags
                    )
                    colors.write(
                        json.dumps(
                            export(
                                {
                                    "frame_index": index,
                                    "track_id": person["track_id"],
                                    "segment_id": row["segment_id"],
                                    "team_id": team,
                                    "audit": audit,
                                }
                            ),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    for variant in (
                        "center",
                        "upper",
                        "lower",
                        "bbox_upper",
                        "bbox_center",
                        "bbox_lower",
                    ):
                        crop = candidate(
                            image, boxes, n, points[n], valid[n], masks, config, variant
                        )
                        if crop is not None:
                            targets.append((person, crop))
                            crops.append(crop["image"])
                if crops:
                    tokens, scores = reader.read(torch.stack(crops))
                    labels = tokens.tolist()
                    for j, (person, crop) in enumerate(targets):
                        number = decode_number(labels[j])
                        allowed = roster.allows(number)
                        read = {
                            "schema_version": 1,
                            "run_id": manifest["run_id"],
                            "decoder": "digits_only_v4",
                            "sampling": "CUDA_interval_recheck_v1",
                            "segment_id": row["segment_id"],
                            "track_id": person["track_id"],
                            "frame_index": index,
                            "processed_frame": index,
                            "timestamp_seconds": now,
                            "crop_bbox": crop["bbox"],
                            "region": crop["region"],
                            "quality": crop["quality"],
                            "mask_audit": crop["mask_audit"],
                            "text": number,
                            "raw_number": number if allowed else None,
                            "number": number if allowed else None,
                            "confidence": scores[j],
                            "status": "vote"
                            if allowed and bool(scores[j] >= 0.8)
                            else "rejected",
                        }
                        key = person["track_id"]
                        if debug:
                            debug.write(
                                json.dumps(export(read), ensure_ascii=False) + "\n"
                            )
                        if key not in recovered or bool(
                            read_rank(read, device) > read_rank(recovered[key], device)
                        ):
                            recovered[key] = read
                    crops_count += len(targets)
                old_reads = previous.pop(index, {})
                for key, read in recovered.items():
                    old = old_reads.get(key)
                    if old is None or bool(
                        read_rank(read, device) > read_rank(old, device)
                    ):
                        old_reads[key] = read
                packed = export(list(old_reads.values()))
                if packed:
                    confidence = tensor(
                        [r["confidence"] for r in packed], device=device
                    )
                    allowed = tensor(
                        [r["number"] is not None for r in packed],
                        device=device,
                        dtype=torch.bool,
                    )
                    accepted += ((confidence >= 0.8) & allowed).sum()
                    vote_count += len(packed)
                    for read in packed:
                        merged.write(json.dumps(read, ensure_ascii=False) + "\n")
                if (index - start) % 30 == 0:
                    print(
                        f"GPU recheck {index - start}/{stop - start}: {crops_count} crops",
                        flush=True,
                    )
                    merged.flush()
                    colors.flush()
                    if progress:
                        progress.update(
                            "recovery", seen - start, stop - start, frames=seen - start
                        )
                        elapsed = time.perf_counter() - started
                        fps = (seen - start) / elapsed
                        progress.metrics(
                            recovery_crops=crops_count,
                            recovery_votes=vote_count,
                            recovery_fps=fps,
                            recovery_remaining_s=(stop - seen) / fps,
                        )
            video.close()
        if seen != stop or previous:
            raise ValueError("Incomplete source recovery interval")
        if progress:
            progress.update("recovery", stop - start, stop - start, frames=stop - start)
        summary = export(
            {
                "source_run": str(run),
                "source_manifest_sha256": file_hash(run / "run.json"),
                "start_frame": start,
                "stop_frame": stop,
                "step": step,
                "crops": crops_count,
                "source_frame_votes": vote_count,
                "accepted_votes": accepted,
                "model": reader.provenance,
                "numerical_device": device,
                "cuda_guard": {
                    **dict(guard.calls),
                    "cpu_tensor_calculations": 0,
                    "fallback_events": [],
                },
                "elapsed_s": time.perf_counter() - started,
            }
        )
    write_json(output / "audit.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--stop", required=True, type=int)
    parser.add_argument("--step", default=3, type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            recover(args.run, args.output, args.start, args.stop, step=args.step),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
