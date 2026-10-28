"""Whole-run CUDA votes and shared conservative spatial identity handoffs."""

import json
from collections import defaultdict
from itertools import islice, pairwise
from pathlib import Path

import torch
import torch.nn.functional as F

from pipeline.cuda.core import export, tensor
from pipeline.cuda.distance import CudaDistance
from pipeline.cuda.vision import majority
from pipeline.numeric import CudaCounter
from training.common.files import write_json


def _split_episodes_reference(frames, colors, device):
    if not frames:
        return []
    tags = list(dict.fromkeys(color for color in colors.values() if color is not None))
    mapping = {tag: index for index, tag in enumerate(tags)}
    codes = tensor(
        [mapping.get(colors.get(frame), -1) for frame in frames],
        device=device,
        dtype=torch.long,
    )
    windows = F.pad(codes, (0, 4), value=-1).unfold(0, 5, 1)
    same = (windows == codes[:, None]).sum(1)
    other = ((windows >= 0) & (windows != codes[:, None])).sum(1)
    stable = (codes >= 0) & (same >= 3) & (other == 0)
    gaps = tensor(frames, device=device, dtype=torch.long).diff() != 1
    known = torch.nonzero(codes >= 0).flatten()
    current = int(codes[known[0]]) if len(known) else -1
    episodes = []
    start = 0
    for index in range(1, len(frames)):
        if bool(gaps[index - 1]):
            episodes.append(frames[start:index])
            start = index
            later = torch.nonzero(codes[index:] >= 0).flatten()
            current = int(codes[index + int(later[0])]) if len(later) else -1
            continue
        color = int(codes[index])
        if color >= 0 and current >= 0 and color != current and bool(stable[index]):
            episodes.append(frames[start:index])
            start = index
            current = color
        elif color >= 0 and current < 0:
            current = color
    episodes.append(frames[start:])
    return episodes


def split_episodes(frames, colors, device):
    from pipeline.cuda.kernels import episode_boundaries

    if not frames:
        return []
    tags = list(dict.fromkeys(color for color in colors.values() if color is not None))
    mapping = {tag: index for index, tag in enumerate(tags)}
    codes = tensor(
        [mapping.get(colors.get(frame), -1) for frame in frames],
        device=device,
        dtype=torch.long,
    )
    windows = F.pad(codes, (0, 4), value=-1).unfold(0, 5, 1)
    same = (windows == codes[:, None]).sum(1)
    other = ((windows >= 0) & (windows != codes[:, None])).sum(1)
    stable = (codes >= 0) & (same >= 3) & (other == 0)
    gaps = tensor(frames, device=device, dtype=torch.long).diff() != 1
    starts = episode_boundaries(codes, stable, gaps) + [len(frames)]
    return [frames[a:b] for a, b in pairwise(starts)]


def _minimal_batch(batch, device):
    """Pack JSON geometry in one transfer; only metadata is handled on host."""
    values, locations = [], []
    for row in batch:
        timestamp = len(values)
        values.append(row["timestamp_s"])
        persons = []
        for person in row["persons"]:
            box = len(values)
            values.extend(person["bbox"])
            point = None
            if person["position_m"] is not None:
                point = len(values)
                values.extend(person["position_m"])
            persons.append((person, box, point))
        locations.append((row, timestamp, persons))
    packed = tensor(values, device=device)
    return [
        {
            "frame_index": row["frame_index"],
            "segment_id": row["segment_id"],
            "timestamp_s": packed[timestamp],
            "scene_cut": row["scene_cut"],
            "included": row["included"],
            "persons": [
                {
                    "role": person["role"],
                    "track_id": person["track_id"],
                    "bbox": packed[box : box + 4],
                    "position_m": packed[point : point + 2]
                    if point is not None
                    else None,
                }
                for person, box, point in persons
            ],
        }
        for row, timestamp, persons in locations
    ]


def _minimal(row, device):
    return {
        "frame_index": row["frame_index"],
        "segment_id": row["segment_id"],
        "timestamp_s": tensor(row["timestamp_s"], device=device),
        "scene_cut": row["scene_cut"],
        "included": row["included"],
        "persons": [
            {
                "role": p["role"],
                "track_id": p["track_id"],
                "bbox": tensor(p["bbox"], device=device),
                "position_m": tensor(p["position_m"], device=device)
                if p["position_m"] is not None
                else None,
            }
            for p in row["persons"]
        ],
    }


def _handoff_index(handoffs):
    result = defaultdict(list)
    for (segment, track, start), item in handoffs.items():
        result[(segment, track)].append((start, item))
    return result


def _select(key, index, direct, assigned, by_track):
    player = direct.get((*key, index))
    if player is None:
        candidate = assigned.get(key)
        if candidate and index in candidate.get("_valid_frames", ()):
            player = candidate
    link = next(
        (
            item
            for start, item in by_track.get(key, ())
            if start <= index <= item["end_frame"]
        ),
        None,
    )
    if player is None and link:
        player = {**link["player"], "resolution_method": "spatial_handoff"}
    return player, link


def _collision_frames_reference(
    rows, direct, assigned, handoffs, device, progress=None
):
    by_track = _handoff_index(handoffs)
    collisions = set()
    if progress:
        progress.update("identity_collisions", 0, len(rows))
    for row in rows.values():
        index = row["frame_index"]
        selected = []
        for person in row["persons"]:
            if person["role"] != "player" or person["track_id"] is None:
                continue
            key = (row["segment_id"], person["track_id"])
            player, _ = _select(key, index, direct, assigned, by_track)
            if player:
                selected.append((key, (player["team_id"], player["number"])))
        counts = CudaCounter((identity for _, identity in selected), device=device)
        for key, identity in selected:
            if bool(counts[identity] > 1):
                collisions.add((index, key))
        if progress and (index + 1) % 300 == 0:
            progress.update("identity_collisions", index + 1, len(rows))
    if progress:
        progress.update("identity_collisions", len(rows), len(rows))
    return collisions


def collision_frames(rows, direct, assigned, handoffs, device, progress=None):
    by_track = _handoff_index(handoffs)
    collisions = set()
    iterator = iter(rows.values())
    completed = 0
    if progress:
        progress.update("identity_collisions", 0, len(rows))
    while batch := list(islice(iterator, 512)):
        selected, identities, encoded = [], {}, []
        for row in batch:
            keys, codes = [], []
            for person in row["persons"]:
                if person["role"] != "player" or person["track_id"] is None:
                    continue
                key = (row["segment_id"], person["track_id"])
                player, _ = _select(key, row["frame_index"], direct, assigned, by_track)
                if player:
                    identity = (player["team_id"], player["number"])
                    code = identities.setdefault(identity, len(identities))
                    keys.append(key)
                    codes.append(code)
            selected.append(keys)
            encoded.append(codes)
        width = max(map(len, encoded), default=0)
        if width:
            codes = tensor(
                [codes + [-1] * (width - len(codes)) for codes in encoded],
                device=device,
                dtype=torch.long,
            )
            repeated = (
                (codes[:, :, None] == codes[:, None, :]) & (codes[:, :, None] >= 0)
            ).sum(2) > 1
            for frame, person in torch.nonzero(repeated).tolist():
                collisions.add((batch[frame]["frame_index"], selected[frame][person]))
        completed += len(batch)
        if progress:
            progress.update("identity_collisions", completed, len(rows))
    return collisions


def resolve(output, roster, device, progress, frames):
    """Load compact evidence; numerical tallies and handoff geometry stay CUDA."""
    output = Path(output)
    rows = {}
    tracks = defaultdict(list)
    reads = defaultdict(list)
    progress.update("identity_colors", 0, frames, frames=frames)
    with (output / "observations.partial.jsonl").open() as stream:
        iterator = (json.loads(line) for line in stream)
        while batch := list(islice(iterator, 512)):
            compact = _minimal_batch(batch, device)
            rows.update((row["frame_index"], row) for row in compact)
            for row in batch:
                frame = row["frame_index"]
                for person in row["persons"]:
                    if person["role"] == "player" and person["track_id"] is not None:
                        tracks[(row["segment_id"], person["track_id"])].append(
                            (frame, person.get("cuda_evidence") or {})
                        )
            progress.update("identity_colors", batch[-1]["frame_index"] + 1, frames)
    progress.update("identity_colors", frames, frames)
    progress.update("identity_ocr", 0, frames)
    all_reads = []
    with (output / "jersey_reads.partial.jsonl").open() as stream:
        iterator = (json.loads(line) for line in stream)
        while batch := list(islice(iterator, 2048)):
            confidence = tensor(
                [read.get("confidence") or 0 for read in batch], device=device
            )
            accepted = (confidence >= 0.8).tolist()
            for index, read in enumerate(batch):
                read["confidence"] = confidence[index]
                all_reads.append(read)
                if read.get("number") and accepted[index]:
                    reads[(read["segment_id"], read["track_id"])].append(read)
            progress.update("identity_ocr", batch[-1]["frame_index"], frames)
    progress.update("identity_ocr", frames, frames)
    direct = {}
    assigned = {}
    team_votes = {}
    artifact = {}
    progress.update("identity_resolution", 0, len(tracks))
    for track_index, (key, points) in enumerate(tracks.items()):
        source_frames = [frame for frame, _ in points]
        raw = {
            frame: evidence.get("team_id")
            for frame, evidence in points
            if evidence.get("team_id")
        }
        counts = CudaCounter(raw.values(), device=device)
        team, share = majority(counts, minimum=5, share=0.5)
        episode_summaries = []
        all_numbers = CudaCounter(device=device)
        named = CudaCounter(device=device)
        local_direct = {}
        reads_at = {read["frame_index"]: read for read in reads.get(key, [])}
        for episode in split_episodes(source_frames, raw, device):
            frame_set = set(episode)
            local_colors = CudaCounter(
                (raw[f] for f in episode if f in raw), device=device
            )
            local_team, local_share = majority(local_colors, minimum=3, share=0.6)
            local_reads = [
                read for frame, read in reads_at.items() if frame in frame_set
            ]
            numbers = CudaCounter(
                (read["number"] for read in local_reads), device=device
            )
            all_numbers.update(numbers)
            number, number_share = majority(numbers, minimum=3, share=0.75)
            selected = None
            if number:
                candidates = roster.identity(number)["candidates"]
                matches = [
                    player for player in candidates if player["team_id"] == local_team
                ]
                if len(matches) == 1:
                    selected = matches[0]
                elif local_team is None and len(candidates) == 1:
                    candidate = candidates[0]
                    strong = CudaCounter(
                        (
                            read["number"]
                            for read in local_reads
                            if bool(read["confidence"] >= 0.95)
                        ),
                        device=device,
                    )
                    if bool(strong.get(number) >= 5) and all(
                        color == candidate["team_id"] for color in local_colors
                    ):
                        selected = candidate
            if selected:
                # Any opposite uniform evidence invalidates that frame's direct
                # identity; pure episodes retain identity through unreadable torsos.
                pure = all(color == selected["team_id"] for color in local_colors)
                for frame in episode:
                    if raw.get(frame) == selected["team_id"] or pure:
                        value = {
                            **selected,
                            "resolution_method": "CUDA_episode_number_and_color_majority",
                        }
                        direct[(*key, frame)] = value
                        local_direct[frame] = value
                        identity = (selected["team_id"], selected["number"])
                        named[identity] = named[identity] + 1
            episode_summaries.append(
                {
                    "start_frame": episode[0],
                    "end_frame": episode[-1],
                    "team_id": local_team,
                    "team_votes": dict(local_colors),
                    "team_majority_share": local_share,
                    "number_majority": number,
                    "number_votes": dict(numbers),
                    "number_majority_share": number_share,
                    "confirmation_frame": None,
                    "number_attempts": tensor(len(local_reads), device=device),
                }
            )
        number, number_share = majority(all_numbers, minimum=3, share=0.6)
        allowed = (
            {frame for frame in source_frames if raw.get(frame) == team}
            if team
            else set()
        )
        team_votes[key] = {
            "team_id": team,
            "counts": counts,
            "raw": raw,
            "allowed_frames": allowed,
            "number_majority": number,
            "mixed": len(counts) > 1,
            "episodes": episode_summaries,
        }
        if named:
            identity, _ = named.most_common(1)[0]
            player = next(
                value
                for value in local_direct.values()
                if (value["team_id"], value["number"]) == identity
            )
            valid = {
                frame
                for frame, value in local_direct.items()
                if (value["team_id"], value["number"]) == identity
            }
            assigned[key] = {**player, "_valid_frames": valid}
        artifact[f"{key[0]}:{key[1]}"] = {
            "frames": source_frames,
            "classified": {str(f): tag for f, tag in raw.items()},
            "counts": dict(counts),
            "team_id": team,
            "majority_share": share,
            "number_votes": dict(all_numbers),
            "number_majority": number,
            "number_majority_share": number_share,
            "episodes": episode_summaries,
        }
        if track_index % 30 == 0:
            progress.update("identity_resolution", track_index, len(tracks))
    progress.update("identity_resolution", len(tracks), len(tracks))
    progress.update("identity_handoffs")
    from pipeline.spatial_identity import (
        expand_direct_identity_spans,
        find_spatial_handoffs,
    )

    expand_direct_identity_spans(None, assigned, team_votes, rows=rows)
    handoffs = find_spatial_handoffs(
        None,
        None,
        assigned,
        team_votes,
        rows=rows,
        all_reads=all_reads,
        progress=lambda completed, total: progress.update(
            "identity_handoffs", completed, total
        ),
    )
    write_json(
        output / "team_votes.partial.json",
        export(
            {
                "schema_version": 1,
                "evidence_version": 6,
                "backend": "native_CUDA",
                "device": device,
                "roster_sha256": roster.sha256,
                "number_frame_step": 2,
                "color_method": "CUDA_segmented_torso",
                "tracks": artifact,
            }
        ),
    )
    collisions = collision_frames(rows, direct, assigned, handoffs, device, progress)
    # Release compact CUDA geometry before replaying full observations.
    del rows, all_reads, tracks, reads
    return direct, assigned, team_votes, handoffs, collisions


def replay(output, roster, config, run_id, progress, frames, identities, palette):
    direct, assigned, team_votes, handoffs, collisions = identities
    by_track = _handoff_index(handoffs)
    device = config["distance"]["device"]
    metric = CudaDistance(config["distance"], run_id)
    metric.spatial_handoffs = [
        {
            "segment_id": key[0],
            "track_id": key[1],
            "start_frame": key[2],
            **{name: value for name, value in item.items() if name != "player"},
        }
        for key, item in handoffs.items()
    ]
    source = Path(output) / "observations.partial.jsonl"
    destination = source.with_suffix(".resolved")
    track_destination = Path(output) / "tracks.partial.jsonl"
    progress.update("statistics", 0, frames)
    with (
        source.open() as stream,
        destination.open("w") as observations,
        track_destination.open("w") as tracks,
    ):
        for line in stream:
            row = json.loads(line)
            index = row["frame_index"]
            row["timestamp_s"] = tensor(row["timestamp_s"], device=device)
            selected = {}
            for person in row["persons"]:
                if person["role"] != "player" or person["track_id"] is None:
                    continue
                key = (row["segment_id"], person["track_id"])
                player, link = _select(key, index, direct, assigned, by_track)
                if player:
                    selected[key] = (player, link)
            for person in row["persons"]:
                if person["position_m"] is not None:
                    person["position_m"] = tensor(person["position_m"], device=device)
                key = (row["segment_id"], person["track_id"])
                evidence = team_votes.get(key)
                person.pop("resolved_identity", None)
                person["majority_jersey_number"] = None
                if person["role"] == "player" and evidence:
                    episode = next(
                        (
                            episode
                            for episode in evidence["episodes"]
                            if episode["start_frame"] <= index <= episode["end_frame"]
                        ),
                        None,
                    )
                    if episode:
                        person["majority_jersey_number"] = episode["number_majority"]
                    tag = evidence["raw"].get(index)
                    if tag:
                        person["resolved_team_id"] = tag
                        person["display_color"] = palette["teams"][tag]
                if key in selected:
                    player, link = selected[key]
                    if (index, key) not in collisions:
                        person["resolved_identity"] = {
                            name: value
                            for name, value in player.items()
                            if not name.startswith("_")
                        }
                        person["display_color"] = palette["teams"][player["team_id"]]
                        if link:
                            person["spatial_handoff"] = {
                                name: value
                                for name, value in link.items()
                                if name not in ("player", "end_frame")
                            }
                    else:
                        person["jersey_number_suppressed"] = True
            for contribution in export(metric.update(row)):
                tracks.write(
                    json.dumps(contribution, ensure_ascii=False, allow_nan=False) + "\n"
                )
            observations.write(
                json.dumps(export(row), ensure_ascii=False, allow_nan=False) + "\n"
            )
            if (index + 1) % 300 == 0:
                progress.update("statistics", index + 1, frames)
    destination.replace(source)
    progress.update("statistics", frames, frames)
    return metric
