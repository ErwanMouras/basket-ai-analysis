"""Conservative CUDA continuity and official-lineup completion for saved runs."""

import re
from collections import defaultdict

import torch

from pipeline.cuda.core import box_iou, tensor
from pipeline.cuda.identity import _handoff_index, _select, split_episodes
from pipeline.cuda.tracking import linear_assignment


def identity_key(player):
    return (player["team_id"], player["number"]) if player else None


def camera_shift(before, after, device):
    """Robust common image translation from persistent tracker IDs."""
    previous = {p["track_id"]: p for p in before if p["track_id"] is not None}
    changes = []
    for p in after:
        old = previous.get(p["track_id"])
        if old is not None:
            changes.append(
                (p["bbox"][:2] + p["bbox"][2:] - old["bbox"][:2] - old["bbox"][2:]) / 2
            )
    return (
        torch.stack(changes).median(0).values
        if len(changes) >= 4
        else torch.zeros(2, device=device, dtype=torch.float64)
    )


def refinement_episodes(frames, colors, rows, key, device, reads):
    """Bridge short safe gaps in one source ID; split sustained number changes."""
    groups = split_episodes(frames, colors, device)
    merged = []
    for group in groups:
        connect = False
        if merged:
            a, b = rows[merged[-1][-1]], rows[group[0]]
            dt = b["timestamp_s"] - a["timestamp_s"]
            first = next(p for p in a["persons"] if p["track_id"] == key[1])
            last = next(p for p in b["persons"] if p["track_id"] == key[1])
            ca = {colors[f] for f in merged[-1] if f in colors}
            cb = {colors[f] for f in group if f in colors}
            height = torch.maximum(
                first["bbox"][3] - first["bbox"][1], last["bbox"][3] - last["bbox"][1]
            ).clamp_min(1)
            delta = (
                last["bbox"][:2]
                + last["bbox"][2:]
                - first["bbox"][:2]
                - first["bbox"][2:]
            ) / 2 - camera_shift(a["persons"], b["persons"], device)
            connect = (
                not b["scene_cut"]
                and not (ca and cb and ca != cb)
                and bool(
                    (dt <= 0.32) & (torch.linalg.vector_norm(delta) / height <= 0.35)
                )
            )
            # A nearby second body makes a hidden identity switch plausible.
            for row, body in ((a, first), (b, last)):
                others = [
                    p["bbox"]
                    for p in row["persons"]
                    if p["role"] == "player" and p["track_id"] != key[1]
                ]
                if others:
                    boxes = torch.stack(others)
                    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
                    center = (body["bbox"][:2] + body["bbox"][2:]) / 2
                    if bool(
                        (
                            torch.linalg.vector_norm(centers - center, dim=1) / height
                            < 0.65
                        ).any()
                    ):
                        connect = False
        if connect:
            merged[-1].extend(group)
        else:
            merged.append(list(group))
    result = []
    for group in merged:
        strong = [
            reads[f]
            for f in group
            if f in reads and bool(reads[f]["confidence"] >= 0.95)
        ]
        boundaries = set()
        current = None
        for i, read in enumerate(strong):
            support = strong[i : i + 5]
            same = [r for r in support if r["number"] == read["number"]]
            if len(same) >= 3 and bool(
                tensor(same[2]["timestamp_seconds"], device=device)
                - tensor(read["timestamp_seconds"], device=device)
                <= 2
            ):
                if current is not None and current != read["number"]:
                    boundaries.add(read["frame_index"])
                current = read["number"]
        part = []
        for frame in group:
            if frame in boundaries and part:
                result.append(part)
                part = []
            part.append(frame)
        result.append(part)
    return result


def match_bodies(before, after, device):
    """Unique one-to-one body links; no names participate in geometric cost."""
    if not before or not after:
        return []
    first = torch.stack([p["bbox"] for p in before])
    second = torch.stack([p["bbox"] for p in after])
    shift = camera_shift(before, after, device)
    moved = first + shift.repeat(2)
    height = torch.maximum(
        (first[:, 3] - first[:, 1])[:, None], (second[:, 3] - second[:, 1])[None]
    ).clamp_min(1)
    delta = (
        moved[:, None, :2]
        + moved[:, None, 2:]
        - second[None, :, :2]
        - second[None, :, 2:]
    ) / 2
    distance = torch.linalg.vector_norm(delta, dim=2) / height
    overlap = box_iou(moved, second)
    cost = torch.where(
        (distance <= 0.45) & (overlap >= 0.25),
        0.6 * (1 - overlap) + 0.4 * distance,
        torch.inf,
    )
    matches, _, _ = linear_assignment(cost, 0.65)
    safe = []
    for i, j in matches:
        other_row = cost[i].clone()
        other_column = cost[:, j].clone()
        other_row[j] = torch.inf
        other_column[i] = torch.inf
        if bool(
            (other_row.amin() - cost[i, j] >= 0.12)
            & (other_column.amin() - cost[i, j] >= 0.12)
        ):
            safe.append((i, j))
    return safe


def propagate_motion(rows, identities, device, progress):
    """Bidirectional bounded propagation. Cuts and opposing anchors terminate it."""
    direct, assigned, teams, handoffs, collisions = identities
    by_track = _handoff_index(handoffs)
    anchors = {}
    for index, row in rows.items():
        for p in row["persons"]:
            if p["role"] != "player" or p["track_id"] is None:
                continue
            key = (row["segment_id"], p["track_id"])
            if (index, key) not in collisions:
                value, _ = _select(key, index, direct, assigned, by_track)
                if value:
                    anchors[(index, p["track_id"])] = value
    proposals = defaultdict(list)
    total = len(rows) * 2
    progress.update("identity_motion", 0, total)
    for backward in (False, True):
        previous = None
        carried = {}
        for n, row in enumerate(
            reversed(list(rows.values())) if backward else rows.values()
        ):
            index = row["frame_index"]
            persons = [
                p
                for p in row["persons"]
                if p["role"] == "player" and p["track_id"] is not None
            ]
            live = {}
            if previous:
                prior, prior_persons = previous
                cut = prior["scene_cut"] if backward else row["scene_cut"]
                if (
                    row["segment_id"] == prior["segment_id"]
                    and not cut
                    and bool((row["timestamp_s"] - prior["timestamp_s"]).abs() <= 0.32)
                ):
                    for i, j in match_bodies(prior_persons, persons, device):
                        old = carried.get(prior_persons[i]["track_id"])
                        if old and bool((row["timestamp_s"] - old[1]).abs() <= 4):
                            live[persons[j]["track_id"]] = old
            for p in persons:
                track = p["track_id"]
                anchor = anchors.get((index, track))
                candidate = live.get(track)
                evidence = teams.get((row["segment_id"], track), {})
                color = evidence.get("raw", {}).get(index)
                if candidate and color not in (None, candidate[0]["team_id"]):
                    live.pop(track, None)
                    candidate = None
                if anchor:
                    live[track] = (anchor, row["timestamp_s"])
                elif candidate:
                    proposals[(row["segment_id"], track, index)].append(candidate[0])
            carried, previous = live, (row, persons)
            if n % 150 == 0:
                progress.update(
                    "identity_motion", n + (len(rows) if backward else 0), total
                )
    for key, candidates in proposals.items():
        if len({identity_key(p) for p in candidates}) == 1:
            direct[key] = {
                **candidates[0],
                "resolution_method": "CUDA_unique_bidirectional_body_continuity",
            }
    progress.update("identity_motion", total, total)
    return len(proposals)


def official_lineups(official, roster):
    """Verified first-period lineup catalogue; no invented clock alignment.

    Period-break changes are not recorded reliably in this source, so subsequent
    periods are excluded instead of assuming the previous five stayed on court.
    """
    lookup = {
        p["name"]: (t["team_id"], p["number"])
        for t in roster.payload["teams"]
        for p in t["players"]
        if p.get("eligible", True)
    }
    current = defaultdict(set)
    for team in official["boxscore"]["players"]:
        for group in team["statistics"]:
            for p in group["athletes"]:
                name = p["athlete"]["displayName"]
                if p.get("starter") and name in lookup:
                    tid, number = lookup[name]
                    current[tid].add(number)
    catalogue = defaultdict(set)
    if len(current) != 2 or any(len(v) != 5 for v in current.values()):
        return {}
    for tid, numbers in current.items():
        catalogue[tid].add(frozenset(numbers))
    for play in official["plays"]:
        if play["period"]["number"] != 1:
            continue
        match = re.fullmatch(r"(.+) enters the game for (.+)", play["text"])
        if not match:
            continue
        incoming, outgoing = (lookup.get(name) for name in match.groups())
        if not incoming or not outgoing or incoming[0] != outgoing[0]:
            return {}  # incomplete official source: disable inference entirely
        tid = incoming[0]
        if outgoing[1] not in current[tid] or incoming[1] in current[tid]:
            return {}
        current[tid].remove(outgoing[1])
        current[tid].add(incoming[1])
        catalogue[tid].add(frozenset(current[tid]))
    return dict(catalogue)


def complete_verified_lineups(rows, identities, catalogue, roster, interval):
    """Complete only a verified Q1 interval, with four named teammates.

    The interval comes from reviewed clock synchronization, not video elapsed
    time. Outside it, official lineups never constrain a player's identity.
    """
    direct, assigned, teams, handoffs, collisions = identities
    by_track = _handoff_index(handoffs)
    count = 0
    players = {
        (p["team_id"], p["number"]): p for n in roster._by_number.values() for p in n
    }
    for index in range(*interval):
        row = rows.get(index)
        if not row:
            continue
        groups = defaultdict(list)
        for p in row["persons"]:
            if p["role"] != "player" or p["track_id"] is None:
                continue
            key = (row["segment_id"], p["track_id"])
            player, _ = _select(key, index, direct, assigned, by_track)
            color = teams.get(key, {}).get("raw", {}).get(index)
            if (index, key) in collisions:
                player = None
            tid = player["team_id"] if player else color
            if tid:
                groups[tid].append((key, player))
        for tid, group in groups.items():
            if len(group) != 5:
                continue
            known = {p["number"] for _, p in group if p}
            missing = [key for key, p in group if p is None]
            if len(known) != 4 or len(missing) != 1:
                continue
            possibilities = {
                number
                for lineup in catalogue.get(tid, ())
                if known <= lineup
                for number in lineup - known
            }
            if len(possibilities) == 1:
                number = next(iter(possibilities))
                direct[(*missing[0], index)] = {
                    **players[(tid, number)],
                    "resolution_method": "verified_Q1_official_lineup_unique_completion",
                }
                count += 1
    return count
