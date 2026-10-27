"""Counts and team shot-origin maps, independent of trajectory reconstruction."""

import math
from statistics import median

from viewer.shot_geometry import number, point


def ground_origin(samples, index, shooter, geometry, boundary):
    """Use only attributed player ground positions in the short pre-release window."""
    length, width = geometry.get("length"), geometry.get("width")
    if geometry.get("units") != "metres" or not all(number(v) and v > 0 for v in (length, width)):
        return {"position_m": None, "reason": "geometry_unavailable"}
    track, team = shooter.get("track_id"), shooter.get("team_id")
    if track is None or team is None:
        return {"position_m": None, "reason": "shooter_unassigned"}
    seed = samples[index]
    positions = []
    methods = set()
    for j in range(index, -1, -1):
        sample = samples[j]
        if seed.time - sample.time > 0.25 or j < index and boundary(sample, samples[j + 1]):
            break
        contact = sample.contact
        if (
            not sample.court
            or not contact
            or contact["ambiguous"]
            or not contact["unique_track"]
            or contact["track_id"] != track
            or contact["team_id"] != team
        ):
            continue
        xy = contact.get("position_m")
        if not point(xy) or not 0 <= xy[0] <= length or not 0 <= xy[1] <= width:
            continue
        positions.append((sample.time, xy))
        methods.add(contact.get("ground_point_method"))
    if len(positions) < 2:
        return {"position_m": None, "reason": "ground_position_unavailable"}
    centre = tuple(median([xy[i] for _, xy in positions]) for i in (0, 1))
    if len(methods) > 1 or max(math.dist(centre, xy) for _, xy in positions) > 1.25:
        return {"position_m": None, "reason": "unstable_ground_position"}
    return {
        "position_m": list(centre),
        "reason": None,
        "method": "pre_release_player_ground",
        "support_frames": len(positions),
        "start_s": positions[-1][0],
        "end_s": positions[0][0],
    }


def assign_directions(events, phases, plans, scoreboard=None):
    """Reuse the learned per-quarter camera directions; never guess from player x."""
    evidence = {(row["plan_id"], row["period"]): row for row in phases.orientation_evidence}
    for event in events:
        event["court_direction"] = None
        team = event["shooter"].get("team_id")
        if team not in phases.teams:
            continue
        plan = next(
            (
                p
                for p in plans
                if p["segment_id"] == event["segment_id"]
                and p["start_s"] <= event["time_s"] <= p["end_s"]
            ),
            None,
        )
        reading = scoreboard.at(event["time_s"]) if scoreboard else None
        period = reading.get("period") if reading else None
        direction = evidence.get((plan["id"], period)) if plan else None
        if direction and direction["direction"] in (-1, 1):
            event["court_direction"] = (
                direction["direction"]
                if team == direction["reference_team_id"]
                else -direction["direction"]
            )


def summarize_shots(events, geometry, roster, available):
    """Map probable and candidate attempts; player totals retain probable shots only."""
    teams = roster.get("teams", [])
    player_counts = {
        f"roster:{t['team_id']}:{p['number']}": 0 for t in teams for p in t.get("players", [])
    }
    length, width = geometry.get("length"), geometry.get("width")
    spatial = geometry.get("units") == "metres" and all(
        number(v) and 1 <= v <= 100 for v in (length, width)
    )
    nx, ny = (math.ceil(length / 2), math.ceil(width)) if spatial else (0, 0)
    maps = {
        t["team_id"]: {
            "team_id": t["team_id"],
            "detected_count": 0,
            "probable_count": 0,
            "candidate_count": 0,
            "located_count": 0,
            "unlocated_count": 0,
            "nx": nx,
            "ny": ny,
            "grid": [0] * (nx * ny),
            "points": [],
        }
        for t in teams
    }
    unassigned = 0
    unassigned_candidates = 0
    for event in events:
        if event["status"] not in ("probable", "candidate"):
            continue
        shooter = event["shooter"]
        team = maps.get(shooter.get("team_id"))
        if not team:
            if event["status"] == "probable":
                unassigned += 1
            else:
                unassigned_candidates += 1
            continue
        subject = shooter.get("subject_id")
        if event["status"] == "probable" and subject in player_counts and subject.startswith(f"roster:{shooter['team_id']}:"):
            player_counts[subject] += 1
        team["detected_count"] += 1
        team[f"{event['status']}_count"] += 1
        xy = (event.get("origin") or {}).get("position_m")
        direction = event.get("court_direction")
        if not spatial or not point(xy) or direction not in (-1, 1):
            team["unlocated_count"] += 1
            continue
        x, y = (length - xy[0], width - xy[1]) if direction == -1 else xy
        if not length / 2 <= x <= length or not 0 <= y <= width:
            team["unlocated_count"] += 1
            continue
        x -= length / 2
        gx, gy = min(nx - 1, int(x / (length / 2) * nx)), min(ny - 1, int(y / width * ny))
        team["grid"][gy * nx + gx] += 1
        team["located_count"] += 1
        team["points"].append(
            {
                "event_id": event["id"],
                "status": event["status"],
                "x": x,
                "y": y,
                "time_s": event["time_s"],
                "subject_id": subject,
            }
        )
    return {
        "player_counts": player_counts if available else None,
        "unassigned_probable_count": unassigned,
        "unassigned_candidate_count": unassigned_candidates,
        "shot_maps": {
            "available": bool(available and spatial),
            "court": (
                {**geometry, "length": length / 2, "full_length": length, "extent": "half"}
                if spatial
                else None
            ),
            "scope": "whole_clip",
            "statuses": ["probable", "candidate"],
            "maps": list(maps.values()),
        },
    }
