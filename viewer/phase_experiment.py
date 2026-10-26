"""Compare phase heuristics on existing artifacts without inference or annotation changes.

Example: python -m viewer.phase_experiment --run runs/analysis/example --output release/phases.json
An optional, source-hash-bound visual-check file provides a development audit,
not an independent estimate of model accuracy.
"""

import argparse
from bisect import bisect_right
import json
from pathlib import Path
from statistics import median

from viewer.phases import PhaseIndex, contact, finite, point, team_of
from viewer.scoreboard import ScoreboardIndex


def zone_only(row, teams, geometry):
    """Deliberately simple baseline: team farther from the occupied-half basket."""
    if not geometry or (row.get("court") or {}).get("status") not in ("fit", "propagated"):
        return None
    xs = {team: [] for team in teams}
    for person in row.get("persons", []):
        team, xy = team_of(person), person.get("position_m")
        if (
            person.get("role") == "player"
            and team in xs
            and point(xy)
            and 0 <= xy[0] <= geometry["length"]
            and 0 <= xy[1] <= geometry["width"]
        ):
            xs[team].append(xy[0])
    if any(len(values) < 2 for values in xs.values()):
        return None
    centres = {team: median(values) for team, values in xs.items()}
    cluster = median([x for values in xs.values() for x in values])
    length = geometry["length"]
    if abs(centres[teams[0]] - centres[teams[1]]) < 0.75 or 0.4 * length < cluster < 0.6 * length:
        return None
    return max(centres, key=centres.get) if cluster < length / 2 else min(centres, key=centres.get)


def experiment(run, review=None):
    manifest = json.loads((run / "run.json").read_text())
    if manifest.get("status") != "completed":
        raise ValueError("A completed run is required")
    roster = json.loads((run / "roster.json").read_text())
    teams = tuple(team["team_id"] for team in roster["teams"])
    if len(teams) != 2:
        raise ValueError("Exactly two roster teams are required")
    geometry = manifest.get("court_coordinate_system") or {}
    if geometry.get("units") != "metres" or not all(
        finite(geometry.get(key)) and 1 <= geometry[key] <= 100 for key in ("length", "width")
    ):
        geometry = None
    config = (manifest.get("config") or {}).get("distance") or {}
    gap = config.get("max_gap_s", 0.2)
    if not finite(gap) or gap <= 0:
        gap = 0.2
    rows = []
    with (run / "observations.jsonl").open() as source:
        for line in source:
            row = json.loads(line)
            # Keep geometry/contacts only; never retain masks or detector payloads.
            row["persons"] = [
                {
                    key: person[key]
                    for key in (
                        "role",
                        "track_id",
                        "bbox",
                        "pose",
                        "position_m",
                        "resolved_identity",
                        "resolved_team_id",
                        "majority_jersey_number",
                        "jersey",
                    )
                    if key in person
                }
                for person in row.get("persons", [])
            ]
            rows.append(row)
    if not rows:
        raise ValueError("The run contains no observations")
    times = [row["timestamp_s"] for row in rows]
    indices = {
        method: PhaseIndex(teams, geometry, gap, method=method, use_court_context=False)
        for method in ("body", "hands")
    }
    scoreboard = ScoreboardIndex.load(run, manifest, roster)
    indices.update(
        {
            "quarter_contacts": PhaseIndex(
                teams, geometry, gap, scoreboard=scoreboard, use_court_context=False
            ),
            "quarter_context": PhaseIndex(teams, geometry, gap, scoreboard=scoreboard),
        }
    )
    unguarded = PhaseIndex(teams, geometry, gap, scoreboard=scoreboard)
    unguarded.score_guard_before_s = unguarded.score_guard_after_s = 0.0
    indices["quarter_context_without_score_guard"] = unguarded
    plan, previous = -1, None
    raw_contacts = []
    zones = []
    for row in rows:
        if (
            previous is None
            or row.get("scene_cut")
            or row.get("segment_id") != previous.get("segment_id")
        ):
            plan += 1
        for index in indices.values():
            index.add_frame(row, plan)
        raw_contacts.append(contact(row, teams, geometry).team)
        zones.append(zone_only(row, teams, geometry))
        previous = row
    for index in indices.values():
        index.finish()
    methods = {}
    for method, index in indices.items():
        methods[f"{method}_temporal"] = {
            key: value
            for key, value in index.metadata().items()
            if key not in ("episodes", "uncertain_intervals")
        }
    result = {
        "run": run.name,
        "source_sha256": manifest["source"].get("sha256"),
        "methods": methods,
        "review": None,
        "limitations": "Development comparison only. No independent accuracy estimate, official possession count or automatic replay removal.",
    }
    if review:
        audit = json.loads(review.read_text())
        if audit.get("source_sha256") != result["source_sha256"]:
            raise ValueError("Visual review belongs to a different source video")
        result["review"] = {
            "sampling": audit["sampling"],
            "review_type": audit["review_type"],
            "limitations": audit["limitations"],
            "checkpoints": [],
            "scores": {},
        }
        for checkpoint in audit["checkpoints"]:
            time = checkpoint["time_s"]
            if not finite(time) or not times[0] <= time <= times[-1]:
                raise ValueError("Invalid visual-review timestamp")
            frame_index = max(0, bisect_right(times, time) - 1)
            predictions = {
                "zone_only": zones[frame_index],
                "hands_instant": raw_contacts[frame_index],
            }
            for method, index in indices.items():
                predictions[f"{method}_temporal"] = next(
                    (span.owner for span in index.spans if span.start <= time < span.end), None
                )
            result["review"]["checkpoints"].append({**checkpoint, "predictions": predictions})
        for method in ("zone_only", "hands_instant", *(f"{name}_temporal" for name in indices)):
            score = {
                "labeled": 0,
                "decided": 0,
                "agreed": 0,
                "disagreed": 0,
                "abstained": 0,
                "unsupported_on_unknown": 0,
            }
            for checkpoint in result["review"]["checkpoints"]:
                expected, predicted = (
                    checkpoint["attacking_team_id"],
                    checkpoint["predictions"][method],
                )
                if expected is None:
                    score["unsupported_on_unknown"] += int(predicted is not None)
                    continue
                score["labeled"] += 1
                score["abstained"] += int(predicted is None)
                if predicted is not None:
                    score["decided"] += 1
                    score["agreed" if predicted == expected else "disagreed"] += 1
            result["review"]["scores"][method] = score
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--review-checkpoints", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = experiment(args.run, args.review_checkpoints)
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
