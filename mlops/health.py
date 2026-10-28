"""Inspect completed analysis artifacts without loading models or source videos."""

import argparse
import json
from collections import Counter
from pathlib import Path

from pipeline.contracts import validate_frame, validate_statistics
from pipeline.distance import identity
from training.common.files import write_json
from training.common.provenance import file_hash


def build_health(run, *, progress=None):
    run = Path(run).resolve(strict=True)
    manifest = json.loads((run / "run.json").read_text())
    if manifest.get("status") not in ("completed", "partial"):
        raise ValueError("Health reporting requires a finalized analysis")
    for name, artifact in manifest.get("artifacts", {}).items():
        path = (run / name).resolve(strict=True)
        if not path.is_relative_to(run) or file_hash(path) != artifact["sha256"]:
            raise ValueError(f"Artifact integrity mismatch: {name}")
    statistics = validate_statistics(json.loads((run / "statistics.json").read_text()))
    counts, exclusions, stage_statuses = Counter(), Counter(), Counter()
    tracks = set()
    previous_index, previous_time = -1, -1.0
    with (run / "observations.jsonl").open() as stream:
        for line in stream:
            row = validate_frame(json.loads(line))
            if row["frame_index"] <= previous_index or row["timestamp_s"] <= previous_time:
                raise ValueError("Observations must have increasing frames and source times")
            previous_index, previous_time = row["frame_index"], row["timestamp_s"]
            counts["frames"] += 1
            if progress and counts["frames"] % 300 == 0:
                progress(counts["frames"])
            counts["included_frames"] += bool(row.get("included", True))
            counts["calibrated_frames"] += (row.get("court") or {}).get("image_to_court") is not None
            counts["cuts"] += bool(row.get("scene_cut"))
            counts["ball_localized_frames"] += row["ball"]["position_px"] is not None
            for name, stage in row["stages"].items():
                stage_statuses[f"{name}:{stage['status']}"] += 1
            for person in row["persons"]:
                if person["role"] != "player":
                    continue
                counts["player_observations"] += 1
                counts["projected_player_observations"] += person["position_m"] is not None
                jersey = person.get("jersey") or {}
                resolved = person.get("resolved_identity") or {}
                named = identity(person) is not None and bool(resolved.get("player_name") or jersey.get("player_name"))
                counts["named_player_observations"] += named
                if person["track_id"] is not None:
                    tracks.add((row["segment_id"], person["track_id"]))
    if counts["frames"] != manifest["frames"]:
        raise ValueError("Manifest frame count differs from observations")
    for person in statistics["players"]:
        exclusions.update(person.get("excluded_intervals", {}))
    def ratio(numerator, denominator):
        return counts[numerator] / counts[denominator] if counts[denominator] else None
    return {"schema_version": 1, "run_id": manifest["run_id"], "status": manifest["status"],
            "counts": dict(counts), "player_track_fragments": len(tracks),
            "ratios": {"calibration_availability": ratio("calibrated_frames", "frames"),
                       "ball_localization_availability": ratio("ball_localized_frames", "frames"),
                       "player_projection_availability": ratio("projected_player_observations", "player_observations"),
                       "named_identity_availability": ratio("named_player_observations", "player_observations")},
            "exclusion_counts": dict(exclusions), "stage_status_counts": dict(stage_statuses),
            "timings": manifest.get("timings"), "resources": manifest.get("resources"),
            "models": manifest.get("models"), "warnings": manifest.get("warnings", []),
            "interpretation": "Availability and integrity only. Track fragments are not unique players. "
                              "These ratios do not measure accuracy or model drift without reviewed ground truth."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new report path")
    result = build_health(args.run)
    write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("models", "stage_status_counts")}, indent=2))


if __name__ == "__main__":
    main()
