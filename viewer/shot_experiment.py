"""Export reproducible shot proposals from a completed run, without video or ML.

Example: python -m viewer.shot_experiment --run runs/analysis/match --output /tmp/shots.json
"""

import argparse
import hashlib
import json
from pathlib import Path

from viewer.shots import ShotIndex


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def experiment(path, review=None):
    manifest = json.loads((path / "run.json").read_text())
    if manifest.get("status") != "completed":
        raise ValueError("Shot experiments require a completed run")
    roster = path / "roster.json"
    detector = ShotIndex(manifest, json.loads(roster.read_text()) if roster.is_file() else None)
    with (path / "observations.jsonl").open() as file:
        for line in file:
            detector.add_frame(json.loads(line))
    detector.finish()
    result = {
        "source_sha256": manifest.get("source", {}).get("sha256"),
        "observations_sha256": digest(path / "observations.jsonl"),
        "manifest_sha256": digest(path / "run.json"),
        "detector_files_sha256": {
            name: digest(Path(__file__).with_name(name))
            for name in ("shots.py", "shot_geometry.py")
        },
        "roster_sha256": digest(roster) if roster.is_file() else None,
        "detection": detector.metadata(),
        "rejections": dict(detector.rejections),
    }
    if review:
        if (
            review.get("source_sha256") != result["source_sha256"]
            or review.get("observations_sha256") != result["observations_sha256"]
        ):
            raise ValueError("Review checkpoints do not match this source and observations")
        used = set()
        checks = []
        for item in review.get("checkpoints", []):
            a, b = item["window_s"]
            matches = [
                event
                for event in detector.events
                if a <= event["time_s"] <= b and event["id"] not in used
            ]
            event = min(matches, key=lambda e: abs(e["time_s"] - (a + b) / 2), default=None)
            if event:
                used.add(event["id"])
            checks.append(
                {
                    **item,
                    "event_id": event["id"] if event else None,
                    "status": event["status"] if event else None,
                }
            )
        result["review"] = {
            "development_only": True,
            "checkpoints": checks,
            "unmatched_events": [e["id"] for e in detector.events if e["id"] not in used],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--compact", action="store_true", help="Omit per-frame image trajectories")
    args = parser.parse_args()
    result = experiment(args.run, json.loads(args.review.read_text()) if args.review else None)
    if args.compact:
        for event in result["detection"]["events"]:
            event.pop("trajectory", None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(
        f"{result['detection']['probable_count']} probable shots, {result['detection']['candidate_count']} candidates"
    )


if __name__ == "__main__":
    main()
