"""Read-only count of declared player annotations; never infer human quality."""
import argparse
import json
from collections import Counter
from pathlib import Path

from training.common.files import write_json
from training.players.contracts import validate_annotations


def inventory(source):
    source = source.resolve(strict=True)
    counts, splits, errors = Counter(), {}, []
    paths = sorted(source.rglob("*.playersann.json"))
    for path in paths:
        try:
            document = validate_annotations(json.loads(path.read_text()))
        except (ValueError, OSError) as exc:
            errors.append({"path": path.relative_to(source).as_posix(), "error": str(exc)})
            continue
        split = document["source"]["split"] or "unknown"
        group = splits.setdefault(split, Counter())
        for frame in document["frames"]:
            counts[frame["review_status"]] += 1
            group[frame["review_status"]] += 1
            if frame["review_status"] == "verified":
                counts["verified_boxes"] += len(frame["boxes"])
                group["verified_boxes"] += len(frame["boxes"])
                counts["verified_negative_frames"] += not frame["boxes"]
                group["verified_negative_frames"] += not frame["boxes"]
    return {"sidecars": len(paths), "declared_frame_counts": dict(counts),
            "by_split": {k: dict(v) for k, v in splits.items()}, "invalid": errors,
            "scope": "Schema and declared review status only; no media integrity or human-quality certification"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.source.resolve()):
        parser.error("Write the report outside the source dataset")
    report = inventory(args.source)
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(report["invalid"]))
