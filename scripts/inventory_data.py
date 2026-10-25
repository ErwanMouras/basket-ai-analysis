"""Read source annotations and decoded video counts; write a dated inventory.

Run from the repository root with the player environment and ffprobe installed.
Sources are never modified. Counts describe declared annotations, not accuracy.
"""

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from training.players.contracts import validate_annotations


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def ranges(indices):
    result = []
    for index in sorted(set(indices)):
        if result and index == result[-1][1] + 1:
            result[-1][1] = index
        else:
            result.append([index, index])
    return result


def hash_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_video(video, root):
    command = ["ffprobe", "-v", "error", "-threads", "1", "-select_streams",
               "v:0", "-count_frames", "-show_entries",
               "stream=width,height,avg_frame_rate,nb_frames,nb_read_frames,duration",
               "-of", "json", str(video)]
    probe = subprocess.run(command, capture_output=True, text=True, check=True)
    stream = json.loads(probe.stdout)["streams"][0]
    decoded = int(stream["nb_read_frames"])
    declared = int(stream["nb_frames"])
    fps = float(Fraction(stream["avg_frame_rate"]))
    metadata = yaml.safe_load(video.with_name(video.name + ".meta.yaml").read_text())
    digest = hash_file(video)
    relative = video.relative_to(root)
    row = {"path": str(video), "clip": relative.parent.as_posix(),
           "split": relative.parts[0], "match_id": metadata.get("match_id"),
           "venue_id": metadata.get("venue_id"), "video_sha256": digest,
           "width": stream["width"], "height": stream["height"], "fps": fps,
           "duration_container_s": float(stream["duration"]),
           "duration_decoded_nominal_s": decoded / fps,
           "frames_declared": declared, "frames_decoded_ffmpeg": decoded,
           "metadata_statuses": metadata.get("statuses", {}),
           "match_yaml": video.with_name("match.yaml").exists(),
           "jersey_json": video.with_name("jersey.json").exists(),
           "historical_gold_files": [str(p) for p in sorted((video.parent / "gold").glob("*"))],
           "probe_diagnostics": probe.stderr}
    ball_path = video.with_name(video.name + ".ballann.json")
    ball = read_json(ball_path)
    annotations = {int(i): ann for i, ann in ball["annotations"].items()}
    position = {i for i, ann in annotations.items() if ann.get("cx") is not None}
    occluded = {i for i in position if annotations[i].get("occluded")}
    unknown = set(annotations) - position
    missing = set(range(decoded)) - set(annotations)
    row["ball"] = {"annotations": len(annotations), "with_position": len(position),
                   "visible_with_position": len(position - occluded),
                   "occluded_with_position": len(occluded), "unknown_position": len(unknown),
                   "missing_decodable": len(missing), "missing_ranges_inclusive": ranges(missing),
                   "unknown_ranges_inclusive": ranges(unknown),
                   "outside_decoded_ranges": ranges(set(annotations) - set(range(decoded))),
                   "legacy_video_path": ball["video"]["path"],
                   "annotation_sha256": hash_file(ball_path)}
    people_path = video.with_name(video.name + ".playersann.json")
    people = {"present": people_path.exists(), "review_status": {},
              "verified_player_boxes": 0, "verified_referee_boxes": 0}
    if people_path.exists():
        document = validate_annotations(read_json(people_path))
        frames = document["frames"]
        verified = [f for f in frames if f["review_status"] == "verified"]
        boxes = [b for f in verified for b in f["boxes"]]
        people.update(review_status=dict(Counter(f["review_status"] for f in frames)),
                      verified_ranges_inclusive=ranges(f["frame_index"] for f in verified),
                      in_progress_ranges_inclusive=ranges(f["frame_index"] for f in frames
                                                          if f["review_status"] == "in_progress"),
                      verified_player_boxes=sum(b["class_id"] == 0 for b in boxes),
                      verified_referee_boxes=sum(b["class_id"] == 1 for b in boxes),
                      verified_boxes_unknown_occlusion=sum(b["occluded"] is None for b in boxes),
                      source_hash_matches=document["source"]["sha256"] == digest,
                      source_path_matches=document["source"]["path"] == relative.as_posix(),
                      frame_count_matches=document["source"]["frame_count"] == declared,
                      missing_decodable=len(set(range(decoded)) - {f["frame_index"] for f in frames}),
                      annotation_sha256=hash_file(people_path))
    row["people"] = people
    draft = read_json(video.with_name(video.name + ".courtann.json"))
    registration = read_json(video.with_name("court_keyframes.json"))
    left = {f["frame_idx"]: f["points"] for f in draft["keyframes"]}
    right = {f["frame_idx"]: f["points"] for f in registration["keyframes"]}
    coordinate_changes = [dict(frame_index=i, landmark=k, draft=left[i][k], registration=right[i][k])
                          for i in sorted(left.keys() & right.keys())
                          for k in sorted(left[i].keys() & right[i].keys()) if left[i][k] != right[i][k]]
    row["court"] = {"draft_keyframes": len(left), "registration_keyframes": len(right),
                    "draft_only_frames": sorted(left.keys() - right.keys()),
                    "registration_only_frames": sorted(right.keys() - left.keys()),
                    "shared_frames_with_different_landmark_sets": [i for i in sorted(left.keys() & right.keys())
                                                                    if left[i].keys() != right[i].keys()],
                    "coordinate_changes": coordinate_changes,
                    "draft_artifact_type": draft.get("artifact_type"),
                    "registration_artifact_type": registration.get("artifact_type")}
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("datas"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.source, args.output
    if output.resolve().is_relative_to(root.resolve()):
        parser.error("Reports must be outside the source dataset")
    if output.exists():
        parser.error("Use a new output directory")
    with ThreadPoolExecutor(max_workers=3) as executor:
        rows = list(executor.map(lambda p: inspect_video(p, root), sorted(root.rglob("*.mp4"))))
    totals = {}
    for split in ("train", "val", "test"):
        clips = [r for r in rows if r["split"] == split]
        totals[split] = {"videos": len(clips),
                         "duration_container_s": sum(r["duration_container_s"] for r in clips),
                         "frames_decoded": sum(r["frames_decoded_ffmpeg"] for r in clips),
                         "ball_annotations": sum(r["ball"]["annotations"] for r in clips),
                         "ball_with_position": sum(r["ball"]["with_position"] for r in clips),
                         "ball_unknown_position": sum(r["ball"]["unknown_position"] for r in clips),
                         "ball_occluded_with_position": sum(r["ball"]["occluded_with_position"] for r in clips),
                         "ball_missing_decodable": sum(r["ball"]["missing_decodable"] for r in clips),
                         "people_verified": sum(r["people"]["review_status"].get("verified", 0) for r in clips),
                         "people_proposed": sum(r["people"]["review_status"].get("proposed", 0) for r in clips),
                         "people_in_progress": sum(r["people"]["review_status"].get("in_progress", 0) for r in clips),
                         "verified_player_boxes": sum(r["people"]["verified_player_boxes"] for r in clips),
                         "verified_referee_boxes": sum(r["people"]["verified_referee_boxes"] for r in clips)}
    report = {"generated_at": datetime.now(ZoneInfo("Europe/Paris")).isoformat(),
              "scope": "Local source inventory; FFmpeg full decoded counts and annotation status/hash checks. "
                       "No visual review, accuracy certification or data changes.",
              "range_convention": "Inclusive endpoints, zero-based frames; nominal time = frame_index / fps.",
              "totals": totals, "videos": rows}
    output.mkdir(parents=True)
    (output / "inventory.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    fields = ["clip", "split", "duration_container_s", "fps", "width", "height", "frames_declared",
              "frames_decoded_ffmpeg", "video_sha256", "ball_annotations", "ball_with_position",
              "ball_unknown_position", "ball_missing_decodable", "people_verified", "people_proposed",
              "people_in_progress", "court_draft_keyframes", "court_registration_keyframes",
              "match_yaml", "jersey_json"]
    with (output / "inventory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {key: row[key] for key in fields if key in row}
            flat.update({"ball_" + key: row["ball"][key] for key in
                         ("annotations", "with_position", "unknown_position", "missing_decodable")})
            flat.update({"people_" + key: row["people"]["review_status"].get(key, 0)
                         for key in ("verified", "proposed", "in_progress")})
            flat.update({"court_" + key: row["court"][key] for key in ("draft_keyframes", "registration_keyframes")})
            writer.writerow(flat)
    print(json.dumps(totals, ensure_ascii=False, indent=2))
    print(f"Reports: {output}")


if __name__ == "__main__":
    main()
