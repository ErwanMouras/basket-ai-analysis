"""Evaluate reviewed trajectories against immutable analysis outputs.

IDF1 uses global identity assignment from all IoU-compatible pairs, as in
TrackEval's Identity metric. Frame matches are used separately for geometry,
identity acceptance, ID switches and distance on shared measurable intervals.
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from pipeline.contracts import finite_point
from pipeline.distance import identity as accepted_identity
from training.common.files import write_json
from training.common.provenance import file_hash, object_hash


def prepare(gold_path, run, iou_threshold):
    gold_path, run = Path(gold_path), Path(run)
    gold = json.loads(gold_path.read_text())
    if gold.get("schema_version") != 1 or gold.get("status") != "reviewed" or gold.get("split") not in ("val", "test"):
        raise ValueError("Gold requires schema_version 1, status reviewed, and val/test split")
    if not gold.get("reviewer") or not gold.get("frames") or not 0 < iou_threshold <= 1:
        raise ValueError("Gold needs reviewer, frames and valid IoU threshold")
    manifest = json.loads((run / "run.json").read_text())
    if manifest.get("status") != "completed" or manifest["source"]["sha256"] != gold.get("source_sha256"):
        raise ValueError("Gold video hash must match a completed analysis")
    seen, previous = set(), None
    for frame in gold["frames"]:
        i, segment, timestamp = frame["frame_index"], frame["segment_id"], frame["timestamp_s"]
        if (type(i) is not int or i < 0 or i in seen or type(segment) is not int or segment < 0
                or type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < 0):
            raise ValueError("Invalid gold frame identity or timestamp")
        if previous and (i <= previous[0] or timestamp <= previous[1] or segment < previous[2]):
            raise ValueError("Gold frames and segments must be ordered")
        seen.add(i)
        previous = (i, timestamp, segment)
        if "scene_cut" in frame and type(frame["scene_cut"]) is not bool:
            raise ValueError("Reviewed scene_cut labels must be boolean")
        for landmark in frame.get("landmarks", []):
            if (landmark.get("holdout") is not True or not finite_point(landmark.get("source_px"), 2)
                    or not finite_point(landmark.get("court_m"), 2)):
                raise ValueError("Independent landmark checks require holdout source pixels and court meters")
        ids = set()
        for person in frame["persons"]:
            if not isinstance(person.get("person_id"), str) or not person["person_id"] or person["person_id"] in ids:
                raise ValueError("Gold person IDs must be unique per frame")
            ids.add(person["person_id"])
            box = person.get("bbox")
            if not finite_point(box, 4) or not (0 <= box[0] < box[2] <= manifest["source"]["width"] and
                                               0 <= box[1] < box[3] <= manifest["source"]["height"]):
                raise ValueError("Invalid gold source-pixel box")
            if person.get("position_m") is not None and not finite_point(person["position_m"], 2):
                raise ValueError("Invalid reference ground position")
            number = person.get("number")
            if number is not None and (not isinstance(number, str) or not number.isascii() or not number.isdigit() or len(number) > 2):
                raise ValueError("Gold numbers must preserve digit strings including 0/00")
    hashes = {name: file_hash(run / name) for name in ("run.json", "observations.jsonl", "tracks.jsonl")}
    for name, digest in hashes.items():
        expected = manifest.get("artifacts", {}).get(name, {}).get("sha256")
        if expected and expected != digest:
            raise ValueError(f"Analysis artifact changed: {name}")
    payload = {"schema_version": 1, "gold_sha256": file_hash(gold_path), "analysis_sha256": hashes,
               "iou_threshold": iou_threshold, "evaluator_sha256": file_hash(Path(__file__)),
               "identity_policy_sha256": file_hash(Path(__file__).resolve().parents[1] / "pipeline/distance.py"),
               "coordinates": "source pixels / ground meters", "scope": "gold frames only; segment-scoped identities"}
    return gold, manifest, {**payload, "recipe_id": object_hash(payload)}


def box_iou(a, b):
    area = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    return area / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area)


def evaluate(gold_path, run, iou_threshold=.5, frozen_recipe=None):
    gold, manifest, protocol = prepare(gold_path, run, iou_threshold)
    if gold["split"] == "test" and (not frozen_recipe or json.loads(Path(frozen_recipe).read_text()) != protocol):
        raise ValueError("Final gold test requires a matching frozen recipe")
    selected = {f["frame_index"] for f in gold["frames"]}
    predictions = {}
    with (Path(run) / "observations.jsonl").open() as stream:
        for line in stream:
            frame = json.loads(line)
            if frame["frame_index"] in selected:
                predictions[frame["frame_index"]] = frame
    intervals = {}
    with (Path(run) / "tracks.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["frame_index"] in selected:
                intervals[(row["segment_id"], row["frame_index"], row["track_id"])] = row
    counts, candidates, states, errors, landmark_errors, distances = Counter(), Counter(), {}, [], [], defaultdict(lambda: [0., 0., 0])
    gt_ids, pred_ids = set(), set()
    for truth in gold["frames"]:
        index, segment = truth["frame_index"], truth["segment_id"]
        prediction = predictions.get(index)
        if prediction and abs(prediction["timestamp_s"] - truth["timestamp_s"]) > 1e-5:
            raise ValueError("Gold timestamps differ from analysis")
        predicted_segment = prediction["segment_id"] if prediction else -1
        if "scene_cut" in truth:
            counts["cut_labeled_frames"] += 1
            actual_cut = bool(prediction and prediction.get("scene_cut"))
            counts["cut_tp"] += truth["scene_cut"] and actual_cut
            counts["cut_fp"] += not truth["scene_cut"] and actual_cut
            counts["cut_fn"] += truth["scene_cut"] and not actual_cut
        matrix = (prediction or {}).get("court") or {}
        matrix = matrix.get("image_to_court")
        for landmark in truth.get("landmarks", []):
            counts["reference_landmarks"] += 1
            if matrix is not None:
                projected = np.asarray(matrix) @ [*landmark["source_px"], 1.]
                if np.isfinite(projected).all() and abs(projected[2]) > 1e-9:
                    landmark_errors.append(math.dist(projected[:2] / projected[2], landmark["court_m"]))
        actual = [p for p in (prediction or {}).get("persons", []) if p["role"] == "player"]
        expected = truth["persons"]
        counts["gt_observations"] += len(expected)
        counts["predicted_observations"] += len(actual)
        similarity = np.array([[box_iou(a["bbox"], b["bbox"]) for b in actual] for a in expected]).reshape(len(expected), len(actual))
        for a, person in enumerate(expected):
            gid = (segment, person["person_id"])
            gt_ids.add(gid)
            for b, predicted in enumerate(actual):
                if predicted["track_id"] is not None and similarity[a, b] >= iou_threshold:
                    candidates[(gid, (predicted_segment, predicted["track_id"]))] += 1
        pred_ids.update((predicted_segment, p["track_id"]) for p in actual if p["track_id"] is not None)
        matches = {}
        if expected and actual:
            scores = (similarity >= iou_threshold) * (1 + similarity / (min(similarity.shape) + 1))
            left, right = linear_sum_assignment(scores, maximize=True)
            matches = {a: b for a, b in zip(left, right) if similarity[a, b] >= iou_threshold}
        counts["spatial_matches"] += len(matches)
        for b in set(range(len(actual))) - set(matches.values()):
            if accepted_identity(actual[b]):
                counts["accepted_identities"] += 1
                counts["accepted_evaluable_identities"] += 1
                counts["unmatched_identity_acceptances"] += 1
        for a, person in enumerate(expected):
            gid = (segment, person["person_id"])
            previous = states.get(gid)
            predicted = actual[matches[a]] if a in matches else None
            raw_track = predicted.get("track_id") if predicted else None
            track = (predicted_segment, raw_track) if raw_track is not None else None
            if track is not None and previous and previous["index"] + 1 == index:
                counts["id_switches"] += previous["last_track"] is not None and previous["last_track"] != track
                counts["fragmentations"] += previous["ever_matched"] and not previous["matched"]
            identity = accepted_identity(predicted) if predicted else None
            known = person.get("team_id") is not None and person.get("number") is not None
            counts["identity_known_gt"] += known
            counts["matched_identity_known_gt"] += known and predicted is not None
            if identity:
                counts["accepted_identities"] += 1
                if known:
                    counts["accepted_evaluable_identities"] += 1
                    counts["accepted_known_gt"] += 1
                    counts["correct_identities"] += identity == (person["team_id"], person["number"])
                else:
                    counts["unverifiable_acceptances"] += 1
                counts["abstention_violations"] += person.get("expected_identity") == "abstain"
            reference, position = person.get("position_m"), predicted.get("position_m") if predicted else None
            counts["reference_positions"] += reference is not None
            if reference is not None and position is not None:
                errors.append(math.dist(reference, position))
            if previous and previous["index"] + 1 == index and reference is not None and previous["position"] is not None:
                counts["reference_intervals"] += 1
                contribution = intervals.get((predicted_segment, index, raw_track))
                if track is not None and track == previous["track"] and contribution and contribution["distance_m"] is not None:
                    shared = distances[gid]
                    shared[0] += math.dist(reference, previous["position"])
                    shared[1] += contribution["distance_m"]
                    shared[2] += 1
                    counts["shared_distance_intervals"] += 1
            states[gid] = {"index": index, "position": reference, "track": track,
                           "last_track": track if track is not None else previous["last_track"] if previous else None,
                           "ever_matched": track is not None or bool(previous and previous["ever_matched"]), "matched": track is not None}
    gt_ids, pred_ids = sorted(gt_ids), sorted(pred_ids)
    matrix = np.array([[candidates[(g, p)] for p in pred_ids] for g in gt_ids]).reshape(len(gt_ids), len(pred_ids))
    idtp = 0
    if matrix.size:
        left, right = linear_sum_assignment(matrix, maximize=True)
        idtp = int(matrix[left, right].sum())
    denominator = counts["gt_observations"] + counts["predicted_observations"]
    if prepare(gold_path, run, iou_threshold)[2] != protocol:
        raise ValueError("Gold or analysis artifacts changed during evaluation")
    def divide(a, b):
        return a / b if b else None
    return {"schema_version": 1, "protocol": protocol, "split": gold["split"], "counts": dict(counts),
            "tracking": {"idtp": idtp, "idfn": counts["gt_observations"] - idtp,
                         "idfp": counts["predicted_observations"] - idtp, "idf1": divide(2 * idtp, denominator),
                         "id_switches": counts["id_switches"], "fragmentations": counts["fragmentations"]},
            "identity": {"accepted_precision": divide(counts["correct_identities"], counts["accepted_evaluable_identities"]),
                         "correct_identity_coverage": divide(counts["correct_identities"], counts["identity_known_gt"]),
                         "accepted_identity_coverage": divide(counts["accepted_known_gt"], counts["identity_known_gt"]),
                         "abstention_rate_on_matched_known": divide(counts["matched_identity_known_gt"] - counts["accepted_known_gt"], counts["matched_identity_known_gt"])},
            "cuts": {"precision": divide(counts["cut_tp"], counts["cut_tp"] + counts["cut_fp"]),
                     "recall": divide(counts["cut_tp"], counts["cut_tp"] + counts["cut_fn"]),
                     "labeled_frames": counts["cut_labeled_frames"]},
            "geometry": {"matched_positions": len(errors), "coverage": divide(len(errors), counts["reference_positions"]),
                         "median_error_m": float(np.median(errors)) if errors else None,
                         "p95_error_m": float(np.percentile(errors, 95)) if errors else None,
                         "holdout_landmarks": len(landmark_errors),
                         "holdout_coverage": divide(len(landmark_errors), counts["reference_landmarks"]),
                         "holdout_median_error_m": float(np.median(landmark_errors)) if landmark_errors else None},
            "distance": {"interval_coverage": divide(counts["shared_distance_intervals"], counts["reference_intervals"]),
                         "trajectories": [{"segment_id": key[0], "person_id": key[1], "reference_shared_m": value[0],
                                           "predicted_shared_m": value[1], "absolute_error_m": abs(value[1] - value[0]),
                                           "relative_error": divide(abs(value[1] - value[0]), value[0]), "shared_intervals": value[2]}
                                          for key, value in sorted(distances.items())]},
            "limitations": ["Reviewed gold required; manual reference has its own uncertainty.",
                            "ID switches and fragmentation use frame-wise IoU matching; HOTA is not computed.",
                            "Distance is compared only on shared consecutive valid intervals; coverage must accompany error."],
            "identity_metric_reference": "https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/identity.py"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--iou", type=float, default=.5)
    parser.add_argument("--frozen-recipe", type=Path)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output", type=Path)
    output.add_argument("--freeze-recipe", type=Path)
    args = parser.parse_args()
    destination = args.output or args.freeze_recipe
    if destination.exists():
        parser.error("Use a new output path")
    if args.freeze_recipe:
        _, _, result = prepare(args.gold, args.run, args.iou)
    else:
        result = evaluate(args.gold, args.run, args.iou, args.frozen_recipe)
    write_json(destination, result)


if __name__ == "__main__":
    main()
