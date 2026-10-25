"""Score human-reviewed, source-aligned masks and OCR labels; never infer gold."""
from collections import Counter
import numpy as np
from training.players.evaluation.metrics import match_frame
from training.segmentation.masks import crop_mask, validate


def score_reviewed(records, predictions, ocr, labels):
    indexed = {r["frame_id"]: r for r in records}
    crops = {(r["frame_id"], r["prediction_index"]): r for r in ocr}
    matches = {}
    for frame_id, r in indexed.items():
        gt = [(i, b) for i, b in enumerate(r["boxes"]) if b["class_id"] == 0]
        ds = predictions.get(frame_id, [])
        people = [(i, d) for i, d in enumerate(ds) if d["role"] == "player"]
        events = match_frame([b["source_bbox"] for _, b in gt], [d for _, d in people], score_threshold=.25, iou_threshold=.5)
        matches.update({(frame_id, gt[e["gt_index"]][1]["object_id"]): people[e["prediction_index"]][0]
                        for e in events if e["kind"] == "TP"})
    counts = {p: Counter() for p in ("disabled", "background_neutral")}
    ious, dice, retention = [], [], []
    seen = set()
    for label in labels:
        if label.get("review_status") != "reviewed":
            continue
        key = label["frame_id"], label["object_id"]
        if key in seen:
            raise ValueError("Duplicate reviewed object")
        seen.add(key)
        r = indexed.get(key[0])
        if r is None or not any(b["object_id"] == key[1] and b["class_id"] == 0 for b in r["boxes"]):
            raise ValueError("Reviewed label does not identify an evaluated player")
        if label.get("image_sha256") != r["image_sha256"]:
            raise ValueError("Reviewed label image hash mismatch")
        pi = matches.get(key)
        detection = predictions[key[0]][pi] if pi is not None else None
        observation = crops.get((key[0], pi)) if pi is not None else None
        if "expected_number" in label:
            expected = label["expected_number"]
            if expected is not None and (not isinstance(expected, str) or not expected.isascii() or not expected.isdigit() or not 1 <= len(expected) <= 2):
                raise ValueError("Expected number must be null or one/two ASCII digits")
            for policy, counter in counts.items():
                counter["reviewed"] += 1
                counter["readable" if expected is not None else "must_abstain"] += 1
                crop = observation["crops"][policy] if observation else {}
                if crop.get("accepted"):
                    counter["correct" if expected is not None and crop["reading"]["text"] == expected else "wrong"] += 1
                else:
                    counter["abstentions"] += 1
        predicted_mask = detection.get("segmentation", {}).get("mask") if detection else None
        for field in ("person_mask", "number_mask"):
            gold = label.get(field)
            if gold is None:
                continue
            validate(gold, r["width"], r["height"])
            # Use one image per reviewed instance; do not retain full-frame masks.
            box = [0, 0, r["width"], r["height"]]
            gm = crop_mask(gold, box)
            pm = crop_mask(predicted_mask, box) if predicted_mask else np.zeros_like(gm)
            if not gm.any():
                raise ValueError("A reviewed gold mask must contain foreground")
            intersection = int((gm & pm).sum())
            if field == "person_mask":
                ious.append(intersection / int((gm | pm).sum()))
                dice.append(2 * intersection / int(gm.sum() + pm.sum()))
            else:
                retention.append(intersection / int(gm.sum()))
    return {"reviewed_objects": len(seen), "ocr": {p: {**dict(c),
        "precision_when_read": c["correct"]/(c["correct"]+c["wrong"]) if c["correct"]+c["wrong"] else None,
        "correct_read_coverage": c["correct"]/c["readable"] if c["readable"] else None} for p, c in counts.items()},
        "mask": {"instances": len(ious), "mean_iou": float(np.mean(ious)) if ious else None,
                 "mean_dice": float(np.mean(dice)) if dice else None,
                 "number_regions": len(retention), "number_pixel_retention": float(np.mean(retention)) if retention else None}}


def main():
    import argparse
    import json
    from pathlib import Path
    from training.common.files import write_json
    from training.common.provenance import file_hash
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,type=Path)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--labels',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new review score output')
    def rows(path): return [json.loads(l) for l in path.read_text().splitlines()]
    predictions={r['frame_id']:r['detections'] for r in rows(args.run/'predictions.jsonl')}
    result=score_reviewed(rows(args.dataset/'frames.jsonl'),predictions,rows(args.run/'ocr.jsonl'),rows(args.labels))
    write_json(args.output,{**result,'labels_sha256':file_hash(args.labels),
        'prediction_sha256':file_hash(args.run/'predictions.jsonl'),'ocr_sha256':file_hash(args.run/'ocr.jsonl')})


if __name__ == '__main__': main()
