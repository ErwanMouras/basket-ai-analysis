"""One COCO bbox evaluator and an explicit score-ordered one-to-one matcher."""

import contextlib
import io

import numpy as np

from training.players.export.geometry import Geometry

AREA_RANGES = {"all": (0, 1e10), "small": (0, 32**2),
               "medium": (32**2, 96**2), "large": (96**2, 1e10)}


def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    return intersection / (area(a) + area(b) - intersection)


def normalize_predictions(record, detections, *, score_floor=0.001, max_detections=100):
    """Invert the export transform, clip padding away, then stable sort and cap."""
    geometry = Geometry(**record["transform"])
    result = []
    for detection in detections:
        box = np.asarray(detection["bbox"], dtype=float)
        score = detection["confidence"]
        if (box.shape != (4,) or not np.isfinite(box).all()
                or type(score) not in (int, float) or not np.isfinite(score)
                or not 0 <= score <= 1 or detection["class_id"] != 0
                or box[2] <= box[0] or box[3] <= box[1]):
            raise ValueError("Invalid detector prediction")
        box = np.clip(geometry.bbox(box, inverse=True), [0, 0, 0, 0],
                      [geometry.source_width, geometry.source_height] * 2).tolist()
        if score >= score_floor and box[2] > box[0] and box[3] > box[1]:
            result.append({"class_id": 0, "bbox": box, "confidence": float(score)})
    return sorted(result, key=lambda d: -d["confidence"])[:max_detections]


def match_frame(boxes, predictions, *, score_threshold, iou_threshold, area_range=None):
    """Highest score first, best unmatched IoU >= threshold; ties use GT order.

    Area slices follow COCO: prefer in-range GT, ignore matches to out-of-range
    GT and unmatched out-of-range predictions. No crowd/reusable GT in players.
    """
    low, high = area_range or AREA_RANGES["all"]
    ignored = {i for i, b in enumerate(boxes) if not low <= area(b) <= high}
    unmatched = set(range(len(boxes)))
    events = []
    for index, prediction in sorted(enumerate(predictions), key=lambda p: -p[1]["confidence"]):
        if prediction["confidence"] < score_threshold:
            continue
        candidates = [(i, iou(prediction["bbox"], boxes[i])) for i in sorted(unmatched)]
        candidates = [(i, overlap) for i, overlap in candidates if overlap >= iou_threshold]
        if candidates:
            chosen, overlap = min(candidates, key=lambda item: (item[0] in ignored, -item[1], item[0]))
            unmatched.remove(chosen)
            if chosen not in ignored:
                events.append({"kind": "TP", "prediction_index": index, "gt_index": chosen, "iou": overlap})
        elif low <= area(prediction["bbox"]) <= high:
            events.append({"kind": "FP", "prediction_index": index, "gt_index": None, "iou": None})
    events += [{"kind": "FN", "prediction_index": None, "gt_index": i, "iou": None}
               for i in sorted(unmatched - ignored)]
    return events


def operating_metrics(records, predictions, *, score_threshold, iou_threshold, area_range=None):
    events = []
    for record in records:
        events.extend({"frame_id": record["frame_id"], **event} for event in match_frame(
            [b["source_bbox"] for b in record["boxes"]], predictions[record["frame_id"]],
            score_threshold=score_threshold, iou_threshold=iou_threshold, area_range=area_range))
    counts = {kind.lower(): sum(e["kind"] == kind for e in events) for kind in ("TP", "FP", "FN")}
    tp, fp, fn = (counts[k] for k in ("tp", "fp", "fn"))
    supported = tp + fn > 0
    return {**counts, "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if supported else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if supported else None}, events


def coco_metrics(records, predictions, *, max_detections=100, area_name="all"):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    images, annotations, detections = [], [], []
    for image_id, record in enumerate(records, 1):
        images.append({"id": image_id})
        for box in record["boxes"]:
            x1, y1, x2, y2 = box["source_bbox"]
            annotations.append({"id": len(annotations) + 1, "image_id": image_id,
                                "category_id": 1, "bbox": [x1, y1, x2-x1, y2-y1],
                                "area": (x2-x1)*(y2-y1), "iscrowd": 0})
        for prediction in predictions[record["frame_id"]]:
            x1, y1, x2, y2 = prediction["bbox"]
            detections.append({"image_id": image_id, "category_id": 1,
                               "bbox": [x1, y1, x2-x1, y2-y1], "score": prediction["confidence"]})
    low, high = AREA_RANGES[area_name]
    support = sum(low <= a["area"] <= high for a in annotations)
    if not support:
        return {"ap50": None, "ap50_95": None, "support": 0, "images": len(records)}, []
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {"images": images, "annotations": annotations,
                      "categories": [{"id": 1, "name": "player"}], "info": {}}
        gt.createIndex()
        if detections:
            dt = gt.loadRes(detections)
        else:
            dt = COCO()
            dt.dataset = {**gt.dataset, "annotations": []}
            dt.createIndex()
        evaluator = COCOeval(gt, dt, "bbox")
        evaluator.params.maxDets = [max_detections]
        evaluator.params.areaRng = [[low, high]]
        evaluator.params.areaRngLbl = [area_name]
        evaluator.evaluate()
        evaluator.accumulate()
    precision = evaluator.eval["precision"][:, :, 0, 0, 0]
    valid = precision[precision >= 0]
    curve = [{"iou": float(threshold), "recall": float(recall),
              "precision": float(precision[t, r]) if precision[t, r] >= 0 else None}
             for t, threshold in enumerate(evaluator.params.iouThrs)
             for r, recall in enumerate(evaluator.params.recThrs)]
    return {"ap50": float(precision[0][precision[0] >= 0].mean()),
            "ap50_95": float(valid.mean()), "support": support, "images": len(records)}, curve


def evaluate(records, predictions, protocol):
    if set(predictions) != {r["frame_id"] for r in records}:
        raise ValueError("Predictions must cover exactly the selected frames, including empty images")

    def measure(items, area_name="all"):
        ap, curve = coco_metrics(items, predictions, max_detections=protocol["max_detections"], area_name=area_name)
        operating, events = operating_metrics(items, predictions, score_threshold=protocol["score_threshold"],
                                             iou_threshold=protocol["iou_threshold"], area_range=AREA_RANGES[area_name])
        return {**ap, **operating}, curve, events

    global_metrics, pr_curve, events = measure(records)
    groups = []
    for key in ("match_id", "source_id", "venue_id"):
        for value in sorted({r[key] for r in records if r[key] is not None}):
            result, _, _ = measure([r for r in records if r[key] == value])
            groups.append({"group": key, "value": value, **result})
        missing = [r for r in records if r[key] is None]
        if missing:
            result, _, _ = measure(missing)
            groups.append({"group": key, "value": "unknown", **result})
    for name in ("small", "medium", "large"):
        result, _, _ = measure(records, name)
        groups.append({"group": "source_area", "value": name, **result})
    # Image cohorts retain every GT in an image; no selective annotation becomes a negative.
    for value in (True, False, None):
        items = [r for r in records if any(b.get("occluded") is value for b in r["boxes"])]
        result, _, _ = measure(items)
        groups.append({"group": "occlusion_image_cohort", "value": str(value), **result})
    thresholds = sorted(set(np.linspace(protocol["score_floor"], 1, 51).tolist() + [protocol["score_threshold"]]))
    score_curve = [{"score_threshold": score, **operating_metrics(
        records, predictions, score_threshold=score, iou_threshold=protocol["iou_threshold"])[0]}
        for score in thresholds]
    return {"global": global_metrics, "groups": groups}, {"coco_pr": pr_curve, "score": score_curve}, events
