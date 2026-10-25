"""Freeze and execute a paired, non-training segmentation experiment."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import time

import cv2
import numpy as np

from pipeline.config import load_config
from pipeline.adapters import People
from training.common.files import write_json, write_jsonl
from training.common.provenance import ROOT, file_hash, object_hash, source_fingerprints, git_state
from training.players.export.verify import verify_export
from training.players.evaluation.metrics import coco_metrics, operating_metrics, match_frame
from training.players.evaluation.benchmark import runtime, summarize
from training.segmentation.adapters import SegmentedPeople

RECIPES = ("baseline", "box_prompt", "direct_l", "direct_x")


def code_identity():
    paths = [p for folder in ("training/segmentation", "training/jersey", "training/players", "training/common", "pipeline")
             for p in (ROOT / folder).rglob("*.py") if "tests" not in p.parts]
    return source_fingerprints(ROOT, paths)


def records_at(root):
    records = [json.loads(l) for l in (root / "frames.jsonl").read_text().splitlines()]
    if not records or any(r["review_status"] != "verified" or r["split"] not in ("train", "val") for r in records):
        raise ValueError("Use verified train/val frames only")
    if any(r["transform"][k] != 1 for r in records for k in ("scale_x", "scale_y")):
        raise ValueError("Experiment requires native source-pixel exports")
    return records


def freeze(dataset, output):
    root = Path(dataset).resolve(strict=True)
    manifest = verify_export(root)
    records = records_at(root)
    configurations = {}
    for name in RECIPES:
        cfg = load_config("configs/segmentation/" + ("box-prompt" if name in ("baseline", "box_prompt") else name.replace("_", "-")) + ".yaml")
        if name == "baseline":
            cfg["segmentation"]["strategy"] = "disabled"
            cfg["segmentation"]["weights"] = None
            cfg["jersey"]["mask_policy"] = "disabled"
        configurations[name] = cfg
    weights = {str(Path(cfg[section]["weights"])): file_hash(Path(cfg[section]["weights"]))
               for cfg in configurations.values() for section in ("people", "pose", "jersey", "segmentation")
               if cfg[section]["weights"]}
    selected = {}
    for source in sorted({r["source_id"] for r in records}):
        items = sorted([r for r in records if r["source_id"] == source], key=lambda r: r["frame_index"])
        selected[source] = {"timing": [r["frame_id"] for r in items[::10]],
                            "ocr": [r["frame_id"] for r in items[::15]]}
    protocol = {
        "schema_version": 1, "status": "frozen_before_inference", "dataset": str(root.relative_to(ROOT)),
        "dataset_id": manifest["dataset_id"], "export_id": manifest["export_id"], "selection_id": object_hash(records),
        "frames": len(records), "cohorts": dict(Counter(f'{r["split"]}/{r["match_id"]}' for r in records)),
        "configurations": configurations, "weights_sha256": weights,
        "code_sha256": code_identity(), "git": git_state(ROOT), "subsets": selected,
        "device": "cuda:0", "cpu_threads": 4, "precision": "fp32", "batch_size": 1,
        "score_floor": 0.001, "score_threshold": 0.25, "iou_threshold": 0.5,
        "max_detections": 100, "warmup": 5, "repeats": 3,
        "role_policy": "COCO person excluding referee detections at score >= .25 / IoU >= .5 is a player proxy; no GT filtering of predictions",
        "ap": "bbox COCO .50:.05:.95, score floor .001, separate low-floor pass; not mask AP",
        "timing": "decoded source BGR -> CPU boxes and ROI-RLE masks; serialized JSON/disk/pose/OCR excluded; synchronized CUDA; stratified every tenth verified frame; three repeats",
        "ocr": "every fifteenth verified frame per source, RTMPose + existing crop gates and PARSeq; paired raw/neutral crops; no roster filter; no temporal vote or wall-clock budget; accepted EOS digits at confidence >= .9",
        "interpretation": "Train/val here describe local data only. Pretrained COCO segmenters were not fine-tuned; no claim of independent unseen footage. Human bbox annotations do not measure mask or jersey accuracy.",
    }
    protocol["protocol_id"] = object_hash(protocol)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", protocol)
    print(json.dumps({"protocol": str(output / "protocol.json"), "frames": len(records), "cohorts": protocol["cohorts"]}), flush=True)


def verify_protocol(protocol, root, records):
    unsigned = {k: v for k, v in protocol.items() if k != "protocol_id"}
    if object_hash(unsigned) != protocol["protocol_id"]:
        raise ValueError("Protocol fingerprint mismatch")
    if code_identity() != protocol["code_sha256"]:
        raise ValueError("Code changed since freeze; freeze a new protocol before inference")
    manifest = verify_export(root)
    if manifest["export_id"] != protocol["export_id"] or object_hash(records) != protocol["selection_id"]:
        raise ValueError("Data changed since freeze")
    if any(file_hash(Path(p)) != sha for p, sha in protocol["weights_sha256"].items()):
        raise ValueError("Weights changed since freeze")


def detection_metrics(records, predictions, protocol):
    groups = {"all": records}
    for key in ("split", "match_id"):
        groups.update({f"{key}/{value}": [r for r in records if r[key] == value] for value in sorted({r[key] for r in records})})
    output = {}
    for cohort, items in groups.items():
        output[cohort] = {}
        for cls, role in ((0, "player"), (1, "referee")):
            metrics, _ = operating_metrics(items, predictions, class_id=cls,
                score_threshold=protocol["score_threshold"], iou_threshold=protocol["iou_threshold"])
            ap, _ = coco_metrics(items, predictions, class_id=cls, max_detections=protocol["max_detections"])
            output[cohort][role] = {**metrics, **ap}
    return output


def run(protocol_path, name, output):
    import torch
    import psutil
    from training.players.pose import PlayerPose
    from training.jersey.crops import candidate
    from training.jersey.reader import ParseqReader
    protocol = json.loads(Path(protocol_path).read_text())
    root = ROOT / protocol["dataset"]
    records = records_at(root)
    verify_protocol(protocol, root, records)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"status": "running", "protocol_id": protocol["protocol_id"], "recipe": name})
    cfg = deepcopy(protocol["configurations"][name])
    environment = runtime(protocol)
    environment["ultralytics_prediction_geometry"] = "native rectangular letterbox at configured imgsz; SAM square 1024"
    load_started = time.perf_counter()
    model = People(cfg["people"]) if name == "baseline" else SegmentedPeople(cfg["people"], cfg["segmentation"])
    torch.cuda.synchronize(protocol["device"])
    load_seconds = time.perf_counter() - load_started
    def read(record):
        path = root / record["image"]
        if file_hash(path) != record["image_sha256"]:
            raise ValueError("Image content changed")
        image = cv2.imread(str(path))
        if image is None or image.shape[:2] != (record["height"], record["width"]):
            raise ValueError("Image dimensions changed")
        return image
    first = read(records[0])
    for _ in range(protocol["warmup"]):
        model.predict(first)
    torch.cuda.synchronize(protocol["device"])
    torch.cuda.reset_peak_memory_stats(protocol["device"])
    timings, rss = [], psutil.Process().memory_info().rss
    timing_ids = {f for group in protocol["subsets"].values() for f in group["timing"]}
    timing_frames = [r for r in records if r["frame_id"] in timing_ids]
    for _ in range(protocol["repeats"]):
        for r in timing_frames:
            image = read(r)
            torch.cuda.synchronize(protocol["device"])
            started = time.perf_counter()
            model.predict(image)
            torch.cuda.synchronize(protocol["device"])
            timings.append(time.perf_counter() - started)
            rss = max(rss, psutil.Process().memory_info().rss)
    performance = {**summarize(timings), "model_load_seconds": load_seconds,
        "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated(protocol["device"]) / 1024**2,
        "peak_cuda_reserved_mb": torch.cuda.max_memory_reserved(protocol["device"]) / 1024**2,
        "sampled_rss_mb": rss / 1024**2, "timing_frame_ids": [r["frame_id"] for r in timing_frames],
        "samples_seconds": timings, "scope": protocol["timing"], "full_pipeline_fps": None}
    write_json(output / "performance.json", performance)
    # Pose/OCR models load only after the perception memory/timing measurement.
    pose = PlayerPose(cfg["pose"])
    reader = ParseqReader(cfg["jersey"])
    ocr_ids = {f for group in protocol["subsets"].values() for f in group["ocr"]}
    ap_predictions, operating_predictions = {}, {}
    crop_rows = []
    with (output / "predictions.partial.jsonl").open("w") as handle:
        for index, r in enumerate(records):
            image = read(r)
            ap_rows = model.predict(image, confidence=protocol["score_floor"]) if name == "baseline" else model.predict(image, masks=False, confidence=protocol["score_floor"])
            ap_predictions[r["frame_id"]] = [{"bbox": d["bbox"], "confidence": d["confidence"],
                "class_id": 1 if d["role"] == "referee" else 0} for d in ap_rows if d["role"] != "unknown"]
            rows = model.predict(image)
            operating_predictions[r["frame_id"]] = [{"bbox": d["bbox"], "confidence": d["confidence"],
                "class_id": 1 if d["role"] == "referee" else 0} for d in rows if d["role"] != "unknown"]
            handle.write(json.dumps({"frame_id": r["frame_id"], "match_id": r["match_id"], "split": r["split"],
                "frame_index": r["frame_index"], "detections": rows, "ap_detections": ap_predictions[r["frame_id"]]}, allow_nan=False) + "\n")
            if r["frame_id"] in ocr_ids:
                posed = pose.predict(image, rows)
                for i, d in enumerate(posed):
                    if d["role"] != "player":
                        continue
                    crops, pending = {}, []
                    for policy in ("disabled", "background_neutral"):
                        crop = candidate(image, d, posed, {**cfg["jersey"], "mask_policy": policy}) if d["confidence"] >= cfg["jersey"]["min_detection_confidence"] else None
                        info = {"eligible": crop is not None, "bbox": crop["bbox"] if crop else None,
                                "mask_audit": crop["mask_audit"] if crop else None}
                        crops[policy] = info
                        if crop:
                            pending.append((policy, crop))
                    if pending:
                        readings = reader.read([crop["image"] for _, crop in pending])
                        for (policy, crop), reading in zip(pending, readings, strict=True):
                            accepted = reading["eos"] and reading["text"].isascii() and reading["text"].isdigit() and 1 <= len(reading["text"]) <= 2 and reading["confidence"] >= cfg["jersey"]["min_ocr_confidence"]
                            crops[policy].update(reading=reading, accepted=accepted)
                    crop_rows.append({"frame_id": r["frame_id"], "prediction_index": i,
                        "match_id": r["match_id"], "bbox": d["bbox"], "crops": crops})
            if index % 50 == 0:
                print(json.dumps({"recipe": name, "frames": index+1, "total": len(records)}), flush=True)
    (output / "predictions.partial.jsonl").rename(output / "predictions.jsonl")
    write_jsonl(output / "ocr.jsonl", crop_rows)
    # COCO AP uses the low-floor pass; operational P/R use exactly the production call.
    ap_metrics = detection_metrics(records, ap_predictions, protocol)
    op_metrics = detection_metrics(records, operating_predictions, protocol)
    for cohort, roles in ap_metrics.items():
        for role, m in roles.items():
            m.update({k: op_metrics[cohort][role][k] for k in ("tp", "fp", "fn", "precision", "recall", "f1")})
    verify_protocol(protocol, root, records)
    artifacts = {p.name: file_hash(p) for p in output.iterdir() if p.name != "status.json"}
    result = {"status": "finished", "recipe": name, "protocol_id": protocol["protocol_id"],
        "model": model.provenance, "environment": environment, "detection": ap_metrics,
        "performance": performance, "ocr": ocr_summary(crop_rows),
        "mask_accuracy": None, "jersey_accuracy": None,
        "missing_accuracy_reason": "No reviewed source-aligned mask or jersey-number labels; confidence/read counts are not accuracy", "artifacts_sha256": artifacts}
    write_json(output / "result.json", result)
    write_json(output / "status.json", {"status": "finished", "protocol_id": protocol["protocol_id"]})
    print(json.dumps({"recipe": name, "player": ap_metrics["all"]["player"], "median_ms": performance["median_ms"], "ocr": result["ocr"]}), flush=True)


def ocr_summary(rows):
    result = {"person_observations": len(rows), "accuracy": None}
    for policy in ("disabled", "background_neutral"):
        crops = [r["crops"][policy] for r in rows]
        eligible = [c for c in crops if c["eligible"]]
        result[policy] = {"eligible": len(eligible), "accepted_reads": sum(c.get("accepted", False) for c in eligible),
            "applied_masks": sum(c["mask_audit"]["status"] == "applied" for c in eligible),
            "raw_fallbacks": sum(c["mask_audit"]["status"] == "raw_fallback" for c in eligible)}
    both = [r for r in rows if all(r["crops"][p].get("accepted") for p in ("disabled", "background_neutral"))]
    result["paired_accepted"] = len(both)
    result["paired_number_changes"] = sum(r["crops"]["disabled"]["reading"]["text"] != r["crops"]["background_neutral"]["reading"]["text"] for r in both)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--dataset", required=True)
    f.add_argument("--output", required=True)
    r = sub.add_parser("run")
    r.add_argument("--protocol", required=True)
    r.add_argument("--recipe", choices=RECIPES, required=True)
    r.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.dataset, args.output)
    else:
        was_present = Path(args.output).exists()
        try:
            run(args.protocol, args.recipe, args.output)
        except BaseException as exc:
            if not was_present and Path(args.output).is_dir():
                write_json(Path(args.output) / "status.json", {"status": "failed", "type": type(exc).__name__, "error": str(exc)})
            raise


if __name__ == "__main__":
    main()
