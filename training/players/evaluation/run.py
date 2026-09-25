"""Evaluate local pretrained/full checkpoints on verified exports, with MLflow."""

import json
from pathlib import Path

import cv2

from training.common.files import write_json, write_jsonl
from training.common.provenance import ROOT, file_hash, object_hash, source_fingerprints
from training.players.export.verify import verify_export
from training.players.learning.tracking import tracked_run
from training.players.models import Detector
from .benchmark import benchmark, runtime
from .config import frozen_payload, require_frozen
from .metrics import evaluate
from training.players.inference import predict_image
from training.players.progress import emit
from .reports import artifacts, write_html


def prepare(config):
    root = Path(config["dataset"]).resolve(strict=True)
    config["dataset"] = str(root)
    output = Path(config["output"]).resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Dataset and output must be disjoint")
    manifest = verify_export(root)
    records = [json.loads(line) for line in (root / "frames.jsonl").read_text().splitlines()]
    records = [r for r in records if r["split"] == config["split"]][:config["max_images"]]
    if not records:
        raise ValueError("Evaluation selection is empty")
    code = source_fingerprints(ROOT, list((ROOT / "training/players/evaluation").glob("*.py"))
                               + [ROOT / "training/players/models.py", ROOT / "training/players/inference.py",
                                  ROOT / "training/players/export/geometry.py"])
    identity = {"dataset_id": manifest["dataset_id"], "selection_id": object_hash(records),
                "checkpoint_sha256": file_hash(Path(config["weights"])), "evaluator_id": object_hash(code)}
    return root, manifest, records, identity, code


def checkpoint_reference(config, expected_sha256):
    import torch

    path = Path(config["weights"])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("players_training") if isinstance(payload, dict) else None
    if state and state["family"] != config["model"]:
        raise ValueError("Checkpoint family does not match evaluation recipe")
    if not state and not config["reference"]:
        raise ValueError("Imported/pretrained weights require an explicit reference describing their origin")
    if file_hash(path) != expected_sha256:
        raise ValueError("Checkpoint changed during loading")
    return {"kind": "players-training" if state else "imported", "reference": config["reference"],
            "training_run_id": state["run_id"] if state else None,
            "training_contract": state["contract"] if state else None,
            "checkpoint_sha256": expected_sha256,
            "smoke": bool(state and state["contract"].get("purpose") == "smoke")}


def run(config):
    root, manifest, records, identity, code = prepare(config)
    require_frozen(config, frozen_payload(config, **identity))
    reference = checkpoint_reference(config, identity["checkpoint_sha256"])
    if config["split"] == "test" and reference["smoke"]:
        raise ValueError("Smoke checkpoints cannot enter the final test")
    environment = runtime(config)
    protocol = {k: config[k] for k in ("split", "precision", "resolution", "cpu_threads", "score_floor",
                                      "score_threshold", "iou_threshold", "max_detections", "warmup", "repeats")}
    protocol.update(schema_version=1, evaluator_id=identity["evaluator_id"], environment=environment,
                    batch_size=1, tf32=False, matching="score descending; best unmatched IoU; GT order tie",
                    coordinates="source XYXY; inverse export transform; clip to source; stable score sort",
                    coco="bbox; IoU .50:.05:.95; 101 recall points; one player class",
                    inference="pinned native preprocess/postprocess; square input; no added NMS; cap after class filter",
                    timing="perf_counter; CUDA sync around calls; forward hooks in separate pass; warm filesystem cache",
                    memory="torch CUDA peak allocated/reserved after warmup; process RSS sampled after each image",
                    occlusion="overlapping image cohorts; retain all GT in each image; unknown explicit")
    comparison = {"dataset_id": identity["dataset_id"], "selection_id": identity["selection_id"], "protocol": protocol}
    tracked_config = {**config, **identity, "export_id": manifest["export_id"], "experiment": "players-evaluation"}
    if reference["smoke"]:
        tracked_config["purpose"] = "smoke"
    with tracked_run(tracked_config, manifest) as (client, run_id, output):
        try:
            write_json(output / "recipe.json", config)
            write_json(output / "protocol.json", protocol)
            write_json(output / "reference.json", reference)
            write_json(output / "code.json", code)
            write_jsonl(output / "selection.jsonl", records)
            if config["frozen_recipe"]:
                write_json(output / "frozen-recipe.json", json.loads(Path(config["frozen_recipe"]).read_text()))
            client.set_tag(run_id, "training_run_id", reference["training_run_id"] or "imported")
            client.set_tag(run_id, "comparison_id", object_hash(comparison))
            detector = Detector(config["model"], config["variant"], config["weights"], device=config["device"],
                                resolution=config["resolution"], source_class=config["source_class"])
            if detector.provenance["checkpoint_sha256"] != identity["checkpoint_sha256"]:
                raise ValueError("Checkpoint changed while constructing detector")
            predictions = {}
            for index, record in enumerate(records):
                image = cv2.imread(str(root / record["image"]))
                if image is None or image.shape[:2] != (record["height"], record["width"]):
                    raise ValueError("Invalid evaluation image")
                predictions[record["frame_id"]] = predict_image(detector, image, config, record)
                emit(batch=index + 1, batches_total=len(records))
            write_jsonl(output / "predictions.jsonl", [
                {"artifact_type": "players_evaluation_predictions", "schema_version": 1,
                 "coordinate_space": "source", "frame_id": r["frame_id"], "source_id": r["source_id"],
                 "frame_index": r["frame_index"], "detections": predictions[r["frame_id"]]} for r in records])
            metrics, curves, events = evaluate(records, predictions, protocol)
            performance = benchmark(detector, records, root, config)
            # Verify the immutable export again before declaring a successful result.
            verify_export(root)
            write_json(output / "performance.json", performance)
            rows, links = artifacts(output, root, records, predictions, metrics, curves, events, config["max_examples"])
            result = {"schema_version": 1, "run_id": run_id, "status": "FINISHED", "model": detector.provenance,
                      "purpose": tracked_config["purpose"], "reference": reference, "comparison": comparison,
                      "comparison_id": object_hash(comparison), "metrics": metrics, "performance": performance}
            write_json(output / "result.json", result)
            write_html(output / "report.html", "Players evaluation", rows, details=result,
                       links=["curves.svg", "predictions.jsonl", "review.csv", *links])
            for key, value in metrics["global"].items():
                if value is not None:
                    client.log_metric(run_id, key, value)
            for scope in ("forward", "adapter", "image_pipeline"):
                for key, value in performance[scope].items():
                    client.log_metric(run_id, f"{scope}.{key}", value)
            for key in ("gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes", "cpu_rss_sampled_peak_bytes"):
                if performance[key] is not None:
                    client.log_metric(run_id, key, performance[key])
        except BaseException as error:
            write_json(output / "error.json", {"type": type(error).__name__, "message": str(error)})
            raise
        finally:
            client.log_artifacts(run_id, str(output), "evaluation")
    return output
