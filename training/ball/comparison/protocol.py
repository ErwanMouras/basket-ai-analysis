"""Freeze a validation comparison before looking at its predictions."""

import json
from pathlib import Path

import yaml

from training.ball.export.dataset import verify_dataset
from training.ball.learning.data import frame_index
from training.common.files import write_json
from training.common.provenance import ROOT, file_hash, object_hash, source_fingerprints


def identity(frame):
    return frame["clip_id"], frame["frame_index"]


def windows(frames, length, cuts=None):
    """Only contiguous past/present frames; reset on clips, gaps and known cuts."""
    cuts = cuts or {}
    pending = []
    for frame in sorted(frames, key=identity):
        if pending and (
            frame["clip_id"] != pending[-1]["clip_id"]
            or frame["frame_index"] != pending[-1]["frame_index"] + 1
            or frame["frame_index"] in cuts.get(frame["clip_id"], [])
        ):
            pending = []
        pending.append(frame)
        pending = pending[-length:]
        if len(pending) == length:
            yield pending.copy()


def evaluator_hashes():
    paths = list((ROOT / "training/ball/comparison").glob("*.py"))
    paths += [ROOT / name for name in (
        "training/ball/evaluation/frames.py", "training/ball/learning/metrics.py",
        "training/ball/learning/references.py", "training/common/provenance.py",
    )]
    return source_fingerprints(ROOT, paths)


def validate(protocol):
    body = {k: v for k, v in protocol.items() if k != "protocol_id"}
    if object_hash(body) != protocol["protocol_id"]:
        raise ValueError("Protocol fingerprint mismatch")
    if protocol["evaluator_sha256"] != evaluator_hashes():
        raise ValueError("Evaluator changed after protocol freeze; freeze a new comparison")
    if protocol["split"] != "val":
        raise ValueError("This selection workflow must not consume test data")


def inventory(uri):
    """Read all MLflow ball training runs, including unsuccessful/smoke runs."""
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=uri)
    experiments = client.search_experiments()
    runs = client.search_runs([e.experiment_id for e in experiments], max_results=50000)
    result = []
    for run in sorted(runs, key=lambda r: r.info.start_time):
        p = run.data.params
        if p.get("model") not in ("yolo", "tracknet_v3", "tracknet_v4", "tracknet_v5", "tracknet_v5_totnet") or "epochs" not in p:
            continue
        limited = any(p.get(k, "None") not in ("None", "0", "null", "")
                      for k in ("max_train_batches", "max_val_batches"))
        reason = ("incomplete run" if run.info.status != "FINISHED" else
                  "limited-batch smoke test" if limited else
                  "historical two-epoch baseline" if int(p["epochs"]) < 30 else
                  "completed full training")
        selected = run.info.status == "FINISHED" and not limited
        result.append({
            "training_run_id": run.info.run_id, "name": run.info.run_name,
            "model": p["model"], "status": run.info.status,
            "epochs": int(p["epochs"]), "training_dataset_id": p.get("dataset_id"),
            "selected": selected, "reason": reason,
            "artifact_uri": run.info.artifact_uri,
            "output": p.get("output"),
        })
    return result


def freeze(output, uri, export_roots):
    from mlflow import MlflowClient

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    exports, canonical = [], None
    for root in export_roots:
        root = Path(root).resolve(strict=True)
        print(f"Verifying export {root}", flush=True)
        verify_dataset(root)
        manifest = json.loads((root / "manifest.json").read_text())
        train_matches = {s["match_id"] for s in manifest["sources"] if s["split"] == "train"}
        val_matches = {s["match_id"] for s in manifest["sources"] if s["split"] == "val"}
        if None in train_matches | val_matches or train_matches & val_matches:
            raise ValueError("Missing match identities or train/validation match overlap")
        frames = sorted(frame_index(root, "val").values(), key=identity)
        image_hashes = {row["path"]: row["sha256"] for row in
                        map(json.loads, (root / "artifacts.jsonl").read_text().splitlines())}
        signature = [{"frame": f, "image_sha256": image_hashes[f["image"]]} for f in frames]
        if canonical is not None and signature != canonical:
            raise ValueError("Exports do not contain identical validation annotations and JPEG bytes")
        canonical = signature
        exports.append({"root": root.relative_to(ROOT).as_posix(),
                        "dataset_id": manifest["dataset_id"], "export_id": manifest["export_id"],
                        "manifest_sha256": file_hash(root / "manifest.json"),
                        "frames_sha256": file_hash(root / "frames.jsonl")})
    if len({e["dataset_id"] for e in exports}) != 1:
        raise ValueError("Exports describe different datasets")
    runs = inventory(uri)
    client = MlflowClient(tracking_uri=uri)
    candidates = []
    for run in runs:
        if not run["selected"]:
            continue
        directory = Path(run["output"]) / run["training_run_id"]
        config = yaml.safe_load((directory / "config.resolved.yaml").read_text())
        model = run["model"]
        label = (Path(config["weights"]).stem if model == "yolo" else model)
        if run["epochs"] < 30:
            label += "_2epochs"
        weights = directory / "yolo/weights" if model == "yolo" else directory
        for filename in ("best.pt", "best_f1.pt"):
            path = weights / filename
            if not path.is_file():
                if filename == "best.pt":
                    raise FileNotFoundError(path)
                continue
            digest = file_hash(path)
            artifact = "checkpoints/" + filename
            logged = Path(client.download_artifacts(run["training_run_id"], artifact))
            if file_hash(logged) != digest:
                raise ValueError("Local checkpoint differs from its MLflow artifact")
            candidates.append({
                "candidate_id": label + "_" + path.stem,
                "model": model, "training_run_id": run["training_run_id"],
                "training_dataset_id": run["training_dataset_id"],
                "training_export_id": config["export_id"],
                "epochs": run["epochs"], "checkpoint": path.relative_to(ROOT).as_posix(),
                "checkpoint_sha256": digest, "mlflow_artifact": artifact,
                "sequence_length": config.get("geometry", {}).get("sequence_length", 1),
                "input_width": config.get("geometry", config).get("input_width"),
                "input_height": config.get("geometry", config).get("input_height"),
                "config": (directory / "config.resolved.yaml").relative_to(ROOT).as_posix(),
                "config_sha256": file_hash(directory / "config.resolved.yaml"),
                "reference_root": ".external/" + ("tracknet_v5" if model == "tracknet_v5_totnet" else model),
            })
    if not candidates:
        raise ValueError("No completed ball models available")
    if any((c["input_width"], c["input_height"]) != (512, 288) for c in candidates):
        raise ValueError("This comparison requires 512x288 checkpoints")
    # The old baseline's training data differs; retain it as an explicit reference.
    longest = max(c["sequence_length"] for c in candidates)
    common = [identity(w[-1]) for w in windows(frames, longest)]
    if not common:
        raise ValueError("No common causal targets")
    protocol = {
        "schema_version": 1, "purpose": "validation_selection", "split": "val",
        "mlflow_uri": uri, "experiment": "ball-model-comparison",
        "exports": exports, "candidates": candidates, "inventory": runs,
        "validation_signature_sha256": object_hash(canonical),
        "common_targets": common, "common_targets_sha256": object_hash(common),
        "temporal_policy": "causal_last", "cut_frames": {},
        "cut_annotations": "unavailable; reset at clip boundaries and missing exported frames only",
        "thresholds": {"heatmap": 0.5, "yolo": 0.25}, "yolo_iou": 0.7,
        "yolo_selection": "highest-confidence class-0 box, then its center",
        "tolerance_px": 15, "tolerance_space": "source",
        "sensitivity_tolerances_px": [5, 10, 15, 25], "unknown_position": "ignore",
        "checkpoint_policy": "evaluate every available best.pt and best_f1.pt; no last.pt; select on validation explicitly",
        "device": "cuda:0", "precision": "fp32", "batch_size": 1,
        "cpu_threads": 4, "seed": 53, "deterministic": True,
        "timing": {"warmup": 5, "windows": 30, "repeats": 3,
                   "scope": "decoded RGB windows to input-coordinate prediction; includes preprocessing, GPU transfer, forward, decoding; excludes disk I/O, loading, source-coordinate restoration and other pipeline stages"},
        "evaluator_sha256": evaluator_hashes(),
    }
    protocol["protocol_id"] = object_hash(protocol)
    write_json(output, protocol)
    print(f"Frozen {len(candidates)} checkpoints, {len(common)}/{len(frames)} common frames: {protocol['protocol_id']}", flush=True)
    return protocol
