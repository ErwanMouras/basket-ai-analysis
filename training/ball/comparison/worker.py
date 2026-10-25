"""Evaluate one checkpoint per process to isolate upstream Python imports."""

import argparse
import json
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from training.ball.evaluation.frames import frame_result, report, summarize
from training.ball.learning.data import frame_index
from training.ball.learning.metrics import center
from training.ball.learning.references import activate_reference
from training.ball.learning.torch_loop import seed_everything
from training.common.files import write_json, write_jsonl
from training.common.provenance import ROOT, file_hash
from training.common.tracking import tracked_run
from .protocol import identity, validate, windows


class Predictor:
    def __init__(self, candidate, protocol):
        self.candidate, self.protocol = candidate, protocol
        self.device = seed_everything(protocol)
        self.geometry = {k: candidate[k] for k in ("input_width", "input_height")}
        self.length = candidate["sequence_length"]
        self.yolo = candidate["model"] == "yolo"
        path = ROOT / candidate["checkpoint"]
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.epoch = checkpoint["epoch"]
        if self.yolo:
            from ultralytics import YOLO

            state = checkpoint["ball_training"]
            if (state["run_id"] != candidate["training_run_id"]
                    or state["contract"]["dataset_id"] != candidate["training_dataset_id"]
                    or state["contract"]["export_id"] != candidate["training_export_id"]):
                raise ValueError("YOLO checkpoint provenance differs from inventory")
            self.model = YOLO(str(path))
            self.model.model.float().to(self.device).eval()
            self.reference = None
        else:
            trained = checkpoint["config"]
            if (checkpoint["run_id"] != candidate["training_run_id"]
                    or trained["dataset_id"] != candidate["training_dataset_id"]
                    or trained["export_id"] != candidate["training_export_id"]
                    or trained["geometry"]["sequence_length"] != self.length):
                raise ValueError("Checkpoint provenance differs from inventory")
            reference = {"model": candidate["model"], "reference_root": str(ROOT / candidate["reference_root"])}
            activate_reference(reference)
            if reference["reference"] != trained["reference"]:
                raise ValueError("Installed upstream differs from checkpoint")
            self.reference = reference["reference"]
            if candidate["model"] == "tracknet_v3":
                from model import TrackNet

                self.model = TrackNet(in_dim=3 * self.length, out_dim=self.length)
            else:
                from models_factory import build_model

                self.model = build_model(trained["architecture"])
            self.model.load_state_dict(checkpoint["model"], strict=True)
            self.model.float().to(self.device).eval()

        def check_input(module, args):
            tensor = args[0]
            expected = (1, 3 * self.length, candidate["input_height"], candidate["input_width"])
            if tuple(tensor.shape) != expected or tensor.dtype != torch.float32:
                raise ValueError("Model received an unexpected shape or precision")

        network = self.model.model if self.yolo else self.model
        network.register_forward_pre_hook(check_input)

    def predict(self, images):
        w, h = self.geometry["input_width"], self.geometry["input_height"]
        if len(images) != self.length:
            raise ValueError("Incorrect context length")
        if self.yolo:
            # Exact rectangular shape, no extra stride padding; RGB -> BGR for
            # the public Ultralytics numpy adapter, which performs RGB conversion.
            image = cv2.cvtColor(cv2.resize(images[-1], (w, h)), cv2.COLOR_RGB2BGR)
            result = self.model.predict(
                image, imgsz=[h, w], rect=True, conf=self.protocol["thresholds"]["yolo"],
                iou=self.protocol["yolo_iou"], device=str(self.device),
                half=False, augment=False, verbose=False, classes=[0], max_det=100,
            )[0]
            boxes = result.boxes
            if len(boxes):
                index = int(boxes.conf.argmax())
                point = boxes.xywh[index, :2].detach().cpu().tolist()
                score = float(boxes.conf[index])
            else:
                point, score = None, None
        else:
            if self.candidate["model"] == "tracknet_v3":
                resized = [np.asarray(Image.fromarray(im).resize((w, h))) for im in images]
            else:
                resized = [cv2.resize(im, (w, h)) for im in images]
            image = np.concatenate(resized, axis=2).transpose(2, 0, 1).copy()
            tensor = torch.from_numpy(image).float().div_(255).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                output = self.model(tensor)
            if tuple(output.shape) != (1, self.length, h, w):
                raise ValueError("Unexpected heatmap shape")
            heatmap = output[0, -1].float().cpu().numpy()
            if not np.isfinite(heatmap).all() or heatmap.min() < 0 or heatmap.max() > 1:
                raise ValueError("Invalid heatmap probabilities")
            point = center(heatmap > self.protocol["thresholds"]["heatmap"])
            point = point.tolist() if point is not None else None
            score = float(heatmap.max())
        if point is not None and not np.isfinite(point).all():
            raise ValueError("Invalid predicted coordinates")
        return {"position_input": point, "confidence": score,
                "window_count": 1, "future_context_frames": 0}


def verify_inputs(candidate, protocol):
    for key in ("checkpoint", "config"):
        if file_hash(ROOT / candidate[key]) != candidate[key + "_sha256"]:
            raise ValueError(f"Changed {key}")
    for export in protocol["exports"]:
        root = ROOT / export["root"]
        for name, key in (("manifest.json", "manifest_sha256"), ("frames.jsonl", "frames_sha256")):
            if file_hash(root / name) != export[key]:
                raise ValueError("Changed export metadata")


def evaluate(protocol_path, candidate_id, output):
    protocol = json.loads(Path(protocol_path).read_text())
    validate(protocol)
    candidate = next(c for c in protocol["candidates"] if c["candidate_id"] == candidate_id)
    verify_inputs(candidate, protocol)
    root = ROOT / protocol["exports"][0]["root"]
    manifest = json.loads((root / "manifest.json").read_text())
    frames = sorted(frame_index(root, "val").values(), key=identity)
    sources = {s["clip_id"]: s for s in manifest["sources"]}
    samples = list(windows(frames, candidate["sequence_length"], protocol["cut_frames"]))
    common = {tuple(key) for key in protocol["common_targets"]}
    artifact_hashes = {row["path"]: row["sha256"] for row in
                       map(json.loads, (root / "artifacts.jsonl").read_text().splitlines())}

    @lru_cache(maxsize=16)
    def read_image(relative):
        path = root / relative
        if file_hash(path) != artifact_hashes[relative]:
            raise ValueError(f"Changed inference image: {relative}")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot decode {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    config = {**protocol, "model": candidate_id, "output": str(output),
              "dataset_id": manifest["dataset_id"], "export_id": manifest["export_id"],
              "purpose": "validation_comparison", "candidate": candidate}
    with tracked_run(config, manifest, root=ROOT,
                     source_paths=[ROOT / p for p in protocol["evaluator_sha256"]]) as (client, run_id, directory):
        write_json(directory / "protocol.json", protocol)
        predictor = Predictor(candidate, protocol)
        predicted = {}
        for index, sample in enumerate(samples):
            images = [read_image(f["image"]) for f in sample]
            predicted[identity(sample[-1])] = predictor.predict(images)
            if (index + 1) % 500 == 0:
                print(f"{candidate_id}: {index + 1}/{len(samples)}", flush=True)
        rows = [frame_result(f, sources[f["clip_id"]], predicted.get(identity(f)),
                             predictor.geometry, protocol) for f in frames]
        common_rows = [r for r in rows if identity(r) in common]
        if len(common_rows) != len(common) or any(r["outcome"] == "uncovered" for r in common_rows):
            raise ValueError("Common cohort is not completely covered")
        metrics = {"common": report(common_rows), "available": report(rows)}
        total = metrics["available"]["overall"]
        metrics["coverage_adjusted_f1"] = 2 * total["tp"] / (
            2 * total["tp"] + total["fp"] + total["fn"] + total["uncovered_frames"])
        metrics["sensitivity"] = {}
        for tolerance in protocol["sensitivity_tolerances_px"]:
            rescored = [{**r, "outcome": ("tp" if r["error_source_px"] <= tolerance else "displaced")}
                        if r["outcome"] in ("tp", "displaced") else r for r in common_rows]
            metrics["sensitivity"][str(tolerance)] = summarize(rescored)
        # A separate, repeated timing pass keeps disk access and initial CUDA/model
        # setup out of adapter latency. All candidates use the same target indices.
        timing = protocol["timing"]
        benchmark = [s for s in samples if identity(s[-1]) in common]
        selection = np.linspace(0, len(benchmark) - 1, timing["windows"], dtype=int)
        sample = benchmark[int(selection[0])]
        images = [read_image(f["image"]) for f in sample]
        for _ in range(timing["warmup"]):
            predictor.predict(images)
        torch.cuda.synchronize(predictor.device)
        torch.cuda.reset_peak_memory_stats(predictor.device)
        durations = []
        for _ in range(timing["repeats"]):
            for index in selection:
                images = [read_image(f["image"]) for f in benchmark[int(index)]]
                torch.cuda.synchronize(predictor.device)
                start = time.perf_counter()
                predictor.predict(images)
                torch.cuda.synchronize(predictor.device)
                durations.append(1000 * (time.perf_counter() - start))
        metrics["latency"] = {"samples": len(durations),
                              "median_ms": float(np.median(durations)),
                              "p95_ms": float(np.percentile(durations, 95)),
                              "peak_allocated_mb": torch.cuda.max_memory_allocated(predictor.device) / 2**20,
                              "hardware": torch.cuda.get_device_name(predictor.device),
                              "scope": timing["scope"]}
        verify_inputs(candidate, protocol)
        validate(protocol)
        result = {"candidate": candidate, "evaluation_run_id": run_id,
                  "protocol_id": protocol["protocol_id"], "checkpoint_epoch_zero_based": predictor.epoch,
                  "reference": predictor.reference, "metrics": metrics}
        write_json(directory / "metrics.json", result)
        write_jsonl(directory / "predictions.jsonl", rows)
        for path in (directory / "protocol.json", directory / "metrics.json", directory / "predictions.jsonl"):
            client.log_artifact(run_id, str(path), "comparison")
        for key in ("precision", "recall", "f1", "error_source_median_px", "error_source_p95_px"):
            value = metrics["common"]["overall"][key]
            if value is not None:
                client.log_metric(run_id, "common." + key, value)
        client.log_metric(run_id, "available.coverage", total["coverage"])
        client.log_metric(run_id, "coverage_adjusted_f1", metrics["coverage_adjusted_f1"])
        for key in ("median_ms", "p95_ms"):
            client.log_metric(run_id, "latency." + key, metrics["latency"][key])
        client.set_tag(run_id, "training_run_id", candidate["training_run_id"])
        client.set_tag(run_id, "protocol_id", protocol["protocol_id"])
        print(json.dumps({"candidate": candidate_id, "common": metrics["common"]["overall"],
                          "latency": metrics["latency"]}), flush=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    evaluate(args.protocol, args.candidate, args.output)


if __name__ == "__main__":
    main()
