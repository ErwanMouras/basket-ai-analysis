"""Rank only finished, supported results within identical data/protocol groups."""

import json
from pathlib import Path

from training.common.files import write_json
from training.common.provenance import object_hash
from .reports import write_csv, write_html


def compare(paths, output, *, include_smoke=False):
    rows, excluded, seen, protocols = [], [], set(), {}
    for path in paths:
        path = Path(path).resolve(strict=True)
        result_path = path / "result.json" if path.is_dir() else path
        summary_path = result_path.parent / "summary.json"
        if not summary_path.exists():
            excluded.append({"run_id": result_path.parent.name, "reason": "unfinished (no summary)"})
            continue
        summary = json.loads(summary_path.read_text())
        if summary["status"] != "FINISHED":
            excluded.append({"run_id": summary["run_id"], "reason": "unfinished"})
            continue
        result = json.loads(result_path.read_text())
        run_id = result["run_id"]
        if run_id in seen:
            continue
        seen.add(run_id)
        reason = None
        if result["comparison_id"] != object_hash(result["comparison"]):
            raise ValueError(f"Invalid comparison identity: {result_path}")
        protocols[result["comparison_id"]] = result["comparison"]
        if summary["run_id"] != run_id or summary["status"] != "FINISHED" or result["status"] != "FINISHED":
            reason = "unfinished"
        elif not include_smoke and (result["purpose"] == "smoke" or result["reference"].get("smoke")):
            reason = "smoke"
        elif result["metrics"]["global"]["ap50_95"] is None:
            reason = "no ground-truth support"
        if reason:
            excluded.append({"run_id": run_id, "reason": reason})
            continue
        contract = result["reference"].get("training_contract") or {}
        model, performance = result["model"], result["performance"]
        protocol = result["comparison"]["protocol"]
        environment = protocol.get("environment", {})
        rows.append({"comparison_id": result["comparison_id"], "run_id": run_id,
                     "dataset_id": result["comparison"]["dataset_id"],
                     "selection_id": result["comparison"].get("selection_id"),
                     "precision": protocol["precision"], "device": environment.get("device"),
                     "hardware": environment.get("gpu") or environment.get("cpu"),
                     "score_floor": protocol.get("score_floor"),
                     "score_threshold": protocol.get("score_threshold"),
                     "iou_threshold": protocol.get("iou_threshold"),
                     "max_detections": protocol.get("max_detections"),
                     "warmup": protocol.get("warmup"), "repeats": protocol.get("repeats"),
                     "family": model["family"], "variant": model["variant"],
                     "resolution": model["resolution"], "source_class": model["source_class"],
                     "purpose": result["purpose"], "reference": result["reference"].get("reference"),
                     "training_run_id": result["reference"].get("training_run_id"),
                     "epochs_budget": contract.get("epochs"), "seed": contract.get("seed"),
                     "training_selection_id": contract.get("selection_id"),
                     "training_max_images": contract.get("max_train_images"),
                     "training_batch_size": contract.get("batch_size"),
                     "training_precision": contract.get("precision"),
                     "training_resolution": contract.get("resolution"),
                     "initialization_sha256": contract.get("initial_weights_sha256"),
                     "checkpoint_sha256": model["checkpoint_sha256"],
                     **result["metrics"]["global"],
                     **{f"{scope}_{key}": performance[scope][key] for scope in ("forward", "adapter", "image_pipeline")
                        for key in ("median_ms", "p95_ms", "images_per_second")},
                     "gpu_peak_allocated_bytes": performance["gpu_peak_allocated_bytes"],
                     "gpu_peak_reserved_bytes": performance["gpu_peak_reserved_bytes"],
                     "cpu_rss_sampled_peak_bytes": performance["cpu_rss_sampled_peak_bytes"]})
    rows.sort(key=lambda r: (r["comparison_id"], -r["ap50_95"], -r["ap50"], r["run_id"]))
    previous, rank, last_score = None, 0, None
    for row in rows:
        if row["comparison_id"] != previous:
            previous, rank, last_score = row["comparison_id"], 0, None
        score = (row["ap50_95"], row["ap50"])
        if score != last_score:
            rank += 1
        row["rank_in_protocol"] = rank
        last_score = score
    output = Path(output)
    if output.exists():
        raise ValueError("Comparison output must be a new directory")
    output.mkdir(parents=True)
    result = {"ranking": "AP50:95 descending, then AP50; ties share rank; independent protocol groups",
              "include_smoke": include_smoke, "protocol_groups": protocols, "rows": rows, "excluded": excluded}
    write_json(output / "comparison.json", result)
    write_csv(output / "comparison.csv", rows, fields=None if rows else ["comparison_id", "run_id", "rank_in_protocol"])
    write_html(output / "comparison.html", "Players comparison — independent protocol groups", rows, details=result)
    return result
