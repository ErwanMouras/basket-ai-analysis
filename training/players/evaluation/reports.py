"""Portable JSON/CSV/HTML reports and bounded FP/FN review artifacts."""

import csv
import html
import json
from pathlib import Path

import cv2

from training.common.files import atomic_writer, write_json
from training.players.export.geometry import Geometry


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with atomic_writer(path) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_html(path, title, rows, *, details=None, links=()):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    escape = lambda value: html.escape(str(value)) if value is not None else "—"
    table = "<tr>" + "".join(f"<th>{escape(k)}</th>" for k in columns) + "</tr>"
    for row in rows:
        table += "<tr>" + "".join(f"<td>{escape(row.get(k))}</td>" for k in columns) + "</tr>"
    with atomic_writer(path) as handle:
        handle.write('<!doctype html><meta charset="utf-8"><title>' + escape(title) + '</title>'
                     '<style>body{font:15px system-ui;margin:2rem}table{border-collapse:collapse}'
                     'td,th{border:1px solid #bbb;padding:.5rem}pre{white-space:pre-wrap}</style>'
                     f'<h1>{escape(title)}</h1><table>{table}</table>'
                     + ''.join(f'<p><a href="{escape(link)}">{escape(link)}</a></p>' for link in links)
                     + '<pre>' + escape(json.dumps(details, indent=2, ensure_ascii=False)) + '</pre>')


def artifacts(output, root, records, predictions, metrics, curves, events, max_examples):
    write_json(output / "metrics.json", metrics)
    write_json(output / "curves.json", curves)
    rows = [{"group": "global", "value": "all", **metrics["global"]}, *metrics["groups"]]
    write_csv(output / "metrics.csv", rows)
    for name, curve in curves.items():
        write_csv(output / f"curve-{name}.csv", curve)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for threshold in (0.5, 0.75, 0.95):
        points = [p for p in curves["coco_pr"] if abs(p["iou"] - threshold) < 1e-6]
        axes[0].plot([p["recall"] for p in points], [p["precision"] for p in points], label=f"IoU {threshold}")
    for key in ("precision", "recall", "f1"):
        axes[1].plot([p["score_threshold"] for p in curves["score"]], [p[key] for p in curves["score"]], label=key)
    for ax, xlabel in zip(axes, ("Recall (COCO)", "Score threshold")):
        ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel=xlabel)
        ax.legend()
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(output / "curves.svg")
    plt.close(fig)

    by_id = {r["frame_id"]: r for r in records}
    review = []
    for event in events:
        record = by_id[event["frame_id"]]
        prediction = predictions[event["frame_id"]][event["prediction_index"]] if event["prediction_index"] is not None else None
        gt = record["boxes"][event["gt_index"]] if event["gt_index"] is not None else None
        review.append({**event, "source_id": record["source_id"], "frame_index": record["frame_index"],
                       "match_id": record["match_id"], "venue_id": record["venue_id"],
                       "score": prediction["confidence"] if prediction else None,
                       "prediction_bbox": json.dumps(prediction["bbox"]) if prediction else None,
                       "gt_bbox": json.dumps(gt["source_bbox"]) if gt else None})
    write_csv(output / "review.csv", review, fields=["frame_id", "kind", "prediction_index", "gt_index", "iou",
              "source_id", "frame_index", "match_id", "venue_id", "score", "prediction_bbox", "gt_bbox"])
    selected = list(dict.fromkeys(e["frame_id"] for e in events if e["kind"] in ("FP", "FN")))[:max_examples]
    links = []
    for frame_id in selected:
        record = by_id[frame_id]
        image = cv2.imread(str(root / record["image"]))
        geometry = Geometry(**record["transform"])
        # Draw on exported pixels; all numerical artifacts remain in source coordinates.
        for event in (e for e in events if e["frame_id"] == frame_id):
            if event["gt_index"] is not None:
                box = record["boxes"][event["gt_index"]]["source_bbox"]
                color = (255, 0, 255) if event["kind"] == "FN" else (0, 200, 0)
            else:
                box = predictions[frame_id][event["prediction_index"]]["bbox"]
                color = (0, 0, 255)
            x1, y1, x2, y2 = [round(v) for v in geometry.bbox(box)]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, event["kind"], (x1, max(12, y1)), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
        path = output / "examples" / f"{frame_id}.png"
        path.parent.mkdir(exist_ok=True)
        if not cv2.imwrite(str(path), image):
            raise RuntimeError("Failed to write review image")
        links.append(path.relative_to(output).as_posix())
    write_json(output / "examples.json", {"available_error_images": len(set(e["frame_id"] for e in events if e["kind"] != "TP")),
                                         "written": links, "max_examples": max_examples})
    return rows, links
