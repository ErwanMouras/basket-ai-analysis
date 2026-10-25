"""Render recorded ball comparisons without rerunning or changing inference."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from training.common.files import write_json


def render(results_path, protocol_path, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recorded = json.loads(results_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    results = recorded["results"]
    if protocol["protocol_id"] != recorded["protocol_id"] or any(r["protocol_id"] != recorded["protocol_id"] for r in results):
        raise ValueError("Mixed comparison protocols")
    expected = {c["candidate_id"]: c for c in protocol["candidates"]}
    actual = {r["candidate"]["candidate_id"]: r["candidate"] for r in results}
    if actual != expected or len(results) != len(expected):
        raise ValueError("Incomplete, duplicated or unexpected checkpoints")
    groups = defaultdict(list)
    for row in results:
        groups[row["candidate"]["training_run_id"]].append(row)
    # This is explicit model selection on validation, never a test score.
    selected = [max(group, key=lambda r: (r["metrics"]["common"]["overall"]["f1"],
                                         r["candidate"]["candidate_id"].endswith("_best")))
                for group in groups.values()]
    selected.sort(key=lambda r: r["metrics"]["common"]["overall"]["f1"], reverse=True)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "results.json", recorded)
    summary = {"protocol_id": recorded["protocol_id"],
               "selection_scope": "highest common-cohort F1 among saved best/best_f1 checkpoints per training run; best.pt wins ties",
               "selected": [{"candidate_id": r["candidate"]["candidate_id"],
                             "training_run_id": r["candidate"]["training_run_id"],
                             "evaluation_run_id": r["evaluation_run_id"],
                             "common": r["metrics"]["common"]["overall"],
                             "coverage": r["metrics"]["available"]["overall"]["coverage"],
                             "coverage_adjusted_f1": r["metrics"]["coverage_adjusted_f1"],
                             "latency": r["metrics"]["latency"]} for r in selected]}
    write_json(output / "selection.json", summary)
    lines = ["| Checkpoint | F1 @15 source px | Precision | Recall | Error median / P95 (px) | Coverage | Adapter median / P95 (ms) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in sorted(results, key=lambda r: r["metrics"]["common"]["overall"]["f1"], reverse=True):
        m, latency = r["metrics"]["common"]["overall"], r["metrics"]["latency"]
        coverage = r["metrics"]["available"]["overall"]["coverage"]
        lines.append(f"| `{r['candidate']['candidate_id']}` | {m['f1']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | "
                     f"{m['error_source_median_px']:.2f} / {m['error_source_p95_px']:.2f} | {100 * coverage:.2f}% | "
                     f"{latency['median_ms']:.2f} / {latency['p95_ms']:.2f} |")
    (output / "checkpoint-table.md").write_text("\n".join(lines) + "\n")
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for index, r in enumerate(selected):
        c, m = r["candidate"], r["metrics"]
        label = c["candidate_id"].replace("_best_f1", "").replace("_best", "")
        label = label.replace("tracknet_", "TrackNet ").replace("_totnet", " + TOTNet").replace("_2epochs", " (2 epochs)")
        label = label.upper() if c["model"] == "yolo" else label.replace("TrackNet v", "TrackNet V")
        x, y = m["latency"]["median_ms"], m["common"]["overall"]["f1"]
        ax.scatter([x], [y], s=75, marker="x" if c["epochs"] < 30 else "o")
        ax.annotate(label, (x, y), xytext=(7, 7 if index % 2 == 0 else -14),
                    textcoords="offset points", fontsize=9)
    hardware = selected[0]["metrics"]["latency"]["hardware"].replace("NVIDIA GeForce ", "")
    targets = selected[0]["metrics"]["common"]["overall"]["frames"]
    values = [r["metrics"]["common"]["overall"]["f1"] for r in selected]
    ax.set(xlabel=f"Adapter latency, median (ms; batch 1, FP32, {hardware})",
           ylabel="Localization F1 at 15 source pixels", ylim=(max(0, min(values) - 0.08), min(1, max(values) + 0.06)),
           title=f"Ball model selection on validation — {targets:,} common causal frames")
    ax.margins(x=0.28)
    ax.set_xlim(left=0)
    ax.grid(alpha=0.2)
    fig.savefig(output / "quality-latency.png", dpi=180)
    plt.close(fig)
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    render(args.results, args.protocol, args.output)
