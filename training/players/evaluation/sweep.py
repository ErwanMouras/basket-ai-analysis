"""Audit and run the ten-model player/referee validation sweep."""

import argparse
import gc
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from training.common.config import read_yaml
from training.common.files import atomic_writer, write_json
from training.common.provenance import ROOT
from training.players.contracts import CLASSES
from training.players.export.verify import verify_export

from .compare import compare
from .config import load_config as load_evaluation_config
from .config import validate_config
from .run import run as evaluate_model

EXPECTED = {(family, variant) for family, variants in (
    ("yolo", [f"yolo26{s}" for s in "nsmlx"] + ["yolov8n"]),
    ("rfdetr", [f"rfdetr_{s}" for s in ("nano", "small", "medium", "large")]),
) for variant in variants}
MODEL_FIELDS = {"name", "model", "variant", "weights", "source_class",
                "referee_weights", "referee_source_class", "reference"}


def _absolute(value):
    return str((ROOT / Path(value).expanduser()).resolve())


def load_plan(path, *, device=None, output=None):
    """Resolve ten explicit recipes without reading weights or creating output."""
    value = read_yaml(Path(path))
    if set(value) != {"schema_version", "base_recipe", "output", "models"} or value["schema_version"] != 1:
        raise ValueError("Sweep requires schema_version, base_recipe, output and models")
    if not isinstance(value["base_recipe"], str) or not value["base_recipe"]:
        raise ValueError("base_recipe must be a nonempty path")
    if not isinstance(value["output"], str) or not value["output"]:
        raise ValueError("output must be a nonempty path")
    base_path = Path(_absolute(value["base_recipe"]))
    base = load_evaluation_config(base_path)
    if base["split"] != "val" or base["purpose"] != "evaluation" or base["max_images"] is not None:
        raise ValueError("The sweep requires the full validation split and evaluation purpose")
    if not isinstance(value["models"], list) or len(value["models"]) != len(EXPECTED):
        raise ValueError("The sweep requires exactly five YOLO26, four RF-DETR and one YOLOv8 model")
    names, variants, models = set(), set(), []
    for item in value["models"]:
        if not isinstance(item, dict) or set(item) - MODEL_FIELDS or \
                not {"name", "model", "variant", "weights", "source_class", "reference"} <= set(item):
            raise ValueError("Each model needs name, model, variant, weights, source_class and reference")
        name = item["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or name in names:
            raise ValueError("Model names must be unique safe path components")
        names.add(name)
        identity = (item["model"], item["variant"])
        if identity not in EXPECTED or identity in variants:
            raise ValueError(f"Unexpected or repeated model: {identity}")
        variants.add(identity)
        if not isinstance(item["weights"], str) or not item["weights"]:
            raise ValueError(f"{name}.weights must be a nonempty path")
        recipe = {**base, **{k: v for k, v in item.items() if k != "name"}}
        recipe["weights"] = _absolute(recipe["weights"])
        if recipe["referee_weights"] is not None:
            recipe["referee_weights"] = _absolute(recipe["referee_weights"])
        if device is not None:
            recipe["device"] = device
        validate_config(recipe)
        models.append({"name": name, "recipe": recipe})
    if variants != EXPECTED:
        raise ValueError(f"Missing model variants: {sorted(EXPECTED - variants)}")
    destination = _absolute(output or value["output"])
    dataset = Path(base["dataset"])
    if Path(destination).is_relative_to(dataset) or dataset.is_relative_to(Path(destination)):
        raise ValueError("Sweep output and dataset must be disjoint")
    return {"schema_version": 1, "output": destination, "dataset": str(dataset),
            "models": models}


def readiness(plan):
    """Read-only audit, collecting every missing file before any model is run."""
    problems, rows = [], []
    dataset = Path(plan["dataset"])
    counts = {name: 0 for name in CLASSES.values()}
    selected = 0
    if not dataset.is_dir():
        problems.append(f"Export absent : {dataset}")
    else:
        try:
            verify_export(dataset)
            for line in (dataset / "frames.jsonl").read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                if record["split"] == "val":
                    selected += 1
                    for box in record["boxes"]:
                        counts[CLASSES[box["class_id"]]] += 1
            if not selected:
                problems.append("L'export ne contient aucune image val vérifiée")
            for name, count in counts.items():
                if not count:
                    problems.append(f"L'export val ne contient aucune annotation {name}")
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            problems.append(f"Export invalide : {error}")
    for item in plan["models"]:
        recipe = item["recipe"]
        weights = Path(recipe["weights"])
        referee = Path(recipe["referee_weights"] or recipe["weights"])
        present = weights.is_file() and weights.stat().st_size > 0
        referee_present = referee.is_file() and referee.stat().st_size > 0
        rows.append({"name": item["name"], "weights": str(weights), "present": present,
                     "referee_weights": str(referee), "referee_present": referee_present})
        if not present:
            problems.append(f"Poids absents : {weights}")
        if not referee_present:
            problems.append(f"Poids arbitres absents : {referee}")
    device = plan["models"][0]["recipe"]["device"]
    if device != "cpu":
        import torch
        index = int(device.split(":")[1])
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            problems.append(f"GPU indisponible : {device}")
    return {"ready": not problems, "dataset": str(dataset), "validation_images": selected,
            "annotations": counts, "models": rows, "problems": list(dict.fromkeys(problems))}


def print_readiness(report):
    print(f"Export : {report['dataset']}")
    print(f"Validation : {report['validation_images']} images, "
          f"{report['annotations']['player']} joueurs, {report['annotations']['referee']} arbitres")
    for row in report["models"]:
        label = "PRÊT" if row["present"] and row["referee_present"] else "MANQUANT"
        print(f"{label:9} {row['name']:15} {row['weights']}")
    for problem in report["problems"]:
        print(f"- {problem}")
    print("Prêt pour l'évaluation." if report["ready"] else "Aucune évaluation ne sera lancée tant que ces éléments manquent.")


def _metric(result, role, key):
    for group in result["metrics"]["groups"]:
        if group["group"] == "class" and group["value"] == role:
            return group[key]
    return None


def dashboard(output, attempts, ranks, *, comparison_available=True):
    """Write a standalone visual summary with links to every detailed report."""
    escape = lambda value: html.escape(str(value), quote=True)
    cards = []
    for attempt in attempts:
        name = escape(attempt["name"])
        if attempt["status"] != "FINISHED":
            cards.append(f'<section class="card failed"><h2>{name}</h2><p>Échec : {escape(attempt["error"])}</p></section>')
            continue
        result = attempt["result"]
        global_ap = result["metrics"]["global"]["ap50_95"]
        player_ap = _metric(result, "player", "ap50_95")
        referee_ap = _metric(result, "referee", "ap50_95")
        def bar(label, value, css):
            shown = "—" if value is None else f"{value:.3f}"
            width = 0 if value is None else max(0, min(100, value * 100))
            return (f'<div class="barrow"><span>{label}</span><div class="track">'
                    f'<div class="fill {css}" style="width:{width:.2f}%"></div></div>'
                    f'<strong>{shown}</strong></div>')
        report = escape(Path(attempt["output"]).relative_to(output).as_posix() + "/report.html")
        rank = ranks.get(result["run_id"])
        rank_text = f"Rang {rank}" if rank is not None else "Hors classement"
        latency = result["performance"]["adapter"]["p95_ms"]
        cards.append(f'<section class="card"><div class="heading"><h2>{name}</h2><span>{rank_text}</span></div>'
                     + bar("AP globale", global_ap, "global")
                     + bar("Joueurs", player_ap, "player")
                     + bar("Arbitres", referee_ap, "referee")
                     + f'<p>P95 inférence complète : <strong>{latency:.1f} ms/image</strong></p>'
                     + f'<a href="{report}">Rapport détaillé →</a></section>')
    count = sum(a["status"] == "FINISHED" for a in attempts)
    comparison_link = ('<a href="comparison/comparison.html">Classement et protocole complets</a>'
                       if comparison_available else "Aucun classement disponible")
    document = ('<!doctype html><html lang="fr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>Comparaison des détecteurs</title><style>'
                'body{font:16px system-ui,sans-serif;background:#101820;color:#edf4f4;margin:0;padding:2rem;max-width:1180px;margin:auto}'
                'h1{font-size:2rem;margin-bottom:.35rem}p{color:#b8c8cc}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1rem}'
                '.card{background:#1b2a33;border:1px solid #38515d;border-radius:14px;padding:1.25rem}.card.failed{border-color:#bf6868}'
                '.heading{display:flex;justify-content:space-between;align-items:center;gap:1rem}.heading h2{margin:.1rem 0 1rem;font-size:1.2rem}'
                '.heading span{color:#a9c0cb;font-size:.85rem}.barrow{display:grid;grid-template-columns:74px 1fr 50px;align-items:center;gap:.6rem;margin:.7rem 0;font-size:.84rem}'
                '.track{height:12px;border-radius:10px;background:#344651;overflow:hidden}.fill{height:100%}.global{background:#70d6ff}.player{background:#a8e88b}.referee{background:#ffbe73}'
                '.barrow strong{text-align:right}a{color:#87d5ff}a:focus-visible{outline:2px solid #fff}</style>'
                f'<h1>Joueurs et arbitres</h1><p>{count}/{len(attempts)} modèles évalués. '
                'AP50:95 par classe et latence P95 sur le même export de validation. '
                + comparison_link + '</p>'
                f'<main class="grid">{"".join(cards)}</main></html>')
    with atomic_writer(output / "overview.html") as handle:
        handle.write(document)


def run_sweep(plan):
    audit = readiness(plan)
    print_readiness(audit)
    if not audit["ready"]:
        raise ValueError("Préparation incomplète ; aucune évaluation lancée")
    destination = Path(plan["output"])
    suite = destination / (datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8])
    suite.mkdir(parents=True, exist_ok=False)
    write_json(suite / "readiness.json", audit)
    attempts = []
    for item in plan["models"]:
        name = item["name"]
        recipe = {**item["recipe"], "output": str(suite / "evaluations" / name)}
        print(f"Évaluation : {name}", flush=True)
        try:
            output = evaluate_model(recipe)
            result = json.loads((output / "result.json").read_text(encoding="utf-8"))
            attempts.append({"name": name, "status": "FINISHED", "output": str(output), "result": result})
        except Exception as error:
            attempts.append({"name": name, "status": "FAILED", "error": f"{type(error).__name__}: {error}"})
            print(f"Échec {name} : {error}", flush=True)
        finally:
            gc.collect()
            write_json(suite / "status.json", {"models": [{k: v for k, v in a.items() if k != "result"}
                                                   for a in attempts]})
    finished = [a for a in attempts if a["status"] == "FINISHED"]
    ranks = {}
    comparison_available = False
    if finished:
        try:
            comparison = compare([a["output"] for a in finished], suite / "comparison")
            ranks = {row["run_id"]: row["rank_in_protocol"] for row in comparison["rows"]}
            comparison_available = True
        except Exception as error:
            print(f"Classement indisponible : {error}", flush=True)
    dashboard(suite, attempts, ranks, comparison_available=comparison_available)
    print(f"Tableau de bord : {suite / 'overview.html'}", flush=True)
    return suite, comparison_available and all(a["status"] == "FINISHED" for a in attempts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run"))
    parser.add_argument("--config", type=Path, default=ROOT / "training/players/configs/evaluate_all.yaml")
    parser.add_argument("--device", help="Override the common device, e.g. cuda:0")
    parser.add_argument("--output", help="Override the parent directory for new sweep runs")
    args = parser.parse_args()
    plan = load_plan(args.config, device=args.device, output=args.output)
    if args.action == "check":
        report = readiness(plan)
        print_readiness(report)
        return 0 if report["ready"] else 2
    _, complete = run_sweep(plan)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
