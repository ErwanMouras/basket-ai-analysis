"""Small declarative pipeline; model recipes retain their phase 4/5 contracts."""

import re
from pathlib import Path

from training.common.config import read_yaml
from training.common.provenance import ROOT
from training.players.export.config import ExportConfig
from training.players.learning.config import load_config as training_config
from training.players.evaluation.config import load_config as evaluation_config


def load_config(path):
    value = read_yaml(Path(path))
    if set(value) - {"schema_version", "source", "output", "export_config", "trials", "include_smoke"}:
        raise ValueError("Unknown pipeline settings")
    if value.get("schema_version") != 1 or not value.get("trials"):
        raise ValueError("A version 1 pipeline needs at least one trial")
    for key in ("source", "output", "export_config"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError(f"Missing pipeline {key}")
        value[key] = str((ROOT / Path(value[key]).expanduser()).resolve())
    source, output = Path(value["source"]), Path(value["output"])
    if source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Pipeline output and sources must be disjoint")
    value["export"] = ExportConfig.from_file(Path(value["export_config"])).to_dict()
    if "test" in value["export"]["splits"]:
        raise ValueError("This selection pipeline excludes final test; use the frozen evaluation command")
    value.setdefault("include_smoke", False)
    if type(value["include_smoke"]) is not bool or not isinstance(value["trials"], list):
        raise ValueError("Invalid pipeline types")
    seen = set()
    for trial in value["trials"]:
        if not isinstance(trial, dict) or set(trial) - {"name", "train", "evaluate"}:
            raise ValueError("Trial supports name, train (optional) and evaluate")
        name = trial.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or name in seen:
            raise ValueError("Trial names must be distinct safe path components")
        seen.add(name)
        for key in ("train", "evaluate"):
            if key == "train" and trial.get(key) is None:
                trial[key] = None
                continue
            if not isinstance(trial.get(key), str):
                raise ValueError(f"Missing {key} recipe")
            trial[key] = str((ROOT / Path(trial[key]).expanduser()).resolve(strict=True))
        trial["evaluation"] = evaluation_config(trial["evaluate"])
        if trial["evaluation"]["split"] != "val":
            raise ValueError("Pipeline selection evaluates val only")
        trial["training"] = None
        if trial["train"]:
            family = read_yaml(Path(trial["train"])).get("model", "yolo")
            trial["training"] = training_config(family, trial["train"])
            if any(trial["training"][k] != trial["evaluation"][k] for k in ("model", "variant")):
                raise ValueError("Training and evaluation must use the same architecture")
            if trial["evaluation"]["source_class"] != 0:
                raise ValueError("Trained player checkpoints use source_class 0")
    return value
