"""Explicit MLflow candidates and audited, reversible alias changes."""

import argparse
import getpass
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

from training.common.config import read_yaml
from training.common.files import write_json
from training.common.provenance import ROOT, file_hash, object_hash, source_fingerprints
from training.players.orchestration.state import inventory, lock

NAME = "players-detector"


def client_for(uri):
    from mlflow import MlflowClient
    uri = os.environ.get("MLFLOW_TRACKING_URI", uri)
    return MlflowClient(tracking_uri=uri, registry_uri=uri), uri


def registry_lock(uri):
    if uri.startswith("sqlite:///"):
        path = Path(uri[len("sqlite:///"):].split("?", 1)[0]).resolve()
        return lock(path.with_name(path.name + ".players-registry.lock"))
    return lock(ROOT / "runs/players/registry-locks" / (object_hash(uri) + ".lock"))


def experiment(client, name):
    found = client.get_experiment_by_name(name)
    return found.experiment_id if found else client.create_experiment(name)


def register_candidate(evaluation, weights, *, mlflow_uri="sqlite:///mlflow.db"):
    import mlflow.pyfunc
    from mlflow.exceptions import MlflowException

    evaluation, weights = Path(evaluation).resolve(strict=True), Path(weights).resolve(strict=True)
    result = json.loads((evaluation / "result.json").read_text())
    summary = json.loads((evaluation / "summary.json").read_text())
    if summary["status"] != "FINISHED" or result["status"] != "FINISHED" or summary["run_id"] != result["run_id"]:
        raise ValueError("Register only a finished evaluation")
    if result["comparison_id"] != object_hash(result["comparison"]):
        raise ValueError("Invalid evaluation protocol identity")
    if file_hash(weights) != result["model"]["checkpoint_sha256"]:
        raise ValueError("Weights differ from the evaluated checkpoint")
    client, uri = client_for(mlflow_uri)
    tracked = client.get_run(result["run_id"])
    if tracked.info.status != "FINISHED" or client.get_experiment(tracked.info.experiment_id).name != "players-evaluation":
        raise ValueError("Evaluation must have a finished players-evaluation MLflow run in this store")
    manifest = json.loads((evaluation / "manifest.json").read_text())
    selection = [json.loads(line) for line in (evaluation / "selection.jsonl").read_text().splitlines()]
    if manifest["dataset_id"] != result["comparison"]["dataset_id"] or object_hash(selection) != result["comparison"]["selection_id"]:
        raise ValueError("Evaluation data provenance mismatch")
    # Compare local review material with the canonical MLflow evaluation artifacts.
    with tempfile.TemporaryDirectory() as temporary:
        for name in ("result.json", "manifest.json", "selection.jsonl", "reference.json", "protocol.json"):
            logged = Path(client.download_artifacts(result["run_id"], "evaluation/" + name, temporary))
            if file_hash(logged) != file_hash(evaluation / name):
                raise ValueError(f"Local evaluation differs from MLflow: {name}")
    sources = [p for p in (ROOT / "training/players").rglob("*.py") if "tests" not in p.parts]
    sources += list((ROOT / "training/common").glob("*.py"))
    code = source_fingerprints(ROOT, sources)
    smoke = result["purpose"] == "smoke" or result["reference"].get("smoke", False)
    metadata = {"schema_version": 1, "model": result["model"], "protocol": result["comparison"]["protocol"],
                "evaluation_run_id": result["run_id"], "training_run_id": result["reference"]["training_run_id"],
                "reference": result["reference"], "dataset_id": manifest["dataset_id"],
                "export_id": manifest["export_id"], "selection_id": result["comparison"]["selection_id"],
                "comparison_id": result["comparison_id"], "metrics": result["metrics"], "smoke": smoke,
                "weights_file": "weights" + weights.suffix, "code": code,
                "requirements_sha256": file_hash(ROOT / "training/players/requirements-cu130.lock")}
    candidate_id = object_hash(metadata)
    metadata["candidate_id"] = candidate_id
    with registry_lock(uri):
        try:
            client.get_registered_model(NAME)
        except MlflowException as exc:
            if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
                raise
            client.create_registered_model(NAME, description="Player detectors with common evaluation provenance; aliases require explicit promotion.")
        existing = client.search_model_versions(f"name = '{NAME}' AND tags.`players.candidate_id` = '{candidate_id}'")
        for version in existing:
            if version.status == "READY":
                return {"name": NAME, "version": str(version.version), "candidate_id": candidate_id,
                        "model_uri": f"models:/{NAME}/{version.version}", "reused": True}
        if existing:
            raise ValueError("This candidate already has an unfinished or failed version; inspect it before retrying")
        # Recover packaging completed before a previous client lost its response.
        exp = experiment(client, "players-registry")
        runs = client.search_runs([exp], filter_string=f"tags.`players.candidate_id` = '{candidate_id}' AND attributes.status = 'FINISHED'")
        if runs:
            run_id = runs[0].info.run_id
        else:
            run_id = client.create_run(exp, tags={"mlflow.runName": "register-player-candidate",
                "players.candidate_id": candidate_id, "evaluation_run_id": result["run_id"],
                "training_run_id": metadata["training_run_id"] or "imported", "purpose": "smoke" if smoke else "candidate"}).info.run_id
            status = "FAILED"
            try:
                with tempfile.TemporaryDirectory(prefix="players-candidate-") as temp:
                    temp = Path(temp)
                    bundle = temp / "bundle"
                    bundle.mkdir()
                    shutil.copy2(weights, bundle / metadata["weights_file"])
                    if file_hash(bundle / metadata["weights_file"]) != metadata["model"]["checkpoint_sha256"]:
                        raise ValueError("Checkpoint changed while packaging")
                    write_json(bundle / "candidate.json", metadata)
                    for name in ("result.json", "manifest.json", "selection.jsonl", "reference.json", "protocol.json"):
                        shutil.copy2(evaluation / name, bundle / name)
                    code_root = temp / "code/training"
                    code_root.mkdir(parents=True)
                    (code_root / "__init__.py").write_text("")
                    for path in sources:
                        target = code_root / path.relative_to(ROOT / "training")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, target)
                        if file_hash(target) != code[path.relative_to(ROOT).as_posix()]:
                            raise ValueError("Code changed during candidate packaging")
                    model_path = temp / "model"
                    mlflow.pyfunc.save_model(path=str(model_path), loader_module="training.players.registry_loader",
                        data_path=str(bundle), code_paths=[str(code_root)],
                        pip_requirements=str(ROOT / "training/players/requirements-cu130.lock"),
                        metadata={"players.candidate_id": candidate_id, "players.smoke": smoke})
                    client.log_artifacts(run_id, str(model_path), "model")
                    write_json(temp / "package.json", inventory([model_path]))
                    client.log_artifact(run_id, str(temp / "package.json"))
                status = "FINISHED"
            except KeyboardInterrupt:
                status = "KILLED"
                raise
            finally:
                client.set_terminated(run_id, status=status)
        tags = {"players.candidate_id": candidate_id, "players.smoke": str(smoke).lower(),
                "evaluation_run_id": result["run_id"], "training_run_id": metadata["training_run_id"] or "imported",
                "dataset_id": metadata["dataset_id"], "selection_id": metadata["selection_id"],
                "checkpoint_sha256": metadata["model"]["checkpoint_sha256"]}
        version = client.create_model_version(NAME, source=f"runs:/{run_id}/model", run_id=run_id, tags=tags,
                    description=f"Evaluated {result['model']['family']} {result['model']['variant']}; {result['comparison_id']}")
        return {"name": NAME, "version": str(version.version), "candidate_id": candidate_id,
                "model_uri": f"models:/{NAME}/{version.version}", "reused": False}


def resolve_version(client, model_uri):
    match = re.fullmatch(r"models:/players-detector(?:/(\d+)|@([a-zA-Z][a-zA-Z0-9_-]*))", model_uri)
    if not match:
        raise ValueError("Use models:/players-detector/VERSION or models:/players-detector@ALIAS")
    return client.get_model_version(NAME, match[1]) if match[1] else client.get_model_version_by_alias(NAME, match[2])


def load_candidate(model_uri, mlflow_uri="sqlite:///mlflow.db", *, device="cpu"):
    from training.players.registry_loader import detector_from_bundle
    client, _ = client_for(mlflow_uri)
    version = resolve_version(client, model_uri)
    if version.status != "READY":
        raise ValueError("Candidate version is not ready")
    with tempfile.TemporaryDirectory(prefix="players-registry-load-") as temporary:
        model_path = Path(client.download_artifacts(version.run_id, "model", temporary))
        bundle = model_path / "data/bundle"
        detector, metadata = detector_from_bundle(bundle, device)
        if metadata["candidate_id"] != version.tags["players.candidate_id"] or metadata["model"]["checkpoint_sha256"] != version.tags["checkpoint_sha256"]:
            raise ValueError("Registered version provenance mismatch")
        return detector, {**metadata, "version": str(version.version), "model_uri": f"models:/{NAME}/{version.version}"}


def promote(alias, version, *, expected_previous, reason, mlflow_uri="sqlite:///mlflow.db"):
    from mlflow.exceptions import MlflowException
    if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]*", alias) or not isinstance(reason, str) or not reason.strip():
        raise ValueError("An explicit alias and reason are required")
    client, uri = client_for(mlflow_uri)
    with registry_lock(uri):
        target = client.get_model_version(NAME, str(version))
        if target.status != "READY" or target.tags.get("players.smoke") != "false":
            raise ValueError("Only a ready, evaluated non-smoke candidate can be promoted")
        # Validate actual packaged provenance too; mutable version tags alone are insufficient.
        with tempfile.TemporaryDirectory() as temp:
            path = Path(client.download_artifacts(target.run_id, "model/data/bundle/candidate.json", temp))
            metadata = json.loads(path.read_text())
            if object_hash({k: v for k, v in metadata.items() if k != "candidate_id"}) != metadata["candidate_id"]:
                raise ValueError("Candidate metadata integrity mismatch")
            if metadata["smoke"] or metadata["candidate_id"] != target.tags.get("players.candidate_id"):
                raise ValueError("Smoke or inconsistent candidate cannot be promoted")
            evaluation = client.get_run(metadata["evaluation_run_id"])
            if evaluation.info.status != "FINISHED":
                raise ValueError("The candidate evaluation is no longer finished")
        try:
            previous = str(client.get_model_version_by_alias(NAME, alias).version)
        except MlflowException as exc:
            if exc.error_code not in ("RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"):
                raise
            previous = None
        expected = str(expected_previous) if expected_previous is not None else None
        if previous != expected:
            raise ValueError(f"Alias changed: expected {expected}, found {previous}")
        if previous == str(version):
            return {"alias": alias, "previous": previous, "version": str(version), "reused": True}
        audit = {"alias": alias, "previous": previous, "version": str(version), "reason": reason,
                 "actor": getpass.getuser(), "time": time.time(), "state": "INTENT",
                 "candidate_id": metadata["candidate_id"]}
        run_id = client.create_run(experiment(client, "players-registry-history"), tags={
            "mlflow.runName": f"alias-{alias}", "alias": alias, "previous_version": previous or "none",
            "target_version": str(version), "actor": audit["actor"], "reason": reason}).info.run_id
        status = "FAILED"
        try:
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "alias-change.json"
                write_json(path, audit)
                client.log_artifact(run_id, str(path), "intent")
                client.set_registered_model_alias(NAME, alias, str(version))
                if str(client.get_model_version_by_alias(NAME, alias).version) != str(version):
                    raise ValueError("Alias changed concurrently; inspect the audit intent")
                audit["state"] = "APPLIED"
                write_json(path, audit)
                client.log_artifact(run_id, str(path), "applied")
            status = "FINISHED"
        except KeyboardInterrupt:
            status = "KILLED"
            raise
        finally:
            client.set_terminated(run_id, status=status)
        return {**audit, "audit_run_id": run_id, "reused": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = read_yaml(args.config)
    action = config.get("action", "register")
    uri = config.get("mlflow_uri", "sqlite:///mlflow.db")
    if action == "register":
        if set(config) - {"action", "evaluation", "weights", "mlflow_uri"}:
            raise ValueError("Unknown candidate settings; registration never promotes an alias")
        result = register_candidate(ROOT / config["evaluation"], ROOT / config["weights"], mlflow_uri=uri)
    elif action == "promote":
        if set(config) - {"action", "alias", "version", "expected_previous", "reason", "mlflow_uri"} or "expected_previous" not in config:
            raise ValueError("Promotion requires an explicit expected_previous and known settings")
        result = promote(config["alias"], config["version"], expected_previous=config["expected_previous"], reason=config["reason"], mlflow_uri=uri)
    else:
        raise ValueError("action must be register or promote; rollback promotes an earlier version explicitly")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
