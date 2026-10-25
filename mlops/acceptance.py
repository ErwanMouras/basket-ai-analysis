"""Exercise evaluation, quality rejection, promotion and rollback without training.

Uses an isolated local MLflow database and existing reviewed player exports.
The baseline and candidate are existing imported checkpoints, not new training.
"""

import argparse
import json
import os
from pathlib import Path

from training.common.config import read_yaml
from training.common.files import write_json
from training.players.evaluation.config import load_config
from training.players.evaluation.run import run
from training.players.registry import NAME, client_for, load_candidate, promote, register_candidate
from mlops.quality import decide, read_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    uri = f"sqlite:///{output / 'mlflow.db'}"
    os.environ["MLFLOW_TRACKING_URI"] = uri
    client, _ = client_for(uri)
    for name in ("players-evaluation", "players-registry", "players-registry-history"):
        client.create_experiment(name, artifact_location=str(output / "artifacts" / name))
    common = {"output": str(output / "evaluations"), "device": args.device}
    baseline_config = load_config("training/players/configs/evaluate_yolo.yaml")
    baseline_config.update(common, variant="yolo26n", weights=str(Path("models/players/yolo26n.pt").resolve()),
                           reference="Local COCO YOLO26 Nano initialization; no training in this acceptance run")
    candidate_config = load_config("training/players/configs/evaluate_yolov8.yaml")
    candidate_config.update(common)
    summary = {"status": "running", "training_executed": False, "annotations_modified": False,
               "mlflow_uri": uri, "scope": "Validation-only operational proof; no held-out accuracy claim"}
    try:
        baseline_path = run(baseline_config)
        summary["baseline_evaluation"] = str(baseline_path)
        write_json(output / "summary.json", summary)
        candidate_path = run(candidate_config)
        summary["candidate_evaluation"] = str(candidate_path)
        policy_path = Path("configs/quality_gates.yaml")
        policy = read_yaml(policy_path)
        accepted = decide(read_result(baseline_path), read_result(candidate_path), policy)
        rejected = decide(read_result(candidate_path), read_result(baseline_path), policy)
        write_json(output / "accepted.json", accepted)
        write_json(output / "rejected.json", rejected)
        if not accepted["accepted"] or rejected["accepted"]:
            raise ValueError("Measured candidates did not demonstrate both acceptance and rejection; inspect actual reports")
        baseline = register_candidate(baseline_path, baseline_config["weights"], mlflow_uri=uri)
        candidate = register_candidate(candidate_path, candidate_config["weights"], mlflow_uri=uri)
        summary["registered"] = {"baseline": baseline, "candidate": candidate}
        first = promote("champion", baseline["version"], expected_previous=None,
                        reason="Bootstrap isolated acceptance baseline against its recorded evaluation",
                        mlflow_uri=uri, quality_policy=policy_path, baseline_evaluation=baseline_path)
        upgrade = promote("champion", candidate["version"], expected_previous=baseline["version"],
                          reason="Candidate passed the recorded validation quality policy", mlflow_uri=uri,
                          quality_policy=policy_path, baseline_evaluation=baseline_path)
        try:
            promote("champion", baseline["version"], expected_previous=candidate["version"],
                    reason="Negative control: quality regression must be refused", mlflow_uri=uri,
                    quality_policy=policy_path, baseline_evaluation=candidate_path)
        except ValueError as exc:
            if "Quality policy rejected" not in str(exc):
                raise
            summary["rejected_promotion"] = str(exc)
        else:
            raise AssertionError("A rejected model was promoted")
        if str(client.get_model_version_by_alias(NAME, "champion").version) != candidate["version"]:
            raise AssertionError("Rejected promotion changed the alias")
        rollback = promote("champion", baseline["version"], expected_previous=candidate["version"],
                           reason="Operational rollback to the previously accepted baseline and its original acceptance reference",
                           mlflow_uri=uri, quality_policy=policy_path, baseline_evaluation=baseline_path)
        detector, metadata = load_candidate("models:/players-detector@champion", uri, device="cpu")
        import cv2
        image = cv2.imread(str(Path(baseline_config["dataset"]) / json.loads((baseline_path / "selection.jsonl").open().readline())["image"]))
        predictions = detector.predict(image, confidence=.25)
        summary.update(status="passed", promotion=upgrade, bootstrap=first, rollback=rollback,
                       restored_checkpoint_sha256=metadata["model"]["checkpoint_sha256"],
                       restored_version=metadata["version"], restored_model_prediction_count=len(predictions))
    except BaseException as exc:
        summary.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        write_json(output / "summary.json", summary)


if __name__ == "__main__":
    main()
