"""Fail-closed model selection on compatible, completed evaluation reports."""

import argparse
import json
import math
from pathlib import Path

import yaml

from training.common.files import write_json
from training.common.provenance import file_hash, object_hash


def read_result(path):
    path = Path(path)
    return json.loads((path / "result.json" if path.is_dir() else path).read_text())


def metric(document, path):
    for key in path:
        document = document[key]
    if type(document) not in (int, float) or not math.isfinite(document):
        raise ValueError("Metric must be a finite number")
    return document


def decide(baseline, candidate, policy):
    if set(policy) != {"schema_version", "description", "rules"} or policy["schema_version"] != 1:
        raise ValueError("Invalid quality policy")
    if not isinstance(policy["rules"], list) or not policy["rules"]:
        raise ValueError("A quality policy needs rules")
    checks = []
    for label, result in (("baseline", baseline), ("candidate", candidate)):
        comparable = result.get("comparison")
        checks.append({"name": f"{label}_integrity", "passed": bool(
            isinstance(comparable, dict) and result.get("comparison_id") == object_hash(comparable))})
        checks.append({"name": f"{label}_completed_non_smoke", "passed":
                       result.get("status") == "FINISHED" and result.get("purpose") == "evaluation"
                       and result.get("reference", {}).get("smoke") is False})
    checks.append({"name": "identical_data_selection_protocol", "passed":
                   isinstance(baseline.get("comparison"), dict)
                   and baseline.get("comparison") == candidate.get("comparison")})
    for rule in policy["rules"]:
        if not isinstance(rule, dict) or set(rule) - {"name", "path", "min", "max", "max_drop", "max_ratio"}:
            raise ValueError("Unknown quality rule settings")
        if not isinstance(rule.get("name"), str) or not isinstance(rule.get("path"), list) or not rule["path"]:
            raise ValueError("Rules need a name and a nonempty metric path")
        bounds = set(rule) & {"min", "max", "max_drop", "max_ratio"}
        if not bounds or any(type(rule[k]) not in (int, float) or not math.isfinite(rule[k]) for k in bounds):
            raise ValueError("Rules need finite acceptance bounds")
        if any(rule[k] < 0 for k in bounds & {"max_drop", "max_ratio"}):
            raise ValueError("Relative bounds must be nonnegative")
        check = {"name": rule["name"], "path": rule["path"], "passed": False}
        try:
            actual = metric(candidate, rule["path"])
            previous = metric(baseline, rule["path"])
            conditions = []
            if "min" in rule:
                conditions.append(actual >= rule["min"])
            if "max" in rule:
                conditions.append(actual <= rule["max"])
            if "max_drop" in rule:
                conditions.append(previous - actual <= rule["max_drop"] + 1e-12)
            if "max_ratio" in rule:
                conditions.append(previous > 0 and actual <= previous * rule["max_ratio"])
            check.update(candidate=actual, baseline=previous, passed=all(conditions))
        except (KeyError, TypeError, ValueError) as exc:
            check["error"] = str(exc)
        checks.append(check)
    payload = {"schema_version": 1, "accepted": all(c["passed"] for c in checks),
               "baseline_run_id": baseline.get("run_id"), "candidate_run_id": candidate.get("run_id"),
               "baseline_checkpoint_sha256": baseline.get("model", {}).get("checkpoint_sha256"),
               "candidate_checkpoint_sha256": candidate.get("model", {}).get("checkpoint_sha256"),
               "policy_id": object_hash(policy), "checks": checks}
    return {**payload, "decision_id": object_hash(payload)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new decision output")
    report = decide(read_result(args.baseline), read_result(args.candidate), yaml.safe_load(args.policy.read_text()))
    report["policy_sha256"] = file_hash(args.policy)
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["accepted"] else 2)


if __name__ == "__main__":
    main()
