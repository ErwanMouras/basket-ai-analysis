import unittest
from copy import deepcopy

from mlops.quality import decide
from training.common.provenance import object_hash


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.result = {"comparison": {"dataset_id": "a", "protocol": {"precision": "fp32"}},
                       "status": "FINISHED", "purpose": "evaluation", "reference": {"smoke": False},
                       "metrics": {"ap": .8}, "run_id": "a", "model": {"checkpoint_sha256": "x"}}
        self.result["comparison_id"] = object_hash(self.result["comparison"])
        self.policy = {"schema_version": 1, "description": "test", "rules": [
            {"name": "ap", "path": ["metrics", "ap"], "min": .5, "max_drop": .01}]}

    def test_accept_and_reject_quality_regression(self):
        self.assertTrue(decide(self.result, self.result, self.policy)["accepted"])
        candidate = deepcopy(self.result)
        candidate["metrics"]["ap"] = .7
        self.assertFalse(decide(self.result, candidate, self.policy)["accepted"])

    def test_missing_nan_bool_metrics_smoke_and_protocol_changes_fail_closed(self):
        for value in (None, float("nan"), True):
            candidate = deepcopy(self.result)
            candidate["metrics"]["ap"] = value
            self.assertFalse(decide(self.result, candidate, self.policy)["accepted"])
        candidate = deepcopy(self.result)
        candidate["comparison"]["dataset_id"] = "other"
        candidate["comparison_id"] = object_hash(candidate["comparison"])
        self.assertFalse(decide(self.result, candidate, self.policy)["accepted"])
        candidate = deepcopy(self.result)
        candidate["reference"]["smoke"] = True
        self.assertFalse(decide(self.result, candidate, self.policy)["accepted"])

    def test_policy_requires_finite_explicit_bounds(self):
        for rule in ({"name": "a", "path": ["metrics", "ap"]},
                     {"name": "a", "path": ["metrics", "ap"], "min": float("nan")}):
            with self.assertRaises(ValueError):
                decide(self.result, self.result, {**self.policy, "rules": [rule]})
