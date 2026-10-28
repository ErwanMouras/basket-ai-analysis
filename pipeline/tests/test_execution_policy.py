"""GPU-only must fail before decoding, model loading or output creation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pipeline.config import load_config
from pipeline.execution_policy import enforce_execution_policy, execution_policy_report
from pipeline.run import analyze


class ExecutionPolicyTests(unittest.TestCase):
    def strict_config(self):
        config = load_config(
            overrides={"compute_policy": "gpu_only", "device_fallback": None}
        )
        config["device_fallback"] = None
        config["distance"]["device"] = "cuda:0"
        config["jersey"]["fallback_device"] = None
        config["ball"]["adapter"] = "yolo"
        config["pose"]["allow_cpu_fallback"] = False
        return config

    def test_cuda_devices_and_no_fallback_do_not_hide_host_calculations(self):
        report = execution_policy_report(self.strict_config(), roster=True)
        self.assertFalse(report["compatible"])
        names = {item["name"] for item in report["blocking_components"]}
        self.assertTrue(
            {
                "tracking",
                "court_calibration",
                "identity_resolution",
                "health_statistics",
            }
            <= names
        )
        self.assertNotIn("distance_duration_coverage", names)

    def test_failure_precedes_models_video_roster_and_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must-not-exist"
            factory = Mock()
            with (
                patch("pipeline.run.Video") as video,
                patch("pipeline.run.Roster.load") as roster,
            ):
                with self.assertRaisesRegex(RuntimeError, "CPU calculations remain"):
                    analyze(
                        "unused.mp4",
                        output,
                        self.strict_config(),
                        "unused.json",
                        models_factory=factory,
                    )
                video.assert_not_called()
                roster.assert_not_called()
                factory.assert_not_called()
            self.assertFalse(output.exists())

    def test_disabling_optional_models_does_not_remove_cpu_tracking(self):
        config = self.strict_config()
        for name in ("ball", "court", "pose", "jersey"):
            config[name]["enabled"] = False
        with self.assertRaisesRegex(RuntimeError, "tracking"):
            enforce_execution_policy(config)

    def test_hybrid_policy_is_explicit_and_lists_cpu_calculations(self):
        report = enforce_execution_policy(load_config())
        self.assertTrue(report["compatible"])
        self.assertEqual(report["compute_policy"], "hybrid")
        self.assertTrue(report["cpu_components"])

    def test_recompute_also_rejects_remaining_cpu_statistics(self):
        with self.assertRaisesRegex(
            RuntimeError, "metric_identity_and_exclusion_counts"
        ):
            enforce_execution_policy(self.strict_config(), recompute=True)

    def test_invalid_policy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "compute_policy"):
            load_config(overrides={"compute_policy": "prefer_gpu"})


class NativeExecutionPolicyTests(unittest.TestCase):
    def config(self):
        config = load_config()
        config.update(
            backend="native_cuda", compute_policy="gpu_only", device_fallback=None
        )
        config["distance"]["device"] = "cuda:0"
        config["jersey"]["fallback_device"] = None
        config["ball"]["adapter"] = "yolo"
        config["segmentation"].update(strategy="box_prompt", segment_referees=True)
        config["tracking"].update(tracker="bytetrack", gmc_method="none")
        return config

    def test_only_native_implemented_components_satisfy_gpu_only(self):
        report = enforce_execution_policy(self.config(), roster=True)
        self.assertTrue(report["compatible"])
        self.assertEqual(report["cpu_components"], [])
        self.assertTrue(
            all(item["device"] == "cuda:0" for item in report["components"])
        )

    def test_native_path_rejects_fallback_missing_roster_and_unsupported_tracker(self):
        for change in ("fallback", "roster", "tracker"):
            with self.subTest(change=change):
                config = self.config()
                if change == "fallback":
                    config["device_fallback"] = "cpu"
                if change == "tracker":
                    config["tracking"]["tracker"] = "botsort"
                self.assertFalse(
                    execution_policy_report(config, roster=change != "roster")[
                        "compatible"
                    ]
                )

    def test_native_recompute_audits_cuda_statistics_without_loading_models(self):
        report = enforce_execution_policy(self.config(), recompute=True)
        self.assertEqual(
            report["components"], [{"name": "statistics", "device": "cuda:0"}]
        )
        self.assertIsNone(report["decoder"])
