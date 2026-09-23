"""CPU training contracts and framework views without importing a detector."""

import copy
import tempfile
import unittest
from pathlib import Path

from training.common.files import write_json
from training.common.provenance import ROOT, object_hash
from training.players.export.dataset import export_dataset
from training.players.export.publication import verify_bundle
from training.players.export.verify import read_json
from training.players.learning.config import (
    load_config,
    resume_contract,
    validate_config,
)
from training.players.learning.data import preflight, prepare
from training.players.learning.run import validate_resume_payload
from training.players.tests.training_fixtures import make_dataset


class TrainingContracts(unittest.TestCase):
    def config(self, family="yolo"):
        return load_config(
            family, ROOT / f"training/players/configs/train_{family}_smoke.yaml"
        )

    def test_all_six_recipes_are_valid_and_use_explicit_local_weights(self):
        for family in ("yolo", "rfdetr"):
            for budget in ("smoke", "quick", "full"):
                config = load_config(
                    family,
                    ROOT / f"training/players/configs/train_{family}_{budget}.yaml",
                )
                self.assertEqual(config["mode"], "finetune")
                self.assertTrue(Path(config["weights"]).is_absolute())
                self.assertEqual(config["workers"], 0)
                self.assertIsNone(config["resume"])

    def test_invalid_training_options_and_ambiguous_initialization_fail(self):
        invalid = (
            {"epochs": True},
            {"epochs": 0},
            {"batch_size": 0},
            {"resolution": 63},
            {"workers": 2},
            {"precision": "bf16"},
            {"device": "auto"},
            {"learning_rate": float("nan")},
            {"mode": "scratch"},
            {"mode": "resume"},
            {"resume": "last.pt"},
            {"weights": None},
            {"variant": "yolo11n"},
            {"deterministic": "true"},
            {"max_train_images": False},
            {"optimizer": "auto"},
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                validate_config({**self.config(), **overrides})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text("yolo:\n  unknown: true\n")
            with self.assertRaisesRegex(ValueError, "Unknown"):
                load_config("yolo", path)

    def test_resume_contract_ignores_paths_but_freezes_budget_data_and_environment(
        self,
    ):
        original = self.config()
        moved = {
            **original,
            "dataset": "/moved",
            "weights": "/other",
            "output": "/other-runs",
            "mode": "resume",
            "resume": "/last.pt",
            "mlflow_uri": "sqlite:///other.db",
        }
        self.assertEqual(resume_contract(original), resume_contract(moved))
        for key, value in (
            ("epochs", 3),
            ("seed", 100),
            ("resolution", 256),
            ("device", "cpu"),
        ):
            self.assertNotEqual(
                resume_contract(original), resume_contract({**original, key: value})
            )

    def test_resume_rejects_weights_only_completed_and_incompatible_payloads(self):
        config = self.config()
        config.update(
            mode="resume",
            resume="last.pt",
            initial_weights_sha256="a" * 64,
            runtime={"torch": "2.11.0"},
            code_sha256={"adapter.py": "b" * 64},
            dataset_id="c" * 64,
        )
        payload = {
            "epoch": 0,
            "optimizer": {"state": {}},
            "scaler": {},
            "ema": {},
            "players_training": {
                "schema_version": 1,
                "family": "yolo",
                "run_id": "run",
                "contract": resume_contract(config),
                "rng": {"python": []},
                "model": {"x": 1},
                "scheduler": {"last_epoch": 0},
                "stopper": {"best_epoch": 0},
                "loader_generator": [],
            },
        }
        validate_resume_payload(copy.deepcopy(config), payload)
        for mutate in (
            lambda p: p.pop("optimizer"),
            lambda p: p.update(epoch=1),
            lambda p: p["players_training"].pop("scheduler"),
            lambda p: p["players_training"]["contract"].update(dataset_id="different"),
            lambda p: p["players_training"]["contract"].update(
                runtime={"torch": "other"}
            ),
            lambda p: p["players_training"]["contract"].update(
                code_sha256={"adapter.py": "changed"}
            ),
        ):
            altered = copy.deepcopy(payload)
            mutate(altered)
            with self.assertRaises(ValueError):
                validate_resume_payload(copy.deepcopy(config), altered)
        with self.assertRaisesRegex(ValueError, "complete player checkpoint"):
            validate_resume_payload(config, {"model": {}})

    def test_private_views_preserve_exports_and_exclude_test_and_keep_negatives(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            exports = make_dataset(base)
            before = verify_bundle(exports)
            for family, fmt in (("yolo", "yolo"), ("rfdetr", "coco")):
                config = self.config(family)
                config.update(dataset=str(exports / fmt), output=str(base / "runs"))
                manifest, records = preflight(config)
                self.assertEqual({r["split"] for r in records}, {"train", "val"})
                self.assertEqual(sum(not r["boxes"] for r in records), 2)
                self.assertEqual(config["selection_id"], object_hash(records))
                out = base / family
                out.mkdir()
                view = prepare(config, records, out)
                self.assertFalse((view / "test").exists())
                self.assertEqual(manifest["dataset_id"], before[fmt]["dataset_id"])
                self.assertFalse(Path(config["dataset"]).is_symlink())
            self.assertEqual(before, verify_bundle(exports))

    def test_preflight_rejects_wrong_format_and_negative_only_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            exports = make_dataset(base)
            config = self.config()
            config.update(dataset=str(exports / "coco"), output=str(base / "runs"))
            with self.assertRaisesRegex(ValueError, "Expected a yolo"):
                preflight(config)
            config.update(
                dataset=str(exports / "yolo"),
                output=str((exports / "yolo").resolve() / "runs"),
            )
            with self.assertRaisesRegex(ValueError, "disjoint"):
                preflight(config)
            sidecar = base / "source/train/0.png.playersann.json"
            document = read_json(sidecar)
            document["frames"][0]["boxes"] = []
            write_json(sidecar, document)
            export_dataset(base / "source", exports)
            config.update(
                dataset=str(exports / "yolo"),
                output=str(base / "runs"),
                max_train_images=1,
            )
            with self.assertRaisesRegex(ValueError, "at least one annotated player"):
                preflight(config)

    def test_rfdetr_resume_requires_optimizer_scheduler_ema_and_amp_scaler(self):
        config = self.config("rfdetr")
        config.update(
            mode="resume",
            resume="last.ckpt",
            initial_weights_sha256="a" * 64,
            precision="amp",
        )
        payload = {
            "epoch": 0,
            "optimizer_states": [{"state": {}}],
            "lr_schedulers": [{"last_epoch": 2}],
            "loops": {},
            "state_dict": {},
            "MixedPrecision": {"scale": 65536},
            "callbacks": {"ema": {"average_model_state_dict": {"weights": 1}}},
            "players_training": {
                "schema_version": 1,
                "family": "rfdetr",
                "run_id": "run",
                "contract": resume_contract(config),
                "rng": {"python": []},
            },
        }
        validate_resume_payload(copy.deepcopy(config), payload)
        for field in (
            "optimizer_states",
            "lr_schedulers",
            "callbacks",
            "MixedPrecision",
        ):
            altered = copy.deepcopy(payload)
            altered[field] = (
                [] if field.endswith("states") or field == "lr_schedulers" else {}
            )
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_resume_payload(copy.deepcopy(config), altered)


if __name__ == "__main__":
    unittest.main()
