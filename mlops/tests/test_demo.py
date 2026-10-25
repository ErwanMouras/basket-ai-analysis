import json
import unittest

from mlops.health import build_health
from training.common.provenance import ROOT, file_hash


class DemoTests(unittest.TestCase):
    def test_public_sample_has_the_required_output_fields_and_matching_source(self):
        run = ROOT / "examples/demo/runs/synthetic"
        manifest = json.loads((run / "run.json").read_text())
        for filename, schema_name, records in (
            ("observations.jsonl", "observation", "persons"),
            ("statistics.json", "statistics", "players"),
        ):
            schema = json.loads((ROOT / f"pipeline/schemas/{schema_name}.schema.json").read_text())
            content = (run / filename).read_text()
            rows = [json.loads(line) for line in content.splitlines()] if filename.endswith("jsonl") else [json.loads(content)]
            for row in rows:
                self.assertLessEqual(set(schema["required"]), row.keys())
                if schema.get("additionalProperties") is False:
                    self.assertLessEqual(row.keys(), schema["properties"].keys())
                for person in row[records]:
                    self.assertLessEqual(set(schema["properties"][records]["items"]["required"]), person.keys())
        stats = json.loads((run / "statistics.json").read_text())
        self.assertEqual(stats["source"], manifest["source"])
        self.assertEqual(stats["capabilities"], manifest["capabilities"])
        self.assertEqual(file_hash(ROOT / "examples/demo/videos/synthetic.mp4"), manifest["source"]["sha256"])
        self.assertEqual(build_health(run)["counts"]["frames"], manifest["frames"])
