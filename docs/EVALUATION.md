# Check, evaluate and package the system

Use this guide after [setting up the pipeline](../pipeline/README.md). Run commands
from the repository root, using a new output directory for each experiment.

Three questions need different evidence:

| Question | How to answer it |
| --- | --- |
| Does the implementation work? | Run regression tests and integrity checks. |
| Are its predictions accurate? | Compare against independently reviewed annotations. |
| Can the same model be run elsewhere? | Verify its configuration, dependencies and packaged weights. |

Passing tests does not establish model accuracy. Likewise, a valid analysis
manifest proves which files were produced, not that every prediction is correct.

## 1. Run the technical checks

```bash
make check-poc CHECK_OUTPUT=runs/checks/new-check
```

Read `summary.json` in that output directory for suites, logs, passed tests and
skips. Optional training integrations are disabled by this runner. The
[CI workflow](../.github/workflows/players.yml) runs the corresponding offline
checks on GitHub.

To check the browser interface, install its dependencies and Chromium:

```bash
npm --prefix viewer ci
node viewer/node_modules/playwright/cli.js install chromium
```

Start `make demo-native PYTHON=.venv-players/bin/python` in another terminal, then run:

```bash
npm --prefix viewer test
npm --prefix viewer run test:browser
```

Browser checks use the scripted sample at <http://127.0.0.1:8766>. They verify
interface behavior, not the accuracy of models on real footage.

## 2. Inspect a completed analysis

New analyses include `health.json`. To check an older run:

```bash
.venv-players/bin/python -m mlops.health \
  --run runs/analysis/my-run --output runs/checks/my-run-health.json
```

The report checks artifact integrity and reports availability: how many frames
have ball positions or court calibration, how many player observations can be
projected, and how many have a resolved identity. Each ratio identifies its
denominator. These counts do not measure accuracy or drift without ground truth.

## 3. Measure model accuracy fairly

Keep training, validation and test **matches** separate and check for duplicate
content. Use validation for tuning and model selection. Freeze the final recipe
before reading test results. Report support per match and temporal coverage:
neighboring video frames are correlated, not independent samples.

Use identical data, source coordinates, hardware and numerical precision when
comparing models. Keep these measurements separate:

| Measurement | Meaning |
| --- | --- |
| Precision | How many accepted detections match a real annotated target |
| Recall | How many annotated targets the model finds |
| F1 | A balance between precision and recall |
| AP | Detection quality summarized across confidence thresholds |
| Coverage | How much of the requested data has a usable prediction or measurement |
| Median / P95 latency | Typical call time and the time below which 95% of calls fall |

### Player and referee detection

Copy an [evaluation recipe](../training/players/configs/evaluate_yolov8.yaml) and
set local weights, their actual source, the verified dataset export and device:

```bash
make evaluate-players CONFIG=/path/to/evaluation.yaml
```

Use `source_class: 2` and `referee_source_class: 3` for E-BARD. A generic COCO
`person` model does not distinguish players, referees and spectators; its source
class must be configured explicitly. YOLO and RF-DETR recipes are also supplied
under [training/players/configs](../training/players/configs/).

Evaluation uses verified frames, including explicit negatives. Predictions and
annotations are compared in source-image coordinates. It reports box AP@50 and
AP@50:95 per role, plus precision/recall/F1 at score 0.25 and intersection-over-union
(IoU) 0.50 by default. IoU measures how much two boxes overlap relative to their
combined area. Duplicate detections count as false positives; missing ground
truth yields unavailable metrics rather than invented scores.

The timing protocol uses FP32, batch 1, five warmups and three repetitions by
default. It separates adapter calls, calls including image decoding and model
forward time. GPU timing synchronizes CUDA; PyTorch memory peaks do not represent
all VRAM usage. These image-based timings are not full video throughput.

Compare completed evaluations with:

```bash
make compare-players \
  RUNS="runs/players/evaluation/run-a runs/players/evaluation/run-b" \
  REPORT_OUTPUT=runs/players/comparison
```

Only matching data, selection and protocols share a ranking. Smoke runs and
incomplete evaluations are excluded by default. Training budgets, pretraining
and seeds remain experimental differences even when evaluation protocols match.

### Ball localization

The ball comparison evaluates **one predicted position per source frame**.
YOLO uses the most confident ball-box center. TrackNet uses the center of the
largest thresholded heatmap contour. An error above 15 source pixels counts as
both a false positive and a false negative. Median/P95 localization errors retain
incorrect predictions, rather than measuring only successful detections.

The comparison fixes confidence thresholds before execution: 0.25 for YOLO and
0.5 for heatmaps. Context is causal: one frame for YOLO, three for V5 and eight
for V3, using the last temporal head. Missing context and camera cuts reduce
coverage. The primary comparison uses the common covered frames; separate results
expose each model's own coverage.

With local exports, checkpoints and pinned source checkouts installed:

```bash
.venv-players/bin/python -m training.ball.comparison freeze \
  --output runs/ball/new-comparison/protocol.json

# Review the frozen protocol before running predictions.
.venv-players/bin/python -m training.ball.comparison run \
  --protocol runs/ball/new-comparison/protocol.json \
  --output runs/ball/new-comparison/evaluations
```

This workflow selects models on validation data and rejects the test split.
Validation selection is not an independent accuracy estimate. Unknown ball
positions are not automatically negative examples; absent-ball accuracy needs
reviewed negatives. Adapter latency excludes weight loading and other pipeline
stages.

### Tracking, identity, geometry and distance

Create independent human annotations linked to the source video's SHA-256. A gold
file uses this structure; replace the placeholder values and complete every
selected frame before changing `status` to `reviewed`:

```json
{
  "schema_version": 1,
  "status": "draft",
  "reviewer": "reviewer name after human review",
  "split": "val",
  "source_sha256": "exact source-video SHA-256",
  "frames": [{
    "frame_index": 450,
    "timestamp_s": 15.0,
    "segment_id": 0,
    "persons": [{
      "person_id": "persistent human identity",
      "bbox": [100, 200, 150, 350],
      "team_id": null,
      "number": null,
      "position_m": null,
      "expected_identity": "abstain"
    }]
  }]
}
```

Annotate all visible target people. Boxes use source pixels; floor positions use
meters in the camera segment's orientation. Human identity labels differ from
tracker IDs. Keep uncertain values `null`, and jersey numbers as strings so `0`
and `00` remain distinct.

```bash
.venv-players/bin/python -m mlops.gold \
  --gold /path/to/reviewed-gold.json --run runs/analysis/my-run \
  --output runs/checks/gold-result.json
```

The report covers identity continuity (IDF1, switches and fragmentation), accepted
identity precision, coverage, projection error and distance error on matching
valid intervals. It penalizes identities assigned to false detections and reports
unverifiable identities separately. It does not calculate HOTA. A manual
reference also has measurement uncertainty.

For a final test, freeze first, then evaluate the same recipe:

```bash
.venv-players/bin/python -m mlops.gold \
  --gold /path/to/test-gold.json --run runs/analysis/my-run \
  --freeze-recipe runs/checks/gold-test.frozen.json
.venv-players/bin/python -m mlops.gold \
  --gold /path/to/test-gold.json --run runs/analysis/my-run \
  --frozen-recipe runs/checks/gold-test.frozen.json \
  --output runs/checks/gold-test-result.json
```

Freezing binds annotations, artifacts, code and thresholds. It does not prove
that nobody previously inspected the test data. Model-specific evaluation CLIs
also expose `--freeze-recipe`; see their `--help` and checked-in configurations.

## 4. Control model promotion and rollback

[quality_gates.yaml](../configs/quality_gates.yaml) defines the provisional rules:
compatible evaluations, minimum support, allowed AP/recall regression and a
maximum P95 latency ratio. Fix the policy and baseline using validation data.

```bash
.venv-players/bin/python -m mlops.quality \
  --baseline /path/to/baseline-evaluation \
  --candidate /path/to/candidate-evaluation \
  --policy configs/quality_gates.yaml --output runs/checks/decision.json
```

Exit code 0 means acceptance; 2 means rejection. Missing metrics, NaN and
incompatible protocols cannot pass implicitly.

Use the [registration recipe](../training/players/configs/register.yaml) and
[promotion recipe](../training/players/configs/promote.yaml) with
`make register-players CONFIG=/path/to/recipe.yaml`. Registration stores the exact
evaluated weights and provenance. Changing the `champion` alias also requires
the quality policy, a baseline evaluation, a reason and `expected_previous`.
Smoke candidates cannot be promoted. Rollback explicitly restores a previous
version and records the reason; it does not establish an accuracy improvement.

To exercise evaluation, registration, acceptance/rejection, rollback and a
prediction from the restored model without training:

```bash
.venv-players/bin/python -m mlops.acceptance --output runs/acceptance/new-check
```

This needs local YOLO26 Nano and E-BARD weights and a verified player export.
It uses its own MLflow database and leaves existing runs intact. Shared-server
alias changes are not a distributed atomic transaction; review incomplete audit
records after an interruption or concurrent external change.

## 5. Package and run CPU inference

```bash
make bundle-models
make verify-model-bundle
make runtime-image
make infer-container VIDEO=/videos/clip.mp4 OUTPUT=runs/container-check MAX_FRAMES=150
```

The bundle contains local weights, hashes, a CPU configuration and the pinned
TrackNet checkout. Creation refuses to overwrite an existing bundle. The image
uses [runtime-cpu.lock](../deploy/runtime-cpu.lock), a Python base image pinned by
digest and a non-root user. Inference uses local volumes with `--network none`;
CPU pose uses ONNX Runtime CPU.

Packaging does not grant redistribution rights to third-party weights or footage.
Keep the source, model terms and hashes with any artifacts you share.
