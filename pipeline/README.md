# Analyze a basketball video

This guide takes you from a local video and model files to a completed analysis
that you can explore in the browser. Run commands from the repository root.

The pipeline processes the video first. The viewer then plays the saved results;
playing a video does not run the models again. To try the interface without
installing models, use the [synthetic demo](../examples/demo/README.md).

## 1. Install the analysis environment

Use Python 3.12 on Linux x86-64 with `uv`. Keep this environment separate from
the older ball-training environments.

For an NVIDIA GPU with a CUDA 13.0-compatible driver:

```bash
uv venv --python 3.12 .venv-players
uv pip sync --python .venv-players/bin/python --torch-backend cu130 \
  training/players/requirements-cu130.lock
uv pip check --python .venv-players/bin/python
```

For CPU-only analysis:

```bash
uv venv --python 3.12 .venv-players-cpu
uv pip sync --python .venv-players-cpu/bin/python --torch-backend cpu \
  training/players/requirements-cpu.lock
uv pip check --python .venv-players-cpu/bin/python
```

On Ubuntu, OpenCV needs `libgl1` and `libglib2.0-0` or `libglib2.0-0t64`.
Tk and a graphical display are needed only for annotation editors. The locks pin
package versions for the stated platform; they do not guarantee bitwise-identical
results across hardware.

The commands below use `.venv-players`. For CPU, use the CPU interpreter and set
every model's `device` to `cpu` in a copy of the analysis configuration.

## 2. Prepare the models and configuration

Start with [studio-sam-yolo26l-ball.yaml](../configs/segmentation/studio-sam-yolo26l-ball.yaml).
It enables player/referee detection, segmentation, ball localization, body pose,
court calibration and jersey-number recognition.

| Component | Model | Installation or required input |
| --- | --- | --- |
| Players and referees | E-BARD YOLOv8n | `make setup-analysis` |
| Segmentation | SAM 2.1 Tiny | Installer shown below |
| Ball | Selected fine-tuned YOLO26L checkpoint | Supply the local checkpoint referenced by the profile |
| Body pose | RTMPose-M | `make setup-players-pose` |
| Court | Pinned court-landmark detector | `make setup-court` |
| Jersey numbers | PARSeq | `make setup-jersey` |

Installation is explicit. These commands may download model files and check their
pinned hashes; analysis itself never downloads missing weights:

```bash
make setup-analysis
make setup-players-pose
make setup-court
make setup-jersey
.venv-players/bin/python -m training.segmentation.setup
```

The segmentation installer installs the experiment checkpoints, including SAM.
For CPU installations, pass `PLAYERS_PYTHON=.venv-players-cpu/bin/python` to Make
and use that interpreter for the last command.

The selected ball weights are **not included in Git**. Copy the profile to a
working configuration, set `ball.weights` to your local checkpoint and keep its
matching `checkpoint_sha256`. The checked-in studio profile pins:

```text
9fa0560664f59083998a05126339e1f5d6b2c2ae0e152de6c2b63c642fcde498
```

A different checkpoint needs its own matching hash and evaluation; it does not
reproduce the published model selection. Missing or mismatched weights stop the
run. Model URLs, revisions and hashes are defined in the installers:
[players/referees](setup.py), [segmentation](../training/segmentation/setup.py),
[pose](../training/players/pose.py), [court](../training/court/setup.py) and
[jersey OCR](../training/jersey/setup.py).

The older [analysis.yaml](../configs/analysis.yaml) uses a local TrackNet V5 ball
checkpoint and the pinned `.external/tracknet_v5` checkout instead. Choose the
configuration that matches your installed assets.

## 3. Supply a roster when identifying players

A roster limits recognized jersey numbers to the two teams playing. Without it,
the hybrid backend can analyze the video with anonymous tracks. The native CUDA
backend requires a roster.

Use [this complete roster](../training/jersey/examples/hou_lal_g1_2026/jersey.json)
as a format example. Replace its teams and players with those in your video.
Each team needs a unique `team_id`, a `name` and a list of players. Each player
needs `number` and `name`. Numbers are strings: `"0"` and `"00"` are distinct.
Optional `eligible: false` excludes an inactive player from recognition.
Optional team `color` and `uniform_color` fields use `#RRGGBB`.

The same number may occur on both teams. Number recognition alone does not
resolve that ambiguity; team and repeated observations contribute evidence.
Conflicting identities remain unresolved.

## 4. Run a short analysis, then the full clip

First check the configuration on a short passage. Use a **new output directory**:

```bash
make analyze-video \
  VIDEO=/path/to/game.mp4 \
  OUTPUT=runs/analysis/first-check \
  ANALYSIS_CONFIG=/path/to/analysis.yaml \
  ROSTER=/path/to/jersey.json \
  MAX_FRAMES=150
```

Inspect the result, then omit `MAX_FRAMES` and choose another output directory
to process the full clip. Omit `ROSTER` only when using the hybrid backend without
player identification. Relative configuration paths resolve from the repository
root.

Monitor a running analysis in another terminal:

```bash
.venv-players/bin/python -m pipeline.monitor --run runs/analysis/first-check
```

`progress.json` covers loading, inference, identity processing, statistics and
integrity checks. Its percentage estimates completed work, not time remaining.
A run reaches 100% only after its final check. Interruptions leave partial files
and diagnostics; check `run.json` for `completed` before consuming final outputs.

## 5. Explore the result

```bash
.venv-players/bin/python -m viewer.server
```

Open <http://127.0.0.1:8765>. Select the completed analysis and choose **EN** for
the English interface.

- **Visualization:** play or pause, step through frames, switch overlays and
  inspect projected court positions.
- **Statistics → Players:** select a player, inspect observed movement and open
  the individual shot-review window. Candidates can be included explicitly.
- **Statistics → Team:** inspect presence and shot-origin maps. **Compare** shows
  both teams and their distribution difference; switch between offense and defense.

Opening statistics pauses playback. Returning preserves the video position and
layer settings. Shot events are proposals, with probable/candidate statuses;
made/missed outcomes are not assigned. Maps describe their displayed time/filter
scope rather than just the current frame.

To serve other directories:

```bash
.venv-players/bin/python -m viewer.server \
  --runs-root /path/to/analyses --video-root /path/to/videos --port 8765
```

`--video-root` limits which source videos the server can expose. If a browser's
playback frames disagree with the analysis after a stream-copy video cut, create
a verified playback copy:

```bash
.venv-players/bin/python -m viewer.prepare_playback --run runs/analysis/first-check
```

This preserves the original video and observations and records the copy's hashes.

## Understand the output files

| File | What to inspect |
| --- | --- |
| `run.json` | Completion status, source, configuration, models, devices, timing and hashes |
| `observations.jsonl` | Detections, masks, pose, ball, calibration and identity evidence for each frame |
| `tracks.jsonl` | Movement contributions and reasons for excluding intervals |
| `statistics.json` | Observed distance, measured duration and coverage by subject |
| `health.json` | Integrity checks and availability of each capability |
| `jersey_reads.jsonl` / `team_votes.json` | OCR readings and evidence used for identity resolution |
| `roster.json` | Frozen roster copy, when supplied |

Boxes, pose and ball positions use source-video pixels. Projected player
positions use meters on an NBA court, 28.6512 × 15.24 m. The reference orientation
belongs to a camera segment; the ball is not projected onto the floor.
The [JSON schemas](schemas/) define the machine-readable contracts.

**Distance means valid observed movement.** The pipeline excludes camera cuts,
detection/projection gaps, identity conflicts, implausible movement and other
invalid intervals. It never adds movement across a break. Smoothing can reduce
turning distance; localization errors can increase it. Off-screen movement is
unmeasured, so these values are not total match distances.

`measured_duration_s` counts valid measurement intervals. `tracked_duration_s`
counts consecutive observations, including those without usable projection.
Their ratio is `coverage_ratio`; it is not the player's percentage of actual
on-court time. Missing measurements are `null`, not zero. A completed run can
contain partial or unavailable player measurements.

## Choose the execution backend

| Backend | Computation | Requirements and fallback |
| --- | --- | --- |
| `legacy` (hybrid) | GPU or CPU perception, with CPU tracking and geometry | Supports anonymous analysis; explicit model fallback can be enabled |
| `native_cuda` | GPU decoding, perception, tracking, geometry and aggregation | All six capabilities, explicit roster and compatible models on one CUDA device; no CPU computation fallback |

The supplied studio profile uses the hybrid path unless configured otherwise.
Setting only model devices or `distance.device` to CUDA does not select the native
backend. Native CUDA uses NVDEC, ByteTrack without camera-motion compensation,
box-prompt SAM and the YOLO ball adapter. The pinned RTMPose graph runs through
Torch CUDA operators. A runtime guard rejects CPU tensor computation. The CPU
still reads packets and files, orchestrates execution and writes outputs.

Install native dependencies in the CUDA environment:

```bash
uv pip install --python .venv-players/bin/python -r pipeline/cuda/requirements.txt
```

Configure `backend: native_cuda` and `compute_policy: gpu_only` in a compatible
profile. Check its static execution policy before running:

```bash
.venv-players/bin/python -m pipeline preflight \
  --config /path/to/native-analysis.yaml --roster
```

Exit code 2 means the profile violates the requested policy. Actual startup also
checks CUDA, NVDEC outputs and weights. The private SAC–SAS profile is under
`datas/test/SAC_SAS/`; it is not supplied by a clean clone. CUDA calibration uses
the same rejection rules as the hybrid path but is not bitwise identical to OpenCV.

Native analysis can resume into a new directory:

```bash
.venv-players/bin/python -m pipeline analyze \
  --video /path/to/game.mp4 --roster /path/to/jersey.json \
  --config /path/to/native-analysis.yaml \
  --resume-from runs/analysis/interrupted \
  --output runs/analysis/resumed
```

Resume retains complete camera segments and replays the final segment to rebuild
state. Video and roster hashes and numerical settings must match. The source
run remains intact. The hybrid backend has no equivalent mid-video resume.

## Verify or recompute an analysis

To recompute statistics from saved observations without loading models:

```bash
.venv-players/bin/python -m pipeline recompute \
  --run runs/analysis/first-check --output runs/analysis/recomputed.json
```

For test suites, model comparison, containers and quality checks, continue with
the [evaluation guide](../docs/EVALUATION.md).
