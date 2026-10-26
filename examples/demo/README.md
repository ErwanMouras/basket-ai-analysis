# Shareable interface demo

`videos/synthetic.mp4` and `runs/synthetic/` were produced by `python -m mlops.demo`.
The scene, six player identities (three per team), positions, poses, ball contacts,
calibration interruption and two camera segments are scripted. Team control
alternates within each segment; the second segment reverses court direction.
A ball-visibility gap exercises the unclassified state. No models were trained or invoked. The sample demonstrates the viewer,
portable artifact paths, distance aggregation, player/team heatmaps and unavailable
measurements. The manifest declares the NBA court dimensions used to script its
positions. The viewer estimates the offensive/defensive passages from those
scripted contacts and aligns their basket direction, without running inference.
The visible control label identifies the generator’s scenario; it is not a model
prediction or an input label consumed by the phase estimator.

These outputs are not evidence of learned-model accuracy. The video includes an
explicit synthetic-data label. No NBA footage or third-party checkpoints are
included. The original generator is [mlops/demo.py](../../mlops/demo.py).

From the repository root: `docker compose up --build -d viewer`, then open
http://127.0.0.1:8766. Stop with `docker compose down`.
