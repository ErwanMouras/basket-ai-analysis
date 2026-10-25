# Shareable interface demo

`videos/synthetic.mp4` and `runs/synthetic/` were produced by `python -m mlops.demo`.
The scene, identities, positions, calibration interruption and camera segment are
scripted. No models were trained or invoked. The sample demonstrates the viewer,
portable artifact paths, distance aggregation and unavailable measurements.

These outputs are not evidence of learned-model accuracy. The video includes an
explicit synthetic-data label. No NBA footage or third-party checkpoints are
included. The original generator is [mlops/demo.py](../../mlops/demo.py).

From the repository root: `docker compose up --build -d viewer`, then open
http://127.0.0.1:8766. Stop with `docker compose down`.
