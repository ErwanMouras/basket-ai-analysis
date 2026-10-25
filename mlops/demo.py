"""Create a shareable, explicitly scripted viewer demo without ML or annotations."""

import argparse
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

from pipeline.contracts import validate_frame, validate_statistics
from pipeline.distance import Distance
from training.common.files import write_json, write_jsonl
from training.common.provenance import file_hash
from training.jersey.display import display_colors


def create_demo(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    videos, run = output / "videos", output / "runs" / "synthetic"
    videos.mkdir()
    run.mkdir(parents=True)
    video = videos / "synthetic.mp4"
    temporary = videos / "frames.avi"
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 360))
    if not writer.isOpened():
        raise ValueError("Cannot create the synthetic source")
    roster = {"schema_version": 1, "match": {"match_id": "SYNTHETIC DEMO — scripted outputs"}, "teams": [
        {"team_id": "blue", "name": "Example Blue", "color": "#45C7F5", "players": [{"number": "7", "name": "Example A"}]},
        {"team_id": "pink", "name": "Example Pink", "color": "#F279C8", "players": [{"number": "12", "name": "Example B"}]}]}
    palette = display_colors(roster)
    metric = Distance({"smoothing_tau_s": 0.0, "deadband_m": 0.0,
                       "max_speed_m_s": 12., "max_gap_s": .2}, "synthetic-demo")
    observations, contributions = [], []
    try:
        for i in range(180):
            image = np.full((360, 640, 3), (35, 55, 45), np.uint8)
            cv2.rectangle(image, (20, 60), (620, 340), (160, 190, 170), 2)
            cv2.line(image, (320, 60), (320, 340), (160, 190, 170), 2)
            cv2.circle(image, (320, 200), 45, (160, 190, 170), 2)
            cv2.putText(image, "SYNTHETIC DEMO / SCRIPTED OUTPUTS", (22, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1)
            cv2.putText(image, "No model inference or accuracy claim", (22, 47), cv2.FONT_HERSHEY_SIMPLEX, .45, (210, 210, 210), 1)
            available = not 60 <= i < 75
            persons = []
            for j, team in enumerate(roster["teams"]):
                x, y = 100 + j * 260 + i * (0.5 if j == 0 else -.35), 150 + j * 50
                cv2.rectangle(image, (round(x), y), (round(x + 26), y + 55), (240, 180, 50) if j == 0 else (200, 120, 240), -1)
                cv2.putText(image, team["players"][0]["number"], (round(x), y + 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
                persons.append({"observation_id": f"synthetic-demo:{i}:{j}", "track_id": j + 1,
                                "role": "player", "confidence": 1., "bbox": [x, y, x + 26, y + 55],
                                "position_m": [(x - 20) / 600 * 28.6512, (y + 55 - 60) / 280 * 15.24] if available else None,
                                "position_status": "projected" if available else "calibration_unavailable",
                                "ground_point_px": [x, y + 55],
                                "ground_point_method": "scripted", "pose": None,
                                "display_color": team["color"],
                                "jersey": {"number": team["players"][0]["number"], "status": "confirmed",
                                           "identity_status": "unique_number", "team_id": team["team_id"],
                                           "player_name": team["players"][0]["name"]}})
            point = [320 + 50 * math.sin(i / 25), 150 + 30 * math.cos(i / 20)] if i % 90 >= 2 else None
            if point:
                cv2.circle(image, tuple(round(v) for v in point), 5, (40, 140, 250), -1)
            if not available:
                cv2.putText(image, "Calibration unavailable", (170, 325), cv2.FONT_HERSHEY_SIMPLEX, .6, (100, 180, 255), 1)
            writer.write(image)
            row = {"schema_version": 1, "artifact_type": "video_observation", "run_id": "synthetic-demo",
                   "frame_index": i, "segment_id": i // 90, "timestamp_s": i / 30,
                   "scene_cut": i == 90, "included": True, "width": 640, "height": 360,
                   "coordinate_space": "source_pixels", "distance_unit": "m", "persons": persons,
                   "court": {"status": "fit" if available else "unavailable", "image_to_court":
                             [[28.6512 / 600, 0., -20 * 28.6512 / 600], [0., 15.24 / 280, -60 * 15.24 / 280], [0., 0., 1.]] if available else None},
                   "ball": {"position_px": point, "confidence": 1. if point else None, "status": "ok" if point else "unavailable"},
                   "stages": {"players": {"status": "ok"}, "court": {"status": "ok" if available else "unavailable"},
                              "ball": {"status": "ok" if point else "unavailable"}}}
            observations.append(validate_frame(row))
            contributions.extend(metric.update(row))
    finally:
        writer.release()
    # H.264 is used for browser playback; MJPEG/mp4v browser support varies.
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(temporary), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", "-y", str(video)], check=True)
    temporary.unlink()
    source = {"path": "synthetic.mp4", "sha256": file_hash(video), "nominal_fps": 30,
              "width": 640, "height": 360, "declared_frames": 180, "decoded_frames": 180}
    capabilities = {"players": "ok", "court": "ok", "ball": "ok"}
    stats = {**metric.result(), "status": "completed", "source": source, "capabilities": capabilities}
    stats["limitations"].append("synthetic_scripted_demo_no_model_inference")
    validate_statistics(stats)
    write_json(run / "statistics.json", stats)
    write_jsonl(run / "observations.jsonl", observations)
    write_jsonl(run / "tracks.jsonl", contributions)
    write_jsonl(run / "jersey_reads.jsonl", [])
    write_json(run / "roster.json", roster)
    manifest = {"schema_version": 1, "run_id": "synthetic-demo", "status": "completed", "frames": 180, "segments": 2,
                "source": source, "capabilities": capabilities,
                "display_colors": palette, "demo": "Scripted synthetic interface demonstration; no learned predictions.",
                "models": {}, "warnings": ["synthetic_scripted_demo"], "artifacts": {}}
    for name in ("statistics.json", "observations.jsonl", "tracks.jsonl", "jersey_reads.jsonl", "roster.json"):
        manifest["artifacts"][name] = {"sha256": file_hash(run / name)}
    write_json(run / "run.json", manifest)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("release/demo"))
    parser.add_argument("--ensure", action="store_true", help="Verify an existing demo instead of recreating it")
    args = parser.parse_args()
    if args.ensure and args.output.exists():
        from mlops.health import build_health
        run = args.output / "runs/synthetic"
        build_health(run)
        manifest = json.loads((run / "run.json").read_text())
        if file_hash(args.output / "videos/synthetic.mp4") != manifest["source"]["sha256"]:
            raise ValueError("Synthetic source hash differs")
        print(args.output)
        return
    print(create_demo(args.output))


if __name__ == "__main__":
    main()
