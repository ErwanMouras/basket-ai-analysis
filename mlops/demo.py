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


def scripted_score(index):
    return {"period": 1 if index < 90 else 3,
            "clock_s": (684 - index // 30) if index < 90 else (710 - (index - 90) // 30),
            "scores": {"blue": 2 if index < 90 else 52,
                       "pink": 0 if index < 45 else 2 if index < 90 else 48 if index < 135 else 50}}


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
    for team_index, team in enumerate(roster["teams"]):
        for extra in (1, 2):
            team["players"].append({"number": str((7 if team_index == 0 else 12) + extra),
                                    "name": f"Example {'A' if team_index == 0 else 'B'}{extra + 1}"})
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
            owner = (i // 45) % 2
            for team_index, team in enumerate(roster["teams"]):
                for player_index, player in enumerate(team["players"]):
                    x = ((240, 315, 365) if team_index == 0 else (415, 465, 480))[player_index] + (i % 90) * .15
                    y = (150, 100, 215)[player_index]
                    if i >= 90:
                        x, y = 640 - (x + 13) - 13, 400 - (y + 55) - 55
                    cx = x + 13
                    keypoints = [[cx, y+6], [cx-3, y+4], [cx+3, y+4], [cx-8, y+6], [cx+8, y+6],
                                 [cx-9, y+16], [cx+9, y+16], [cx-14, y+26], [cx+14, y+26],
                                 [cx-15, y+32], [cx+15, y+32], [cx-6, y+33], [cx+6, y+33],
                                 [cx-8, y+44], [cx+8, y+44], [cx-8, y+55], [cx+8, y+55]]
                    cv2.rectangle(image, (round(x), y), (round(x + 26), y + 55),
                                  (240, 180, 50) if team_index == 0 else (200, 120, 240), -1)
                    cv2.putText(image, player["number"], (round(x), y + 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
                    persons.append({"observation_id": f"synthetic-demo:{i}:{team_index}:{player_index}",
                                    "track_id": 1 + team_index * 3 + player_index,
                                    "role": "player", "confidence": 1., "bbox": [x, y, x + 26, y + 55],
                                    "position_m": [(cx - 20) / 600 * 28.6512, (y + 55 - 60) / 280 * 15.24] if available else None,
                                    "position_status": "projected" if available else "calibration_unavailable",
                                    "ground_point_px": [cx, y + 55], "ground_point_method": "scripted",
                                    "pose": {"format": "coco17", "coordinate_space": "source", "keypoints": keypoints,
                                             "scores": [1.] * 17, "valid": [True] * 17},
                                    "display_color": team["color"],
                                    "jersey": {"number": player["number"], "status": "confirmed",
                                               "identity_status": "unique_number", "team_id": team["team_id"],
                                               "player_name": player["name"]}})
            carrier = persons[owner * 3]
            point = [carrier["pose"]["keypoints"][9][0] + 2 * math.sin(i / 3),
                     carrier["pose"]["keypoints"][9][1]] if not 20 <= i < 41 else None
            if point:
                cv2.circle(image, tuple(round(v) for v in point), 5, (40, 140, 250), -1)
            cv2.putText(image, f"Scripted control: {'Blue' if owner == 0 else 'Pink'}", (22, 355),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
            if not available:
                cv2.putText(image, "Calibration unavailable", (170, 325), cv2.FONT_HERSHEY_SIMPLEX, .6, (100, 180, 255), 1)
            score = scripted_score(i)
            cv2.rectangle(image, (145, 316), (505, 342), (27, 30, 35), -1)
            clock = score["clock_s"]
            label = f"BLU {score['scores']['blue']}   PNK {score['scores']['pink']}   Q{score['period']}   {clock//60}:{clock%60:02d}"
            cv2.putText(image, label, (155, 335), cv2.FONT_HERSHEY_SIMPLEX, .52, (245, 245, 245), 1)
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
    write_json(run / "scoreboard.json", {
        "schema_version": 1, "source_sha256": source["sha256"], "sample_s": .5,
        "teams": [{"team_id": "blue", "label": "BLU"}, {"team_id": "pink", "label": "PNK"}],
        "provenance": {"method": "scripted_demo_no_ocr_or_accuracy_claim"},
        "samples": [{"time_s": i/30, **scripted_score(i)} for i in [*range(0,180,15),179]],
    })
    manifest = {"schema_version": 1, "run_id": "synthetic-demo", "status": "completed", "frames": 180, "segments": 2,
                "source": source, "capabilities": capabilities,
                "display_colors": palette, "demo": "Scripted synthetic interface demonstration; no learned predictions.",
                "models": {}, "warnings": ["synthetic_scripted_demo"], "artifacts": {}}
    for name in ("statistics.json", "observations.jsonl", "tracks.jsonl", "jersey_reads.jsonl", "roster.json", "scoreboard.json"):
        manifest["artifacts"][name] = {"sha256": file_hash(run / name)}
    manifest["court_coordinate_system"] = {
        "league": "NBA", "units": "metres", "length": 28.6512, "width": 15.24,
        "orientation_scope": "camera_segment",
    }
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
