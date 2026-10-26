"""Read a fixed NBA broadcast scorebug offline using OpenCV and Tesseract.

Example: python -m viewer.scoreboard_extract --run runs/analysis/example
         --left-team houston_rockets --right-team los_angeles_lakers
The ESPN profile is specific to this graphic, not a universal NBA layout.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

from viewer.scoreboard import confirm_readings

# Coordinates on the verified 1920x1080 ESPN playoffs layout. The crop scales
# with the source resolution; a different graphic needs an explicit profile.
ESPN = {
    "name": "espn-playoffs-bottom-v1",
    "reference_size": [1920, 1080],
    "fields": {
        "left_team": [625, 959, 733, 997],
        "right_team": [893, 959, 1006, 997],
        "left_score": [767, 959, 850, 997],
        "right_score": [1033, 959, 1118, 997],
        "period": [1140, 957, 1202, 998],
        "clock": [1206, 958, 1377, 1001],
    },
}


def parse_clock(text):
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", text)
    if match and int(match[2]) < 60:
        return int(match[1]) * 60 + int(match[2])
    # Under one minute ESPN can show seconds and tenths, without a colon.
    if re.fullmatch(r"\d{1,2}\.\d", text) and float(text) < 60:
        return float(text)
    return None


def parse_period(text):
    match = re.fullmatch(r"([1-4])(?:ST|ND|RD|TH|87)?", text.upper())
    if match:
        return int(match[1])
    if text.upper() == "OT":
        return 5
    match = re.fullmatch(r"([1-9])OT", text.upper())
    return 4 + int(match[1]) if match else None


def ocr_field(frame, box, kind, temporary):
    import cv2

    x, y, right, bottom = box
    crop = frame[y:bottom, x:right]
    # Green separates purple digits from the yellow Lakers panel particularly
    # well, and preserves white lettering on the black panel.
    gray = cv2.resize(crop[:, :, 1], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if binary.mean() < 127:
        binary = 255 - binary
    if kind in ("score", "clock"):
        count, labels, stats, _ = cv2.connectedComponentsWithStats((binary < 128).astype("uint8"))
        h, w = binary.shape
        for label in range(1, count):
            left, top, width, height, area = stats[label]
            if left == 0 or top == 0 or left + width == w or top + height == h or area < 5:
                binary[labels == label] = 255
    # Tighten around ink, removing padding that confuses a single zero with Oo.
    ys, xs = (binary < 128).nonzero()
    if not len(xs):
        return ""
    binary = binary[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    binary = cv2.copyMakeBorder(binary, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    cv2.imwrite(str(temporary), binary)
    whitelist = (
        "0123456789Oo"
        if kind == "score"
        else "0123456789:." if kind == "clock" else "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )
    # A very bold isolated digit can fall outside Tesseract's preferred glyph
    # size. Retry at half size only when the primary read is unavailable.
    for psm, scale in ((7, 1), (8, 1), (7, 0.5), (8, 0.5)):
        if scale != 1:
            cv2.imwrite(str(temporary), cv2.resize(binary, None, fx=scale, fy=scale))
        result = subprocess.run(
            [
                "tesseract",
                str(temporary),
                "stdout",
                "-l",
                "eng",
                "--psm",
                str(psm),
                "-c",
                "tessedit_char_whitelist=" + whitelist,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        text = result.stdout.strip().replace(" ", "")
        if kind == "score":
            text = text.replace("O", "0").replace("o", "0")
            if re.fullmatch(r"\d{1,3}", text):
                return text
        elif kind == "clock":
            value = parse_clock(text)
            if value is not None and value <= 720:
                return text
        else:
            return text
    return ""


def extract(run, video, left, right, profile, sample_s=0.5):
    import cv2

    manifest = json.loads((run / "run.json").read_text())
    roster = json.loads((run / "roster.json").read_text())
    ids = {t["team_id"] for t in roster["teams"]}
    if len(ids) != 2 or {left[0], right[0]} != ids:
        raise ValueError("The two panel teams must match the run roster")
    digest = hashlib.sha256()
    with video.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != manifest["source"]["sha256"]:
        raise ValueError("Video SHA256 differs from run source")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError("Cannot open source video")
    rw, rh = profile["reference_size"]
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    boxes = {
        key: [round(v * (width / rw if i % 2 == 0 else height / rh)) for i, v in enumerate(box)]
        for key, box in profile["fields"].items()
    }
    if any(not (0 <= b[0] < b[2] <= width and 0 <= b[1] < b[3] <= height) for b in boxes.values()):
        raise ValueError("Scorebug crop outside source")
    end = min(
        (manifest["frames"] - 1) / manifest["source"]["nominal_fps"],
        (capture.get(cv2.CAP_PROP_FRAME_COUNT) - 1) / capture.get(cv2.CAP_PROP_FPS),
    )
    samples, raw = [], []
    with tempfile.TemporaryDirectory(prefix="vyneros-scoreboard-") as directory:
        temporary = Path(directory) / "field.png"
        time = 0.0
        while time <= end + 1e-8:
            capture.set(cv2.CAP_PROP_POS_MSEC, time * 1000)
            ok, frame = capture.read()
            if not ok:
                break
            fields = {
                key: ocr_field(
                    frame,
                    box,
                    "score" if key.endswith("score") else "clock" if key == "clock" else "text",
                    temporary,
                )
                for key, box in boxes.items()
            }
            period, clock = parse_period(fields["period"]), parse_clock(fields["clock"])
            scores = None
            if (
                fields["left_team"] == left[1]
                and fields["right_team"] == right[1]
                and all(re.fullmatch(r"\d{1,3}", fields[k]) for k in ("left_score", "right_score"))
            ):
                scores = {left[0]: int(fields["left_score"]), right[0]: int(fields["right_score"])}
            samples.append(
                {"time_s": round(time, 6), "period": period, "clock_s": clock, "scores": scores}
            )
            raw.append({"time_s": round(time, 6), **fields})
            time += sample_s
    capture.release()
    version = subprocess.run(
        ["tesseract", "--version"], capture_output=True, text=True, check=True
    ).stdout.splitlines()[0]
    return {
        "schema_version": 1,
        "source_sha256": digest.hexdigest(),
        "sample_s": sample_s,
        "teams": [{"team_id": tid, "label": label} for tid, label in (left, right)],
        "provenance": {
            "method": "fixed_crop_ocr_temporal_consensus_v1",
            "engine": version,
            "profile": profile,
            "score_confirmation_reads": 2,
            "training": False,
        },
        "samples": confirm_readings(samples, ids, sample_s),
        "raw_readings": raw,
    }


def summarize(document, checkpoints=None):
    samples = document["samples"]
    result = {
        "source_sha256": document["source_sha256"],
        "provenance": document["provenance"],
        "sample_count": len(samples),
        "readable": {
            key: sum(row[key] is not None for row in samples)
            for key in ("period", "clock_s", "scores")
        },
    }
    if checkpoints is not None:
        if checkpoints["source_sha256"] != document["source_sha256"]:
            raise ValueError("Review checkpoints belong to another video")
        indexed = {row["time_s"]: row for row in samples}
        audit = {}
        for key in ("period", "clock_s", "scores"):
            counts = dict(agree=0, wrong=0, unavailable=0, expected_missing=0)
            for expected in checkpoints["readings"]:
                actual = indexed.get(expected["time_s"], {}).get(key)
                if expected[key] is None:
                    counts["expected_missing"] += 1
                    counts["wrong"] += actual is not None
                elif actual is None:
                    counts["unavailable"] += 1
                else:
                    counts["agree" if actual == expected[key] else "wrong"] += 1
            audit[key] = counts
        result["development_audit"] = audit
        result["independent_accuracy_validation"] = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--video", type=Path, help="Defaults to the absolute source path from run.json"
    )
    parser.add_argument(
        "--left-team",
        required=True,
        help="Roster team_id:graphic abbreviation, e.g. houston_rockets:HOU",
    )
    parser.add_argument("--right-team", required=True)
    parser.add_argument(
        "--profile", type=Path, help="JSON with reference_size and field crop rectangles"
    )
    parser.add_argument("--sample-s", type=float, default=0.5)
    parser.add_argument(
        "--review-checkpoints", type=Path, help="Source-bound manual development checkpoints"
    )
    parser.add_argument(
        "--report", type=Path, help="Write coverage and optional development audit as JSON"
    )
    args = parser.parse_args()
    if not 0.1 <= args.sample_s <= 2:
        parser.error("sample-s must be between .1 and 2")
    manifest = json.loads((args.run / "run.json").read_text())
    if args.review_checkpoints and not args.report:
        parser.error("--review-checkpoints requires --report")
    checkpoints = (
        json.loads(args.review_checkpoints.read_text()) if args.review_checkpoints else None
    )
    if checkpoints and checkpoints.get("source_sha256") != manifest["source"]["sha256"]:
        parser.error("Review checkpoints belong to another video")
    video = args.video or Path(manifest["source"]["path"])
    if not video.is_absolute():
        parser.error("A relative source requires --video with its resolved path")
    profile = json.loads(args.profile.read_text()) if args.profile else ESPN

    def team(value):
        tid, label = value.split(":", 1)
        if not tid or not re.fullmatch(r"[A-Z]{2,5}", label):
            parser.error("Expected team_id:ABBREVIATION")
        return tid, label

    document = extract(
        args.run, video, team(args.left_team), team(args.right_team), profile, args.sample_s
    )
    output = args.run / "scoreboard.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    temporary.replace(output)
    if args.report:
        args.report.write_text(json.dumps(summarize(document, checkpoints), indent=2) + "\n")
    print(
        json.dumps(
            {
                "path": str(output),
                "samples": len(document["samples"]),
                "scores": sum(r["scores"] is not None for r in document["samples"]),
                "periods": sum(r["period"] is not None for r in document["samples"]),
            }
        )
    )


if __name__ == "__main__":
    main()
