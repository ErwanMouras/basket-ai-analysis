"""JSON-native observations with explicit units and referential validation."""

import json
import math

VERSION = 1
STATUSES = {"ok", "not_detected", "not_scheduled", "disabled", "unavailable", "error"}


def finite_point(value, size):
    return isinstance(value, list) and len(value) == size and all(
        type(x) in (int, float) and math.isfinite(x) for x in value)


def validate_frame(row):
    if type(row.get("schema_version")) is not int or row["schema_version"] != VERSION or row.get("artifact_type") != "video_observation":
        raise ValueError("Unsupported observation schema")
    for name in ("frame_index", "segment_id"):
        if type(row[name]) is not int or row[name] < 0:
            raise ValueError(f"Invalid {name}")
    t = row["timestamp_s"]
    if type(t) not in (int, float) or not math.isfinite(t) or t < 0:
        raise ValueError("Invalid source timestamp")
    if row["coordinate_space"] != "source_pixels" or row["distance_unit"] != "m":
        raise ValueError("Invalid coordinate units")
    if any(type(row[k]) is not int or row[k] < 1 for k in ("width", "height")):
        raise ValueError("Invalid source dimensions")
    ids, tracks = set(), set()
    for p in row["persons"]:
        if p["observation_id"] in ids or p["role"] not in ("player", "referee", "unknown"):
            raise ValueError("Duplicated observation or invalid role")
        ids.add(p["observation_id"])
        if not finite_point(p["bbox"], 4) or not (0 <= p["bbox"][0] < p["bbox"][2] <= row["width"] and 0 <= p["bbox"][1] < p["bbox"][3] <= row["height"]):
            raise ValueError("Invalid source box")
        tid = p["track_id"]
        if tid is not None:
            if type(tid) is not int or tid < 1 or tid in tracks:
                raise ValueError("Invalid or repeated track")
            tracks.add(tid)
        if p["position_m"] is not None and not finite_point(p["position_m"], 2):
            raise ValueError("Invalid metric position")
        if type(p["confidence"]) not in (float, int) or not 0 <= p["confidence"] <= 1:
            raise ValueError("Invalid confidence")
        pose = p.get("pose")
        if pose is not None and (pose["format"] != "coco17" or len(pose["keypoints"]) != 17 or
                                 any(v is not None and not finite_point(v, 2) for v in pose["keypoints"])):
            raise ValueError("Invalid COCO17 pose")
        jersey = p.get("jersey")
        if jersey and jersey.get("number") is not None:
            from training.jersey.config import number
            if number(jersey["number"]) is None:
                raise ValueError("Jersey number must preserve its digit string")
    ball = row["ball"]
    point = ball["position_px"]
    if point is not None and (not finite_point(point, 2) or not (0 <= point[0] < row["width"] and 0 <= point[1] < row["height"])):
        raise ValueError("Invalid source ball position")
    if ball.get("confidence") is not None and not 0 <= ball["confidence"] <= 1:
        raise ValueError("Invalid ball confidence")
    matrix = (row.get("court") or {}).get("image_to_court")
    if matrix is not None and (not isinstance(matrix, list) or len(matrix) != 3 or any(not finite_point(v, 3) for v in matrix)):
        raise ValueError("Invalid court homography")
    if any(v["status"] not in STATUSES for v in row["stages"].values()):
        raise ValueError("Invalid stage status")
    json.dumps(row, allow_nan=False)
    return row


def validate_statistics(document):
    if document.get("schema_version") != VERSION or document.get("metric") != "observed_distance":
        raise ValueError("Unsupported statistics schema")
    ids = set()
    for person in document["players"]:
        if person["subject_id"] in ids:
            raise ValueError("Repeated statistic subject")
        ids.add(person["subject_id"])
        tracked, measured = person["tracked_duration_s"], person["measured_duration_s"]
        if not (0 <= measured <= tracked + 1e-9):
            raise ValueError("Measured duration must be a subset of tracked duration")
        distance = person["observed_distance_m"]
        if distance is not None and (type(distance) not in (float, int) or distance < 0 or measured <= 0):
            raise ValueError("Distance requires a measured interval")
        if (distance is None) != (person["measurement_status"] == "unavailable"):
            raise ValueError("Unavailable distance must be null")
    json.dumps(document, allow_nan=False)
    return document
