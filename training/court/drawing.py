"""Projected NBA markings and a metric, aspect-correct tactical view."""

import cv2
import numpy as np

from training.court.geometry import LENGTH, WIDTH, markings, project


def color_for(track_id):
    return ((60 + track_id * 67 % 196, 60 + track_id * 131 % 196,
             60 + track_id * 43 % 196) if track_id is not None else (0, 220, 0))


def draw_court(image, record):
    height, width = image.shape[:2]
    if record["image_to_court"] is not None:
        inverse = np.linalg.inv(record["image_to_court"])
        for line in markings():
            projected = project(inverse, line)
            for a, b in zip(projected[:-1], projected[1:]):
                if not np.isfinite([a, b]).all() or np.max(np.abs([a, b])) > 1e6:
                    continue
                visible, start, end = cv2.clipLine((0, 0, width, height), tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)))
                if visible:
                    cv2.line(image, start, end, (255, 210, 30), 2, cv2.LINE_AA)
    label = f"NBA court: {record['status']} | segment {record['segment_id']}"
    cv2.putText(image, label, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, label, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1, cv2.LINE_AA)


def tactical_view(positions, record, *, width=720):
    scale = (width - 40) / LENGTH
    height = int(round(WIDTH * scale)) + 70
    height += height % 2
    image = np.full((height, width, 3), (37, 46, 50), np.uint8)
    transform = np.array([[scale, 0, 20], [0, scale, 45], [0, 0, 1]])
    for line in markings():
        pts = np.round(project(transform, line)).astype(np.int32)
        cv2.polylines(image, [pts], False, (165, 182, 185), 1, cv2.LINE_AA)
    for row in positions:
        if row["position_m"] is None:
            continue
        p = tuple(np.round(project(transform, [row["position_m"]])[0]).astype(int))
        color = color_for(row["track_id"])
        cv2.circle(image, p, 6, color, -1, cv2.LINE_AA)
        label = str(row["track_id"]) if row["track_id"] is not None else "?"
        cv2.putText(image, label, (p[0] + 8, p[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
    cv2.putText(image, f"NBA 28.6512 x 15.24 m | {record['status']}", (18, 25),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (235, 235, 235), 1, cv2.LINE_AA)
    return image
