"""NBA Rule 1 / 2025–26 official court diagram, measured in metres.

Regulation dimensions use their specified inside/outside edges, not rounded
FIBA dimensions. Landmark locations follow these nominal reference edges;
the learned detector does not resolve the 2-inch paint thickness reliably.
The origin is the far-left inside corner in the broadcast tactical view.
"""

import math

import numpy as np

FOOT = 0.3048
INCH = FOOT / 12
LENGTH = 94 * FOOT
WIDTH = 50 * FOOT
LINE_WIDTH = 2 * INCH
LANE_WIDTH = 16 * FOOT
FREE_THROW_X = 19 * FOOT
RIM_X = 5.25 * FOOT
THREE_RADIUS = 23.75 * FOOT
CORNER_DISTANCE = 22 * FOOT
RESTRICTED_RADIUS = 4 * FOOT
SOURCE = "https://cdn.nba.com/manage/2026/01/Official-2025-26-NBA-Playing-Rules.pdf"

# The reference checkpoint's 18 labels, NOT COCO person keypoints.
LANDMARK_NAMES = (
    "left_far_corner", "left_far_three", "left_far_lane", "left_near_lane",
    "left_near_three", "left_near_corner", "center_near", "center_far",
    "left_free_throw_far", "left_free_throw_near", "right_near_corner",
    "right_near_three", "right_near_lane", "right_far_lane", "right_far_three",
    "right_far_corner", "right_free_throw_far", "right_free_throw_near",
)
LANDMARKS = np.array([
    (0, 0), (0, 3), (0, 17), (0, 33), (0, 47), (0, 50),
    (47, 50), (47, 0), (19, 17), (19, 33),
    (94, 50), (94, 47), (94, 33), (94, 17), (94, 3), (94, 0),
    (75, 17), (75, 33),
], dtype=np.float64) * FOOT
CORNERS = np.array([(0, 0), (LENGTH, 0), (LENGTH, WIDTH), (0, WIDTH)])


def project(matrix, points):
    """Homogeneous projection; invalid/horizon points are NaN, never clipped."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    q = np.column_stack((pts, np.ones(len(pts)))) @ np.asarray(matrix).T
    result = np.full((len(pts), 2), np.nan)
    good = np.isfinite(q).all(axis=1) & (np.abs(q[:, 2]) > 1e-9)
    result[good] = q[good, :2] / q[good, 2, None]
    return result


def markings():
    """Ground-plane polylines, including arcs (never the elevated basket)."""
    lines = [np.vstack((CORNERS, CORNERS[0])),
             np.array([[LENGTH / 2, 0], [LENGTH / 2, WIDTH]])]
    theta = np.linspace(0, 2 * math.pi, 121)
    for radius in (6 * FOOT, 2 * FOOT):
        lines.append(np.column_stack((LENGTH / 2 + radius * np.cos(theta),
                                      WIDTH / 2 + radius * np.sin(theta))))
    half_angle = math.asin(CORNER_DISTANCE / THREE_RADIUS)
    arc = np.linspace(-half_angle, half_angle, 121)
    join_x = RIM_X + math.sqrt(THREE_RADIUS**2 - CORNER_DISTANCE**2)
    for mirrored in (False, True):
        half = [
            np.array([[0, 17 * FOOT], [FREE_THROW_X, 17 * FOOT],
                      [FREE_THROW_X, 33 * FOOT], [0, 33 * FOOT]]),
            np.column_stack((FREE_THROW_X + 6 * FOOT * np.cos(theta),
                             WIDTH / 2 + 6 * FOOT * np.sin(theta))),
            np.array([[0, 3 * FOOT], [join_x, 3 * FOOT]]),
            np.array([[0, 47 * FOOT], [join_x, 47 * FOOT]]),
            np.column_stack((RIM_X + THREE_RADIUS * np.cos(arc),
                             WIDTH / 2 + THREE_RADIUS * np.sin(arc))),
        ]
        restricted = np.linspace(-math.pi / 2, math.pi / 2, 61)
        half.append(np.vstack((
            [4 * FOOT, WIDTH / 2 - RESTRICTED_RADIUS],
            np.column_stack((RIM_X + RESTRICTED_RADIUS * np.cos(restricted),
                             WIDTH / 2 + RESTRICTED_RADIUS * np.sin(restricted))),
            [4 * FOOT, WIDTH / 2 + RESTRICTED_RADIUS])))
        for line in half:
            if mirrored:
                line[:, 0] = LENGTH - line[:, 0]
            lines.append(line)
    return lines


def metadata():
    return {"league": "NBA", "units": "metres", "length": LENGTH, "width": WIDTH,
            "line_width": LINE_WIDTH, "lane_width": LANE_WIDTH,
            "free_throw_baseline_distance": FREE_THROW_X, "rim_baseline_distance": RIM_X,
            "three_point_radius": THREE_RADIUS, "corner_three_distance": CORNER_DISTANCE,
            "restricted_radius": RESTRICTED_RADIUS, "source": SOURCE,
            "origin": "far_left_inside_corner", "x_axis": "broadcast_left_to_right",
            "y_axis": "far_sideline_to_near_sideline", "orientation_scope": "camera_segment",
            "landmark_names": list(LANDMARK_NAMES), "landmarks_m": LANDMARKS.tolist(),
            "measurement_convention": "nominal_rule_dimensions_at_specified_inside_or_outside_edges"}
