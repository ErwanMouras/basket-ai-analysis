"""Robust point fitting and bounded temporal propagation on the floor plane."""

import math

import cv2
import numpy as np

from training.court.config import settings
from training.court.geometry import CORNERS, LANDMARKS, LENGTH, WIDTH, metadata, project
from training.court.model import CourtKeypoints


def hull(points):
    return cv2.convexHull(np.asarray(points, np.float32)).reshape(-1, 2)


def normalize(matrix):
    if matrix is None or not np.isfinite(matrix).all() or abs(matrix[2, 2]) < 1e-10:
        return None
    matrix = matrix / matrix[2, 2]
    if abs(np.linalg.det(matrix)) < 1e-12:
        return None
    return matrix


def fit_keypoints(keypoints, shape, cfg):
    """Fit court -> pixels so RANSAC's threshold is in pixels, then invert."""
    values = np.asarray(keypoints, dtype=np.float64)
    if values.shape != (18, 3) or not np.isfinite(values).all():
        raise ValueError("Court detector must return finite 18 x 3 source-pixel keypoints")
    height, width = shape[:2]
    valid = ((values[:, 2] >= cfg["keypoint_confidence"])
             & (values[:, 0] > 1) & (values[:, 0] < width - 1)
             & (values[:, 1] > 1) & (values[:, 1] < height - 1))
    # The reference model sometimes predicts both half-court labels at the
    # same pixel. Keep the more confident observation, not contradictory pairs.
    ids = []
    for i in np.argsort(-values[:, 2]):
        if valid[i] and all(np.linalg.norm(values[i, :2] - values[j, :2]) > 8 * height / 1080 for j in ids):
            ids.append(int(i))
    report = {"candidate_ids": ids, "inlier_ids": [], "rejection": None}

    def reject(reason):
        report["rejection"] = reason
        return None, report

    if len(ids) < cfg["min_inliers"]:
        return reject("insufficient_keypoints")
    world, pixels = LANDMARKS[ids], values[ids, :2]
    if cv2.contourArea(hull(world)) < cfg["min_support_area_m2"]:
        return reject("degenerate_court_support")
    if cv2.contourArea(hull(pixels)) < 0.002 * height * width:
        return reject("degenerate_image_support")
    matrix, mask = cv2.findHomography(world, pixels, cv2.RANSAC,
                                     cfg["ransac_pixels"] * height / 1080,
                                     maxIters=3000, confidence=0.999)
    matrix = normalize(matrix)
    if matrix is None or mask is None:
        return reject("singular_homography")
    inliers = mask.ravel().astype(bool)
    report["inlier_ids"] = [i for i, keep in zip(ids, inliers) if keep]
    report["inlier_ratio"] = float(inliers.mean())
    if inliers.sum() < cfg["min_inliers"] or inliers.mean() < cfg["min_inlier_ratio"]:
        return reject("insufficient_inliers")
    world, pixels = world[inliers], pixels[inliers]
    if cv2.contourArea(hull(world)) < cfg["min_support_area_m2"]:
        return reject("degenerate_inlier_support")
    matrix = normalize(cv2.findHomography(world, pixels, 0)[0])
    if matrix is None:
        return reject("singular_refit")
    residuals = np.linalg.norm(project(matrix, world) - pixels, axis=1)
    rmse = float(np.sqrt(np.mean(residuals**2)))
    report["reprojection_rmse_px"] = rmse if math.isfinite(rmse) else None
    if not math.isfinite(rmse) or rmse > cfg["max_reprojection_pixels"] * height / 1080:
        return reject("reprojection_error")
    # A continuous court must not cross the projective horizon.
    denominators = np.column_stack((CORNERS, np.ones(4))) @ matrix[2]
    if np.min(denominators) * np.max(denominators) <= 0:
        return reject("horizon_crosses_court")
    center = world.mean(axis=0)
    axes = project(matrix, [center, center + [0.1, 0], center + [0, 0.1]])
    dx, dy = axes[1] - axes[0], axes[2] - axes[0]
    if abs(dx[0]) < 1e-3 or dy[1] <= 0 or abs(dx[0] * dy[1] - dx[1] * dy[0]) < 0.01:
        return reject("unsupported_camera_orientation")
    # The model can swap left/right labels on a mirrored broadcast half.
    # Explicitly canonicalize x along screen-left -> screen-right. This is
    # per camera segment, not a guarantee of a fixed arena compass direction.
    reflected = bool(dx[0] < 0)
    if reflected:
        reflection = np.array([[-1., 0, LENGTH], [0, 1, 0], [0, 0, 1]])
        matrix = normalize(matrix @ reflection)
        world = world.copy()
        world[:, 0] = LENGTH - world[:, 0]
    inverse = normalize(np.linalg.inv(matrix))
    if inverse is None:
        return reject("singular_inverse")
    report["reflected_model_labels_x"] = reflected
    return {"image_to_court": inverse, "support": hull(world), "fit": report}, report


def motion_between(previous, current, mask, cfg):
    """Forward/backward LK tracks on masked floor texture; prev -> current."""
    points = cv2.goodFeaturesToTrack(previous, maxCorners=500, qualityLevel=0.01,
                                    minDistance=10, mask=mask, blockSize=7)
    if points is None or len(points) < cfg["motion_min_inliers"]:
        return None, {"inliers": 0, "reason": "insufficient_features"}
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    following, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None,
                                                 winSize=(21, 21), maxLevel=3, criteria=criteria)
    if following is None:
        return None, {"inliers": 0, "reason": "flow_failed"}
    backward, back_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, following, None,
                                                     winSize=(21, 21), maxLevel=3, criteria=criteria)
    if backward is None:
        return None, {"inliers": 0, "reason": "backward_flow_failed"}
    p, q = points.reshape(-1, 2), following.reshape(-1, 2)
    keep = ((status.ravel() == 1) & (back_status.ravel() == 1)
            & (np.linalg.norm(p - backward.reshape(-1, 2), axis=1) < 1.5)
            & np.isfinite(q).all(axis=1)
            & (q[:, 0] >= 0) & (q[:, 0] < current.shape[1])
            & (q[:, 1] >= 0) & (q[:, 1] < current.shape[0]))
    p, q = p[keep], q[keep]
    if len(p) < cfg["motion_min_inliers"]:
        return None, {"inliers": 0, "reason": "inconsistent_flow"}
    transform, inliers = cv2.findHomography(p, q, cv2.RANSAC, 2.5, maxIters=1000)
    transform = normalize(transform)
    if transform is None or inliers is None:
        return None, {"inliers": 0, "reason": "motion_fit_failed"}
    good = inliers.ravel().astype(bool)
    report = {"inliers": int(good.sum()), "inlier_ratio": float(good.mean()), "reason": None}
    if (good.sum() < cfg["motion_min_inliers"] or good.mean() < 0.65
            or cv2.contourArea(hull(p[good])) < 0.015 * previous.size):
        return None, {**report, "reason": "weak_motion_support"}
    h, w = previous.shape
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
    moved = project(transform, corners)
    ratio = abs(cv2.contourArea(moved.astype(np.float32))) / (w * h)
    if (not np.isfinite(moved).all() or not 0.65 < ratio < 1.5
            or np.max(np.linalg.norm(moved - corners, axis=1)) > 0.3 * math.hypot(w, h)):
        return None, {**report, "reason": "implausible_motion"}
    return transform, report


class CourtCalibrator:
    def __init__(self, config, *, fps, model=None):
        self.config = settings(config)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Court calibration requires a positive frame rate")
        self.fps = fps
        self.model = model if model is not None else CourtKeypoints(self.config)
        self.previous_gray = self.previous_mask = self.previous_thumbnail = None
        self.previous_histogram = None
        self.state = None
        self.segment_id = 0
        self.next_frame = 0
        self.last_fit = None
        self.shape = None
        self.stats = {"fit": 0, "propagated": 0, "unavailable": 0, "cuts": 0, "model_calls": 0}
        self.provenance = {"model": self.model.provenance, "geometry": metadata(),
                           "config": self.config, "fps": fps,
                           "homography_direction": "source_pixels_to_court_metres",
                           "propagation": "masked_floor_forward_backward_LK_RANSAC",
                           "lens_distortion": "not_estimated"}

    def _mask(self, image, detections, scaled_shape):
        height, width = image.shape[:2]
        mask = np.zeros((height, width), np.uint8)
        if self.state is not None:
            corners = project(np.linalg.inv(self.state["image_to_court"]), CORNERS)
            if np.isfinite(corners).all() and np.max(np.abs(corners)) < 1e6:
                cv2.fillConvexPoly(mask, np.round(corners).astype(np.int32), 255)
        else:
            mask[int(height * .25):] = 255
        for detection in detections:
            if detection["confidence"] < 0.15:
                continue
            x1, y1, x2, y2 = detection["bbox"]
            cv2.rectangle(mask, (int(x1) - 8, int(y1) - 8), (int(x2) + 8, int(y2) + 8), 0, -1)
        for x1, y1, x2, y2 in self.config["excluded_regions"]:
            mask[int(y1 * height):int(y2 * height), int(x1 * width):int(x2 * width)] = 0
        return cv2.resize(mask, scaled_shape, interpolation=cv2.INTER_NEAREST)

    def update(self, image, detections, *, frame_index, force_cut=False):
        if frame_index != self.next_frame:
            raise ValueError("Court calibration requires consecutive frames starting at zero")
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Court calibration requires uint8 BGR frames")
        if self.shape is not None and image.shape != self.shape:
            raise ValueError("Court video dimensions changed")
        self.shape = image.shape
        cfg = self.config
        height, width = image.shape[:2]
        scaled_width = min(width, cfg["motion_width"])
        scaled_height = max(1, round(height * scaled_width / width))
        gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (scaled_width, scaled_height))
        thumbnail = cv2.resize(image, (64, 36))
        hsv = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
        hist = cv2.normalize(hist, None, norm_type=cv2.NORM_L1)
        motion, motion_report = None, {"inliers": 0, "reason": "first_frame"}
        difference = histogram_distance = 0.0
        structural_correlation = None
        if self.previous_gray is not None:
            motion, motion_report = motion_between(self.previous_gray, gray, self.previous_mask, cfg)
            difference = float(np.mean(cv2.absdiff(thumbnail, self.previous_thumbnail))) / 255
            histogram_distance = float(cv2.compareHist(hist, self.previous_histogram, cv2.HISTCMP_BHATTACHARYYA))
            a = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2GRAY).astype(float).ravel()
            b = cv2.cvtColor(self.previous_thumbnail, cv2.COLOR_BGR2GRAY).astype(float).ravel()
            a -= a.mean()
            b -= b.mean()
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            structural_correlation = float(np.dot(a, b) / denom) if denom > 1e-6 else 0.0
        # Arena photography flashes change brightness/histograms abruptly but
        # preserve scene structure. They must not erase player identities.
        flash_like = structural_correlation is not None and structural_correlation > .75
        automatic_cut = self.previous_gray is not None and cfg["auto_cuts"] and not flash_like and (
            (difference > cfg["cut_pixel_difference"])
            or (difference > .12 and histogram_distance > cfg["cut_histogram_distance"])
            or (difference > .20 and motion is None))
        cut = frame_index > 0 and (force_cut or automatic_cut)
        if cut:
            self.segment_id += 1
            self.stats["cuts"] += 1
            self.state = None
            self.last_fit = None
            motion = None
        propagated = None
        if (self.state is not None and motion is not None and self.last_fit is not None
                and (frame_index - self.last_fit) / self.fps <= cfg["max_propagation_seconds"]):
            scale = np.diag([scaled_width / width, scaled_height / height, 1.])
            full_motion = np.linalg.inv(scale) @ motion @ scale
            matrix = normalize(self.state["image_to_court"] @ np.linalg.inv(full_motion))
            if matrix is not None:
                propagated = {**self.state, "image_to_court": matrix}
        keypoints = None
        fit_report = None
        candidate = None
        if frame_index % cfg["detect_every"] == 0 or propagated is None:
            keypoints = self.model.predict(image)
            self.stats["model_calls"] += 1
            candidate, fit_report = fit_keypoints(keypoints, image.shape, cfg)
            if candidate is not None and propagated is not None:
                anchors = project(np.linalg.inv(propagated["image_to_court"]), propagated["support"])
                disagreement = np.linalg.norm(project(candidate["image_to_court"], anchors) - propagated["support"], axis=1)
                if not np.isfinite(disagreement).all() or np.median(disagreement) > cfg["max_temporal_jump_m"]:
                    fit_report["rejection"] = "temporal_disagreement"
                    candidate = None
        status = "unavailable"
        self.state = None
        if candidate is not None:
            self.state = candidate
            self.last_fit = frame_index
            status = "fit"
        elif propagated is not None:
            self.state = propagated
            status = "propagated"
        self.stats[status] += 1
        self.previous_gray, self.previous_thumbnail, self.previous_histogram = gray, thumbnail, hist
        self.previous_mask = self._mask(image, detections, (scaled_width, scaled_height))
        self.next_frame += 1
        return {"frame_index": frame_index, "timestamp_seconds": frame_index / self.fps,
                "segment_id": self.segment_id, "status": status, "scene_cut": cut,
                "cut_reason": "explicit" if cut and force_cut else "automatic" if cut else None,
                "cut_pixel_difference": difference, "cut_histogram_distance": histogram_distance,
                "cut_structure_correlation": structural_correlation,
                "image_to_court": self.state["image_to_court"].tolist() if self.state else None,
                "support_polygon_m": self.state["support"].tolist() if self.state else None,
                "last_fit_frame": self.last_fit,
                "fit_age_seconds": (frame_index - self.last_fit) / self.fps if self.last_fit is not None else None,
                "keypoints": np.asarray(keypoints).tolist() if keypoints is not None else None,
                "fit_attempt": fit_report, "accepted_fit": self.state["fit"] if self.state else None,
                "motion": motion_report}

    def project_players(self, detections, record, shape):
        """Keep every observation, with null coordinates and reason on rejection."""
        height, width = shape[:2]
        result = []
        for index, detection in enumerate(detections):
            x1, y1, x2, y2 = detection["bbox"]
            point = np.array([(x1 + x2) / 2, y2], dtype=float)
            method = "bbox_bottom_center"
            reason = None
            if detection["confidence"] < self.config["min_player_confidence"]:
                reason = "low_player_confidence"
            elif y2 >= height - 2:
                reason = "feet_out_of_frame"
            if self.config["footpoint"] == "ankles":
                pose = detection.get("pose")
                if (pose and pose.get("format") == "coco17" and len(pose.get("valid", [])) == 17
                        and pose["valid"][15] and pose["valid"][16]):
                    ankles = np.asarray([pose["keypoints"][15], pose["keypoints"][16]], dtype=float)
                    if np.isfinite(ankles).all() and np.all(ankles[:, 0] >= x1) and np.all(ankles[:, 0] <= x2):
                        point = ankles.mean(axis=0)
                        method = "ankles_midpoint_approximation"
            xy = None
            if reason is None and record["image_to_court"] is None:
                reason = "calibration_unavailable"
            if reason is None:
                xy = project(record["image_to_court"], [point])[0]
                margin = self.config["court_margin_m"]
                if not np.isfinite(xy).all():
                    reason = "projection_at_infinity"
                elif not (-margin <= xy[0] <= LENGTH + margin and -margin <= xy[1] <= WIDTH + margin):
                    reason = "outside_court"
                elif cv2.pointPolygonTest(np.asarray(record["support_polygon_m"], np.float32), tuple(xy), True) < -self.config["max_extrapolation_m"]:
                    reason = "outside_calibrated_support"
            result.append({"detection_index": index, "track_id": detection.get("track_id"),
                           "bbox": list(detection["bbox"]), "confidence": detection["confidence"],
                           "ground_point_px": point.tolist(), "ground_point_method": method,
                           "position_m": xy.tolist() if reason is None else None,
                           "status": "projected" if reason is None else reason})
        return result
