"""Conservative, causal grouping of jersey torso colors within one video."""

from collections import deque

import cv2
import numpy as np


def dominant_color(rgb):
    """Return a coarse uniform color, or None for a mixed/unreadable torso."""
    height, width = rgb.shape[:2]
    core = rgb[height // 10 : height - height // 10,
               width // 10 : width - width // 10]
    hsv = cv2.cvtColor(core, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    colored = (saturation >= 75) & (value >= 50)
    if colored.mean() >= 0.70:
        hist = np.bincount((hue[colored] // 10).ravel(), minlength=18)
        centre = int(hist.argmax())
        inliers = ((hue // 10 - centre + 1) % 18 <= 2) & colored
        if inliers.sum() / colored.sum() >= 0.80:
            angles = hue[inliers].astype(np.float32) * (2 * np.pi / 180)
            angle = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
            return ("hue", float(angle % (2 * np.pi) * 180 / (2 * np.pi)))
    if (value <= 85).mean() >= 0.70:
        return ("dark", 0.0)
    if ((value >= 175) & (saturation <= 60)).mean() >= 0.70:
        return ("light", 0.0)
    return None


def color_distance(a, b):
    if a[0] != b[0]:
        return 180.0
    if a[0] != "hue":
        return 0.0
    delta = abs(a[1] - b[1])
    return min(delta, 180 - delta)


class TeamColors:
    """Two anonymous color groups; every track needs repeated clean evidence."""

    def __init__(self):
        self.groups = []

    def observe(self, state, crop, timestamp):
        if state.get("color_conflict"):
            return
        if crop["overlap"] > 0.10:
            return
        if timestamp - state["last_color_sample"] < 0.35:
            return
        state["last_color_sample"] = timestamp
        sample = dominant_color(crop.get("color_image", crop["image"]))
        if sample is None:
            return
        votes = state["color_votes"]
        votes.append((timestamp, sample))
        while votes and timestamp - votes[0][0] > 12:
            votes.popleft()
        if len(votes) < 3:
            return
        recent = [value for _, value in list(votes)[-3:]]
        if any(color_distance(recent[0], value) > 10 for value in recent[1:]):
            state["team_group"] = None
            return
        signature = recent[-1]
        distances = [color_distance(signature, group) for group in self.groups]
        if distances and min(distances) <= 10:
            group = f"team_{distances.index(min(distances)) + 1}"
        elif all(distance >= 20 for distance in distances) and len(self.groups) < 2:
            self.groups.append(signature)
            group = f"team_{len(self.groups)}"
        else:
            state["team_group"] = None
            return
        if state.get("first_team_group") not in (None, group):
            state["color_conflict"] = True
            state["team_group"] = None
            return
        state["first_team_group"] = group
        state["team_group"] = group

    @staticmethod
    def new_votes():
        return deque(maxlen=8)
