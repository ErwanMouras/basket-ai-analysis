"""Offline shot-attempt proposals from observed ball, wrists and court motion.

No outcome, official shot count, physical acceleration or made/missed label is
inferred. Future observations within a short flight confirm an earlier release.
"""

from collections import Counter
from dataclasses import dataclass
from statistics import median
import math

from viewer.heatmaps import player_subject
from viewer.phases import team_of
from viewer.shot_geometry import inverse, matrix, number, point, quadratic, rim_anchors, warp

VERSION = "shot-attempts-v1"
RULES = {
    "min_ball_confidence": 0.35,
    "max_gap_s": 0.2,
    "max_flight_s": 2.4,
    "min_free_flight_share": 0.65,
    "rim_radius_body_heights": 0.65,
    "candidate_fit_rmse_body_heights": 0.12,
    "probable_fit_rmse_body_heights": 0.08,
}


@dataclass(slots=True)
class Sample:
    frame: int
    time: float
    segment: int
    width: int
    height: int
    ball: tuple | None
    court: list | None
    contact: dict | None
    body_height: float
    shoulder_y: float
    cut: bool
    included: bool


def hand_contact(row, ball):
    """Associate visible wrists, keeping unknown players/referees as competitors."""
    if ball is None:
        return None
    tracks = Counter(p.get("track_id") for p in row.get("persons", []))
    subjects = Counter(player_subject(p) for p in row.get("persons", []))
    candidates = []
    for p in row.get("persons", []):
        box, pose = p.get("bbox"), p.get("pose") or {}
        if (
            not isinstance(box, (list, tuple))
            or len(box) != 4
            or not all(map(number, box))
            or box[3] <= box[1]
            or pose.get("format") != "coco17"
            or pose.get("coordinate_space") != "source"
        ):
            continue
        keys, scores, valid = (
            pose.get("keypoints") or [],
            pose.get("scores") or [],
            pose.get("valid"),
        )

        def key(i):
            return (
                keys[i]
                if len(keys) > i
                and point(keys[i])
                and len(scores) > i
                and number(scores[i])
                and scores[i] >= 0.35
                and (valid is None or len(valid) > i and valid[i])
                else None
            )

        wrists = [key(i) for i in (9, 10) if key(i)]
        if not wrists:
            continue
        size = box[3] - box[1]
        distance = min(math.dist(ball, w) for w in wrists) / size
        if distance > 0.24:
            continue
        shoulders = [key(i)[1] for i in (5, 6) if key(i)]
        raised = bool(shoulders and ball[1] <= sum(shoulders) / len(shoulders) - 0.04 * size)
        candidates.append(
            {
                "distance": distance,
                "height": size,
                "raised": raised,
                "role": p.get("role"),
                "track_id": p.get("track_id"),
                "team_id": team_of(p),
                "subject_id": player_subject(p) if subjects[player_subject(p)] == 1 else None,
                "unique_track": p.get("track_id") is not None and tracks[p["track_id"]] == 1,
            }
        )
    candidates.sort(key=lambda p: p["distance"])
    if not candidates:
        return None
    result = candidates[0]
    if len(candidates) > 1 and candidates[1]["distance"] - result["distance"] < 0.045:
        return {**result, "ambiguous": True}
    return {**result, "ambiguous": False}


class ShotIndex:
    def __init__(self, manifest, roster=None):
        self.geometry = manifest.get("court_coordinate_system") or {}
        self.source = manifest.get("source") or {}
        self.roster = roster or {}
        self.samples = []
        self.events = []
        self.valid_ball = 0
        self.pose_frames = 0
        self.fallback_frames = 0
        self.rejections = Counter()

    def add_frame(self, row):
        raw = row.get("ball") or {}
        ball = raw.get("position_px")
        width, height = row.get("width", self.source.get("width")), row.get(
            "height", self.source.get("height")
        )
        valid = (
            row.get("included", True)
            and raw.get("status") == "ok"
            and point(ball)
            and number(raw.get("confidence"))
            and raw["confidence"] >= RULES["min_ball_confidence"]
            and number(width)
            and number(height)
            and width > 0
            and height > 0
            and 0 <= ball[0] <= width
            and 0 <= ball[1] <= height
        )
        ball = tuple(ball) if valid else None
        self.valid_ball += int(valid)
        contact = hand_contact(row, ball)
        self.pose_frames += int(
            any((p.get("pose") or {}).get("format") == "coco17" for p in row.get("persons", []))
        )
        court = row.get("court") or {}
        homography = (
            court.get("image_to_court") if court.get("status") in ("fit", "propagated") else None
        )
        boxes = [
            p["bbox"]
            for p in row.get("persons", [])
            if p.get("role") == "player"
            and isinstance(p.get("bbox"), (list, tuple))
            and len(p["bbox"]) == 4
            and all(map(number, p["bbox"]))
            and p["bbox"][3] > p["bbox"][1]
        ]
        body_height = median([box[3] - box[1] for box in boxes]) if boxes else 0
        shoulder_y = (
            median([box[1] + 0.25 * (box[3] - box[1]) for box in boxes])
            if boxes
            else (height / 2 if number(height) else 0)
        )
        self.fallback_frames += int(bool(valid and boxes and matrix(homography)))
        self.samples.append(
            Sample(
                row["frame_index"],
                float(row["timestamp_s"]),
                row.get("segment_id", 0),
                width,
                height,
                ball,
                homography if matrix(homography) else None,
                contact,
                body_height,
                shoulder_y,
                bool(row.get("scene_cut")),
                row.get("included", True),
            )
        )

    @staticmethod
    def boundary(a, b):
        return (
            b.cut
            or not a.included
            or not b.included
            or b.segment != a.segment
            or b.frame != a.frame + 1
            or not 0 < b.time - a.time <= RULES["max_gap_s"] + 1e-9
            or (a.width, a.height) != (b.width, b.height)
        )

    def finish(self):
        proposals = []
        for i, seed in enumerate(self.samples[:-4]):
            contact = seed.contact
            if (
                not seed.ball
                or not contact
                or contact["role"] != "player"
                or contact["ambiguous"]
                or not contact["unique_track"]
                or not contact["raised"]
            ):
                continue
            following = self.samples[i + 1]
            if self.boundary(seed, following):
                continue
            if (
                following.contact
                and following.contact["track_id"] == contact["track_id"]
                and not following.contact["ambiguous"]
            ):
                continue
            # Require recent observed support for the release attribution. Never
            # carry an identity across a cut, lost track or conflicting hand.
            support = 0
            for j in range(i, max(-1, i - 9), -1):
                s = self.samples[j]
                if seed.time - s.time > 0.25 or j < i and self.boundary(s, self.samples[j + 1]):
                    break
                if (
                    s.contact
                    and not s.contact["ambiguous"]
                    and s.contact["track_id"] == contact["track_id"]
                ):
                    support += 1
            event = self.flight(i, support)
            if event:
                proposals.append(event)
        # A tip-in or occluded release may have no trustworthy wrist association.
        # Propose it only with a clean arc AND an observed rim approach, always
        # for review and without inventing a shooter or an exact release time.
        for i in range(0, len(self.samples) - 6, 3):
            seed, following = self.samples[i : i + 2]
            if (
                not seed.ball
                or seed.contact
                or not following.ball
                or not seed.court
                or seed.body_height <= 0
                or seed.ball[1] > seed.shoulder_y - 0.05 * seed.body_height
                or self.boundary(seed, following)
            ):
                continue
            inverse_seed = inverse(seed.court)
            after = warp(inverse_seed, following.court, following.ball) if inverse_seed else None
            if after and after[1] < seed.ball[1] - 0.015 * seed.body_height:
                event = self.flight(i, 0)
                if event and event["evidence"]["arc"] and event["evidence"]["rim_approach"]:
                    proposals.append(event)
        # Overlapping release associations are one proposal, not several shots.
        # A later, independently released putback after a catch can remain separate.
        for event in sorted(proposals, key=lambda e: (e["time_s"], -e["quality"])):
            duplicate = next(
                (
                    old
                    for old in self.events
                    if old["segment_id"] == event["segment_id"]
                    and (
                        abs(old["time_s"] - event["time_s"]) < 0.65
                        or len(
                            {p["frame_index"] for p in old["trajectory"]}
                            & {p["frame_index"] for p in event["trajectory"]}
                        )
                        / min(len(old["trajectory"]), len(event["trajectory"]))
                        > 0.6
                    )
                    and event["time_s"] < old["end_s"]
                ),
                None,
            )
            if duplicate:
                if event["quality"] > duplicate["quality"]:
                    self.events.remove(duplicate)
                    self.events.append(event)
            else:
                self.events.append(event)
        self.events.sort(key=lambda e: e["time_s"])
        for i, event in enumerate(self.events):
            event["id"] = f"shot-{i+1}"
            event.pop("quality")
        # Keep only compact evidence after indexing; no raw pose/mask duplication.
        self.samples.clear()

    def flight(self, index, support):
        seed = self.samples[index]
        contact = seed.contact
        size = contact["height"] if contact else seed.body_height
        ref_inverse = inverse(seed.court)
        track = []
        catch_start = None
        last_observed = seed.time
        ending = "window_end"
        for j in range(index, min(len(self.samples), index + 180)):
            s = self.samples[j]
            if j > index and self.boundary(self.samples[j - 1], s):
                ending = "boundary"
                break
            if s.time - seed.time > RULES["max_flight_s"]:
                break
            if s.time - last_observed > RULES["max_gap_s"] + 1e-8:
                ending = "ball_gap"
                break
            if s.ball is None:
                continue
            xy = (
                warp(ref_inverse, s.court, s.ball)
                if ref_inverse and s.court
                else (s.ball if not ref_inverse else None)
            )
            if xy is None:
                continue
            if track and math.dist(xy, track[-1]["xy"]) / size / (s.time - last_observed) > 25:
                ending = "track_jump"
                break
            rims = rim_anchors(s.court, self.geometry, s.width, s.height)
            rims = [warp(ref_inverse, s.court, p) if ref_inverse else p for p in rims]
            rims = [p for p in rims if p]
            touching = bool(s.contact and s.contact["distance"] < 0.16)
            if j > index + 1 and touching:
                if catch_start is None:
                    catch_start = s.time
                if s.time - catch_start >= 0.10:
                    ending = "contact"
                    break
            else:
                catch_start = None
            track.append({"sample": s, "xy": xy, "touching": touching, "rims": rims})
            last_observed = s.time
            # Stop shortly after the first rim approach, before a rebound makes
            # a second arc part of this shot's evidence.
            near = [
                p
                for p in track[1:]
                if p["rims"] and min(math.dist(p["xy"], r) for r in p["rims"]) < 0.45 * size
            ]
            descent = xy[1] - min(p["xy"][1] for p in track)
            if near and s.time - near[0]["sample"].time >= 0.23 and descent >= 0.08 * size:
                ending = "rim_window"
                break
        if len(track) < 6 or track[-1]["sample"].time - seed.time < 0.18:
            self.rejections["insufficient_flight"] += 1
            return None
        # A bounce can start a second curve before a receiver is associated.
        # Keep the longest clean initial arc instead of fitting through a rebound.
        clean = []
        for stop in range(6, len(track) + 1):
            prefix = track[:stop]
            fit_prefix = quadratic(
                [p["sample"].time for p in prefix], [p["xy"][1] / size for p in prefix]
            )
            if not fit_prefix:
                continue
            aa, bb, _, residual = fit_prefix
            span = prefix[-1]["sample"].time - seed.time
            low = min(p["xy"][1] for p in prefix) / size
            if (
                0.25 < aa < 12
                and bb < -0.15
                and 0.04 < -bb / (2 * aa) < span - 0.03
                and prefix[0]["xy"][1] / size - low >= 0.18
                and prefix[-1]["xy"][1] / size - low >= 0.08
                and residual <= RULES["probable_fit_rmse_body_heights"]
            ):
                clean = prefix
        if clean and len(clean) < len(track):
            track = clean
            ending = "arc_fit_end"
        times = [p["sample"].time for p in track]
        ys = [p["xy"][1] / size for p in track]
        fit = quadratic(times, ys)
        if fit is None:
            return None
        a, b, c, error = fit
        duration = times[-1] - times[0]
        rise = ys[0] - min(ys)
        fall = ys[-1] - min(ys)
        free = sum(not p["touching"] for p in track[1:]) / max(1, len(track) - 1)
        arc = (
            0.25 < a < 12
            and b < -0.15
            and 0.04 < -b / (2 * a) < duration - 0.03
            and rise >= 0.18
            and fall >= 0.08
            and error <= RULES["candidate_fit_rmse_body_heights"]
        )
        near = [
            (i, min(math.dist(p["xy"], r) for r in p["rims"]) / size)
            for i, p in enumerate(track)
            if p["rims"]
        ]
        near_distance = min((distance for _, distance in near), default=math.inf)
        rim_near = any(
            i >= 2 and distance <= RULES["rim_radius_body_heights"] for i, distance in near
        )
        # A nearby rising finish may not expose a full parabola. Keep it as a
        # review candidate instead of pretending that a dunk/layup was confirmed.
        close_finish = rim_near and rise >= 0.12 and duration <= 0.9 and error <= 0.18
        if free < RULES["min_free_flight_share"] or not (arc or close_finish):
            self.rejections["held_pass_or_no_arc"] += 1
            return None
        # An observed receiver before the ball approaches a rim is a pass.
        if ending == "contact" and not rim_near:
            self.rejections["receiver_before_rim"] += 1
            return None
        stable = bool(ref_inverse and all(p["sample"].court for p in track))
        probable = (
            arc
            and rim_near
            and stable
            and support >= 2
            and error <= RULES["probable_fit_rmse_body_heights"]
        )
        shooter = {
            k: contact[k] if contact else None for k in ("track_id", "team_id", "subject_id")
        }
        if shooter["team_id"] not in {t["team_id"] for t in self.roster.get("teams", [])}:
            shooter["team_id"] = None
            shooter["subject_id"] = None
        recent = []
        for j in range(index, -1, -1):
            sample = self.samples[j]
            if (
                seed.time - sample.time > 0.25
                or j < index
                and self.boundary(sample, self.samples[j + 1])
            ):
                break
            if sample.contact and sample.contact["track_id"] == shooter["track_id"]:
                recent.append(sample.contact)
        if len({item["subject_id"] for item in recent if item["subject_id"]}) > 1:
            shooter["subject_id"] = None
        if len({item["team_id"] for item in recent if item["team_id"]}) > 1:
            shooter["team_id"] = None
            shooter["subject_id"] = None
        valid_subjects = {
            f"roster:{team['team_id']}:{p['number']}"
            for team in self.roster.get("teams", [])
            for p in team.get("players", [])
        }
        if shooter["subject_id"] not in valid_subjects:
            shooter["subject_id"] = None
        event = {
            "type": "shot_attempt",
            "status": "probable" if probable else "candidate",
            "time_s": seed.time,
            "start_s": max(0, seed.time - 0.6),
            "end_s": times[-1],
            "release_window_s": self.release_window(index, contact is not None),
            "release_observed": contact is not None,
            "frame_index": seed.frame,
            "end_frame_index": track[-1]["sample"].frame,
            "segment_id": seed.segment,
            "shooter": shooter,
            "outcome": "unknown",
            "evidence": {
                "release_observations": support,
                "free_flight_share": free,
                "flight_s": duration,
                "rise_body_heights": rise,
                "fall_body_heights": fall,
                "arc": arc,
                "fit_rmse_body_heights": error,
                "curvature_body_heights_s2": 2 * a,
                "rim_approach": rim_near,
                "rim_distance_body_heights": near_distance if near else None,
                "camera_compensated": stable,
                "ending": ending,
                "observations": len(track),
            },
            "trajectory": [
                {
                    "frame_index": p["sample"].frame,
                    "time_s": p["sample"].time,
                    "position_px": list(p["sample"].ball),
                }
                for p in track
            ],
            "quality": int(probable) * 10 + int(arc) * 2 + int(rim_near) + free - error,
        }
        return event

    def release_window(self, index, observed):
        seed = self.samples[index]
        first = seed.time
        for j in range(index - 1, -1, -1):
            previous = self.samples[j]
            if seed.time - previous.time > (0.15 if observed else 0.3) or self.boundary(
                previous, self.samples[j + 1]
            ):
                break
            first = previous.time
        return [first, self.samples[index + 1].time if observed else seed.time]

    def metadata(self):
        available = self.valid_ball >= 6 and (self.pose_frames >= 2 or self.fallback_frames >= 6)
        return {
            "version": VERSION,
            "rules": RULES.copy(),
            "available": available,
            "reason": None if available else "ball_or_context_unavailable",
            "events": self.events,
            "probable_count": sum(e["status"] == "probable" for e in self.events),
            "candidate_count": sum(e["status"] == "candidate" for e in self.events),
            "observed_ball_frames": self.valid_ball,
            "pose_frames": self.pose_frames,
            "fallback_context_frames": self.fallback_frames,
            "offline_confirmation": True,
            "outcomes_available": False,
            "automatic_replay_exclusion": False,
        }
