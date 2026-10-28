"""Learn basket directions, then retain only installed attack/defense phases.

Possession and installed play are separate: a confirmed turnover enters a neutral
transition until the new controller reaches its attacking half. Neutral and
unobserved intervals remain internal and never contribute to the heatmaps.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from statistics import median

from viewer.settled import classify_settled


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def point(value):
    return isinstance(value, (list, tuple)) and len(value) == 2 and all(finite(x) for x in value)


def team_of(person):
    if (person.get("jersey") or {}).get("team_group_conflict"):
        return None
    resolved = person.get("resolved_identity") or {}
    if resolved.get("team_id"):
        return resolved["team_id"]
    if "resolved_team_id" in person:
        return person["resolved_team_id"]
    if "majority_jersey_number" in person:
        return None
    jersey = person.get("jersey") or {}
    return jersey.get("team_id") if jersey.get("status") == "confirmed" else None


@dataclass(slots=True)
class Contact:
    team: str | None = None
    method: str = "none"
    reason: str = "no_contact"
    holder_x: float | None = None
    direction: int | None = None


def contact(row, teams, geometry=None, method="hands", min_ball_confidence=0.35):
    """Normalized image-space proximity, with competing unknown/referee contacts."""
    ball = row.get("ball") or {}
    position = ball.get("position_px")
    if (
        not row.get("included", True)
        or ball.get("status") != "ok"
        or not point(position)
        or not finite(ball.get("confidence"))
        or ball["confidence"] < min_ball_confidence
    ):
        return Contact(reason="ball_unavailable")
    if any(
        finite(row.get(size)) and not 0 <= value <= row[size]
        for value, size in zip(position, ("width", "height"))
    ):
        return Contact(reason="ball_outside_image")
    candidates = []
    tracks = Counter(
        p.get("track_id") for p in row.get("persons", []) if p.get("track_id") is not None
    )
    for person in row.get("persons", []):
        box = person.get("bbox")
        if (
            not isinstance(box, (list, tuple))
            or len(box) != 4
            or not all(finite(v) for v in box)
            or box[3] <= box[1]
            or box[2] <= box[0]
        ):
            continue
        height = box[3] - box[1]
        if not (
            box[0] - 0.35 * height <= position[0] <= box[2] + 0.35 * height
            and box[1] - 0.25 * height <= position[1] <= box[3] + 0.1 * height
        ):
            continue
        pose = person.get("pose") or {}
        wrists = []
        if (
            method == "hands"
            and pose.get("format") == "coco17"
            and pose.get("coordinate_space") == "source"
        ):
            for index in (9, 10):
                keys, scores, valid = (
                    pose.get("keypoints") or [],
                    pose.get("scores") or [],
                    pose.get("valid"),
                )
                if (
                    len(keys) > index
                    and point(keys[index])
                    and len(scores) > index
                    and finite(scores[index])
                    and scores[index] >= 0.45
                    and (valid is None or len(valid) > index and valid[index])
                ):
                    wrists.append(math.dist(position, keys[index]) / height)
        if wrists:
            distance, evidence, threshold = min(wrists), "hands", 0.22
        else:
            distance = math.dist(position, ((box[0] + box[2]) / 2, box[1] + 0.5 * height)) / height
            evidence, threshold = "body", 0.28
        candidates.append(
            (
                distance + (0 if evidence == "hands" else 0.08),
                distance <= threshold,
                person,
                evidence,
            )
        )
    if not candidates:
        return Contact()
    candidates.sort(key=lambda value: value[0])
    score, accepted, holder, evidence = candidates[0]
    team = team_of(holder)
    if not accepted:
        return Contact()
    if holder.get("role") != "player":
        return Contact(reason="referee_contact")
    if team not in teams or holder.get("track_id") is None or tracks[holder["track_id"]] != 1:
        return Contact(reason="unassigned_contact")
    competitor = min(
        (
            other[0]
            for other in candidates[1:]
            if team_of(other[2]) != team or other[2].get("role") != "player"
        ),
        default=math.inf,
    )
    if competitor - score < (0.055 if evidence == "hands" else 0.10):
        return Contact(reason="contested_contact")
    result = Contact(team=team, method=evidence, reason="direct_contact")
    court = row.get("court") or {}
    if (
        not geometry
        or court.get("status") not in ("fit", "propagated")
        or court.get("image_to_court") is None
    ):
        return result
    length, width = geometry["length"], geometry["width"]

    def valid_position(person):
        xy = person.get("position_m")
        return xy if point(xy) and 0 <= xy[0] <= length and 0 <= xy[1] <= width else None

    xy = valid_position(holder)
    if xy:
        result.holder_x = xy[0]
    opponents = [
        valid_position(p)
        for p in row.get("persons", [])
        if p.get("role") == "player" and team_of(p) in teams and team_of(p) != team
    ]
    opponents = [p[0] for p in opponents if p]
    # Near-baseline contacts can be either finishing at the opposing rim or
    # inbounding beneath one's own rim. They never establish direction alone.
    if xy and 0.20 * length <= xy[0] <= 0.80 * length and len(opponents) >= 2:
        defending_x = median(opponents)
        delta = defending_x - xy[0]
        if (defending_x < 0.38 * length and delta <= -1.5) or (
            defending_x > 0.62 * length and delta >= 1.5
        ):
            result.direction = 1 if delta > 0 else -1
    return result


def occupied_half(row, teams, geometry):
    """A settled, calibrated group in an outer third; never project the ball.

    Both teams must be represented. Requiring their medians in the same outer
    third avoids treating a team crossing midcourt as a settled attack.
    """
    court = row.get("court") or {}
    if (
        not geometry
        or len(teams) != 2
        or not row.get("included", True)
        or court.get("status") not in ("fit", "propagated")
        or court.get("image_to_court") is None
    ):
        return None
    people = [p for p in row.get("persons", []) if p.get("role") == "player"]
    tracks = Counter(p.get("track_id") for p in people)
    xs = {team: [] for team in teams}
    all_x = []
    length, width = geometry["length"], geometry["width"]
    for person in people:
        xy, track = person.get("position_m"), person.get("track_id")
        if (
            track is None
            or tracks[track] != 1
            or not point(xy)
            or not 0 <= xy[0] <= length
            or not 0 <= xy[1] <= width
        ):
            continue
        x = xy[0] / length
        all_x.append(x)
        team = team_of(person)
        if team in xs:
            xs[team].append(x)
    if any(len(values) < 2 for values in xs.values()):
        return None
    centres = [median(values) for values in xs.values()]
    if all(x <= 0.35 for x in centres) and sum(x < 0.45 for x in all_x) / len(all_x) >= 0.8:
        return -1
    if all(x >= 0.65 for x in centres) and sum(x > 0.55 for x in all_x) / len(all_x) >= 0.8:
        return 1
    return None


@dataclass(slots=True)
class Evidence:
    frame: int
    time: float
    plan: int
    included: bool
    contact: Contact
    period: int | None = None
    half: int | None = None
    court_visible: bool = False


@dataclass(slots=True)
class Span:
    start: float
    end: float
    plan: int
    owner: str | None
    method: str
    reason: str
    direction: int | None = None
    episode: int = -1
    period: int | None = None


class PhaseIndex:
    window_s = 0.4
    min_evidence_s = 0.16
    quarter_anchor_s = 0.75
    reversed_view_agreement = 0.9

    def __init__(
        self,
        teams,
        geometry,
        max_gap=0.2,
        method="hands",
        min_ball_confidence=0.35,
        scoreboard=None,
        use_court_context=True,
    ):
        self.teams = tuple(teams)
        self.geometry = geometry
        self.max_gap = max_gap
        self.method = method
        self.min_ball_confidence = min_ball_confidence
        self.frames = []
        self.spans = []
        self.episodes = []
        self.directions = {}
        self.uncertain_intervals = []
        self.duration_s = 0.0
        self.orientation_evidence = []
        self.scoreboard = scoreboard
        self.period_directions = []
        self.use_court_context = use_court_context
        events = scoreboard.events if scoreboard else []
        self.score_changes = defaultdict(list)
        for previous, current in zip(events, events[1:]):
            if previous["period"] == current["period"] and previous["scores"] != current["scores"]:
                self.score_changes[current["period"]].append(current["time_s"])

    def add_frame(self, row, plan):
        reading = self.scoreboard.at(row["timestamp_s"]) if self.scoreboard else None
        self.frames.append(
            Evidence(
                row["frame_index"],
                row["timestamp_s"],
                plan,
                row.get("included", True),
                contact(row, self.teams, self.geometry, self.method, self.min_ball_confidence),
                reading.get("period") if reading else None,
                occupied_half(row, self.teams, self.geometry) if self.use_court_context else None,
                (row.get("court") or {}).get("status") in ("fit", "propagated")
                and (row.get("court") or {}).get("image_to_court") is not None,
            )
        )

    def finish(self):
        times = [frame.time for frame in self.frames if finite(frame.time)]
        self.duration_s = max(times) - min(times) if times else 0.0
        blocks, current = [], []
        for frame in self.frames:
            previous = current[-1] if current else None
            if (
                not frame.included
                or not finite(frame.time)
                or previous
                and (
                    frame.plan != previous.plan
                    or frame.period != previous.period
                    or frame.frame != previous.frame + 1
                    or not 0 < frame.time - previous.time <= self.max_gap + 1e-9
                )
            ):
                if current:
                    blocks.append(current)
                current = []
            if frame.included and finite(frame.time):
                current.append(frame)
        if current:
            blocks.append(current)
        direction_votes = defaultdict(Counter)
        for block in blocks:
            pairs = [(a.time, b.time, b) for a, b in zip(block, block[1:])]
            ends = [item[1] for item in pairs]
            starts = [item[0] for item in pairs]
            for start, end, frame in pairs:
                middle = (start + end) / 2
                lo, hi = middle - self.window_s, middle + self.window_s
                votes, support_frames = Counter(), Counter()
                for a, b, other in pairs[bisect_right(ends, lo) : bisect_left(starts, hi)]:
                    if other.contact.team in self.teams:
                        weight = max(0.0, min(hi, b) - max(lo, a))
                        votes[other.contact.team] += weight
                        support_frames[other.contact.team] += int(weight > 0)
                owner = None
                reason = frame.contact.reason
                if votes and len(self.teams) == 2:
                    winner, weight = votes.most_common(1)[0]
                    hard_unknown = frame.contact.reason in (
                        "referee_contact",
                        "unassigned_contact",
                        "contested_contact",
                    )
                    if (
                        not hard_unknown
                        and weight >= self.min_evidence_s - 1e-9
                        and support_frames[winner] >= 2
                        and weight / sum(votes.values()) >= 0.8
                        and frame.contact.team in (None, winner)
                    ):
                        owner = winner
                        reason = (
                            "repeated_contact"
                            if frame.contact.team == winner
                            else "short_continuity"
                        )
                span = Span(
                    start, end, frame.plan, owner, frame.contact.method, reason, period=frame.period
                )
                self.spans.append(span)
                if owner and frame.contact.team == owner and frame.contact.direction:
                    base_direction = (
                        frame.contact.direction
                        if owner == self.teams[0]
                        else -frame.contact.direction
                    )
                    direction_votes[(frame.period, frame.plan)][base_direction] += end - start
        local_directions = {}
        for key, votes in direction_votes.items():
            direction, weight = votes.most_common(1)[0]
            if sum(votes.values()) >= 0.25 and weight / sum(votes.values()) >= 0.8:
                local_directions[key] = direction
                self.directions[key[1]] = direction
        # A quarter keeps its first well-supported reference view. A strongly
        # reversed camera still uses its own orientation. Only contact evidence
        # establishes that reference: spatial completion cannot vote for itself.
        locked = {}
        for (period, plan), votes in direction_votes.items():
            if (
                period is not None
                and period not in locked
                and sum(votes.values()) >= self.quarter_anchor_s
                and (period, plan) in local_directions
            ):
                locked[period] = local_directions[(period, plan)]
                self.period_directions.append(
                    {
                        "period": period,
                        "reference_team_id": self.teams[0],
                        "direction": locked[period],
                        "anchor_plan_id": plan,
                        "contact_s": sum(votes.values()),
                        "vote_agreement": max(votes.values()) / sum(votes.values()),
                    }
                )
        effective_directions = dict(local_directions)
        sources = {}
        for span in self.spans:
            key = (span.period, span.plan)
            if span.period not in locked:
                sources[key] = "camera_plan"
                continue
            votes = direction_votes[key]
            opposite = -locked[span.period]
            total = sum(votes.values())
            if (
                total >= self.quarter_anchor_s
                and votes[opposite] / total >= self.reversed_view_agreement
            ):
                effective_directions[key] = opposite
                sources[key] = "reversed_view"
            elif total >= self.quarter_anchor_s and votes[locked[span.period]] / total < 0.8:
                effective_directions.pop(key, None)
                sources[key] = "conflicting_view"
            else:
                effective_directions[key] = locked[span.period]
                sources[key] = "quarter_lock"
        for period, plan in dict.fromkeys([*direction_votes, *effective_directions]):
            votes = direction_votes[(period, plan)]
            if not votes and (period, plan) not in effective_directions:
                continue
            self.orientation_evidence.append(
                {
                    "plan_id": plan,
                    "period": period,
                    "reference_team_id": self.teams[0],
                    "left_contact_s": votes[-1],
                    "right_contact_s": votes[1],
                    "vote_agreement": max(votes.values()) / sum(votes.values()) if votes else None,
                    "direction": effective_directions.get((period, plan)),
                    "source": sources.get((period, plan), "camera_plan"),
                }
            )
        if self.use_court_context:
            classify_settled(
                blocks,
                self.spans,
                self.teams,
                self.geometry,
                effective_directions,
                self.score_changes,
            )
        for span in self.spans:
            direction = effective_directions.get((span.period, span.plan))
            if direction and span.owner:
                span.direction = direction if span.owner == self.teams[0] else -direction
            seconds = span.end - span.start
            if not span.owner:
                previous = self.uncertain_intervals[-1] if self.uncertain_intervals else None
                if (
                    previous
                    and previous["plan_id"] == span.plan
                    and previous["period"] == span.period
                    and abs(previous["end_s"] - span.start) < 1e-6
                ):
                    previous["end_s"] = span.end
                    previous["duration_s"] += seconds
                else:
                    self.uncertain_intervals.append(
                        {
                            "plan_id": span.plan,
                            "period": span.period,
                            "start_s": span.start,
                            "end_s": span.end,
                            "duration_s": seconds,
                        }
                    )
                continue
            previous = self.episodes[-1] if self.episodes else None
            # Installed-play passages are contiguous. The legacy contact-only
            # experiment can still group short gaps for historical comparison.
            if (
                previous
                and previous["plan_id"] == span.plan
                and previous["period"] == span.period
                and previous["attacking_team_id"] == span.owner
                and (
                    abs(span.start - previous["end_s"]) < 1e-6
                    if self.use_court_context
                    else span.start - previous["end_s"] <= 0.6 + 1e-9
                )
            ):
                previous["end_s"] = span.end
                previous["duration_s"] = span.end - previous["start_s"]
                previous["classified_s"] += seconds
            else:
                self.episodes.append(
                    {
                        "id": len(self.episodes),
                        "plan_id": span.plan,
                        "period": span.period,
                        "start_s": span.start,
                        "end_s": span.end,
                        "duration_s": seconds,
                        "classified_s": seconds,
                        "attacking_team_id": span.owner,
                        "direction": span.direction,
                        "orientation_ready": span.direction is not None,
                        "orientation_source": sources.get((span.period, span.plan), "camera_plan"),
                    }
                )
            span.episode = self.episodes[-1]["id"]
        self.frames.clear()

    def metadata(self):
        eligible = self.duration_s
        classified = sum(s.end - s.start for s in self.spans if s.owner)
        oriented = sum(s.end - s.start for s in self.spans if s.owner and s.direction)
        unknown_reasons = Counter()
        for span in self.spans:
            if not span.owner:
                unknown_reasons[span.reason] += span.end - span.start
        unknown_reasons['cuts_and_gaps'] = max(0.0, eligible - sum(s.end - s.start for s in self.spans))
        return {
            "experimental": True,
            "orientation_evidence": self.orientation_evidence,
            "period_directions": self.period_directions,
            "orientation_policy": "quarter_lock_with_verified_camera_reversals",
            "method": (
                "settled_halfcourt_state_machine_v3"
                if self.use_court_context
                else "image_contacts_temporal_layout_v1"
            ),
            "classification_sources": {
                "contacts_s": sum(
                    s.end - s.start
                    for s in self.spans
                    if s.owner and s.method not in ("court", "settled")
                ),
                "settled_s": sum(
                    s.end - s.start for s in self.spans if s.owner and s.method == "settled"
                ),
                "court_context_s": sum(
                    s.end - s.start for s in self.spans if s.owner and s.method == "court"
                ),
            },
            "phase_policy": "installed_attack_neutral_installed_opponent",
            "neutral_s": sum(s.end - s.start for s in self.spans if s.method == "neutral"),
            "eligible_s": eligible,
            "classified_s": classified,
            "unknown_s": eligible - classified,
            "unknown_reasons_s": dict(unknown_reasons),
            "oriented_s": oriented,
            "episodes": self.episodes,
            "uncertain_intervals": self.uncertain_intervals,
            "classification_coverage": classified / eligible if eligible else None,
            "automatic_replay_exclusion": False,
        }

    def timeline(self):
        """Compact half-open intervals, preserving cuts and missing source time."""
        rows = []
        for span in self.spans:
            if not span.owner:
                continue
            source = (
                "settled"
                if span.method == "settled"
                else (
                    "court_context"
                    if span.method == "court"
                    else "contacts" if span.owner else None
                )
            )
            context = {
                "plan_id": span.plan,
                "period": span.period,
                "attacking_team_id": span.owner,
                "direction": span.direction,
                "source": source,
            }
            previous = rows[-1] if rows else None
            if (
                previous
                and all(previous[key] == value for key, value in context.items())
                and abs(previous["end_s"] - span.start) < 1e-6
            ):
                previous["end_s"] = span.end
            else:
                rows.append({"start_s": span.start, "end_s": span.end, **context})
        return rows
