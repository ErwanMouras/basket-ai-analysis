"""Time-weighted, camera-plan-local spatial summaries of completed observations.

No ball projection, cross-cut interpolation or synthetic player locations.
Attack/defense labels are conservative estimates from the separate phase index. The heatmaps are distributions of the data we observed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

from viewer.phases import PhaseIndex, team_of


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def player_subject(person):
    if person.get("jersey_number_suppressed"):
        return None
    resolved = person.get("resolved_identity") or {}
    if resolved.get("team_id") and resolved.get("number") is not None:
        return f"roster:{resolved['team_id']}:{resolved['number']}"
    if "majority_jersey_number" in person:
        return None
    jersey = person.get("jersey") or {}
    if jersey.get("identity_status") == "unique_number" and jersey.get("status") == "confirmed":
        return f"roster:{jersey.get('team_id')}:{jersey.get('number')}"
    return None


def union_duration(spans):
    """Wall-clock seconds, distinct from the sum of player-seconds."""
    total = 0.0
    stop = -math.inf
    for start, end in sorted(spans):
        total += max(0.0, end - max(start, stop))
        stop = max(stop, end)
    return total


@dataclass(slots=True)
class Interval:
    plan: int
    start: float
    end: float
    team: str
    subject: str | None
    first: tuple | None
    last: tuple | None
    distance: float | None = None
    owner: str | None = None
    direction: int | None = None
    episode: int = -1


class HeatmapIndex:
    """Compact intervals built alongside the viewer's existing observation index."""

    def __init__(self, manifest, roster, scoreboard=None):
        raw = manifest.get("court_coordinate_system") or {}
        self.geometry = (
            raw
            if (
                raw.get("units") == "metres"
                and all(
                    finite(raw.get(key)) and 1 <= raw[key] <= 100 for key in ("length", "width")
                )
            )
            else None
        )
        self.subjects = {}
        self.team_ids = set()
        for team in (roster or {}).get("teams", []):
            team_id = team["team_id"]
            self.team_ids.add(team_id)
            self.subjects[f"team:{team_id}"] = {
                "id": f"team:{team_id}",
                "kind": "team",
                "team_id": team_id,
                "label": team["name"],
            }
            for player in team.get("players", []):
                key = f"roster:{team_id}:{player['number']}"
                self.subjects[key] = {
                    "id": key,
                    "kind": "player",
                    "team_id": team_id,
                    "label": player["name"],
                    "number": str(player["number"]),
                }
        config = (manifest.get("config") or {}).get("distance") or {}
        gap, speed = config.get("max_gap_s"), config.get("max_speed_m_s")
        self.max_gap = gap if finite(gap) and gap > 0 else 0.2
        self.max_speed = speed if finite(speed) and speed > 0 else 12.0
        self.phases = PhaseIndex(
            tuple(team["team_id"] for team in (roster or {}).get("teams", [])),
            self.geometry,
            self.max_gap,
            scoreboard=scoreboard,
        )
        self.scoreboard = scoreboard
        self.plans = []
        self.intervals = []
        self.previous = {}
        self.previous_row = None
        self.track_intervals = {}

    def point(self, person, row):
        point = person.get("position_m")
        if (
            not self.geometry
            or (row.get("court") or {}).get("status") not in ("fit", "propagated")
            or (row.get("court") or {}).get("image_to_court") is None
            or not isinstance(point, (list, tuple))
            or len(point) != 2
            or not all(finite(value) for value in point)
        ):
            return None
        if not (
            0 <= point[0] <= self.geometry["length"] and 0 <= point[1] <= self.geometry["width"]
        ):
            return None
        return tuple(point)

    def add_frame(self, row):
        now, segment = row["timestamp_s"], row.get("segment_id", 0)
        if not finite(now) or now < 0:
            self.previous = {}
            self.previous_row = None
            return
        previous = self.previous_row
        cut = previous is None or row.get("scene_cut") or segment != previous.get("segment_id", 0)
        if cut:
            self.plans.append(
                {"id": len(self.plans), "segment_id": segment, "start_s": now, "end_s": now}
            )
        plan = self.plans[-1]
        plan["end_s"] = now
        self.phases.add_frame(row, plan["id"])
        people = [
            person
            for person in row.get("persons", [])
            if person.get("role") == "player" and person.get("track_id") is not None
        ]
        subjects = Counter(player_subject(person) for person in people)
        tracks = Counter(person["track_id"] for person in people)
        current = {}
        for person in people:
            subject = player_subject(person)
            # Ambiguous duplicates must never double the time or create a false named map.
            if tracks[person["track_id"]] > 1 or (subject and subjects[subject] > 1):
                continue
            team = team_of(person)
            if team not in self.team_ids:
                continue
            if subject not in self.subjects:
                subject = None
            item = {
                "subject": subject,
                "team": team,
                "point": self.point(person, row),
                "method": person.get("ground_point_method"),
            }
            current[person["track_id"]] = item
            before = self.previous.get(person["track_id"])
            if (
                cut
                or previous is None
                or not before
                or not row.get("included", True)
                or not previous.get("included", True)
                or row["frame_index"] != previous["frame_index"] + 1
                or not 0 < now - previous["timestamp_s"] <= self.max_gap + 1e-9
                or (item["subject"], team) != (before["subject"], before["team"])
            ):
                continue
            start = previous["timestamp_s"]
            first, last = before["point"], item["point"]
            if before["method"] != item["method"] or (
                first and last and math.dist(first, last) / (now - start) > self.max_speed
            ):
                first = last = None
            interval = Interval(plan["id"], start, now, team, subject, first, last)
            self.intervals.append(interval)
            self.track_intervals[(segment, person["track_id"], row["frame_index"])] = interval
        self.previous, self.previous_row = current, {
            "frame_index": row["frame_index"],
            "timestamp_s": now,
            "segment_id": segment,
            "included": row.get("included", True),
        }

    def add_track(self, row):
        interval = self.track_intervals.get(
            (row.get("segment_id"), row.get("track_id"), row.get("frame_index"))
        )
        distance = row.get("distance_m")
        if (
            not interval
            or not interval.first
            or not interval.last
            or not finite(distance)
            or distance < 0
            or row.get("reason") is not None
            or not finite(row.get("interval_start_s"))
            or abs(row["interval_start_s"] - interval.start) > 1e-6
            or not finite(row.get("measured_duration_s"))
            or abs(row["measured_duration_s"] - (interval.end - interval.start)) > 1e-6
            or (interval.subject and row.get("subject_id") != interval.subject)
        ):
            return
        interval.distance = float(distance)

    def finish(self):
        self.phases.finish()
        spans = {(span.plan, span.start, span.end): span for span in self.phases.spans}
        for interval in self.intervals:
            span = spans.get((interval.plan, interval.start, interval.end))
            if span:
                interval.owner, interval.direction, interval.episode = (
                    span.owner,
                    span.direction,
                    span.episode,
                )
        # No need to retain the join table or previous-frame data after indexing.
        self.track_intervals.clear()
        self.previous.clear()
        self.previous_row = None

    def metadata(self):
        return {
            "available": bool(self.geometry and self.subjects),
            "reason": "geometry" if not self.geometry else "roster" if not self.subjects else None,
            "court": self.geometry,
            "subjects": list(self.subjects.values()),
            "plans": self.plans,
            "bounds": (
                {"start_s": self.plans[0]["start_s"], "end_s": self.plans[-1]["end_s"]}
                if self.plans
                else None
            ),
            "phase_analysis": self.phases.metadata(),
            "scoreboard": (
                self.scoreboard.metadata()
                if self.scoreboard
                else {"available": False, "periods": []}
            ),
        }

    def phase_summary(self, team, start, end, episode=None):
        durations = {
            "offense_s": 0.0,
            "defense_s": 0.0,
            "unknown_s": 0.0,
            "orientation_missing_s": 0.0,
        }
        for span in self.phases.spans:
            if episode is not None and span.owner and span.episode != episode:
                continue
            seconds = max(0.0, min(end, span.end) - max(start, span.start))
            role = "unknown" if not span.owner else "offense" if span.owner == team else "defense"
            durations[role + "_s"] += seconds
            if span.owner and not span.direction:
                durations["orientation_missing_s"] += seconds
        durations["unknown_s"] = max(
            0.0, end - start - durations["offense_s"] - durations["defense_s"]
        )
        return durations

    def summarize(
        self, subject_id, plan_id, start, end, metric="presence", phase=None, episode=None
    ):
        subject = self.subjects[subject_id]
        length, width = self.geometry["length"], self.geometry["width"]
        nx, ny = math.ceil(length), math.ceil(width)
        grid, zones = [0.0] * (nx * ny), [0.0] * 9
        observed = localized = measured = 0.0
        spans = []
        for interval in self.intervals:
            if (
                (phase is None and interval.plan != plan_id)
                or (episode is not None and interval.episode != episode)
                or (phase == "offense" and interval.owner != subject["team_id"])
                or (
                    phase == "defense"
                    and (not interval.owner or interval.owner == subject["team_id"])
                )
                or (phase == "unknown" and interval.owner is not None)
                or (subject["kind"] == "team" and interval.team != subject["team_id"])
                or (subject["kind"] == "player" and interval.subject != subject_id)
            ):
                continue
            a, b = max(start, interval.start), min(end, interval.end)
            if b <= a:
                continue
            seconds = b - a
            observed += seconds
            if (
                not interval.first
                or not interval.last
                or (phase is not None and interval.direction is None)
            ):
                continue
            localized += seconds
            spans.append((a, b))
            if metric == "movement" and interval.distance is None:
                continue
            measured += seconds
            weight = (
                seconds
                if metric == "presence"
                else interval.distance * seconds / (interval.end - interval.start)
            )
            if not weight:
                continue
            lo = (a - interval.start) / (interval.end - interval.start)
            hi = (b - interval.start) / (interval.end - interval.start)

            def point(fraction):
                xy = tuple(x + fraction * (y - x) for x, y in zip(interval.first, interval.last))
                return (
                    (length - xy[0], width - xy[1])
                    if phase is not None and interval.direction == -1
                    else xy
                )

            first, last = point(lo), point(hi)
            # Presence is time, not FPS. Distribute moving intervals along their short path;
            # every contribution is clipped and conserves its total mass at cell borders.
            count = max(1, math.ceil(math.dist(first, last) / 0.20))
            for index in range(count):
                fraction = (index + 0.5) / count
                px, py = (x + fraction * (y - x) for x, y in zip(first, last))
                gx, gy = min(nx - 1, int(px / length * nx)), min(ny - 1, int(py / width * ny))
                zx, zy = min(2, int(px / length * 3)), min(2, int(py / width * 3))
                grid[gy * nx + gx] += weight / count
                zones[zy * 3 + zx] += weight / count
        mass = sum(grid)
        return {
            "subject": subject,
            "start_s": start,
            "end_s": end,
            "nx": nx,
            "ny": ny,
            "grid": grid,
            "zones": zones,
            "total": mass,
            "unit": (
                "m" if metric == "movement" else "player_s" if subject["kind"] == "team" else "s"
            ),
            "observed_s": observed,
            "localized_s": localized,
            "measured_s": measured,
            "localized_video_s": union_duration(spans),
            "coverage": localized / observed if observed else None,
            "movement_coverage": measured / localized if localized else None,
            "phase": phase,
            "phase_durations": (
                self.phase_summary(subject["team_id"], start, end, episode) if phase else None
            ),
        }

    def query(
        self,
        subject,
        plan=0,
        start=None,
        end=None,
        mode="presence",
        compare=None,
        phase=None,
        episode=None,
    ):
        if not self.geometry or not self.subjects:
            raise ValueError("Heatmaps unavailable without court dimensions and a run roster")
        if subject not in self.subjects or mode not in (
            "presence",
            "movement",
            "zones",
            "comparison",
            "evolution",
        ):
            raise ValueError("Invalid heatmap subject or mode")
        if phase is not None:
            if phase not in ("offense", "defense", "unknown") or not self.plans:
                raise ValueError("Invalid game phase")
            bounds = {"start_s": self.plans[0]["start_s"], "end_s": self.plans[-1]["end_s"]}
            if episode is not None:
                if type(episode) is not int or not 0 <= episode < len(self.phases.episodes):
                    raise ValueError("Invalid game passage")
                bounds = self.phases.episodes[episode]
        else:
            if type(plan) is not int or not 0 <= plan < len(self.plans):
                raise ValueError("Invalid camera plan")
            bounds = self.plans[plan]
        start = bounds["start_s"] if start is None else start
        end = bounds["end_s"] if end is None else end
        if (
            not finite(start)
            or not finite(end)
            or start < bounds["start_s"] - 1e-6
            or end > bounds["end_s"] + 1e-6
            or end <= start
        ):
            raise ValueError("Heatmap interval must stay inside the selected scope")
        if mode == "comparison" and (
            compare not in self.subjects
            or compare == subject
            or self.subjects[compare]["kind"] != self.subjects[subject]["kind"]
        ):
            raise ValueError("Compare two distinct players or two distinct teams")
        metric = "movement" if mode == "movement" else "presence"
        if mode == "evolution":
            middle = (start + end) / 2
            maps = [
                self.summarize(subject, plan, start, middle, phase=phase, episode=episode),
                self.summarize(subject, plan, middle, end, phase=phase, episode=episode),
            ]
        else:
            maps = [self.summarize(subject, plan, start, end, metric, phase, episode)]
            if mode == "comparison":
                maps.append(self.summarize(compare, plan, start, end, phase=phase, episode=episode))
        return {
            "court": self.geometry,
            "plan": bounds if phase is None else None,
            "scope": {
                "kind": "phase" if phase else "plan",
                "phase": phase,
                "episode": episode,
                "start_s": start,
                "end_s": end,
                "normalized_basket": "right" if phase in ("offense", "defense") else None,
            },
            "phase_analysis": (
                self.phase_summary(self.subjects[subject]["team_id"], start, end, episode)
                if phase
                else None
            ),
            "mode": mode,
            "maps": maps,
            "normalization": "share_of_observed_mass",
            "grid_target_m": 1.0,
            "automatic_replay_exclusion": False,
        }
