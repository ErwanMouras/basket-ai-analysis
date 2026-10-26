"""Validated, source-bound broadcast readings. No OCR dependency in the viewer."""

from bisect import bisect_right
import json
import math
from pathlib import Path


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def reading_valid(row, team_ids):
    period = row.get("period")
    clock = row.get("clock_s")
    scores = row.get("scores")
    return (
        isinstance(period, int)
        and not isinstance(period, bool)
        and 1 <= period <= 20
        and (clock is None or number(clock) and 0 <= clock <= (720 if period <= 4 else 300))
        and (
            scores is None
            or isinstance(scores, dict)
            and set(scores) == set(team_ids)
            and all(
                isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 250
                for v in scores.values()
            )
        )
    )


def confirm_readings(raw, team_ids, sample_s=0.5):
    """Two adjacent reads confirm period/score; clock stays an observed value.

    Never infer scoring events, interpolate game time, or bridge missing graphics.
    A stable score decrease/period rewind is a replay or correction candidate;
    without further evidence it stays unavailable rather than rewriting history.
    """
    valid = [r if reading_valid(r, team_ids) else None for r in raw]
    output, previous_period, previous_scores = [], None, None
    for i, row in enumerate(raw):
        item = {"time_s": row["time_s"], "period": None, "clock_s": None, "scores": None}
        candidate = valid[i]
        neighbors = [
            valid[j]
            for j in (i - 1, i + 1)
            if 0 <= j < len(raw) and abs(raw[j]["time_s"] - row["time_s"]) <= sample_s * 1.51
        ]
        if candidate and any(n and n["period"] == candidate["period"] for n in neighbors):
            period = candidate["period"]
            # Periods are independent of clock availability. An edited clip
            # can skip game time; source elapsed time never fills that gap.
            if previous_period is None or period >= previous_period:
                item["period"] = period
                clock = candidate.get("clock_s")
                if number(clock) and any(
                    n
                    and n["period"] == period
                    and number(n.get("clock_s"))
                    and abs(n["clock_s"] - clock) <= sample_s + 1.1
                    for n in neighbors
                ):
                    item["clock_s"] = clock
                scores = candidate.get("scores")
                if scores and any(
                    n and n["period"] == period and n.get("scores") == scores for n in neighbors
                ):
                    if previous_scores is None or all(
                        scores[t] >= previous_scores[t] for t in team_ids
                    ):
                        item["scores"] = scores
                        previous_scores = scores
                previous_period = period
        output.append(item)
    return output


class ScoreboardIndex:
    def __init__(self, document=None, source_sha=None, team_ids=()):
        self.document = None
        self.reason = "missing"
        self.samples, self.times, self.events = [], [], []
        self.periods = []
        if document is None:
            return
        try:
            if not isinstance(document, dict):
                raise ValueError("invalid_document")
            if (
                document.get("schema_version") != 1
                or not source_sha
                or document.get("source_sha256") != source_sha
            ):
                raise ValueError("source_mismatch")
            teams = document["teams"]
            ids = [t["team_id"] for t in teams]
            if len(ids) != 2 or len(set(ids)) != 2 or set(ids) != set(team_ids):
                raise ValueError("team_mismatch")
            if not all(
                isinstance(t.get("label"), str) and 1 <= len(t["label"]) <= 12 for t in teams
            ):
                raise ValueError("invalid_teams")
            sample_s = document["sample_s"]
            if not number(sample_s) or not 0.1 <= sample_s <= 2:
                raise ValueError("invalid_sampling")
            previous = -1
            for row in document["samples"]:
                time = row["time_s"]
                if not number(time) or time < 0 or time <= previous:
                    raise ValueError("invalid_times")
                if row.get("period") is not None and not reading_valid(row, ids):
                    raise ValueError("invalid_reading")
                if row.get("period") is None and (
                    row.get("scores") is not None or row.get("clock_s") is not None
                ):
                    raise ValueError("invalid_reading")
                self.samples.append(
                    {k: row.get(k) for k in ("time_s", "period", "clock_s", "scores")}
                )
                previous = time
            self.times = [r["time_s"] for r in self.samples]
            self.hold_s = min(2.5, sample_s * 1.51)
            self.document = {
                "teams": teams,
                "provenance": document.get("provenance", {}),
                "sample_s": sample_s,
            }
            self.reason = None
            previous_score = None
            for i, row in enumerate(self.samples):
                if row["scores"] is not None:
                    key = (row["period"], row["scores"])
                    if key != previous_score:
                        self.events.append(row)
                    previous_score = key
                if row["period"] is not None:
                    end = min(
                        row["time_s"] + self.hold_s,
                        self.samples[i + 1]["time_s"] if i + 1 < len(self.samples) else math.inf,
                    )
                    period = self.periods[-1] if self.periods else None
                    if (
                        period
                        and period["period"] == row["period"]
                        and abs(row["time_s"] - period["end_s"]) < 1e-6
                    ):
                        period["end_s"] = end
                    else:
                        self.periods.append(
                            {
                                "period": row["period"],
                                "start_s": row["time_s"],
                                "end_s": end,
                            }
                        )
        except (KeyError, TypeError, ValueError) as error:
            self.document = None
            self.samples, self.times, self.events, self.periods = [], [], [], []
            self.reason = str(error)

    @classmethod
    def load(cls, run, manifest, roster):
        path = Path(run) / "scoreboard.json"
        try:
            document = (
                json.loads(path.read_text())
                if path.is_file() and path.stat().st_size <= 5_000_000
                else None
            )
        except (OSError, ValueError):
            document = {}
        return cls(
            document,
            (manifest.get("source") or {}).get("sha256"),
            [t["team_id"] for t in (roster or {}).get("teams", [])],
        )

    def at(self, time):
        i = bisect_right(self.times, time + 1e-8) - 1
        return self.samples[i] if i >= 0 and time - self.times[i] <= self.hold_s else None

    def metadata(self):
        return {
            "available": self.document is not None,
            "reason": self.reason,
            **(self.document or {}),
            "samples": self.samples,
            "events": self.events,
            "periods": self.periods,
            "max_age_s": getattr(self, "hold_s", 0),
            "interpretation": "observed_broadcast_scores_not_official_scoring_events",
        }
