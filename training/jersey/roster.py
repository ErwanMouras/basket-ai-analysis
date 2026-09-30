"""Validated per-video allowlist; shared numbers never imply a unique identity."""

import copy
import hashlib
import json
import re
from pathlib import Path

from training.jersey.config import number

def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Invalid JSON constant: {value}")


def _text(value, field):
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value) > 200
    ):
        raise ValueError(
            f"{field} must be a nonempty trimmed string (max 200 characters)"
        )
    return value


class UnrestrictedRoster:
    """Explicit anonymous mode: numeric OCR evidence without an identity list."""

    provenance = {"mode": "unrestricted", "sha256": None}
    sha256 = None

    def allows(self, value):
        return number(value) is not None

    def identity(self, value):
        return {"player_name": None, "team_id": None, "team_name": None,
                "candidates": [], "identity_status": "anonymous"}


class Roster:
    def __init__(self, payload, *, source=None, sha256=None):
        if (
            not isinstance(payload, dict)
            or type(payload.get("schema_version")) is not int
            or payload["schema_version"] != 1
        ):
            raise ValueError("jersey.json requires schema_version: 1")
        unknown = set(payload) - {
            "schema_version",
            "teams",
            "match",
            "sources",
            "notes",
        }
        if unknown:
            raise ValueError(f"Unknown jersey.json fields: {sorted(unknown)}")
        teams = payload.get("teams")
        if not isinstance(teams, list) or len(teams) != 2:
            raise ValueError("jersey.json must contain the two teams playing the match")
        self._payload = copy.deepcopy(payload)
        self._by_number = {}
        team_ids = set()
        total = eligible = 0
        for team in self._payload["teams"]:
            if (not isinstance(team, dict) or set(team) - {"team_id", "name", "players", "color", "uniform_color"}
                    or not {"team_id", "name", "players"} <= set(team)):
                raise ValueError("Each team requires team_id, name and players; color is optional")
            tid = _text(team["team_id"], "team_id")
            name = _text(team["name"], "team name")
            for color_field in ("color", "uniform_color"):
                if color_field in team and (not isinstance(team[color_field], str)
                                            or not re.fullmatch(r"#[0-9A-Fa-f]{6}", team[color_field])):
                    raise ValueError(f"Team {color_field} must be a #RRGGBB hex string")
            if tid in team_ids:
                raise ValueError(f"Duplicate team_id: {tid}")
            team_ids.add(tid)
            players = team["players"]
            if not isinstance(players, list) or not 1 <= len(players) <= 64:
                raise ValueError("Each team requires 1 to 64 players")
            numbers = set()
            team_eligible = 0
            for player in players:
                if (
                    not isinstance(player, dict)
                    or set(player) - {"number", "name", "eligible", "status"}
                    or not {"number", "name"} <= set(player)
                ):
                    raise ValueError(
                        "Players require number/name; optional eligible/status"
                    )
                n = number(player["number"])
                if n is None:
                    raise ValueError(
                        "Jersey numbers must be strings of one or two ASCII digits"
                    )
                if n in numbers:
                    raise ValueError(f"Duplicate jersey number {n!r} in team {tid}")
                numbers.add(n)
                pname = _text(player["name"], "player name")
                allowed = player.get("eligible", True)
                if type(allowed) is not bool:
                    raise ValueError("Player eligible must be boolean")
                if "status" in player and player["status"] not in (
                    "played",
                    "dnp",
                    "dnd",
                    "inactive",
                ):
                    raise ValueError("Unknown match player status")
                total += 1
                if allowed:
                    eligible += 1
                    team_eligible += 1
                    self._by_number.setdefault(n, []).append(
                        {
                            "team_id": tid,
                            "team_name": name,
                            "number": n,
                            "player_name": pname,
                        }
                    )
            if not team_eligible:
                raise ValueError("Each team must contain at least one eligible player")
        self.sha256 = (
            sha256
            or hashlib.sha256(
                json.dumps(self._payload, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
        )
        self.provenance = {
            "path": str(source) if source is not None else None,
            "sha256": self.sha256,
            "players": total,
            "eligible_players": eligible,
            "allowed_numbers": sorted(self._by_number),
        }

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        try:
            with path.open("rb") as handle:
                data = handle.read(1024 * 1024 + 1)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Jersey recognition requires {path}; create jersey.json beside the video"
            ) from exc
        if len(data) > 1024 * 1024:
            raise ValueError("jersey.json exceeds 1 MiB")
        payload = json.loads(
            data,
            object_pairs_hook=_unique_keys,
            parse_constant=_invalid_constant,
        )
        return cls(payload, source=path, sha256=hashlib.sha256(data).hexdigest())

    @property
    def payload(self):
        return copy.deepcopy(self._payload)

    def allows(self, value):
        return value in self._by_number

    def identity(self, value):
        candidates = [dict(p) for p in self._by_number.get(value, [])]
        unique = candidates[0] if len(candidates) == 1 else {}
        return {
            "player_name": unique.get("player_name"),
            "team_id": unique.get("team_id"),
            "team_name": unique.get("team_name"),
            "candidates": candidates,
            "identity_status": "unique_number"
            if len(candidates) == 1
            else "ambiguous"
            if candidates
            else "unresolved",
        }
