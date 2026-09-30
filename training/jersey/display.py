"""Stable display colors for runs and the lightweight local viewer."""

TEAM_DISPLAY_COLORS = ("#45C7F5", "#F279C8")
GROUP_DISPLAY_COLORS = {"team_1": "#A5B4FC", "team_2": "#FACC65"}
REFEREE_DISPLAY_COLOR = "#FFBD78"
UNASSIGNED_DISPLAY_COLOR = "#C9F579"


def display_colors(roster=None):
    """Anonymous groups do not imply a named roster team."""
    payload = roster.payload if hasattr(roster, "payload") else roster
    teams = {}
    if payload is not None:
        teams = {team["team_id"]: team.get("color", TEAM_DISPLAY_COLORS[i])
                 for i, team in enumerate(payload["teams"])}
    return {"teams": teams, "groups": dict(GROUP_DISPLAY_COLORS),
            "referee": REFEREE_DISPLAY_COLOR, "unassigned": UNASSIGNED_DISPLAY_COLOR}


def person_display_color(person, palette):
    if person.get("role") == "referee":
        return palette["referee"]
    if person.get("role") != "player":
        return palette["unassigned"]
    jersey = person.get("jersey") or {}
    team = jersey.get("team_id")
    if team in palette["teams"]:
        return palette["teams"][team]
    group = jersey.get("team_group")
    return palette["groups"].get(group, palette["unassigned"])
