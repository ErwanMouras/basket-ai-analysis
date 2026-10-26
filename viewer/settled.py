"""Installed half-court play, separated from possession and neutral transitions.

Contacts confirm who controls the ball. Court sides decide whether that team
has reached its attacking half. A turnover under the old basket ends the phase;
it does not start an attack for the other team beneath its own basket.
"""

from bisect import bisect_left, bisect_right
from collections import Counter


def contact_owner(pairs, starts, ends, end, since, teams):
    """Causal contact confirmation; a lone contrary detection cannot flip play."""
    lo = max(since, end - 0.4)
    votes, frames = Counter(), Counter()
    for a, b, frame in pairs[bisect_right(ends, lo) : bisect_right(starts, end - 1e-9)]:
        if frame.contact.team in teams:
            weight = max(0.0, min(end, b) - max(lo, a))
            votes[frame.contact.team] += weight
            frames[frame.contact.team] += int(weight > 0)
    current = pairs[bisect_left(ends, end)][2].contact
    if not votes or current.reason in (
        "referee_contact",
        "unassigned_contact",
        "contested_contact",
    ):
        return None
    winner, weight = votes.most_common(1)[0]
    if (
        weight >= 0.16 - 1e-9
        and frames[winner] >= 2
        and weight / sum(votes.values()) >= 0.8
        and current.team in (None, winner)
    ):
        return winner
    return None


def settled_half(pairs, starts, ends, end, since):
    """Require a stable observed group in one half, using only past frames."""
    lo = max(since, end - 0.4)
    votes, frames = Counter(), Counter()
    for a, b, frame in pairs[bisect_right(ends, lo) : bisect_right(starts, end - 1e-9)]:
        if frame.half:
            weight = max(0.0, min(end, b) - max(lo, a))
            votes[frame.half] += weight
            frames[frame.half] += int(weight > 0)
    if not votes:
        return None
    half, weight = votes.most_common(1)[0]
    if weight >= 0.25 - 1e-9 and frames[half] >= 3 and weight / sum(votes.values()) >= 0.9:
        return half
    return None


def classify_settled(blocks, spans, teams, geometry, directions, score_changes):
    """Apply the three-state model: team A installed, neutral, team B installed.

    Evidence resets at cuts/gaps/period boundaries. A confirmed pending team can
    survive a cut within the same quarter, so the old half cannot restart it.
    No interval is synthesized.
    Within a continuous view the installed phase survives a hidden ball, until
    contrary control is confirmed or the ball is observed back outside that half.
    """
    indexed = {(s.plan, s.start, s.end): s for s in spans}
    previous, waiting = None, None
    for block in blocks:
        pairs = [(a.time, b.time, b) for a, b in zip(block, block[1:])]
        starts, ends = [a for a, _, _ in pairs], [b for _, b, _ in pairs]
        same_quarter_cut = (
            previous is not None
            and block[0].period is not None
            and block[0].period == previous.period
            and block[0].frame == previous.frame + 1
            and 0 < block[0].time - previous.time <= 0.2 + 1e-9
        )
        active = blocked_half = None
        possession = pending = waiting if same_quarter_cut else None
        since = block[0].time
        for start, end, frame in pairs:
            span = indexed[(frame.plan, start, end)]
            span.owner, span.direction, span.method, span.reason = None, None, "none", "unobserved"
            direction = directions.get((frame.period, frame.plan))
            if len(teams) != 2 or not geometry or not direction:
                continue
            changes = score_changes.get(frame.period, ())
            score_changed = bisect_right(changes, end) > bisect_right(changes, start)
            if score_changed:
                blocked_half = (
                    (direction if active == teams[0] else -direction) if active else frame.half
                )
                active = possession = pending = None
                since = end
            confirmed = contact_owner(pairs, starts, ends, end, since, teams)
            if confirmed:
                if (active and confirmed != active) or (possession and confirmed != possession):
                    active = None
                    pending = confirmed
                    since = end
                elif active is None:
                    pending = confirmed
                possession = confirmed
            if frame.contact.reason == "referee_contact":
                active = None
                pending = possession
                since = end
                span.method, span.reason = "neutral", "transition"
                continue
            half = settled_half(pairs, starts, ends, end, since)
            candidate = teams[0] if half == direction else teams[1] if half else None
            # A detected carrier in its attacking half is sufficient even if
            # several off-ball players are outside the camera/calibrated area.
            carrier = frame.contact.team
            x = frame.contact.holder_x
            carrier_direction = direction if carrier == teams[0] else -direction
            frontcourt = (
                carrier in teams
                and x is not None
                and (
                    x >= 0.55 * geometry["length"]
                    if carrier_direction == 1
                    else x <= 0.45 * geometry["length"]
                )
            )
            if (
                active
                and carrier == active
                and confirmed == active
                and x is not None
                and not frontcourt
            ):
                active = None
                pending = possession
                since = end
            if active is None:
                if pending:
                    # The confirmed new team stays neutral at its own basket.
                    if (
                        candidate == pending or carrier == pending and frontcourt
                    ) and frame.contact.team in (None, pending):
                        active = pending
                elif candidate and half != blocked_half and frame.contact.team in (None, candidate):
                    active = candidate
                if confirmed and carrier == confirmed and frontcourt:
                    active = confirmed
                if active:
                    pending = blocked_half = None
            elif (
                candidate
                and candidate != active
                and confirmed in (None, candidate)
                and frame.contact.team in (None, candidate)
            ):
                active = candidate
            if not frame.court_visible or frame.contact.reason in (
                "unassigned_contact",
                "contested_contact",
            ):
                continue
            if active:
                span.owner = active
                span.direction = direction if active == teams[0] else -direction
                span.method, span.reason = "settled", "settled_attack"
            elif pending or blocked_half or score_changed:
                span.method, span.reason = "neutral", "transition"
        waiting = pending if active is None else None
        previous = block[-1]
