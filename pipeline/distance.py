"""Conservative metric integration; no interpolation across missing observations."""

from collections import Counter
import math


def identity(person):
    if person.get("jersey_number_suppressed"):
        return None
    resolved = person.get("resolved_identity") or {}
    if resolved.get("team_id") and resolved.get("number") is not None:
        return (resolved["team_id"], resolved["number"])
    if "majority_jersey_number" in person:
        return None
    jersey = person.get("jersey") or {}
    if jersey.get("identity_status") == "unique_number" and jersey.get("status") == "confirmed":
        return (jersey["team_id"], jersey["number"])
    return None


class Distance:
    def __init__(self, config, run_id):
        self.config, self.run_id = config, run_id
        self.previous = {}
        self.rows = {}
        self.last_frame = -1
        self.last_timestamp = -1.0
        self.last_segment = 0

    def update(self, frame):
        index, now, segment = frame["frame_index"], frame["timestamp_s"], frame["segment_id"]
        if index != self.last_frame + 1 or now <= self.last_timestamp or segment < self.last_segment:
            raise ValueError("Distance requires ordered, consecutive observations")
        self.last_frame, self.last_timestamp = index, now
        self.last_segment = segment
        counts = Counter(identity(p) for p in frame["persons"] if p["role"] == "player" and p["track_id"] is not None)
        current, contributions = {}, []
        for p in frame["persons"]:
            if p["track_id"] is None or p["role"] != "player" or not frame["included"]:
                continue
            track = (segment, p["track_id"])
            who = identity(p)
            conflict = who is not None and counts[who] > 1
            if conflict:
                who = None
            subject = "roster:" + ":".join(who) if who else f"anonymous-s{segment}-t{p['track_id']}"
            key = (subject, segment, p["track_id"])
            jersey = p.get("jersey") or {}
            resolved = p.get("resolved_identity") or {}
            row = self.rows.setdefault(key, {
                "subject_id": subject, "player_id": subject if who else None,
                "player_name": (resolved.get("player_name") or jersey.get("player_name")) if who else None,
                "team_id": who[0] if who else p.get("resolved_team_id"),
                "jersey_number": who[1] if who else None,
                "team_group": jersey.get("team_group"),
                "identity_status": "confirmed" if who else "anonymous",
                "track_refs": [{"run_id": self.run_id, "segment_id": segment, "track_id": p["track_id"]}],
                "distance": 0.0, "tracked_duration_s": 0.0, "measured_duration_s": 0.0,
                "valid_pairs": 0, "warnings": set(), "excluded_intervals": Counter(),
                "last_frame": index,
            })
            if row["last_frame"] < index - 1:
                row["warnings"].add("tracking_gap")
                row["excluded_intervals"]["tracking_gap"] += 1
            row["last_frame"] = index
            if not who:
                row["warnings"].add("identity_not_resolved")
                if p.get("jersey_number_suppressed"):
                    row["warnings"].add("jersey_number_suppressed")
                number = (None if p.get("jersey_number_suppressed") else
                          p.get("majority_jersey_number") if "majority_jersey_number" in p else
                          jersey.get("number") if jersey.get("status") == "confirmed" else None)
                if number is not None and "jersey_number_conflict" not in row["warnings"]:
                    if row["jersey_number"] is not None and row["jersey_number"] != number:
                        row["jersey_number"] = None
                        row["warnings"].add("jersey_number_conflict")
                    else:
                        row["jersey_number"] = number
                if jersey.get("identity_status") == "ambiguous":
                    row["warnings"].add("roster_identity_ambiguous")
            if jersey.get("team_group_conflict"):
                row["team_group"] = None
                row["warnings"].add("team_group_conflict")
            elif jersey.get("team_group") is None:
                row["team_group"] = None
            elif "team_group_conflict" not in row["warnings"]:
                if row["team_group"] is None:
                    row["team_group"] = jersey["team_group"]
                elif row["team_group"] != jersey["team_group"]:
                    row["team_group"] = None
                    row["warnings"].add("team_group_conflict")
            if conflict:
                row["warnings"].add("identity_conflict")
            prev = self.previous.get(track)
            position = p["position_m"]
            state = {"index": index, "time": now, "point": position, "filtered": position,
                     "anchor": position, "method": p.get("ground_point_method"), "subject": subject,
                     "conflict": conflict}
            reason = None
            distance = None
            dt = now - prev["time"] if prev else 0
            if prev:
                # Denominator includes unmeasurable intervals while the same subject is visible.
                if prev["subject"] == subject:
                    row["tracked_duration_s"] += dt
                if prev["subject"] != subject or conflict or prev["conflict"]:
                    reason = "identity_transition_or_conflict"
                elif dt > self.config["max_gap_s"]:
                    reason = "timestamp_gap"
                elif position is None or prev["point"] is None:
                    reason = p.get("position_status", "position_unavailable")
                elif prev["method"] != state["method"]:
                    reason = "ground_point_method_change"
                elif math.dist(position, prev["point"]) / dt > self.config["max_speed_m_s"]:
                    reason = "implausible_speed"
                else:
                    tau = self.config["smoothing_tau_s"]
                    alpha = 1 - math.exp(-dt / tau) if tau else 1.0
                    filtered = [a + alpha * (b - a) for a, b in zip(prev["filtered"], position)]
                    step = math.dist(filtered, prev["anchor"])
                    distance = step if step >= self.config["deadband_m"] else 0.0
                    state["filtered"] = filtered
                    state["anchor"] = filtered if distance else prev["anchor"]
                    row["distance"] += distance
                    row["valid_pairs"] += 1
                    row["measured_duration_s"] += dt
            if reason:
                row["warnings"].add(reason)
                row["excluded_intervals"][reason] += 1
            if position is None:
                row["warnings"].add(p.get("position_status", "position_unavailable"))
            current[track] = state
            contributions.append({"schema_version": 1, "run_id": self.run_id,
                                  "segment_id": segment, "track_id": p["track_id"],
                                  "subject_id": subject, "frame_index": index, "timestamp_s": now,
                                  "position_m": position, "filtered_position_m": state["filtered"],
                                  "interval_start_s": prev["time"] if prev else None,
                                  "distance_m": distance, "measured_duration_s": dt if distance is not None else 0.0,
                                  "reason": reason or ("interval_start" if prev is None else None)})
        # Missing/role-changed/excluded tracks cannot bridge even one missing frame.
        self.previous = current
        return contributions

    def result(self):
        groups = {}
        for row in self.rows.values():
            subject = row["subject_id"]
            if subject not in groups:
                groups[subject] = {**row, "track_refs": [], "warnings": set(),
                                   "excluded_intervals": Counter(), "distance": 0.0,
                                   "tracked_duration_s": 0.0, "measured_duration_s": 0.0, "valid_pairs": 0}
            group = groups[subject]
            group["track_refs"].extend(row["track_refs"])
            group["warnings"].update(row["warnings"])
            group["excluded_intervals"].update(row["excluded_intervals"])
            for key in ("distance", "tracked_duration_s", "measured_duration_s", "valid_pairs"):
                group[key] += row[key]
        result = []
        for row in groups.values():
            valid = row.pop("valid_pairs")
            row.pop("last_frame")
            distance = row.pop("distance")
            row["observed_distance_m"] = distance if valid else None
            duration = row["tracked_duration_s"]
            row["coverage_ratio"] = row["measured_duration_s"] / duration if duration else None
            row["measurement_status"] = "unavailable" if not valid else "partial" if row["measured_duration_s"] < duration - 1e-9 else "ok"
            row["warnings"] = sorted(row["warnings"])
            row["excluded_intervals"] = dict(row["excluded_intervals"])
            result.append(row)
        return {"schema_version": 1, "run_id": self.run_id, "metric": "observed_distance",
                "distance_unit": "m", "time_basis": "video", "scope": "included_video_intervals",
                "method": self.config, "players": result,
                "limitations": ["no_offscreen_extrapolation", "no_automatic_replay_exclusion",
                                "ground_contact_approximation", "metric_accuracy_not_validated"]}
