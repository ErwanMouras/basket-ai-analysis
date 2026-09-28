"""Causal recognition with bounded state, source-time and wall-time OCR budgets."""

import math
import time
from collections import deque

from training.jersey.config import number, settings
from training.jersey.colors import TeamColors
from training.jersey.crops import candidate
from training.jersey.reader import ParseqReader, ResourceDeferred


class JerseyRecognizer:
    def __init__(self, config, *, roster=None, reader=None, clock=time.monotonic):
        from training.jersey.roster import UnrestrictedRoster
        self.config = settings(config)
        self.roster = roster if roster is not None else UnrestrictedRoster()
        self.reader = reader if reader is not None else ParseqReader(self.config)
        self.clock = clock
        self.states = {}
        self.team_colors = TeamColors()
        self.segment = None
        self.frame = -1
        self.timestamp = -1.0
        self.tokens = float(self.config["batch_size"])
        self.next_wall = 0.0
        self.stats = {
            "roster_rejections": 0,
            "ocr_batches": 0,
            "ocr_crops": 0,
            "ocr_seconds": 0.0,
            "resource_deferrals": 0,
            "quality_rejections": 0,
            "capacity_rejections": 0,
            "peak_tracks": 0,
            "peak_pending_crops": 0,
            "peak_crop_bytes": 0,
        }
        self.provenance = {
            "reader": self.reader.provenance,
            "roster": self.roster.provenance,
            "config": self.config,
            "fusion": "causal weighted agreement with consecutive confirmation",
            "budget": "source-time token bucket and wall-time duty cooldown",
            "team_colors": "two anonymous groups from repeated dominant torso colors",
        }

    def _decision(self, state, now):
        if state.get("color_votes") and now - state["color_votes"][-1][0] > 12:
            state["team_group"] = None
        group = {"team_group": state.get("team_group"),
                 "team_group_conflict": state.get("color_conflict", False)}
        votes = state["votes"]
        while (
            votes
            and now - votes[0]["timestamp_seconds"]
            > self.config["evidence_ttl_seconds"]
        ):
            votes.popleft()
        totals, counts = {}, {}
        for v in votes:
            n = v["number"]
            totals[n] = totals.get(n, 0) + v["weight"]
            counts[n] = counts.get(n, 0) + 1
        if not totals:
            return {
                "number": None,
                "status": "unknown",
                "votes": 0,
                "agreement": 0.0,
                **group,
                **self.roster.identity(None),
            }
        best = max(totals, key=lambda n: (totals[n], n))
        agreement = totals[best] / sum(totals.values())
        # A fresh contradiction immediately revokes a displayed number.
        recent = list(votes)[-self.config["min_votes"] :]
        confirmed = (
            counts[best] >= self.config["min_votes"]
            and len(recent) >= self.config["min_votes"]
            and all(v["number"] == best for v in recent)
            and agreement >= self.config["min_agreement"]
        )
        return {
            **self.roster.identity(best if confirmed else None),
            "number": best if confirmed else None,
            "status": "confirmed" if confirmed else "uncertain",
            "votes": counts[best],
            "agreement": agreement,
            **group,
        }

    def _finish(self, tid, now, reason):
        s = self.states.pop(tid)
        return {
            "track_id": tid,
            "segment_id": self.segment,
            "first_frame": s["first_frame"],
            "last_frame": s["last_frame"],
            "reason": reason,
            "jersey": self._decision(s, now),
            "evidence": list(s["votes"]),
        }

    def finish(self):
        return [
            self._finish(tid, self.timestamp, "end_of_video")
            for tid in list(self.states)
        ]

    def update(self, image, detections, *, frame_index, timestamp_seconds, segment_id):
        cfg = self.config
        if (
            type(frame_index) is not int
            or frame_index <= self.frame
            or not math.isfinite(timestamp_seconds)
            or timestamp_seconds < 0
            or timestamp_seconds <= self.timestamp
            or type(segment_id) is not int
            or segment_id < 0
            or (self.segment is not None and segment_id < self.segment)
        ):
            raise ValueError(
                "Jersey frames/time must increase and segments must not decrease"
            )
        now = timestamp_seconds
        ended = []
        if self.segment is not None and segment_id != self.segment:
            ended = [
                self._finish(tid, self.timestamp, "scene_reset")
                for tid in list(self.states)
            ]
        self.segment = segment_id
        elapsed = max(0, now - max(0, self.timestamp))
        self.tokens = min(
            float(cfg["batch_size"]),
            self.tokens + elapsed * cfg["crops_per_video_second"],
        )
        self.timestamp, self.frame = now, frame_index
        for tid in list(self.states):
            if now - self.states[tid]["last_seen"] > cfg["state_ttl_seconds"]:
                ended.append(self._finish(tid, now, "expired"))
        tracked_ids = [
            d.get("track_id") for d in detections if d.get("track_id") is not None
        ]
        if any(type(tid) is not int or tid < 1 for tid in tracked_ids) or len(
            set(tracked_ids)
        ) != len(tracked_ids):
            raise ValueError("Expected distinct positive track IDs within each frame")
        visible = set(tracked_ids)
        for d in detections:
            tid = d.get("track_id")
            if tid is None:
                continue
            if tid not in self.states:
                if d["confidence"] < cfg["min_detection_confidence"]:
                    continue
                if len(self.states) >= cfg["max_tracks"]:
                    inactive = [key for key in self.states if key not in visible]
                    if inactive:
                        oldest = min(
                            inactive, key=lambda key: self.states[key]["last_seen"]
                        )
                        ended.append(self._finish(oldest, now, "capacity_eviction"))
                if len(self.states) >= cfg["max_tracks"]:
                    self.stats["capacity_rejections"] += 1
                    continue
                self.states[tid] = {
                    "first_frame": frame_index,
                    "last_frame": frame_index,
                    "last_seen": now,
                    "last_read": -1e9,
                    "last_sample": -1e9,
                    "pending": None,
                    "votes": deque(maxlen=cfg["max_evidence"]),
                    "color_votes": TeamColors.new_votes(),
                    "last_color_sample": -1e9,
                    "team_group": None,
                    "first_team_group": None,
                    "color_conflict": False,
                }
            s = self.states[tid]
            s.update(last_seen=now, last_frame=frame_index)
            decision = self._decision(s, now)
            interval = (
                cfg["confirmed_interval_seconds"]
                if decision["number"] is not None
                else cfg["read_interval_seconds"]
            )
            ocr_ready = (
                now - s["last_read"] >= interval
                and now - s["last_sample"] >= cfg["sample_interval_seconds"]
            )
            color_ready = now - s["last_color_sample"] >= 0.35
            if not ocr_ready and not color_ready:
                continue
            if d["confidence"] < cfg["min_detection_confidence"]:
                continue
            crop = candidate(image, d, detections, cfg)
            if crop is None:
                self.stats["quality_rejections"] += 1
                continue
            if color_ready:
                self.team_colors.observe(s, crop, now)
            if not ocr_ready:
                continue
            s["last_sample"] = now
            crop.update(
                frame_index=frame_index,
                timestamp_seconds=now,
                selected_since=s["pending"]["selected_since"] if s["pending"] else now,
            )
            if s["pending"] is None or crop["quality"] > s["pending"]["quality"]:
                s["pending"] = crop
        # Never retain crops from a previous sighting indefinitely.
        for s in self.states.values():
            if (
                s["pending"]
                and now - s["pending"]["timestamp_seconds"]
                > cfg["read_interval_seconds"]
            ):
                s["pending"] = None
        pending = [s["pending"] for s in self.states.values() if s["pending"]]
        self.stats["peak_tracks"] = max(self.stats["peak_tracks"], len(self.states))
        self.stats["peak_pending_crops"] = max(
            self.stats["peak_pending_crops"], len(pending)
        )
        self.stats["peak_crop_bytes"] = max(
            self.stats["peak_crop_bytes"], sum(p["image"].nbytes for p in pending)
        )
        ready = [
            (tid, s)
            for tid, s in self.states.items()
            if tid in visible
            and s["pending"]
            and now - s["pending"]["selected_since"] >= cfg["selection_window_seconds"]
        ]
        ready.sort(
            key=lambda row: (
                row[1]["last_read"],
                row[1]["pending"]["selected_since"],
                row[0],
            )
        )
        picked = ready[: min(cfg["batch_size"], int(self.tokens + 1e-9))]
        readings = []
        if picked and self.clock() >= self.next_wall:
            started = self.clock()
            try:
                results = self.reader.read([s["pending"]["image"] for _, s in picked])
            except ResourceDeferred as exc:
                self.stats["resource_deferrals"] += 1
                self.next_wall = self.clock() + 0.5
                readings.append(
                    {
                        "status": "resource_deferred",
                        "reason": str(exc),
                        "frame_index": frame_index,
                    }
                )
            else:
                duration = max(0, self.clock() - started)
                self.next_wall = max(
                    started + cfg["min_batch_interval_seconds"],
                    self.clock() + duration * (1 / cfg["max_duty_cycle"] - 1),
                )
                if len(results) != len(picked):
                    raise RuntimeError("OCR output count does not match selected crops")
                self.tokens = max(0, self.tokens - len(picked))
                self.stats["ocr_batches"] += 1
                self.stats["ocr_crops"] += len(picked)
                self.stats["ocr_seconds"] += duration
                for (tid, s), result in zip(picked, results):
                    crop = s["pending"]
                    confidence = result["confidence"]
                    if (
                        type(confidence) not in (int, float)
                        or not math.isfinite(confidence)
                        or not 0 <= confidence <= 1
                    ):
                        raise RuntimeError("Invalid OCR confidence")
                    n = number(result.get("text")) if result.get("eos") else None
                    permitted = n is not None and self.roster.allows(n)
                    accepted = permitted and confidence >= cfg["min_ocr_confidence"]
                    reason = (
                        "invalid_text"
                        if n is None
                        else "not_in_roster"
                        if not permitted
                        else "low_ocr_confidence"
                        if not accepted
                        else None
                    )
                    if n is not None and not permitted:
                        self.stats["roster_rejections"] += 1
                    reading = {
                        "track_id": tid,
                        "segment_id": segment_id,
                        "processed_frame": frame_index,
                        "frame_index": crop["frame_index"],
                        "timestamp_seconds": crop["timestamp_seconds"],
                        "crop_bbox": crop["bbox"],
                        "region": crop["region"],
                        "quality": crop["quality"],
                        "text": result["text"],
                        "number": n if permitted else None,
                        "raw_number": n,
                        "rejection_reason": reason,
                        "confidence": confidence,
                        "status": "vote" if accepted else "rejected",
                    }
                    readings.append(reading)
                    if accepted:
                        s["votes"].append(
                            {
                                "number": n,
                                "confidence": confidence,
                                "weight": confidence * crop["quality"],
                                "frame_index": crop["frame_index"],
                                "timestamp_seconds": crop["timestamp_seconds"],
                            }
                        )
                    s.update(last_read=now, pending=None)
        enriched = []
        for d in detections:
            s = self.states.get(d.get("track_id"))
            jersey = (
                self._decision(s, now)
                if s is not None
                else {
                    "number": None,
                    "status": "untracked"
                    if d.get("track_id") is None
                    else "below_detection_confidence"
                    if d["confidence"] < cfg["min_detection_confidence"]
                    else "capacity_limit",
                    "votes": 0,
                    "agreement": 0.0,
                    "team_group": None,
                    "team_group_conflict": False,
                }
            )
            jersey = {**jersey, **self.roster.identity(jersey["number"])}
            enriched.append({**d, "jersey": jersey})
        return enriched, readings, ended
