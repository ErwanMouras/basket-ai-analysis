"""Use an explicitly verified active lineup, OCR and CUDA motion assignment.

This is restricted to an interval whose lineup is known. A tied initial
assignment fails; unknown identities are never filled by arbitrary list order.
"""

import torch

from pipeline.cuda.core import box_iou, tensor
from pipeline.cuda.tracking import Kalman, Track, linear_assignment, xyah, xyxy


def seed_assignment(track_ids, teams, players, reads, device):
    lookup = {track: i for i, track in enumerate(track_ids)}
    scores = torch.zeros(
        (len(track_ids), len(players)), device=device, dtype=torch.float64
    )
    for read in reads:
        i = lookup.get(read["track_id"])
        if i is None or read.get("number") is None:
            continue
        probability = tensor(read["confidence"], device=device)
        for j, player in enumerate(players):
            if read["number"] == player["number"]:
                scores[i, j] += torch.where(probability >= 0.8, probability, 0)
    compatible = tensor(
        [[team == player["team_id"] for player in players] for team in teams],
        device=device,
        dtype=torch.bool,
    )
    cost = torch.where(compatible, -scores, torch.inf)
    matches, left, right = linear_assignment(cost, 1e6)
    if left or right:
        raise ValueError("The seed frame must contain the complete verified lineup")
    indices = tensor(matches, device=device, dtype=torch.long)
    optimum = cost[indices[:, 0], indices[:, 1]].sum()
    margins = []
    for i, j in matches:
        alternative = cost.clone()
        alternative[i, j] = torch.inf
        other, unmatched, _ = linear_assignment(alternative, 1e6)
        if unmatched:
            margin = torch.full((), torch.inf, device=device, dtype=torch.float64)
        else:
            ix = tensor(other, device=device, dtype=torch.long)
            margin = alternative[ix[:, 0], ix[:, 1]].sum() - optimum
        margins.append(margin)
    margins = torch.stack(margins)
    if not bool((margins >= 1).all()):
        raise ValueError("OCR and lineup do not uniquely identify the seed players")
    return {track_ids[i]: j for i, j in matches}, scores, margins


class LineupTracker:
    def __init__(self, persons, players, seeds, device):
        self.players, self.device, self.seeds = players, device, dict(seeds)
        present = {p["track_id"] for p in persons if p["role"] == "player"}
        if present != set(seeds) or len(present) != len(players):
            raise ValueError(
                "The seed frame must contain each reviewed player exactly once"
            )
        self.kalman = Kalman(device)
        self.tracks = []
        for person in persons:
            if person["role"] != "player":
                continue
            j = seeds[person["track_id"]]
            mean, covariance = self.kalman.initiate(
                xyah(tensor(person["bbox"], device=device))
            )
            self.tracks.append(Track(j + 1, 2, mean, covariance, 0, 0, "tracked", True))
        self.tracks.sort(key=lambda t: t.id)

    def update(self, persons, frame, source_teams):
        indices = [i for i, p in enumerate(persons) if p["role"] == "player"]
        if not indices:
            return {}, {}
        values = tensor(
            [persons[i]["bbox"] + [persons[i]["confidence"], 2] for i in indices],
            device=self.device,
        )
        if frame:
            self.kalman.predict_many(self.tracks)
        predicted = torch.stack([xyxy(t.mean) for t in self.tracks])
        wh = (predicted[:, 2:] - predicted[:, :2]).clamp_min(1)
        centers = (predicted[:, :2] + predicted[:, 2:]) / 2
        candidate_centers = (values[:, :2] + values[:, 2:4]) / 2
        distance = (
            torch.linalg.vector_norm(centers[:, None] - candidate_centers[None], dim=2)
            / wh[:, 1:2]
        )
        iou = box_iou(predicted, values[:, :4])
        old = [persons[i]["track_id"] for i in indices]
        anchor = tensor(
            [
                [self.seeds.get(track) == j for track in old]
                for j in range(len(self.players))
            ],
            device=self.device,
            dtype=torch.bool,
        )
        compatible = tensor(
            [
                [source_teams.get(track) in (None, p["team_id"]) for track in old]
                for p in self.players
            ],
            device=self.device,
            dtype=torch.bool,
        )
        cost = 0.6 * (1 - iou) + 0.4 * distance - 0.35 * anchor
        cost = torch.where(compatible & (distance < 1.2), cost, torch.inf)
        matches, _, _ = linear_assignment(cost, 1.0)
        updates = [(self.tracks[j], n) for j, n in matches]
        self.kalman.update_many(updates, values)
        identities, evidence = {}, {}
        for j, n in matches:
            i = indices[n]
            identities[i] = j
            evidence[i] = {
                "assignment_cost": cost[j, n],
                "source_track_id": persons[i]["track_id"],
            }
            self.tracks[j].last = frame
            self.tracks[j].state = "tracked"
        matched = {j for j, _ in matches}
        for j, t in enumerate(self.tracks):
            if j not in matched:
                t.state = "lost"
        return identities, evidence
