"""ByteTrack's two confidence stages, XYAH Kalman filter and LAP on CUDA.

Track lifecycles and ID allocation are Python control metadata. All costs,
linear assignment potentials, predictions and covariance arithmetic are CUDA.
No SciPy, LAP, NumPy or OpenCV calculation is called.
"""

from dataclasses import dataclass

import torch

from pipeline.cuda.core import box_iou, tensor


def _linear_assignment_reference(cost, threshold):
    n, m = cost.shape
    if not n or not m:
        return [], list(range(n)), list(range(m))
    device = cost.device
    # One dummy column per row makes unmatched assignments explicit. Their
    # cost equals the cost limit, the constant-baseline equivalent of LAPJV's
    # square extension with two half-cost dummy edges.
    work = torch.cat(
        (
            cost.double(),
            torch.full((n, n), threshold, device=device, dtype=torch.float64),
        ),
        1,
    )
    cols = m + n
    u = torch.zeros(n + 1, dtype=torch.float64, device=device)
    v = torch.zeros(cols + 1, dtype=torch.float64, device=device)
    p = torch.zeros(cols + 1, dtype=torch.long, device=device)
    way = torch.zeros_like(p)
    for row in range(1, n + 1):
        p[0] = row
        column = 0
        minimum = torch.full((cols + 1,), torch.inf, device=device, dtype=torch.float64)
        used = torch.zeros(cols + 1, device=device, dtype=torch.bool)
        while True:
            used[column] = True
            active_row = int(p[column])
            reduced = work[active_row - 1] - u[active_row] - v[1:]
            better = (~used[1:]) & (reduced < minimum[1:])
            minimum[1:] = torch.where(better, reduced, minimum[1:])
            way[1:] = torch.where(better, column, way[1:])
            delta, next_column = torch.where(used[1:], torch.inf, minimum[1:]).min(0)
            if not bool(torch.isfinite(delta)):
                raise RuntimeError("CUDA assignment has no finite augmenting path")
            u.index_add_(0, p[used], delta.expand_as(p[used]))
            v[used] -= delta
            minimum[~used] -= delta
            column = int(next_column) + 1
            if int(p[column]) == 0:
                break
        while column:
            previous = int(way[column])
            p[column] = p[previous]
            column = previous
    matched = []
    for column in range(1, m + 1):
        row = int(p[column]) - 1
        if row >= 0 and bool(cost[row, column - 1] <= threshold):
            matched.append((row, column - 1))
    matched_rows, matched_cols = {r for r, _ in matched}, {c for _, c in matched}
    return (
        matched,
        [r for r in range(n) if r not in matched_rows],
        [c for c in range(m) if c not in matched_cols],
    )


def linear_assignment(cost, threshold):
    from pipeline.cuda.kernels import assignment

    n, m = cost.shape
    if not n or not m:
        return [], list(range(n)), list(range(m))
    # The entire augmenting-path solver runs on CUDA. Export only final IDs.
    owners = assignment(cost, threshold).tolist()
    matched = [(row, column) for column, row in enumerate(owners) if row >= 0]
    matched_rows = {row for row, _ in matched}
    return (
        matched,
        [row for row in range(n) if row not in matched_rows],
        [column for column, row in enumerate(owners) if row < 0],
    )


def xyah(boxes):
    wh = boxes[..., 2:] - boxes[..., :2]
    return torch.cat(
        (
            (boxes[..., :2] + boxes[..., 2:]) / 2,
            (wh[..., 0] / wh[..., 1])[..., None],
            wh[..., 1:],
        ),
        -1,
    )


def xyxy(mean):
    h = mean[..., 3]
    half = torch.stack((mean[..., 2] * h, h), -1) / 2
    return torch.cat((mean[..., :2] - half, mean[..., :2] + half), -1)


class Kalman:
    def __init__(self, device):
        self.device = device
        self.motion = torch.eye(8, device=device, dtype=torch.float64)
        self.motion[:4, 4:] = torch.eye(4, device=device, dtype=torch.float64)
        self.predict_factors = tensor(
            [0.05, 0.05, 0, 0.05, 0.00625, 0.00625, 0, 0.00625], device=device
        )
        self.update_factors = tensor([0.05, 0.05, 0, 0.05], device=device)

    def initiate(self, measurement):
        mean = torch.cat(
            (
                measurement.double(),
                torch.zeros(4, device=self.device, dtype=torch.float64),
            )
        )
        factors = tensor(
            [0.1, 0.1, 0, 0.1, 0.0625, 0.0625, 0, 0.0625], device=self.device
        )
        std = factors * mean[3]
        std[2], std[6] = 1e-2, 1e-5
        return mean, std.square().diag()

    def predict(self, mean, covariance, *, lost=False):
        mean = mean.clone()
        if lost:
            mean[7] = 0
        std = (
            tensor(
                [0.05, 0.05, 0, 0.05, 0.00625, 0.00625, 0, 0.00625], device=self.device
            )
            * mean[3]
        )
        std[2], std[6] = 1e-2, 1e-5
        return (
            self.motion @ mean,
            self.motion @ covariance @ self.motion.T + std.square().diag(),
        )

    def update(self, mean, covariance, measurement):
        std = tensor([0.05, 0.05, 0, 0.05], device=self.device) * mean[3]
        std[2] = 0.1
        projected = covariance[:4, :4] + std.square().diag()
        gain = torch.linalg.solve(projected, covariance[:4]).T
        return mean + gain @ (
            measurement.double() - mean[:4]
        ), covariance - gain @ projected @ gain.T

    def predict_many(self, tracks):
        if not tracks:
            return
        means = torch.stack([track.mean for track in tracks])
        covariance = torch.stack([track.covariance for track in tracks])
        lost = torch.tensor(
            [track.state == "lost" for track in tracks],
            device=self.device,
            dtype=torch.bool,
        )
        means[:, 7] = torch.where(lost, 0, means[:, 7])
        std = self.predict_factors[None] * means[:, 3:4]
        std[:, 2], std[:, 6] = 1.0e-2, 1.0e-5
        following = means @ self.motion.T
        uncertainty = self.motion @ covariance @ self.motion.T + torch.diag_embed(
            std.square()
        )
        for index, track in enumerate(tracks):
            track.mean, track.covariance = following[index], uncertainty[index]

    def update_many(self, matches, rows):
        if not matches:
            return
        means = torch.stack([track.mean for track, _ in matches])
        covariance = torch.stack([track.covariance for track, _ in matches])
        indices = tensor(
            [index for _, index in matches], device=self.device, dtype=torch.long
        )
        measurements = xyah(rows[indices, :4]).double()
        std = self.update_factors[None] * means[:, 3:4]
        std[:, 2] = 0.1
        projected = covariance[:, :4, :4] + torch.diag_embed(std.square())
        solution, info = torch.linalg.solve_ex(
            projected, covariance[:, :4], check_errors=False
        )
        if bool((info != 0).any()):
            raise RuntimeError("Singular CUDA Kalman update")
        gain = solution.transpose(1, 2)
        following = means + (gain @ (measurements - means[:, :4])[:, :, None])[:, :, 0]
        uncertainty = covariance - gain @ projected @ gain.transpose(1, 2)
        for index, (track, _) in enumerate(matches):
            track.mean, track.covariance = following[index], uncertainty[index]


@dataclass
class Track:
    id: int
    role: int
    mean: torch.Tensor
    covariance: torch.Tensor
    start: int
    last: int
    state: str
    activated: bool


class Tracker:
    def __init__(self, config, fps, device):
        if config["tracker"] != "bytetrack" or config["gmc_method"] != "none":
            raise ValueError(
                "Native CUDA tracking requires tracker=bytetrack and gmc_method=none"
            )
        self.config = config
        self.device = device
        self.kalman = Kalman(device)
        self.buffer = int(
            torch.ceil(
                tensor(fps, device=device) * config["track_buffer_seconds"]
            ).item()
        )
        self.tracks = []
        self.counter = 0
        self.provenance = {
            "backend": "native_CUDA_ByteTrack",
            "kalman": "XYAH_CUDA_float64",
            "assignment": "CUDA_shortest_augmenting_path_cost_limit",
            "device": device,
            "association": "independent_per_role",
            "gmc_method": "none",
            "config": config,
        }

    def _associate(self, tracks, rows, indices, threshold, fuse=False):
        if tracks and indices:
            boxes = torch.stack([xyxy(t.mean) for t in tracks])
            selected = tensor(indices, device=self.device, dtype=torch.long)
            cost = 1 - box_iou(boxes, rows[selected, :4].double())
            if fuse:
                cost = 1 - (1 - cost) * rows[selected, 4][None]
            roles = tensor(
                [t.role for t in tracks], device=self.device, dtype=torch.long
            )
            cost = torch.where(
                roles[:, None] == rows[selected, 5][None], cost, torch.inf
            )
        else:
            cost = torch.empty(
                (len(tracks), len(indices)), device=self.device, dtype=torch.float64
            )
        matches, unmatched_tracks, unmatched_rows = linear_assignment(cost, threshold)
        return (
            [(tracks[a], indices[b]) for a, b in matches],
            [tracks[a] for a in unmatched_tracks],
            [indices[b] for b in unmatched_rows],
        )

    def update(self, rows, frame, *, cut=False):
        if cut:
            self.tracks.clear()
        self.tracks = [t for t in self.tracks if frame - t.last <= self.buffer]
        high = (
            torch.nonzero(rows[:, 4] >= self.config["track_high_thresh"])
            .flatten()
            .tolist()
        )
        low = (
            torch.nonzero(
                (rows[:, 4] > self.config["track_low_thresh"])
                & (rows[:, 4] < self.config["track_high_thresh"])
            )
            .flatten()
            .tolist()
        )
        confirmed = [t for t in self.tracks if t.activated]
        unconfirmed = [t for t in self.tracks if not t.activated]
        self.kalman.predict_many(confirmed)
        first, unmatched, remaining = self._associate(
            confirmed,
            rows,
            high,
            self.config["match_thresh"],
            self.config["fuse_score"],
        )
        second, missed, _ = self._associate(
            [t for t in unmatched if t.state == "tracked"], rows, low, 0.5
        )
        third, removed, remaining = self._associate(
            unconfirmed, rows, remaining, 0.7, self.config["fuse_score"]
        )
        assignments = [None] * len(rows)
        matches = first + second + third
        self.kalman.update_many(matches, rows)
        for track, index in matches:
            track.last, track.state, track.activated = frame, "tracked", True
            assignments[index] = track.id
        for track in missed:
            track.state = "lost"
        removed_ids = {t.id for t in removed}
        self.tracks = [t for t in self.tracks if t.id not in removed_ids]
        for index in remaining:
            if bool(rows[index, 4] < self.config["new_track_thresh"]):
                continue
            mean, covariance = self.kalman.initiate(xyah(rows[index, :4]))
            self.counter += 1
            track = Track(
                self.counter,
                int(rows[index, 5]),
                mean,
                covariance,
                frame,
                frame,
                "tracked",
                frame == 0 or cut,
            )
            self.tracks.append(track)
            if track.activated:
                assignments[index] = track.id
        # ByteTrack duplicate removal keeps the longer-lived track.
        active = [t for t in self.tracks if t.state == "tracked" and t.activated]
        lost = [t for t in self.tracks if t.state == "lost"]
        drop = set()
        if active and lost:
            overlaps = box_iou(
                torch.stack([xyxy(t.mean) for t in active]),
                torch.stack([xyxy(t.mean) for t in lost]),
            )
            for a, b in torch.nonzero(overlaps > 0.85).tolist():
                if active[a].role != lost[b].role:
                    continue
                drop.add(
                    lost[b].id
                    if frame - active[a].start > frame - lost[b].start
                    else active[a].id
                )
        self.tracks = [t for t in self.tracks if t.id not in drop]
        return assignments
