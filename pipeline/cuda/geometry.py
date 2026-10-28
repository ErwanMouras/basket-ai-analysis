"""CUDA homography, masked sparse Lucas–Kanade motion and court projection."""

import torch
import torch.nn.functional as F

from pipeline.cuda.core import project, resize, tensor


def normalize(matrix):
    if not bool(torch.isfinite(matrix).all()) or bool(matrix[2, 2].abs() < 1e-10):
        return None
    result = matrix / matrix[2, 2]
    return result if bool(torch.linalg.det(result).abs() >= 1e-12) else None


def dlt(first, second):
    """Hartley-normalized DLT, accepting a batch of point sets on CUDA."""
    single = first.ndim == 2
    if single:
        first, second = first[None], second[None]

    def prepare(points):
        centre = points.mean(1, keepdim=True)
        scale = tensor(
            2.0, device=str(points.device)
        ).sqrt() / torch.linalg.vector_norm(points - centre, dim=2).mean(1).clamp_min(
            1e-12
        )
        t = torch.eye(3, dtype=points.dtype, device=points.device).repeat(
            len(points), 1, 1
        )
        t[:, 0, 0] = t[:, 1, 1] = scale
        t[:, :2, 2] = -centre[:, 0] * scale[:, None]
        return (points - centre) * scale[:, None, None], t

    a, ta = prepare(first.double())
    b, tb = prepare(second.double())
    x, y = a.unbind(-1)
    u, v = b.unbind(-1)
    ones, zeros = torch.ones_like(x), torch.zeros_like(x)
    row1 = torch.stack((-x, -y, -ones, zeros, zeros, zeros, u * x, u * y, u), -1)
    row2 = torch.stack((zeros, zeros, zeros, -x, -y, -ones, v * x, v * y, v), -1)
    design = torch.stack((row1, row2), 2).flatten(1, 2)
    _, _, vh = torch.linalg.svd(design, full_matrices=first.shape[1] == 4)
    matrix = torch.linalg.inv(tb) @ vh[:, -1].reshape(-1, 3, 3) @ ta
    denominator = matrix[:, 2:3, 2:3]
    matrix = matrix / torch.where(
        denominator.abs() > 1e-12, denominator, torch.ones_like(denominator)
    )
    return matrix[0] if single else matrix


def ransac(first, second, threshold, *, samples=2048):
    from pipeline.cuda.kernels import four_point

    count = len(first)
    if count < 4:
        return None, torch.zeros(count, device=first.device, dtype=torch.bool)
    indices = (
        torch.rand((samples, count), device=first.device)
        .topk(4, dim=1, largest=False)
        .indices
    )
    models = four_point(first[indices], second[indices])
    valid = torch.isfinite(models).all((1, 2)) & (
        torch.linalg.det(models).abs() > 1e-12
    )
    projected = project(models, first[None].expand(samples, -1, -1))
    errors = torch.linalg.vector_norm(projected - second[None], dim=2)
    inliers = (errors <= threshold) & valid[:, None]
    score = inliers.sum(1).double() * 1e6 - torch.where(inliers, errors, 0).sum(1)
    best = score.argmax()
    keep = inliers[best]
    if int(keep.sum()) < 4:
        return None, keep
    model = normalize(dlt(first[keep], second[keep]))
    if model is not None:
        keep = (
            torch.linalg.vector_norm(project(model, first) - second, dim=1) <= threshold
        )
        if int(keep.sum()) >= 4:
            model = normalize(dlt(first[keep], second[keep]))
    return model, keep


def _hull_reference(points):
    order = points[:, 1].argsort(stable=True)
    order = order[points[order, 0].argsort(stable=True)].tolist()

    def side(indices):
        result = []
        for index in indices:
            while len(result) >= 2:
                a, b = (
                    points[result[-1]] - points[result[-2]],
                    points[index] - points[result[-1]],
                )
                if bool(a[0] * b[1] - a[1] * b[0] > 0):
                    break
                result.pop()
            result.append(index)
        return result

    result = side(order)[:-1] + side(reversed(order))[:-1]
    return points[tensor(result, device=str(points.device), dtype=torch.long)]


def hull(points):
    from pipeline.cuda.kernels import convex_hull

    return convex_hull(points)


def optical_flow(previous, current, points):
    from pipeline.cuda.kernels import lk_level

    source, target = [previous.contiguous()], [current.contiguous()]
    for _ in range(3):
        source.append(F.avg_pool2d(source[-1][None, None], 2)[0, 0])
        target.append(F.avg_pool2d(target[-1][None, None], 2)[0, 0])
    following = points
    valid = torch.ones(len(points), device=points.device, dtype=torch.bool)
    for level in (3, 2, 1, 0):
        following = lk_level(
            source[level],
            target[level],
            gradients(source[level]),
            points,
            following,
            valid,
            level,
        )
    return following, valid


def area(polygon):
    return (
        (
            polygon[:, 0] * polygon.roll(-1, 0)[:, 1]
            - polygon[:, 1] * polygon.roll(-1, 0)[:, 0]
        ).sum()
        / 2
    ).abs()


def polygon_distance(points, polygon):
    edges = polygon.roll(-1, 0) - polygon
    offset = points[:, None] - polygon[None]
    cross = edges[None, :, 0] * offset[:, :, 1] - edges[None, :, 1] * offset[:, :, 0]
    inside = (cross >= 0).all(1) | (cross <= 0).all(1)
    alpha = (offset * edges[None]).sum(-1) / edges.square().sum(-1).clamp_min(1e-12)[
        None
    ]
    nearest = polygon[None] + alpha.clamp(0, 1)[..., None] * edges[None]
    distance = torch.linalg.vector_norm(points[:, None] - nearest, dim=-1).amin(1)
    return torch.where(inside, distance, -distance)


def gray(image):
    weights = tensor(
        [0.299, 0.587, 0.114], device=str(image.device), dtype=torch.float32
    )
    return (image.float() * weights[:, None, None]).sum(0)


def gradients(image):
    kernel = (
        tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            device=str(image.device),
            dtype=torch.float32,
        )
        / 8
    )
    return F.conv2d(
        image[None, None], torch.stack((kernel, kernel.T))[:, None], padding=1
    )[0]


def sample(image, coordinates):
    height, width = image.shape
    grid = (coordinates + 0.5) / tensor(
        [width, height], device=str(image.device), dtype=torch.float32
    ) * 2 - 1
    return F.grid_sample(
        image[None, None], grid[None], align_corners=False, padding_mode="reflection"
    )[0, 0]


def features(image, mask):
    dx, dy = gradients(image)
    covariance = F.avg_pool2d(
        torch.stack((dx.square(), dy.square(), dx * dy))[None], 7, 1, 3
    )[0]
    a, b, c = covariance
    response = (a + b - ((a - b).square() + 4 * c.square()).sqrt()) / 2
    response = torch.where(mask, response, 0)
    local = F.max_pool2d(response[None, None], 21, 1, 10)[0, 0]
    response = torch.where(
        (response == local) & (response >= response.amax() * 0.01), response, 0
    )
    scores, indices = response.flatten().topk(min(500, response.numel()))
    indices = indices[scores > 1e-6]
    return torch.stack(
        (
            indices.remainder(image.shape[1]),
            indices.div(image.shape[1], rounding_mode="floor"),
        ),
        1,
    ).float()


def _optical_flow_reference(previous, current, points):
    source, target = [previous], [current]
    for _ in range(3):
        source.append(F.avg_pool2d(source[-1][None, None], 2)[0, 0])
        target.append(F.avg_pool2d(target[-1][None, None], 2)[0, 0])
    offsets = torch.arange(-10, 11, device=points.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
    patch = torch.stack((xx.flatten(), yy.flatten()), -1)
    following = points / 8
    valid = torch.ones(len(points), device=points.device, dtype=torch.bool)
    for level in (3, 2, 1, 0):
        if level < 3:
            following = following * 2
        anchors = points / (2**level)
        coordinates = anchors[:, None] + patch[None]
        reference = sample(source[level], coordinates)
        dx, dy = gradients(source[level])
        gx, gy = sample(dx, coordinates), sample(dy, coordinates)
        a, b, c = gx.square().sum(1), (gx * gy).sum(1), gy.square().sum(1)
        determinant = a * c - b.square()
        valid &= determinant > 1e-4
        safe = determinant.clamp_min(1e-4)
        active = valid.clone()
        step_sizes = tensor(
            [1.0, 0.5, 0.25, 0.125], device=str(points.device), dtype=torch.float32
        )
        point_indices = torch.arange(len(points), device=points.device)
        for _ in range(30):
            error = reference - sample(target[level], following[:, None] + patch[None])
            bx, by = (gx * error).sum(1), (gy * error).sum(1)
            delta = torch.stack(((c * bx - b * by) / safe, (a * by - b * bx) / safe), 1)
            delta = torch.where(active[:, None], delta, 0).clamp(-5, 5)
            # Per-point backtracking prevents a converged patch from drifting
            # when interpolation makes the linearized step overshoot. Every
            # accepted update must lower the actual photometric residual.
            trials = following[None] + step_sizes[:, None, None] * delta[None]
            coordinates = trials[:, :, None] + patch[None, None]
            sampled = sample(target[level], coordinates.flatten(0, 1)).reshape(
                4, len(points), -1
            )
            costs = (reference[None] - sampled).square().mean(2)
            best_cost, best = costs.min(0)
            improved = active & (best_cost < error.square().mean(1))
            following = torch.where(
                improved[:, None], trials[best, point_indices], following
            )
            active &= improved & (
                torch.linalg.vector_norm(delta, dim=1) * step_sizes[best] >= 0.01
            )
            if not bool(active.any()):
                break
    valid &= torch.isfinite(following).all(1)
    return following, valid


def motion(previous, current, mask, minimum):
    points = features(previous, mask)
    if len(points) < minimum:
        return None, {"inliers": 0, "reason": "insufficient_features"}
    following, good = optical_flow(previous, current, points)
    backward, back_good = optical_flow(current, previous, following)
    keep = good & back_good & (torch.linalg.vector_norm(points - backward, dim=1) < 1.5)
    keep &= (
        (following[:, 0] >= 0)
        & (following[:, 0] < current.shape[1])
        & (following[:, 1] >= 0)
        & (following[:, 1] < current.shape[0])
    )
    first, second = points[keep].double(), following[keep].double()
    if len(first) < minimum:
        return None, {"inliers": 0, "reason": "inconsistent_flow"}
    matrix, inliers = ransac(first, second, 2.5, samples=1024)
    report = {
        "inliers": inliers.sum(),
        "inlier_ratio": inliers.double().mean(),
        "reason": None,
    }
    if matrix is None:
        return None, {**report, "reason": "motion_fit_failed"}
    if bool((inliers.sum() < minimum) | (inliers.double().mean() < 0.65)):
        return None, {**report, "reason": "weak_motion_support"}
    if bool(area(hull(first[inliers])) < previous.numel() * 0.015):
        return None, {**report, "reason": "weak_motion_support"}
    h, w = previous.shape
    corners = tensor([[0, 0], [w, 0], [w, h], [0, h]], device=str(previous.device))
    moved = project(matrix, corners)
    ratio = area(moved) / (w * h)
    diagonal = tensor([w, h], device=str(previous.device)).norm()
    if not bool(torch.isfinite(moved).all()) or bool(
        (ratio <= 0.65)
        | (ratio >= 1.5)
        | (torch.linalg.vector_norm(moved - corners, dim=1).amax() > 0.3 * diagonal)
    ):
        return None, {**report, "reason": "implausible_motion"}
    return matrix, report


class Court:
    def __init__(self, config, model, device):
        self.config, self.model, self.device = config, model, device
        self.landmarks = (
            tensor(
                [
                    (0, 0),
                    (0, 3),
                    (0, 17),
                    (0, 33),
                    (0, 47),
                    (0, 50),
                    (47, 50),
                    (47, 0),
                    (19, 17),
                    (19, 33),
                    (94, 50),
                    (94, 47),
                    (94, 33),
                    (94, 17),
                    (94, 3),
                    (94, 0),
                    (75, 17),
                    (75, 33),
                ],
                device=device,
            )
            * 0.3048
        )
        self.dimensions = tensor([94, 50], device=device) * 0.3048
        self.excluded = tensor(config["excluded_regions"], device=device).reshape(-1, 4)
        length, width = self.dimensions.unbind()
        zero = torch.zeros((), device=device, dtype=torch.float64)
        self.corners = torch.stack(
            (
                torch.stack((zero, zero)),
                torch.stack((length, zero)),
                torch.stack((length, width)),
                torch.stack((zero, width)),
            )
        )
        self.state = self.previous = self.last_fit_time = None
        self.segment = 0
        self.last_fit_frame = None
        self.provenance = {
            "backend": "native_CUDA_normalized_DLT_RANSAC_LK",
            "device": device,
            "propagation": "masked_floor_forward_backward_pyramidal_LK_RANSAC",
            "numerical_equivalence": "same quality bounds; not OpenCV bit-identical",
        }

    def fit(self, keypoints, shape):
        cfg = self.config
        height, width = shape
        values = keypoints.double()
        valid = (
            (values[:, 2] >= cfg["keypoint_confidence"])
            & (values[:, 0] > 1)
            & (values[:, 0] < width - 1)
            & (values[:, 1] > 1)
            & (values[:, 1] < height - 1)
        )
        ids = []
        for index in values[:, 2].argsort(descending=True).tolist():
            if not bool(valid[index]):
                continue
            if ids and not bool(
                (
                    torch.linalg.vector_norm(values[ids, :2] - values[index, :2], dim=1)
                    > tensor(height, device=self.device) * 8 / 1080
                ).all()
            ):
                continue
            ids.append(index)
        report = {"candidate_ids": ids, "inlier_ids": [], "rejection": None}

        def reject(reason):
            return None, {**report, "rejection": reason}

        if len(ids) < cfg["min_inliers"]:
            return reject("insufficient_keypoints")
        world, pixels = self.landmarks[ids], values[ids, :2]
        if bool(area(hull(world)) < cfg["min_support_area_m2"]):
            return reject("degenerate_court_support")
        if bool(
            area(hull(pixels)) < tensor(height * width, device=self.device) * 0.002
        ):
            return reject("degenerate_image_support")
        matrix, inliers = ransac(
            world,
            pixels,
            tensor(height, device=self.device) * cfg["ransac_pixels"] / 1080,
            samples=3072,
        )
        report.update(
            inlier_ids=[ids[i] for i in torch.nonzero(inliers).flatten().tolist()],
            inlier_ratio=inliers.double().mean(),
        )
        if matrix is None:
            return reject("singular_homography")
        if bool(
            (inliers.sum() < cfg["min_inliers"])
            | (inliers.double().mean() < cfg["min_inlier_ratio"])
        ):
            return reject("insufficient_inliers")
        world, pixels = world[inliers], pixels[inliers]
        if bool(area(hull(world)) < cfg["min_support_area_m2"]):
            return reject("degenerate_inlier_support")
        error = (project(matrix, world) - pixels).square().sum(1).mean().sqrt()
        report["reprojection_rmse_px"] = error if bool(torch.isfinite(error)) else None
        if not bool(torch.isfinite(error)) or bool(
            error
            > tensor(height, device=self.device) * cfg["max_reprojection_pixels"] / 1080
        ):
            return reject("reprojection_error")
        denominator = (
            torch.cat((self.corners, torch.ones_like(self.corners[:, :1])), 1)
            @ matrix[2]
        )
        if bool(denominator.amin() * denominator.amax() <= 0):
            return reject("horizon_crosses_court")
        centre = world.mean(0)
        axes = project(
            matrix,
            centre[None] + tensor([[0, 0], [0.1, 0], [0, 0.1]], device=self.device),
        )
        dx, dy = axes[1] - axes[0], axes[2] - axes[0]
        if bool(
            (dx[0].abs() < 1e-3)
            | (dy[1] <= 0)
            | ((dx[0] * dy[1] - dx[1] * dy[0]).abs() < 0.01)
        ):
            return reject("unsupported_camera_orientation")
        reflected = bool(dx[0] < 0)
        if reflected:
            reflection = torch.eye(3, device=self.device, dtype=torch.float64)
            reflection[0, 0] = -1
            reflection[0, 2] = self.dimensions[0]
            matrix = normalize(matrix @ reflection)
            world = world.clone()
            world[:, 0] = self.dimensions[0] - world[:, 0]
        inverse = normalize(torch.linalg.inv(matrix))
        if inverse is None:
            return reject("singular_inverse")
        report["reflected_model_labels_x"] = reflected
        return {"matrix": inverse, "support": hull(world), "fit": report}, report

    def _mask_reference(self, image, detections, shape):
        height, width = shape
        yy, xx = torch.meshgrid(
            torch.arange(height, device=self.device),
            torch.arange(width, device=self.device),
            indexing="ij",
        )
        coordinates = torch.stack((xx.flatten(), yy.flatten()), 1).double()
        scale = tensor(
            [image.shape[2] / width, image.shape[1] / height], device=self.device
        )
        coordinates = coordinates * scale
        if self.state:
            polygon = project(torch.linalg.inv(self.state["matrix"]), self.corners)
            mask = polygon_distance(coordinates, polygon) >= 0
        else:
            mask = (
                coordinates[:, 1] >= tensor(image.shape[1], device=self.device) * 0.25
            )
        for row in detections:
            if bool(row[4] < 0.15):
                continue
            box = row[:4]
            excluded = (
                (coordinates[:, 0] >= box[0] - 8)
                & (coordinates[:, 0] <= box[2] + 8)
                & (coordinates[:, 1] >= box[1] - 8)
                & (coordinates[:, 1] <= box[3] + 8)
            )
            mask &= ~excluded
        for region in self.config["excluded_regions"]:
            bounds = tensor(region, device=self.device) * tensor(
                [image.shape[2], image.shape[1]] * 2, device=self.device
            )
            mask &= ~(
                (coordinates[:, 0] >= bounds[0])
                & (coordinates[:, 0] <= bounds[2])
                & (coordinates[:, 1] >= bounds[1])
                & (coordinates[:, 1] <= bounds[3])
            )
        return mask.reshape(height, width)

    def _mask(self, image, detections, shape):
        from pipeline.cuda.kernels import floor_mask

        polygon = (
            project(torch.linalg.inv(self.state["matrix"]), self.corners)
            if self.state
            else None
        )
        return floor_mask(polygon, detections, self.excluded, shape, image.shape[-2:])

    def update(self, image, detections, index, now, *, force_cut=False):
        from kornia.color import rgb_to_hsv

        cfg = self.config
        height, width = image.shape[-2:]
        scaled_width = min(width, cfg["motion_width"])
        scaled_height = round(height * scaled_width / width)
        current = gray(resize(image, scaled_height, scaled_width))
        thumbnail = resize(image, 36, 64)
        hsv = rgb_to_hsv(thumbnail / 255)
        bins = (hsv[0] / (2 * torch.pi) * 24).long().clamp(0, 23) * 16 + (
            hsv[1] * 16
        ).long().clamp(0, 15)
        histogram = torch.bincount(bins.flatten(), minlength=384).double()
        histogram /= histogram.sum()
        transformation = None
        motion_report = {"inliers": 0, "reason": "first_frame"}
        difference = tensor(0.0, device=self.device)
        hist_distance = tensor(0.0, device=self.device)
        correlation = None
        if self.previous:
            prev, prev_mask, prev_thumb, prev_hist = self.previous
            transformation, motion_report = motion(
                prev, current, prev_mask, cfg["motion_min_inliers"]
            )
            difference = (thumbnail - prev_thumb).abs().mean() / 255
            hist_distance = (
                (1 - (histogram * prev_hist).sqrt().sum()).clamp_min(0).sqrt()
            )
            a, b = gray(thumbnail).flatten(), gray(prev_thumb).flatten()
            a = a - a.mean()
            b = b - b.mean()
            denominator = a.norm() * b.norm()
            correlation = torch.where(
                denominator > 1e-6, (a * b).sum() / denominator, 0
            )
        flash = correlation is not None and bool(correlation > 0.75)
        auto = (
            self.previous is not None
            and cfg["auto_cuts"]
            and not flash
            and bool(
                (difference > cfg["cut_pixel_difference"])
                | (
                    (difference > 0.12)
                    & (hist_distance > cfg["cut_histogram_distance"])
                )
                | ((difference > 0.20) & (transformation is None))
            )
        )
        cut = index > 0 and (force_cut or auto)
        if cut:
            self.segment += 1
            self.state = None
            self.last_fit_time = None
            self.last_fit_frame = None
            transformation = None
        propagated = None
        if (
            self.state
            and transformation is not None
            and self.last_fit_time is not None
            and bool(now - self.last_fit_time <= cfg["max_propagation_seconds"])
        ):
            scale = tensor(
                [scaled_width / width, scaled_height / height, 1], device=self.device
            ).diag()
            full_motion = torch.linalg.inv(scale) @ transformation @ scale
            inverse = normalize(self.state["matrix"] @ torch.linalg.inv(full_motion))
            if inverse is not None:
                propagated = {**self.state, "matrix": inverse}
        keypoints = None
        fit_report = None
        candidate = None
        if index % cfg["detect_every"] == 0 or propagated is None:
            detected = self.model.predict(image, confidence=0.25, max_det=1)
            keypoints = (
                detected[0, 6:].reshape(18, 3)
                if len(detected)
                else torch.zeros((18, 3), device=self.device)
            )
            candidate, fit_report = self.fit(keypoints, (height, width))
            if candidate and propagated:
                anchors = project(
                    torch.linalg.inv(propagated["matrix"]), propagated["support"]
                )
                disagreement = torch.linalg.vector_norm(
                    project(candidate["matrix"], anchors) - propagated["support"], dim=1
                )
                if not bool(torch.isfinite(disagreement).all()) or bool(
                    disagreement.median() > cfg["max_temporal_jump_m"]
                ):
                    fit_report["rejection"] = "temporal_disagreement"
                    candidate = None
                elif cfg["smoothing_seconds"]:
                    world = self.landmarks[fit_report["inlier_ids"]].clone()
                    if fit_report["reflected_model_labels_x"]:
                        world[:, 0] = self.dimensions[0] - world[:, 0]
                    direct = project(torch.linalg.inv(candidate["matrix"]), world)
                    predicted = project(torch.linalg.inv(propagated["matrix"]), world)
                    alpha = -torch.expm1(
                        -(now - self.last_fit_time) / cfg["smoothing_seconds"]
                    )
                    smoothed = normalize(
                        dlt(world, predicted + alpha * (direct - predicted))
                    )
                    error = (
                        (
                            project(smoothed, world)
                            - keypoints[fit_report["inlier_ids"], :2]
                        )
                        .square()
                        .sum(1)
                        .mean()
                        .sqrt()
                        if smoothed is not None
                        else tensor(torch.inf, device=self.device)
                    )
                    valid_smoothing = False
                    if smoothed is not None:
                        denominators = (
                            torch.cat(
                                (self.corners, torch.ones_like(self.corners[:, :1])), 1
                            )
                            @ smoothed[2]
                        )
                        centre = world.mean(0)
                        axes = project(
                            smoothed,
                            centre[None]
                            + tensor([[0, 0], [0.1, 0], [0, 0.1]], device=self.device),
                        )
                        dx, dy = axes[1] - axes[0], axes[2] - axes[0]
                        valid_smoothing = bool(
                            (denominators.amin() * denominators.amax() > 0)
                            & (dx[0] > 1e-3)
                            & (dy[1] > 0)
                            & ((dx[0] * dy[1] - dx[1] * dy[0]).abs() > 0.01)
                        )
                    if valid_smoothing and bool(
                        torch.isfinite(error)
                        & (
                            error
                            <= tensor(height, device=self.device)
                            * cfg["max_reprojection_pixels"]
                            / 1080
                        )
                    ):
                        candidate["matrix"] = normalize(torch.linalg.inv(smoothed))
                        fit_report.update(
                            smoothing_applied=True,
                            smoothing_alpha=alpha,
                            stabilized_reprojection_rmse_px=error,
                        )
        self.state = candidate or propagated
        if candidate:
            self.last_fit_time = now.clone()
            self.last_fit_frame = index
        status = "fit" if candidate else "propagated" if propagated else "unavailable"
        self.previous = (
            current,
            self._mask(image, detections, current.shape),
            thumbnail,
            histogram,
        )
        return {
            "frame_index": index,
            "timestamp_seconds": now,
            "segment_id": self.segment,
            "status": status,
            "scene_cut": cut,
            "cut_reason": "explicit"
            if cut and force_cut
            else "automatic"
            if cut
            else None,
            "cut_pixel_difference": difference,
            "cut_histogram_distance": hist_distance,
            "cut_structure_correlation": correlation,
            "image_to_court": self.state["matrix"] if self.state else None,
            "support_polygon_m": self.state["support"] if self.state else None,
            "last_fit_frame": self.last_fit_frame,
            "fit_age_seconds": now - self.last_fit_time
            if self.last_fit_time is not None
            else None,
            "keypoints": keypoints,
            "fit_attempt": fit_report,
            "accepted_fit": self.state["fit"] if self.state else None,
            "motion": motion_report,
        }

    def _positions_reference(self, rows, pose, valid, record, shape):
        result = []
        for index, row in enumerate(rows):
            box = row[:4]
            point = torch.stack(((box[0] + box[2]) / 2, box[3])).double()
            method = "bbox_bottom_center"
            reason = None
            if bool(row[4] < self.config["min_player_confidence"]):
                reason = "low_player_confidence"
            elif bool(box[3] >= shape[0] - 2):
                reason = "feet_out_of_frame"
            if self.config["footpoint"] == "ankles" and bool(valid[index, 15:17].all()):
                ankles = pose[index, 15:17]
                if bool(
                    torch.isfinite(ankles).all()
                    & (ankles[:, 0] >= box[0]).all()
                    & (ankles[:, 0] <= box[2]).all()
                ):
                    point = ankles.mean(0).double()
                    method = "ankles_midpoint_approximation"
            xy = None
            if reason is None and not self.state:
                reason = "calibration_unavailable"
            if reason is None:
                xy = project(self.state["matrix"], point[None])[0]
                margin = self.config["court_margin_m"]
                if not bool(torch.isfinite(xy).all()):
                    reason = "projection_at_infinity"
                elif not bool(
                    ((xy >= -margin) & (xy <= self.dimensions + margin)).all()
                ):
                    reason = "outside_court"
                elif bool(
                    polygon_distance(xy[None], self.state["support"])[0]
                    < -self.config["max_extrapolation_m"]
                ):
                    reason = "outside_calibrated_support"
            result.append(
                {
                    "ground_point_px": point,
                    "ground_point_method": method,
                    "position_m": xy if reason is None else None,
                    "position_status": "projected" if reason is None else reason,
                }
            )
        return result

    def positions(self, rows, pose, valid, record, shape):
        if not len(rows):
            return []
        boxes = rows[:, :4]
        ankles = pose[:, 15:17]
        use_ankles = (
            valid[:, 15:17].all(1)
            & torch.isfinite(ankles).all((1, 2))
            & (ankles[:, :, 0] >= boxes[:, None, 0]).all(1)
            & (ankles[:, :, 0] <= boxes[:, None, 2]).all(1)
        )
        if self.config["footpoint"] != "ankles":
            use_ankles = torch.zeros_like(use_ankles)
        bottom = torch.stack(((boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]), 1)
        points = torch.where(use_ankles[:, None], ankles.mean(1), bottom).double()
        reasons = torch.where(
            rows[:, 4] < self.config["min_player_confidence"],
            1,
            torch.where(boxes[:, 3] >= shape[0] - 2, 2, 0),
        )
        xy = None
        if not self.state:
            reasons = torch.where(reasons == 0, 3, reasons)
        else:
            xy = project(self.state["matrix"], points)
            reasons = torch.where(
                (reasons == 0) & ~torch.isfinite(xy).all(1), 4, reasons
            )
            margin = self.config["court_margin_m"]
            inside = ((xy >= -margin) & (xy <= self.dimensions + margin)).all(1)
            reasons = torch.where((reasons == 0) & ~inside, 5, reasons)
            supported = (
                polygon_distance(xy, self.state["support"])
                >= -self.config["max_extrapolation_m"]
            )
            reasons = torch.where((reasons == 0) & ~supported, 6, reasons)
        labels = [
            "projected",
            "low_player_confidence",
            "feet_out_of_frame",
            "calibration_unavailable",
            "projection_at_infinity",
            "outside_court",
            "outside_calibrated_support",
        ]
        decisions = torch.stack((reasons, use_ankles.long()), 1).tolist()
        return [
            {
                "ground_point_px": points[i],
                "ground_point_method": "ankles_midpoint_approximation"
                if ankle
                else "bbox_bottom_center",
                "position_m": xy[i] if reason == 0 else None,
                "position_status": labels[reason],
            }
            for i, (reason, ankle) in enumerate(decisions)
        ]
