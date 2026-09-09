"""TOTNet-inspired components for the TrackNetV5 SDK (occlusion-aware training).

These are standalone (numpy/cv2/torch only) so they unit-test without the SDK.
Adapted from NBA/tracknet/totnet_ext.py at commit
b3a151edba8a10b6975a318f7861137f584e881b (source SHA-256:
b0f256d1d21bfbe2499f87922bfc06f542339be3dbf10be9d55a03d907c442c1).
See TRAINING.md for intentional geometry and RNG corrections.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn


class OcclusionAugment:
    """Randomly mask the ball patch in the current frame to force temporal inference.

    Insert after ``LoadMultiImagesFromPaths`` and before ``Resize`` so it operates at
    export resolution, where ``results["coords"]`` align with the loaded image. Use in
    the train pipeline only.

    Expects an H x W x C uint8 image array (``results[image_key]``); the ``noise`` fill
    generates random pixels assuming exactly 3 channels (``image.shape[2]``).
    """

    def __init__(
        self,
        prob: float = 0.5,
        patch_scale: int = 14,
        fill: str = "mean",
        only_when_visible: bool = True,
        image_key: str = "path",
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= prob <= 1.0:
            raise ValueError("prob must be in [0, 1]")
        if patch_scale < 1:
            raise ValueError("patch_scale must be >= 1")
        if fill not in {"mean", "black", "noise"}:
            raise ValueError("fill must be 'mean', 'black' or 'noise'")
        self.prob = prob
        self.patch_scale = patch_scale
        self.fill = fill
        self.only_when_visible = only_when_visible
        self.image_key = image_key
        self._rng = np.random.default_rng(seed)

    def __call__(self, results: dict) -> dict:
        if self._rng.random() >= self.prob:
            return results
        coords = results.get("coords")
        if not coords or len(coords) < 2:
            return results
        cx, cy = coords[1]
        if cx is None or cy is None or not (math.isfinite(cx) and math.isfinite(cy)):
            return results
        visibility = results.get("visibility")
        # When asked to mask only visible balls, skip whenever the current-frame
        # visibility is unknown (missing/short list) or explicitly not visible.
        if self.only_when_visible and (
            not visibility or len(visibility) < 2 or int(visibility[1]) == 0
        ):
            return results
        image = results[self.image_key]
        height, width = image.shape[:2]
        cx_i, cy_i = round(cx), round(cy)
        x1 = max(0, cx_i - self.patch_scale)
        y1 = max(0, cy_i - self.patch_scale)
        x2 = min(width, cx_i + self.patch_scale + 1)
        y2 = min(height, cy_i + self.patch_scale + 1)
        if x2 <= x1 or y2 <= y1:
            return results
        if self.fill == "black":
            image[y1:y2, x1:x2] = 0
        elif self.fill == "mean":
            image[y1:y2, x1:x2] = image.mean(axis=(0, 1)).astype(image.dtype)
        else:  # noise
            image[y1:y2, x1:x2] = self._rng.integers(
                0, 256, size=(y2 - y1, x2 - x1, image.shape[2]), dtype=np.uint8
            )
        results[self.image_key] = image
        return results


class VisibilityWeightedTrackNetLoss(nn.Module):
    """TrackNetV2 WBCE focal loss, up-weighting frames whose target contains a ball.

    Reproduces the SDK ``TrackNetV2Loss`` element-wise loss, then scales each
    ``(sample, frame)`` by ``occluded_weight`` when that frame's target heatmap has a
    ball (max pixel == 255). ``occluded_weight == 1.0`` reproduces TrackNetV2Loss.

    Note: ``occluded_weight`` up-weights ALL ball-present frames equally — both
    fully-visible and occluded-with-position. It does not, on its own, distinguish
    occluded from visible frames; pair it with ``OcclusionAugment`` (which masks the
    ball in the input while keeping the target blob) so the up-weighted frames become
    the hard, effectively-occluded cases.
    """

    def __init__(self, occluded_weight: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        if occluded_weight <= 0:
            raise ValueError("occluded_weight must be > 0")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum' or 'none'")
        self.occluded_weight = occluded_weight
        self.reduction = reduction

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        # `logits` are already post-sigmoid predictions, matching the SDK TrackNetV2Loss convention.
        prob = logits
        y = torch.where(targets == 255, 1.0, 0.0)
        pos_weight = (1.0 - prob).pow(2)
        neg_weight = prob.pow(2)
        eps = 1e-6
        prob = torch.clamp(prob, eps, 1.0 - eps)
        per_elem = -(
            pos_weight * y * torch.log(prob + eps)
            + neg_weight * (1.0 - y) * torch.log(1.0 - prob + eps)
        )
        has_ball = (targets == 255).flatten(start_dim=2).any(dim=2)  # [B, F]
        frame_weight = has_ball.to(per_elem.dtype) * (self.occluded_weight - 1.0) + 1.0
        per_elem = per_elem * frame_weight[:, :, None, None]
        if self.reduction == "mean":
            return per_elem.mean()
        if self.reduction == "sum":
            return per_elem.sum()
        return per_elem


_FRAME_KEYS = ("path_prev", "path", "path_next")


class TripletPhotometricAugment:
    """Photometric jitter applied coherently to the 3 frames of a triplet.

    One parameter draw per triplet (same brightness/contrast/hue/blur/jpeg for
    the 3 frames, mimicking a camera/lighting change); gaussian noise is drawn
    per frame (sensor noise is temporally independent). Coordinates are
    untouched, so pre-rendered GT heatmaps stay valid. Train pipeline only,
    right after ``LoadMultiImagesFromPaths``.

    Each sub-augmentation is a dict with at least ``prob``; ranges as listed in
    ``training/ball/configs/ball_augmentation.yaml``.
    """

    def __init__(
        self,
        brightness: dict | None = None,
        contrast: dict | None = None,
        hue_saturation: dict | None = None,
        gaussian_noise: dict | None = None,
        motion_blur: dict | None = None,
        jpeg: dict | None = None,
        seed: int | None = None,
    ) -> None:
        import cv2  # local import to keep module import light

        self._cv2 = cv2
        self.brightness = brightness
        self.contrast = contrast
        self.hue_saturation = hue_saturation
        self.gaussian_noise = gaussian_noise
        self.motion_blur = motion_blur
        self.jpeg = jpeg
        self._rng = np.random.default_rng(seed)

    def _fires(self, sub: dict | None) -> bool:
        return sub is not None and self._rng.random() < float(sub.get("prob", 0.0))

    def __call__(self, results: dict) -> dict:
        cv2 = self._cv2
        frames = {key: results[key] for key in _FRAME_KEYS if key in results}
        if not frames:
            return results

        if self._fires(self.brightness):
            delta = self._rng.uniform(-1.0, 1.0) * float(self.brightness["max_delta"])  # type: ignore[index]
            for key, img in frames.items():
                frames[key] = np.clip(img.astype(np.float32) + delta, 0, 255).astype(
                    np.uint8
                )

        if self._fires(self.contrast):
            low, high = self.contrast["range"]  # type: ignore[index]
            factor = self._rng.uniform(float(low), float(high))
            for key, img in frames.items():
                shifted = (img.astype(np.float32) - 128.0) * factor + 128.0
                frames[key] = np.clip(shifted, 0, 255).astype(np.uint8)

        if self._fires(self.hue_saturation):
            hue_delta = self._rng.uniform(-1.0, 1.0) * float(
                self.hue_saturation.get("hue_delta_deg", 0.0)  # type: ignore[union-attr]
            )
            sat_low, sat_high = self.hue_saturation.get("saturation_range", (1.0, 1.0))  # type: ignore[union-attr]
            sat_factor = self._rng.uniform(float(sat_low), float(sat_high))
            # OpenCV hue is in [0, 180) (degrees / 2).
            hue_shift = hue_delta / 2.0
            for key, img in frames.items():
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180.0
                hsv[..., 1] = np.clip(hsv[..., 1] * sat_factor, 0, 255)
                frames[key] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        if self._fires(self.motion_blur):
            k_low, k_high = self.motion_blur["kernel_range"]  # type: ignore[index]
            ksize = (
                int(self._rng.integers(int(k_low), int(k_high) + 1)) | 1
            )  # force odd
            angle = self._rng.uniform(0.0, 180.0)
            kernel = np.zeros((ksize, ksize), dtype=np.float32)
            kernel[ksize // 2, :] = 1.0
            center = ((ksize - 1) / 2.0, (ksize - 1) / 2.0)
            rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
            kernel = cv2.warpAffine(kernel, rotation, (ksize, ksize))
            kernel /= max(kernel.sum(), 1e-6)
            for key, img in frames.items():
                frames[key] = cv2.filter2D(img, -1, kernel)

        if self._fires(self.gaussian_noise):
            s_low, s_high = self.gaussian_noise["sigma_range"]  # type: ignore[index]
            sigma = self._rng.uniform(float(s_low), float(s_high))
            for key, img in frames.items():
                noise = self._rng.normal(0.0, sigma, size=img.shape)
                frames[key] = np.clip(img.astype(np.float32) + noise, 0, 255).astype(
                    np.uint8
                )

        if self._fires(self.jpeg):
            q_low, q_high = self.jpeg["quality_range"]  # type: ignore[index]
            quality = int(self._rng.integers(int(q_low), int(q_high) + 1))
            params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            for key, img in frames.items():
                ok, encoded = cv2.imencode(".jpg", img[..., ::-1], params)
                if ok:
                    frames[key] = cv2.imdecode(encoded, cv2.IMREAD_COLOR)[..., ::-1]

        for key, img in frames.items():
            results[key] = img
        return results


class TripletHorizontalFlip:
    """Horizontal flip of the concatenated input, the GT heatmaps and coords.

    Must run AFTER ``ConcatChannels`` (image is H x W x 3F numpy) and AFTER
    ``LoadAndFormatMultiTargets`` (target is a torch [F, H, W] tensor), so all
    three representations flip together. ``coord_width`` is the width of the
    coordinate space of ``results["coords"]`` (the export resolution).
    """

    def __init__(
        self,
        prob: float = 0.5,
        coord_width: int | None = None,
        image_key: str = "image",
        target_key: str = "target",
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= prob <= 1.0:
            raise ValueError("prob must be in [0, 1]")
        self.prob = prob
        self.coord_width = coord_width
        self.image_key = image_key
        self.target_key = target_key
        self._rng = np.random.default_rng(seed)

    def __call__(self, results: dict) -> dict:
        if self._rng.random() >= self.prob:
            return results
        image = results.get(self.image_key)
        if image is not None:
            results[self.image_key] = np.ascontiguousarray(image[:, ::-1, :])
        target = results.get(self.target_key)
        if target is not None:
            results[self.target_key] = torch.flip(target, dims=[2])
        coords = results.get("coords")
        width = self.coord_width
        if coords and width:
            flipped = []
            for point in coords:
                x, y = point
                if (
                    x is None
                    or y is None
                    or not (math.isfinite(x) and math.isfinite(y))
                ):
                    flipped.append(point)
                else:
                    flipped.append((float(width - 1) - float(x), float(y)))
            results["coords"] = flipped
        return results


class TripletRandomZoomCrop:
    """Random zoom-in crop (then resize back) of input, targets and coords.

    Simulates closer camera framings. The crop keeps the current-frame ball
    inside when its position is known. Must run AFTER ``ConcatChannels`` and
    ``LoadAndFormatMultiTargets``; coordinates are in the export resolution
    (``coord_width`` x ``coord_height``). Balls of context frames falling
    outside the crop get visibility 0 and NaN coords.
    """

    def __init__(
        self,
        prob: float = 0.3,
        scale_range: tuple[float, float] = (0.7, 1.0),
        coord_width: int | None = None,
        coord_height: int | None = None,
        image_key: str = "image",
        target_key: str = "target",
        seed: int | None = None,
    ) -> None:
        import cv2

        if not 0.0 <= prob <= 1.0:
            raise ValueError("prob must be in [0, 1]")
        low, high = float(scale_range[0]), float(scale_range[1])
        if not 0.2 <= low <= high <= 1.0:
            raise ValueError("scale_range must satisfy 0.2 <= low <= high <= 1.0")
        self._cv2 = cv2
        self.prob = prob
        self.scale_range = (low, high)
        self.coord_width = coord_width
        self.coord_height = coord_height
        self.image_key = image_key
        self.target_key = target_key
        self._rng = np.random.default_rng(seed)

    def __call__(self, results: dict) -> dict:
        if self._rng.random() >= self.prob:
            return results
        image = results.get(self.image_key)
        target = results.get(self.target_key)
        if (
            image is None
            or target is None
            or not self.coord_width
            or not self.coord_height
        ):
            return results
        scale = self._rng.uniform(*self.scale_range)
        if scale >= 0.999:
            return results

        coords = list(results.get("coords") or [])
        visibility = list(results.get("visibility") or [])
        # Relative position of the current-frame ball, to keep it inside the crop.
        anchor = None
        if len(coords) >= 2:
            x, y = coords[1]
            if (
                x is not None
                and y is not None
                and math.isfinite(x)
                and math.isfinite(y)
                and (len(visibility) < 2 or int(visibility[1]) != 0)
            ):
                anchor = (float(x) / self.coord_width, float(y) / self.coord_height)

        def _offset(rel: float | None) -> float:
            limit = 1.0 - scale
            if rel is None:
                return self._rng.uniform(0.0, limit)
            low = max(0.0, rel - scale)
            high = min(limit, rel)
            if high < low:
                return min(max(rel, 0.0), limit)
            return self._rng.uniform(low, high)

        ox = _offset(anchor[0] if anchor else None)
        oy = _offset(anchor[1] if anchor else None)

        cv2 = self._cv2
        height, width = image.shape[:2]
        x0, y0 = round(ox * width), round(oy * height)
        x1, y1 = round((ox + scale) * width), round((oy + scale) * height)
        crop = image[y0:y1, x0:x1]
        results[self.image_key] = cv2.resize(crop, (width, height))

        target_np = target.numpy()
        flipped_frames = []
        t_frames, t_height, t_width = target_np.shape
        tx0, ty0 = round(ox * t_width), round(oy * t_height)
        tx1, ty1 = round((ox + scale) * t_width), round((oy + scale) * t_height)
        for frame_idx in range(t_frames):
            crop_t = target_np[frame_idx, ty0:ty1, tx0:tx1]
            flipped_frames.append(
                cv2.resize(crop_t, (t_width, t_height), interpolation=cv2.INTER_NEAREST)
            )
        results[self.target_key] = torch.from_numpy(
            np.stack(flipped_frames, axis=0)
        ).to(target.dtype)

        new_coords = []
        new_visibility = list(visibility) if visibility else [1] * len(coords)
        for idx, point in enumerate(coords):
            x, y = point
            if x is None or y is None or not (math.isfinite(x) and math.isfinite(y)):
                new_coords.append(point)
                continue
            rel_x = float(x) / self.coord_width * width - x0
            rel_y = float(y) / self.coord_height * height - y0
            if not (0.0 <= rel_x < x1 - x0 and 0.0 <= rel_y < y1 - y0):
                new_coords.append((float("nan"), float("nan")))
                if idx < len(new_visibility):
                    new_visibility[idx] = 0
                continue
            new_coords.append(
                (
                    rel_x / (x1 - x0) * self.coord_width,
                    rel_y / (y1 - y0) * self.coord_height,
                )
            )
        if coords:
            results["coords"] = new_coords
            results["visibility"] = new_visibility
            # A cropped-out position must not leave a partial positive target.
            for index, visible in enumerate(new_visibility):
                if not visible:
                    results[self.target_key][index].zero_()
        return results


class TripletAugmentation:
    """NBA transform order, with deterministic independent worker/epoch streams."""

    def __init__(self, config):
        self.seed = config["seed"]
        self.epoch = 0
        self.stream = None
        options = config["augmentation"]
        self.photometric = TripletPhotometricAugment(
            **{
                key: value
                for key, value in options.items()
                if key not in ("occlusion", "horizontal_flip", "zoom_crop")
            }
        )
        occlusion = dict(options["occlusion"])
        if not config["geometry"]["occlusion_augmentation"]:
            occlusion["prob"] = 0.0
        self.occlusion = OcclusionAugment(**occlusion)
        self.flip = TripletHorizontalFlip(**options["horizontal_flip"])
        self.crop = TripletRandomZoomCrop(**options["zoom_crop"])
        self.input_width = config["geometry"]["input_width"]
        self.input_height = config["geometry"]["input_height"]

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.stream = None

    def before_resize(self, results):
        worker = torch.utils.data.get_worker_info()
        stream = (self.epoch, worker.id if worker else 0)
        if self.stream != stream:
            for index, transform in enumerate(
                (self.photometric, self.occlusion, self.flip, self.crop)
            ):
                transform._rng = np.random.default_rng(
                    np.random.SeedSequence([self.seed, *stream, index])
                )
            self.stream = stream
        height, width = results["path"].shape[:2]
        results = self.photometric(results)
        results = self.occlusion(results)
        # Transform coordinates to the same grid as images/targets before geometry augmentation.
        results["coords"] = [
            (x * self.input_width / width, y * self.input_height / height)
            for x, y in results["coords"]
        ]
        return results

    def after_resize(self, results):
        self.flip.coord_width = self.input_width
        self.crop.coord_width = self.input_width
        self.crop.coord_height = self.input_height
        return self.crop(self.flip(results))
