"""CUDA torso colors, segmentation-aware OCR crops and online numeric votes."""

import torch
import torch.nn.functional as F

from pipeline.cuda.core import resize, tensor
from pipeline.cuda.geometry import gray
from pipeline.numeric import CudaCounter


def _dominant_color_reference(image, foreground=None):
    from kornia.color import rgb_to_hsv

    height, width = image.shape[-2:]
    if height < 2 or width < 2:
        return None
    sy, sx = height // 10, width // 10
    core = image[:, sy : height - sy, sx : width - sx].float() / 255
    hsv = rgb_to_hsv(core)
    if foreground is not None:
        keep = foreground[sy : height - sy, sx : width - sx]
        if bool((keep.sum() < 16) | (keep.float().mean() < 0.5)):
            return None
        hsv = hsv[:, keep]
    hue = hsv[0] * (180 / (2 * torch.pi))
    saturation = hsv[1] * 255
    value = hsv[2] * 255
    colored = (saturation >= 75) & (value >= 50)
    zero = tensor(0.0, device=str(image.device))
    if bool(colored.float().mean() >= 0.7):
        bins = (hue[colored] / 10).floor().long().clamp(0, 17)
        centre = torch.bincount(bins.flatten(), minlength=18).argmax()
        # Match the original's circular three-bin support test.
        inliers = (((hue / 10).floor() - centre + 1).remainder(18) <= 2) & colored
        if bool(inliers.sum() / colored.sum() >= 0.8):
            angle = hue[inliers] * (2 * torch.pi / 180)
            result = (
                torch.atan2(angle.sin().mean(), angle.cos().mean()).remainder(
                    2 * torch.pi
                )
                * 180
                / (2 * torch.pi)
            )
            return "hue", result
    if bool((value <= 85).float().mean() >= 0.7):
        return "dark", zero
    if bool(((value >= 175) & (saturation <= 60)).float().mean() >= 0.7):
        return "light", zero
    return None


def dominant_color(image, foreground=None):
    from pipeline.cuda.kernels import torso_color

    height, width = image.shape[-2:]
    if height < 2 or width < 2:
        return None
    if (height - 2 * (height // 10)) * (width - 2 * (width // 10)) > 32768:
        return _dominant_color_reference(image, foreground)
    result = torso_color(image, foreground)
    kind = int(result[0])  # final category/dispatch metadata, not color arithmetic
    return (("hue", "dark", "light")[kind - 1], result[1]) if kind else None


def color_distance(first, second):
    if first[0] != second[0]:
        return torch.full_like(first[1], 180)
    if first[0] != "hue":
        return torch.zeros_like(first[1])
    difference = (first[1] - second[1]).abs()
    return torch.minimum(difference, 180 - difference)


def torso(box, points, valid):
    wh = box[2:] - box[:2]
    if bool(valid[[5, 6, 11, 12]].all()):
        corners = points[[5, 6, 11, 12]]
        lo = corners.amin(0)
        hi = corners.amax(0)
        span = hi - lo
        return torch.cat(
            (
                lo
                + span * tensor([0.10, 0.12], device=str(box.device), dtype=box.dtype),
                hi
                - span * tensor([0.10, 0.08], device=str(box.device), dtype=box.dtype),
            )
        )
    return torch.cat(
        (
            box[:2]
            + wh * tensor([0.22, 0.20], device=str(box.device), dtype=box.dtype),
            box[:2]
            + wh * tensor([0.78, 0.56], device=str(box.device), dtype=box.dtype),
        )
    )


def bounds(box, shape):
    roi = box.round().long()
    roi[[0, 2]] = roi[[0, 2]].clamp(0, shape[1])
    roi[[1, 3]] = roi[[1, 3]].clamp(0, shape[0])
    return roi, roi.tolist()


def candidate(image, rows, index, points, valid, masks, config, variant):
    row = rows[index]
    box = row[:4]
    wh = box[2:] - box[:2]
    device = str(image.device)
    if bool(row[4] < config["min_detection_confidence"]):
        return None
    if variant == "lower":
        roi = torch.cat(
            (
                box[:2] + wh * tensor([0.24, 0.30], device=device, dtype=box.dtype),
                box[:2] + wh * tensor([0.76, 0.53], device=device, dtype=box.dtype),
            )
        )
        region = "bbox_number"
    else:
        roi = torso(box, torch.zeros_like(points), torch.zeros_like(valid))
        region = "bbox_number_upper"
    if bool(valid[[5, 6, 11, 12]].all()):
        corners = points[[5, 6, 11, 12]]
        lo = corners.amin(0)
        hi = corners.amax(0)
        span = hi - lo
        if bool((span[0] < 0.15 * wh[0]) | (span[1] < 0.12 * wh[1])) or not bool(
            ((lo >= box[:2]) & (hi <= box[2:])).all()
        ):
            return None
        if variant == "center":
            roi = torch.cat(
                (
                    lo + span * tensor([0.05, 0.37], device=device, dtype=box.dtype),
                    lo + span * tensor([0.95, 0.78], device=device, dtype=box.dtype),
                )
            )
            region = "pose_number_center"
        elif variant == "upper":
            roi = torso(box, points, valid)
            region = "pose_number_upper"
        else:
            roi = torch.cat(
                (
                    lo + span * tensor([0.1, 0.48], device=device, dtype=box.dtype),
                    lo + span * tensor([0.9, 1.12], device=device, dtype=box.dtype),
                )
            )
            region = "pose_number"
            size = roi[2:] - roi[:2]
            if bool(
                (size[0] < config["min_crop_width"])
                | (size[1] < config["min_crop_height"])
            ):
                roi = torso(box, points, valid)
                region = "pose_number_upper"
    elif variant == "center":
        return None
    roi, (a, b, c, d) = bounds(roi, image.shape[-2:])
    size = roi[2:] - roi[:2]
    if bool(
        (size[0] < config["min_crop_width"]) | (size[1] < config["min_crop_height"])
    ):
        return None
    area = size.prod().float()
    foreground = masks[index, b:d, a:c] if masks is not None else None
    coverage = foreground.float().mean() if foreground is not None else None
    usable = coverage is not None and bool(coverage >= config["min_mask_coverage"])
    lo = torch.maximum(rows[:, :2], roi[:2])
    hi = torch.minimum(rows[:, 2:4], roi[2:])
    overlaps = (hi - lo).clamp_min(0).prod(1) / area
    if masks is not None and usable:
        overlaps = torch.where(
            overlaps > 0, masks[:, b:d, a:c].float().mean((1, 2)), overlaps
        )
    overlaps[index] = 0
    overlaps = torch.where(
        rows[:, 4] >= config["min_detection_confidence"], overlaps, 0
    )
    overlap = overlaps.amax() if len(rows) else tensor(0.0, device=device)
    if bool(overlap > config["max_overlap"]):
        return None
    crop = image[:, b:d, a:c].float()
    scale = torch.minimum(
        tensor(1.0, device=device), tensor(160.0, device=device) / size.amax()
    )
    small = resize(
        crop,
        max(1, int((size[1] * scale).round())),
        max(1, int((size[0] * scale).round())),
    )
    grayscale = gray(small)
    laplace = tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=device, dtype=torch.float32
    )[None, None]
    sharpness = F.conv2d(
        F.pad(grayscale[None, None], (1, 1, 1, 1), mode="reflect"), laplace
    ).var(correction=0)
    contrast = grayscale.std(correction=0)
    if bool(
        (sharpness < config["min_sharpness"]) | (contrast < config["min_contrast"])
    ):
        return None
    quality = (
        0.4 * (sharpness / 120).clamp_max(1)
        + 0.3 * (contrast / 48).clamp_max(1)
        + 0.3 * (area.sqrt() / 96).clamp_max(1)
    ) * (1 - overlap)
    policy = config["mask_policy"]
    if policy != "disabled":
        if not usable and policy == "require_mask":
            return None
        if usable:
            crop = torch.where(foreground[None], crop, 127)
    return {
        "image": resize(crop, 32, 128, mode="bicubic").clamp(0, 255),
        "bbox": roi,
        "region": region,
        "quality": quality,
        "sharpness": sharpness,
        "contrast": contrast,
        "overlap": overlap,
        "mask_audit": {
            "policy": policy,
            "status": "applied" if usable and policy != "disabled" else "raw_fallback",
            "coverage": coverage,
        },
    }


def majority(counter, minimum=3, share=0.75):
    if not counter:
        return None, None
    total = torch.stack(list(counter.values())).sum()
    key, count = counter.most_common(1)[0]
    ratio = count / total
    return (
        key
        if bool((total >= minimum) & (count > total / 2) & (ratio >= share))
        else None
    ), ratio


class Evidence:
    def __init__(self, config, roster, reader, device):
        self.config, self.roster, self.reader, self.device = (
            config,
            roster,
            reader,
            device,
        )
        self.teams = roster.payload["teams"]
        self.signatures = []
        for team in self.teams:
            rgb = tensor(
                [int(team["uniform_color"][i : i + 2], 16) for i in (1, 3, 5)],
                device=device,
                dtype=torch.float32,
            )
            self.signatures.append(
                dominant_color(rgb[:, None, None].expand(3, 32, 128))
            )
        self.states = {}

    def update(self, image, rows, tracks, segment, index, now, points, valid, masks):
        data = [None] * len(rows)
        reads = []
        jobs = []
        crop_images = []
        for n, row in enumerate(rows):
            if int(row[5]) != 2 or tracks[n] is None:
                continue
            _roi, (a, b, c, d) = bounds(
                torso(row[:4], points[n], valid[n]), image.shape[-2:]
            )
            signature = (
                dominant_color(
                    image[:, b:d, a:c],
                    masks[n, b:d, a:c] if masks is not None else None,
                )
                if c > a and d > b
                else None
            )
            matches = [
                k
                for k, value in enumerate(self.signatures)
                if signature and value and bool(color_distance(signature, value) <= 6)
            ]
            team = matches[0] if len(matches) == 1 else None
            key = (segment, tracks[n])
            state = self.states.get(key)
            if state is None or state["last"] != index - 1:
                state = {
                    "start": index,
                    "team": team,
                    "last_color": team,
                    "streak": tensor(0.0, device=self.device),
                    "votes": CudaCounter(device=self.device),
                    "confirmed": False,
                }
                self.states[key] = state
            if team is not None:
                state["streak"] = (
                    state["streak"] + 1
                    if state["last_color"] == team
                    else tensor(1.0, device=self.device)
                )
                state["last_color"] = team
                if team != state["team"] and bool(state["streak"] >= 3):
                    state.update(
                        start=index,
                        team=team,
                        votes=CudaCounter(device=self.device),
                        confirmed=False,
                    )
            state["last"] = index
            data[n] = {
                "team_id": self.teams[team]["team_id"] if team is not None else None,
                "number": None,
                "confidence": None,
            }
            if state["confirmed"] or (index - state["start"]) % 2:
                continue
            offsets = []
            for variant in ("center", "upper", "lower"):
                crop = candidate(
                    image, rows, n, points[n], valid[n], masks, self.config, variant
                )
                if crop is not None:
                    offsets.append((len(crop_images), crop))
                    crop_images.append(crop["image"])
            if offsets:
                jobs.append((n, key, state, offsets))
        if crop_images:
            tokens, confidence = self.reader.read(torch.stack(crop_images))
            for n, key, state, offsets in jobs:
                choices = tensor(
                    [offset for offset, _ in offsets],
                    device=self.device,
                    dtype=torch.long,
                )
                choice = int(confidence[choices].argmax())
                offset, crop = offsets[choice]
                token = tokens[offset]
                probability = confidence[offset]
                codes = token.tolist()
                alphabet = {i: str(i - 1) for i in range(1, 11)}
                number = (
                    alphabet.get(codes[0], "")
                    if codes[1] == 0
                    else alphabet.get(codes[0], "") + alphabet.get(codes[1], "")
                )
                allowed = self.roster.allows(number) and bool(probability > 0)
                accepted = allowed and bool(probability >= 0.8)
                if accepted:
                    state["votes"][number] = state["votes"][number] + 1
                winner, _ = majority(state["votes"])
                if winner is not None:
                    state["confirmed"] = True
                data[n].update(
                    number=number if allowed else None, confidence=probability
                )
                reads.append(
                    {
                        "schema_version": 1,
                        "decoder": "digits_only_v4",
                        "sampling": "CUDA_every_2_frames_until_majority",
                        "segment_id": segment,
                        "track_id": key[1],
                        "frame_index": index,
                        "processed_frame": index,
                        "timestamp_seconds": now,
                        "crop_bbox": crop["bbox"],
                        "region": crop["region"],
                        "quality": crop["quality"],
                        "mask_audit": crop["mask_audit"],
                        "text": number,
                        "raw_number": number if allowed else None,
                        "number": number if allowed else None,
                        "confidence": probability,
                        "status": "vote" if accepted else "rejected",
                        "rejection_reason": None
                        if accepted
                        else "not_in_roster"
                        if not allowed
                        else "low_ocr_confidence",
                    }
                )
        # State TTL is frame/ID bookkeeping; avoid retaining inference state for a full match.
        if index % 300 == 0:
            self.states = {
                key: value
                for key, value in self.states.items()
                if key[0] == segment and index - value["last"] < 300
            }
        return data, reads
