"""GPU tensor primitives and an executable ban on CPU tensor calculations."""

import json
import subprocess
from collections import Counter
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from training.common.provenance import file_hash


class CudaOnly(TorchDispatchMode):
    """Check both tensor inputs and outputs, including factories without inputs.

    Device copies and scalar/JSON exports are I/O, and are allowed. Computation
    on an exported CPU tensor is forbidden. This mode is active during every
    numerical stage, including final statistics, not just model inference.
    """

    def __init__(self):
        super().__init__()
        self.calls = Counter()
        self._operators = {}

    @staticmethod
    def _devices(value, result):
        # ATen arguments are tensors, tuples/lists, or simple keyword dicts.
        # Avoid rebuilding PyTree specifications at every dispatch.
        if isinstance(value, torch.Tensor):
            if value.is_cuda:
                result[0] = True
            else:
                result[1] = True
                if value.ndim:
                    result[2] = True
        elif isinstance(value, (tuple, list)):
            for item in value:
                CudaOnly._devices(item, result)
        elif isinstance(value, dict):
            for item in value.values():
                CudaOnly._devices(item, result)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        operator = self._operators.get(func)
        if operator is None:
            name = str(func)
            transfer = name.startswith(
                (
                    "aten._to_copy.",
                    "aten.to.",
                    "aten.copy_.",
                    "aten.detach.",
                    "aten.lift_fresh.",
                    "aten.lift_fresh_copy.",
                )
            )
            operator = (
                name,
                transfer,
                name.startswith(("aten.index.", "aten.index_put_.")),
            )
            self._operators[func] = operator
        name, transfer, index = operator
        devices = [False, False, False]  # CUDA, CPU, non-scalar CPU
        self._devices(args, devices)
        if kwargs:
            self._devices(kwargs, devices)
        index_metadata = index and devices[0]
        # A CPU zero-dimensional constant used as a CUDA kernel parameter
        # (e.g. context[:,0] = 95) does not perform host arithmetic.
        scalar_parameters = devices[0] and not devices[2]
        if not transfer and not index_metadata and not scalar_parameters and devices[1]:
            raise RuntimeError(f"GPU-only violation: CPU input to {name}")
        output = func(*args, **kwargs)
        devices = [False, False, False]
        if not transfer:
            self._devices(output, devices)
        if not transfer and devices[1]:
            # torch.tensor(..., device=cuda) first lifts host data, then copies
            # it. Arithmetic is still prohibited on those staging tensors.
            raise RuntimeError(f"GPU-only violation: CPU output from {name}")
        self.calls["tensor_dispatches"] += 1
        if transfer:
            self.calls["io_transfers"] += 1
        return output


def _constant_key(value):
    if type(value) in (int, float, bool):
        return value
    if isinstance(value, (tuple, list)) and len(value) <= 16:
        keys = tuple(_constant_key(item) for item in value)
        if all(key is not None for key in keys):
            return keys
    return None


@lru_cache(maxsize=512)
def _constant(value, device, dtype, stream_id):
    # stream_id is part of the cache key: reuse never crosses CUDA streams.
    return torch.as_tensor(value, device=device, dtype=dtype)


def tensor(value, *, device="cuda:0", dtype=torch.float64):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    if str(device).startswith("cuda:"):
        key = _constant_key(value)
        if key is not None and not torch.cuda.is_current_stream_capturing():
            stream_id = torch.cuda.current_stream(device).cuda_stream
            # Preserve fresh-allocation semantics: callers may mutate tensors.
            return _constant(key, str(device), dtype, stream_id).clone()
    return torch.as_tensor(value, device=device, dtype=dtype)


def export(value):
    """Pack CUDA tensors by dtype before exporting JSON, synchronizing once.

    List reconstruction on the host is serialization, including shape/index
    bookkeeping. All tensor flattening and packing takes place on CUDA.
    """
    groups, locations = {}, {}

    def collect(item):
        if isinstance(item, torch.Tensor) and item.is_cuda:
            if id(item) not in locations:
                key = (item.device, item.dtype)
                group = groups.setdefault(key, [[], 0])
                locations[id(item)] = (key, group[1], tuple(item.shape))
                group[0].append(item.reshape(-1))
                group[1] += item.numel()
        elif isinstance(item, dict):
            for entry in item.values():
                collect(entry)
        elif isinstance(item, (list, tuple)):
            for entry in item:
                collect(entry)

    collect(value)
    packed = {
        key: torch.cat(parts).detach().tolist() for key, (parts, _) in groups.items()
    }

    def restore(item):
        if isinstance(item, torch.Tensor):
            if not item.is_cuda:
                return item.detach().tolist()
            key, offset, shape = locations[id(item)]
            values = iter(packed[key][offset : offset + item.numel()])

            def shaped(dimensions):
                if not dimensions:
                    return next(values)
                return [shaped(dimensions[1:]) for _ in range(dimensions[0])]

            return shaped(shape)
        if isinstance(item, dict):
            return {key: restore(entry) for key, entry in item.items()}
        if isinstance(item, (list, tuple)):
            return [restore(entry) for entry in item]
        return item

    return restore(value)


def resize(image, height, width, *, mode="bilinear"):
    batch = image[None].float()
    return F.interpolate(
        batch,
        size=(height, width),
        mode=mode,
        align_corners=False if mode in ("bilinear", "bicubic") else None,
    )[0]


def letterbox(image, size, *, center=True, square=False):
    height, width = image.shape[-2:]
    ratio = min(size / height, size / width)
    h, w = round(height * ratio), round(width * ratio)
    target_h = size if square else ((h + 31) // 32) * 32
    target_w = size if square else ((w + 31) // 32) * 32
    top = round((target_h - h) / 2 - 0.1) if center else 0
    left = round((target_w - w) / 2 - 0.1) if center else 0
    out = F.pad(
        resize(image, h, w),
        (left, target_w - w - left, top, target_h - h - top),
        value=114,
    )
    return out, ratio, left, top


def box_iou(first, second):
    a = (first[:, 2:] - first[:, :2]).clamp_min(0).prod(1)
    b = (second[:, 2:] - second[:, :2]).clamp_min(0).prod(1)
    intersection = (
        (
            torch.minimum(first[:, None, 2:], second[None, :, 2:])
            - torch.maximum(first[:, None, :2], second[None, :, :2])
        )
        .clamp_min(0)
        .prod(2)
    )
    return intersection / (a[:, None] + b[None] - intersection).clamp_min(1e-12)


def project(matrix, points):
    homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), -1)
    result = homogeneous @ matrix.transpose(-1, -2)
    valid = torch.isfinite(result).all(-1) & (result[..., 2].abs() > 1e-9)
    return torch.where(valid[..., None], result[..., :2] / result[..., 2:], torch.nan)


class Video:
    """NVDEC and GPU RGB conversion; compressed packet reads remain host I/O."""

    def __init__(self, path, device="cuda:0", batch_size=4):
        import PyNvVideoCodec as nvc

        self.path = str(Path(path).resolve(strict=True))
        self.batch_size = batch_size
        container = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=avg_frame_rate,nb_frames,width,height,time_base:packet=pts,flags",
                    "-of",
                    "json",
                    self.path,
                ]
            )
        )
        metadata = container["streams"][0]
        numerator, denominator = metadata["avg_frame_rate"].split("/")
        self.fps = float(numerator) / float(
            denominator
        )  # container metadata, not video analysis
        self.decoder = nvc.SimpleDecoder(
            self.path,
            gpu_id=int(device.split(":")[1]),
            use_device_memory=True,
            output_color_type=nvc.OutputColorType.RGBP,
        )
        # SimpleDecoder 2.2.3 exposes decode-order packet counters as timestamp
        # on this OBS recording. Read presentation timestamps from the container
        # instead; demuxing does not decode pixels. Sorting/scaling happens CUDA.
        packets = container["packets"]
        packet_pts = tensor(
            [packet["pts"] for packet in packets], device=device, dtype=torch.int64
        )
        display = tensor(
            ["D" not in packet.get("flags", "") for packet in packets],
            device=device,
            dtype=torch.bool,
        )
        # AV_PKT_FLAG_DISCARD packets may be needed as codec references but
        # must not be presented. Container nb_frames can include them. NVDEC
        # handles reference decoding; count/map only presentation packets.
        pts = packet_pts[display].sort().values
        time_base = tensor(
            [int(value) for value in metadata["time_base"].split("/")], device=device
        )
        self.timestamps = pts.double() * time_base[0] / time_base[1]
        if (
            len(packets) != len(self.decoder)
            or not len(pts)
            or not bool((self.timestamps.diff() > 0).all())
            or bool(self.timestamps[0] < 0)
        ):
            raise ValueError(
                "Container PTS do not map uniquely to decoded presentation frames"
            )
        self.source = {
            "path": self.path,
            "sha256": file_hash(Path(self.path)),
            "nominal_fps": self.fps,
            "declared_frames": len(self.decoder),
            "presentation_frames": len(pts),
            "discarded_packets": len(packets) - len(pts),
            "decoded_frames": 0,
            "width": int(metadata["width"]),
            "height": int(metadata["height"]),
            "timestamp_policy": "container_non_discard_packet_PTS_presentation_order",
            "decoder": "NVIDIA_NVDEC",
            "pixel_memory": device,
            "color_conversion": "NVIDIA_device_RGBP",
        }

    def frames(self, limit=None, *, start=0):
        total = (
            self.source["presentation_frames"]
            if limit is None
            else min(limit, self.source["presentation_frames"])
        )
        # Sequential batches avoid random seeking and keep the decode queue bounded.
        self.decoder.seek_to_index(start)
        index = start
        while index < total:
            batch = self.decoder.get_batch_frames(min(self.batch_size, total - index))
            if not batch:
                raise RuntimeError(f"NVDEC ended at frame {index}, expected {total}")
            for frame in batch:
                image = torch.from_dlpack(frame)
                if not image.is_cuda:
                    raise RuntimeError(
                        "NVDEC returned a host frame; CPU fallback forbidden"
                    )
                # PTS is container metadata; seconds are calculated on CUDA.
                timestamp = self.timestamps[index]
                # Presentation timestamps were checked once at startup.
                self.source["decoded_frames"] = index + 1
                yield index, timestamp, image
                index += 1

    def close(self):
        self.decoder = None
