"""Execute the pinned RTMPose ONNX graph using CUDA Torch operators only.

This is a small interpreter for the exact, hash-pinned graph, not an ONNX
Runtime provider policy. Unknown operators fail before inference. Shape
metadata is exported for allocation; tensor arithmetic stays on CUDA.
"""

from collections import Counter
from pathlib import Path
from typing import ClassVar

import torch
import torch.nn.functional as F

from pipeline.cuda.core import tensor
from pipeline.cuda.graphs import CapturedCall
from training.common.provenance import file_hash
from training.players.pose import MODEL_SHA256


class PoseGraph:
    OPS: ClassVar[set[str]] = {
        "Conv",
        "Sigmoid",
        "Mul",
        "Add",
        "Concat",
        "GlobalAveragePool",
        "HardSigmoid",
        "MaxPool",
        "Shape",
        "Slice",
        "Reshape",
        "ReduceSum",
        "Sqrt",
        "Clip",
        "Div",
        "MatMul",
        "Split",
        "Unsqueeze",
        "Squeeze",
        "Transpose",
        "Relu",
    }

    def __init__(self, path, device):
        import onnx
        from onnx import helper, numpy_helper

        path = Path(path)
        if file_hash(path) != MODEL_SHA256:
            raise ValueError("CUDA RTMPose requires the pinned checkpoint")
        graph = onnx.load(path).graph
        unsupported = {node.op_type for node in graph.node} - self.OPS
        if unsupported:
            raise ValueError(f"Unsupported CUDA ONNX operators: {unsupported}")
        self.device = device
        # numpy_helper only deserializes tensor bytes; no NumPy calculations.
        self.weights = {
            item.name: torch.from_numpy(numpy_helper.to_array(item).copy()).to(device)
            for item in graph.initializer
        }
        self.nodes = [
            (
                node.op_type,
                tuple(node.input),
                tuple(node.output),
                {a.name: helper.get_attribute_value(a) for a in node.attribute},
            )
            for node in graph.node
        ]
        self.inputs = [
            item.name for item in graph.input if item.name not in self.weights
        ]
        self.outputs = [item.name for item in graph.output]
        self.uses = Counter(name for _, inputs, _, _ in self.nodes for name in inputs)
        self.allocation_metadata = {}
        self.provenance = {
            "model": "rtmpose-m",
            "checkpoint_sha256": MODEL_SHA256,
            "backend": "pinned_ONNX_graph_native_Torch_CUDA",
            "device": device,
            "cpu_graph_partitions": 0,
        }

    def __call__(self, batch):
        values = {**self.weights, self.inputs[0]: batch}
        allocation = self.allocation_metadata.setdefault(tuple(batch.shape), {})
        uses = self.uses.copy()  # graph bookkeeping, not tensor calculations
        for node_index, (op, inputs, outputs, attrs) in enumerate(self.nodes):
            args = [values[name] if name else None for name in inputs]
            x = args[0]
            if op == "Conv":
                pads = attrs.get("pads", [0, 0, 0, 0])
                if pads[:2] != pads[2:]:
                    raise RuntimeError("Asymmetric RTMPose convolution is unsupported")
                result = F.conv2d(
                    x,
                    args[1],
                    args[2] if len(args) > 2 else None,
                    stride=tuple(attrs.get("strides", [1, 1])),
                    padding=tuple(pads[:2]),
                    dilation=tuple(attrs.get("dilations", [1, 1])),
                    groups=attrs.get("group", 1),
                )
            elif op == "Sigmoid":
                result = x.sigmoid()
            elif op == "Relu":
                result = x.relu()
            elif op == "Mul":
                result = x * args[1]
            elif op == "Add":
                result = x + args[1]
            elif op == "Div":
                result = x / args[1]
            elif op == "MatMul":
                result = x @ args[1]
            elif op == "Sqrt":
                result = x.sqrt()
            elif op == "HardSigmoid":
                result = (x * attrs.get("alpha", 0.2) + attrs.get("beta", 0.5)).clamp(
                    0, 1
                )
            elif op == "Concat":
                result = torch.cat(args, attrs["axis"])
            elif op == "GlobalAveragePool":
                result = x.mean(tuple(range(2, x.ndim)), keepdim=True)
            elif op == "MaxPool":
                result = F.max_pool2d(
                    x,
                    tuple(attrs["kernel_shape"]),
                    tuple(attrs["strides"]),
                    tuple(attrs["pads"][:2]),
                    ceil_mode=bool(attrs.get("ceil_mode", 0)),
                )
            elif op == "Shape":
                if node_index not in allocation:
                    allocation[node_index] = tensor(
                        x.shape, device=self.device, dtype=torch.int64
                    )
                result = allocation[node_index]
            elif op == "Slice":
                if node_index not in allocation:
                    starts, ends = args[1].tolist(), args[2].tolist()
                    axes = (
                        args[3].tolist() if len(args) > 3 else list(range(len(starts)))
                    )
                    steps = args[4].tolist() if len(args) > 4 else [1] * len(starts)
                    slices = [slice(None)] * x.ndim
                    for axis, start, end, step in zip(axes, starts, ends, steps):
                        slices[axis] = slice(start, end, step)
                    allocation[node_index] = tuple(slices)
                result = x[allocation[node_index]]
            elif op == "Reshape":
                if node_index not in allocation:
                    allocation[node_index] = tuple(args[1].tolist())
                result = x.reshape(allocation[node_index])
            elif op == "ReduceSum":
                result = x.sum(
                    tuple(attrs["axes"]), keepdim=bool(attrs.get("keepdims", 1))
                )
            elif op == "Clip":
                result = x
                if len(args) > 1 and args[1] is not None:
                    result = torch.maximum(result, args[1])
                if len(args) > 2 and args[2] is not None:
                    result = torch.minimum(result, args[2])
            elif op == "Split":
                result = torch.split(x, attrs["split"], attrs["axis"])
            elif op == "Unsqueeze":
                result = x
                for axis in sorted(attrs["axes"]):
                    result = result.unsqueeze(axis)
            elif op == "Squeeze":
                result = x
                for axis in sorted(attrs["axes"], reverse=True):
                    result = result.squeeze(axis)
            elif op == "Transpose":
                result = x.permute(attrs["perm"])
            else:
                raise RuntimeError(op)
            for name, value in zip(
                outputs, result if isinstance(result, tuple) else (result,), strict=True
            ):
                values[name] = value
            for name in inputs:
                uses[name] -= 1
                if (
                    uses[name] == 0
                    and name not in self.weights
                    and name not in self.outputs
                ):
                    values.pop(name, None)
        return [values[name] for name in self.outputs]


class Pose:
    def __init__(self, config):
        self.config = config
        self.device = config["device"]
        self.model = PoseGraph(config["weights"], self.device)
        self.provenance = self.model.provenance
        self.captured = {}

    def prepare_graphs(self):
        limit = self.config["batch_size"]
        for size in sorted({1, min(4, limit), min(8, limit), limit}):
            self.captured[size] = CapturedCall(
                self.model, torch.zeros((size, 3, 256, 192), device=self.device)
            )
        self.provenance["cuda_graph_batch_sizes"] = list(self.captured)
        self.provenance["cuda_graph_details"] = {
            str(size): graph.metadata for size, graph in self.captured.items()
        }

    def predict(self, image, detections):
        if not len(detections):
            return (
                torch.empty((0, 17, 2), device=self.device),
                torch.empty((0, 17), device=self.device),
                torch.empty((0, 17), device=self.device, dtype=torch.bool),
            )
        boxes = detections[:, :4]
        center = (boxes[:, :2] + boxes[:, 2:]) / 2
        scale = (boxes[:, 2:] - boxes[:, :2]) * 1.25
        scale = torch.stack(
            (
                torch.maximum(scale[:, 0], scale[:, 1] * 0.75),
                torch.maximum(scale[:, 1], scale[:, 0] / 0.75),
            ),
            1,
        )
        yy, xx = torch.meshgrid(
            torch.arange(256, device=self.device),
            torch.arange(192, device=self.device),
            indexing="ij",
        )
        # Align the source center with (96,128), reproducing the affine contract.
        coordinates = torch.stack(((xx - 96) / 192, (yy - 128) / 256), -1).float()
        grid = center[:, None, None] + coordinates[None] * scale[:, None, None]
        height, width = image.shape[-2:]
        grid = (grid + 0.5) / tensor(
            [width, height], device=self.device, dtype=torch.float32
        ) * 2 - 1
        # The pinned model normalizes BGR, not RGB.
        batch = F.grid_sample(
            image.flip(0)[None].float().expand(len(boxes), -1, -1, -1),
            grid,
            align_corners=False,
            padding_mode="zeros",
        )
        mean = tensor(
            [123.675, 116.28, 103.53], device=self.device, dtype=torch.float32
        )[None, :, None, None]
        std = tensor([58.395, 57.12, 57.375], device=self.device, dtype=torch.float32)[
            None, :, None, None
        ]
        outputs = [[], []]
        for start in range(0, len(batch), self.config["batch_size"]):
            normalized = (batch[start : start + self.config["batch_size"]] - mean) / std
            if self.captured:
                size = next(size for size in self.captured if size >= len(normalized))
                padded = F.pad(
                    normalized, (0, 0, 0, 0, 0, 0, 0, size - len(normalized))
                )
                x, y = self.captured[size](padded)
                x, y = x[: len(normalized)], y[: len(normalized)]
            else:
                x, y = self.model(normalized)
            outputs[0].append(x)
            outputs[1].append(y)
        x, y = [torch.cat(parts) for parts in outputs]
        confidence = (x.amax(-1) + y.amax(-1)) / 2
        points = torch.stack((x.argmax(-1), y.argmax(-1)), -1).float()
        points = (
            points
            / tensor([384, 512], device=self.device, dtype=torch.float32)
            * scale[:, None]
            + center[:, None]
            - scale[:, None] / 2
        )
        valid = (confidence >= self.config["keypoint_min_confidence"]) & (
            confidence > 0
        )
        valid &= (
            (points[..., 0] >= 0)
            & (points[..., 0] < width)
            & (points[..., 1] >= 0)
            & (points[..., 1] < height)
        )
        valid &= (detections[:, 4] >= self.config["min_detection_confidence"])[:, None]
        valid &= (
            (boxes[:, 2:] - boxes[:, :2]).prod(1) >= self.config["min_bbox_area"]
        )[:, None]
        return points, confidence, valid
