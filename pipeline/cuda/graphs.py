"""Replay audited, fixed-shape model calculations as CUDA graphs.

Capture executes under CudaOnly: any CPU tensor calculation fails there.
Replay launches precisely that captured GPU graph, without per-layer Python.
"""

import torch


class CapturedCall:
    def __init__(self, function, sample):
        if not sample.is_cuda:
            raise RuntimeError("GPU-only violation: CUDA graph input is on CPU")
        self.input = sample.clone()
        self.signature = (sample.shape, sample.dtype, sample.device)
        self.metadata = {
            "input_shape": list(sample.shape),
            "dtype": str(sample.dtype),
            "device": str(sample.device),
            "replays": 0,
            "audit": "CUDA kernels captured under CudaOnly",
        }
        stream = torch.cuda.Stream(device=sample.device)
        stream.wait_stream(torch.cuda.current_stream(sample.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                function(self.input)
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.output = function(self.input)
        torch.cuda.current_stream(sample.device).wait_stream(stream)

    def __call__(self, value):
        if (value.shape, value.dtype, value.device) != self.signature:
            raise ValueError("CUDA graph input signature changed")
        self.input.copy_(value)
        self.graph.replay()
        self.metadata["replays"] += 1

        def copy(item):
            if isinstance(item, torch.Tensor):
                return item.clone()
            if isinstance(item, list):
                return [copy(value) for value in item]
            if isinstance(item, tuple):
                return tuple(copy(value) for value in item)
            if isinstance(item, dict):
                return {key: copy(value) for key, value in item.items()}
            return item

        # Consumers may update detection/feature tensors; preserve graph buffers.
        return copy(self.output)
