"""Independent model execution on CUDA streams with explicit dependencies."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import torch

from pipeline.cuda.core import CudaOnly
from pipeline.numeric import cuda_numbers


def _record(value, stream):
    if isinstance(value, torch.Tensor):
        value.record_stream(stream)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _record(item, stream)
    elif isinstance(value, dict):
        for item in value.values():
            _record(item, stream)


class ModelsInFlight:
    def __init__(self, device, workers=3):
        self.device = device
        self.pool = ThreadPoolExecutor(
            max_workers=min(3, workers), thread_name_prefix="cuda-model"
        )
        self.streams = {
            key: torch.cuda.Stream(device=device)
            for key in ("pose", "segmentation", "ball")
        }
        self.calls = Counter()

    def submit(self, key, function, *args, **kwargs):
        stream = self.streams[key]
        stream.wait_stream(torch.cuda.current_stream(self.device))

        def execute():
            guard = CudaOnly()
            with (
                torch.inference_mode(),
                torch.cuda.stream(stream),
                cuda_numbers(self.device),
                guard,
            ):
                _record(args, stream)
                _record(kwargs, stream)
                output = function(*args, **kwargs)
                ready = torch.cuda.Event()
                ready.record(stream)
            return output, ready, guard.calls

        return self.pool.submit(execute)

    def result(self, future):
        output, ready, calls = future.result()  # only waits for host submission
        stream = torch.cuda.current_stream(self.device)
        stream.wait_event(ready)
        _record(output, stream)
        self.calls.update(calls)
        return output

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
