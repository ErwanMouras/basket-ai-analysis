"""Device-scoped scalar arithmetic for shared identity/metric control logic."""

import builtins
import math
from collections import Counter as HostCounter
from contextlib import contextmanager
from contextvars import ContextVar

DEVICE = ContextVar("analysis_numeric_device", default=None)


@contextmanager
def cuda_numbers(device):
    token = DEVICE.set(device)
    try:
        yield
    finally:
        DEVICE.reset(token)


class CudaCounter(dict):
    def __init__(self, iterable=(), *, device=None):
        self.device = device or DEVICE.get()
        super().__init__()
        self.update(iterable)

    def _number(self, value):
        import torch

        if (
            isinstance(value, torch.Tensor)
            and value.dtype == torch.float64
            and str(value.device) == self.device
        ):
            return value
        return torch.as_tensor(value, device=self.device, dtype=torch.float64)

    def __missing__(self, key):
        return self._number(0)

    def __setitem__(self, key, value):
        super().__setitem__(key, self._number(value))

    def get(self, key, default=0):
        # dict.get evaluates its default eagerly: do not allocate a CUDA scalar
        # for every lookup of a key that already exists.
        return self[key] if key in self else self._number(default)

    def update(self, iterable=(), **kwargs):
        if hasattr(iterable, "items"):
            for key, value in iterable.items():
                self[key] = self[key] + self._number(value)
        else:
            import torch

            # Category encoding and ordering are metadata. The histogram, like
            # all vote arithmetic, is computed on the GPU in one batch.
            keys = {}
            encoded = []
            for key in iterable:
                if key not in keys:
                    keys[key] = len(keys)
                encoded.append(keys[key])
            if encoded:
                codes = torch.as_tensor(encoded, device=self.device, dtype=torch.long)
                counts = torch.bincount(codes, minlength=len(keys)).double()
                for key, index in keys.items():
                    self[key] = (
                        self[key] + counts[index] if key in self else counts[index]
                    )
        if kwargs:
            self.update(kwargs)

    def most_common(self, n=None):
        import torch

        keys = list(self)
        if not keys:
            return []
        order = (
            torch.stack(list(self.values()))
            .argsort(descending=True, stable=True)
            .tolist()
        )
        return [(keys[i], self[keys[i]]) for i in (order if n is None else order[:n])]

    def elements(self):
        for key, value in self.items():
            for _ in range(int(value)):
                yield key


def Counter(iterable=()):
    return CudaCounter(iterable) if DEVICE.get() else HostCounter(iterable)


def distance(first, second):
    device = DEVICE.get()
    if device is None:
        return math.dist(first, second)
    import torch

    def vector(values):
        return torch.stack(
            [
                torch.as_tensor(value, dtype=torch.float64, device=device)
                for value in values
            ]
        )

    return torch.linalg.vector_norm(vector(first) - vector(second))


def rounded(value, decimals=0):
    if DEVICE.get():
        import torch

        return torch.round(
            torch.as_tensor(value, device=DEVICE.get(), dtype=torch.float64),
            decimals=decimals,
        )
    return builtins.round(value, decimals)


def fraction(first, second):
    if DEVICE.get():
        import torch

        return torch.as_tensor(first, device=DEVICE.get(), dtype=torch.float64) / second
    return first / second
