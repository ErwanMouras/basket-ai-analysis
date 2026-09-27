"""Persistent bounded executor, per-model exclusion and per-device admission."""

from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock, Semaphore
from time import perf_counter


class Runtime:
    def __init__(self, config):
        self.pool = ThreadPoolExecutor(max_workers=config["workers"]) if config["mode"] == "parallel" else None
        self.devices = defaultdict(lambda: Semaphore(config["gpu_concurrency"]))
        self.models = defaultdict(Lock)
        self.timings = defaultdict(lambda: {"calls": 0, "seconds": 0.0, "wait_seconds": 0.0})
        self.guard = Lock()

    def call(self, name, device, fn, *args, **kwargs):
        # Build locks before submitting work; no model instance is called concurrently.
        with self.guard:
            model_lock = self.models[name]
            device_lock = self.devices[device] if device.startswith("cuda") else None

        queued = perf_counter()

        def work():
            with model_lock:
                if device_lock:
                    device_lock.acquire()
                started = perf_counter()
                try:
                    return fn(*args, **kwargs)
                finally:
                    # Adapters return CPU values, so the measured inference is complete.
                    elapsed = perf_counter() - started
                    if device_lock:
                        device_lock.release()
                    with self.guard:
                        row = self.timings[name]
                        row["calls"] += 1
                        row["seconds"] += elapsed
                        row["wait_seconds"] += started - queued
        if self.pool:
            return self.pool.submit(work)
        future = Future()
        try:
            future.set_result(work())
        except Exception as exc:
            future.set_exception(exc)
        return future

    def close(self):
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=True)
