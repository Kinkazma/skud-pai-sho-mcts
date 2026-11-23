"""Execution settings only: never alter model arithmetic, batches or RNG order."""
import json
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def defaults(device=None, torch_version=None):
    path = Path(__file__).resolve().parents[1] / "runtime-profile.json"
    if path.exists():
        profile = json.loads(path.read_text())
        data = profile["selected"]
        result = {"engine_workers": min(64, data["engine_workers"]),
                  "prefetch": bool(data["prefetch"]), "cpu_threads":None, "execution":{}}
        if str(device) == profile["device"] and torch_version == profile["torch"]:
            result["cpu_threads"] = data["cpu_threads"]
            result["execution"] = profile.get("execution", {})
        return result
    return {"engine_workers": min(os.cpu_count() or 1, 8), "prefetch": True, "cpu_threads":None, "execution":{}}


class StageTimes:
    def __init__(self):
        self.values = {}

    def call(self, stage, fn, *args, **kwargs):
        begin = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            value = self.values.setdefault(stage, {"calls":0,"seconds":0.0})
            value["calls"] += 1
            value["seconds"] += time.perf_counter()-begin


class Lookahead:
    """One bounded CPU lookahead; submission never advances a training cursor."""
    def __init__(self, enabled=True):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="skud-input") if enabled else None
        self.key = self.future = None

    def submit(self, key, fn, *args):
        if self.executor is not None and self.future is None:
            self.key, self.future = key, self.executor.submit(fn, *args)

    def get(self, key, fn, *args):
        if self.future is not None:
            saved_key, future = self.key, self.future
            self.key = self.future = None
            value = future.result()  # Exceptions propagate; no silently skipped data.
            if saved_key == key:
                return value
        return fn(*args)

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
        self.key = self.future = None
