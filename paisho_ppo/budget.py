"""Whole-project disk budget, including dependencies and build products."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
from .engine import ROOT

PROJECT_LIMIT_BYTES = 40_000_000_000  # 40 decimal GB, not 40 GiB.
METADATA_RESERVE = 1024*1024


class SpaceUnavailable(RuntimeError):
    """Managed space pressure; training may continue without a new checkpoint."""


def usage(directory):
    total = 0
    stack = [Path(directory)]
    while stack:
        path = stack.pop()
        info = path.lstat()
        total += max(info.st_size, info.st_blocks*512)
        if path.is_dir() and not path.is_symlink():
            stack.extend(path.iterdir())
    return total


def check(reserve=0):
    occupied = usage(ROOT)
    if occupied + reserve > PROJECT_LIMIT_BYTES:
        raise SpaceUnavailable(f"GPT-SKUD's 40 GB project limit leaves insufficient space for this operation "
                           f"({occupied/1e9:.3f} GB used; {reserve/1e9:.3f} GB requested)")
    return occupied


def run_path(directory):
    directory = Path(directory).resolve()
    parent = (ROOT / "runs").resolve()
    if not directory.is_relative_to(parent) or directory == parent:
        raise ValueError("training directories must be inside GPT-SKUD/runs")
    return directory


@contextmanager
def project_write_guard(enforce=True):
    # Serialize managed disk allocations across every run in this project.
    with (ROOT / ".storage.lock").open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            if enforce:
                check()
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class CappedWriter:
    """Reject an oversized serialization before its bytes reach the filesystem."""
    def __init__(self, stream, limit):
        self.stream, self.limit, self.written = stream, limit, 0

    def write(self, data):
        if self.written + len(data) > self.limit:
            raise SpaceUnavailable("checkpoint write would exceed the project or run storage budget")
        amount = self.stream.write(data)
        self.written += amount
        return amount

    def __getattr__(self, name):
        return getattr(self.stream, name)


def writer_limit(directory, run_limit_mb):
    project_remaining = PROJECT_LIMIT_BYTES-check()-METADATA_RESERVE
    run_remaining = run_limit_mb*1024*1024-usage(directory)-METADATA_RESERVE
    limit = min(project_remaining, run_remaining)
    if limit <= 0:
        raise SpaceUnavailable("no checkpoint space remains within GPT-SKUD's disk budget")
    return limit


def report():
    occupied = usage(ROOT)
    return {"project": str(ROOT), "used_bytes": occupied, "limit_bytes": PROJECT_LIMIT_BYTES,
            "used_gb": round(occupied/1e9, 3), "limit_gb": 40,
            "remaining_gb": round((PROJECT_LIMIT_BYTES-occupied)/1e9, 3),
            "within_limit": occupied <= PROJECT_LIMIT_BYTES,
            "checkpoint_slots_per_run": 2, "per_run_default_limit_mb": 1024}


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--reserve-gb", type=float, default=0)
    args = parser.parse_args()
    check(int(args.reserve_gb*1e9))
    print(json.dumps(report(), indent=2))
