"""Two checkpoint slots and an atomic commit manifest; no growing replay store."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import torch
from .engine import ROOT
from . import budget


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def source_identity():
    paths = sorted((ROOT / "paisho_ppo").glob("*.py")) + [ROOT / "native/src/main.rs",
             ROOT / "native/Cargo.toml", ROOT / "native/Cargo.lock", ROOT / "requirements-lock.txt"]
    return hashlib.sha256("\n".join(f"{p.relative_to(ROOT)}:{digest(p)}" for p in paths).encode()).hexdigest()


def _atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, data):
    with budget.project_write_guard():
        budget.check(reserve=len(json.dumps(data).encode())+budget.METADATA_RESERVE)
        _atomic_json(path, data)


@contextmanager
def run_lock(directory):
    directory = budget.run_path(directory)
    with budget.project_write_guard():
        budget.check(reserve=budget.METADATA_RESERVE)
        directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another process owns this training directory") from exc
        try:
            # Interrupted writes are never valid recovery points.
            for name in ("checkpoint.tmp", "state.json.tmp", "previous-state.json.tmp"):
                (directory / name).unlink(missing_ok=True)
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return value


class _CountWriter:
    def __init__(self):
        self.bytes = 0

    def write(self, data):
        self.bytes += len(data)
        return len(data)

    def flush(self):
        pass


def _valid_metadata(directory):
    for name in ('state.json', 'previous-state.json'):
        try:
            row = json.loads((directory/name).read_text())
            if row['schema'] != 1 or row['file'] not in ('checkpoint-0.pt', 'checkpoint-1.pt'):
                continue
            path = directory/row['file']
            if path.stat().st_size == row['bytes'] and digest(path) == row['sha256']:
                return row
        except (OSError, ValueError, KeyError):
            pass
    return None


def _reclaim(directory, protected, required, limit_mb):
    """Old checkpoint files are expendable; the current run's valid one is not."""
    def fits():
        try:
            return budget.writer_limit(directory, limit_mb) >= required
        except budget.SpaceUnavailable:
            return False

    if fits():
        return {'removed_files':0, 'removed_bytes':0}
    removed = {'removed_files':0, 'removed_bytes':0}
    protected_path = directory/protected['file'] if protected else None

    def remove(path):
        if not path.exists():
            return
        size = max(path.stat().st_size, path.stat().st_blocks*512)
        path.unlink()
        removed['removed_files'] += 1
        removed['removed_bytes'] += size

    # Reclaim the older local rotating slot first. No committed latest model
    # is destroyed to make room for a checkpoint that has not succeeded yet.
    for path in sorted(directory.glob('checkpoint-*.pt'), key=lambda p:p.stat().st_mtime):
        if path != protected_path:
            remove(path)
            previous = directory/'previous-state.json'
            if previous.exists() and json.loads(previous.read_text()).get('file') == path.name:
                remove(previous)
        if fits():
            return removed

    raise budget.SpaceUnavailable('checkpoint deferred: the current run has no replaceable slot left')


def _save(directory, payload, summary, limit_mb):
    directory = budget.run_path(directory)
    manifest_path = directory / "state.json"
    old = _valid_metadata(directory)
    generation = (old["generation"] if old else 0) + 1
    slot = generation % 2
    filename = f"checkpoint-{slot}.pt"
    temporary = directory / "checkpoint.tmp"
    try:
        prepared = cpu_tree(payload)
        counter = _CountWriter()
        torch.save(prepared,counter)
        reclamation = _reclaim(directory,old,counter.bytes,limit_mb)
        with temporary.open("wb") as stream:
            capped = budget.CappedWriter(stream, budget.writer_limit(directory, limit_mb))
            torch.save(prepared, capped)
            stream.flush()
            os.fsync(stream.fileno())
        charged = budget.usage(directory)
        if charged > limit_mb * 1024*1024:
            raise budget.SpaceUnavailable("checkpoint exceeds configured run-directory storage limit")
        size, checksum = temporary.stat().st_size, digest(temporary)
        os.replace(temporary, directory / filename)
        metadata = {"schema": 1, "generation": generation, "file": filename,
                    "sha256": checksum, "bytes": size,
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                    "storage_reclamation":reclamation, **summary}
        if old:
            _atomic_json(directory / "previous-state.json", old)
        _atomic_json(manifest_path, metadata)
        budget.check()
        return metadata
    finally:
        temporary.unlink(missing_ok=True)


def save(directory, payload, summary, limit_mb):
    with budget.project_write_guard(enforce=False):
        return _save(directory, payload, summary, limit_mb)


def load(directory):
    directory = budget.run_path(directory)
    errors = []
    for name in ("state.json", "previous-state.json"):
        try:
            metadata = json.loads((directory / name).read_text())
            if metadata["schema"] != 1 or metadata["file"] not in ("checkpoint-0.pt", "checkpoint-1.pt"):
                raise ValueError("invalid checkpoint manifest")
            path = directory / metadata["file"]
            if path.stat().st_size != metadata["bytes"] or digest(path) != metadata["sha256"]:
                raise ValueError("checkpoint checksum mismatch")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict) or payload.get("schema") not in (1, 2, 3):
                raise ValueError("invalid checkpoint payload")
            return payload, metadata, name != "state.json"
        except (OSError, ValueError, KeyError, RuntimeError, EOFError) as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError("no valid checkpoint: " + "; ".join(errors))
