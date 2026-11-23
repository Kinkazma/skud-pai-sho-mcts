import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import orjson

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "native/target/release/paisho-selfplay-engine"


def engine_identity():
    manifest = json.loads((ROOT / "engine-manifest.json").read_text())
    vendor = ROOT / "native/vendor/paisho-core"
    for filename, digest in manifest["files"].items():
        if hashlib.sha256((vendor / filename).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"vendored rules changed: {filename}")
    return hashlib.sha256(json.dumps(manifest["files"], sort_keys=True).encode()).hexdigest()


def build(test=False):
    from . import budget
    engine_identity()
    env = os.environ.copy()
    command = ["cargo", "test" if test else "build", "--release", "--locked"]
    if test:
        command.append("--workspace")
    if not (ROOT / "native/Cargo.lock").exists():
        command.remove("--locked")
    with budget.project_write_guard():
        budget.check(reserve=2_000_000_000)
        subprocess.run(command, cwd=ROOT / "native", env=env, check=True)
        budget.check()


class Engine:
    def __init__(self, workers=None, compact=True, json_backend="orjson"):
        engine_identity()
        if not BINARY.is_file():
            raise RuntimeError("Run cargo build --release --locked from the native directory")
        if (ROOT / "native/src/main.rs").stat().st_mtime > BINARY.stat().st_mtime:
            raise RuntimeError("Rules bridge source changed. Rebuild from the native directory.")
        if workers is None:
            from .runtime import defaults
            workers = defaults()["engine_workers"]
        if type(workers) is not int or not 1 <= workers <= 64:
            raise ValueError("engine workers must be in 1..64")
        self.workers = workers
        self.compact = compact
        if json_backend not in ("orjson", "stdlib"):
            raise ValueError("unknown JSON codec")
        self.json_backend = json_backend
        self.timings = {}
        self._histories = []
        self._lock = threading.Lock()
        env = os.environ.copy()
        env["GPT_SKUD_ENGINE_WORKERS"] = str(workers)
        self.process = subprocess.Popen([str(BINARY)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, env=env)
        self.description = self.request("describe")
        if self.description["protocol"] != 1:
            self.close()
            raise RuntimeError("engine protocol mismatch")

    def request(self, op, **data):
        # A replay prefetch and an exact checkpoint export can share the bridge.
        with self._lock:
            started = time.perf_counter()
            if op == "step" and self.compact:
                data["compact"] = True
            request = {"op":op, **data}
            encoded = (orjson.dumps(request) if self.json_backend == "orjson" else
                       json.dumps(request,separators=(",", ":")).encode())
            self.process.stdin.write(encoded+b"\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(f"rules bridge exited ({self.process.poll()})")
            reply = (orjson.loads(line) if self.json_backend == "orjson" else json.loads(line))
            if not reply["ok"]:
                raise ValueError(reply["error"])
            result = reply["data"]
            if op in ("reset", "restore"):
                self._histories = [o["history"] for o in result]
            elif op == "load_record":
                self._histories = [result["history"]]
            elif op == "reset_slots":
                for item in result:
                    self._histories[item["slot"]] = item["observation"]["history"]
            elif op == "step":
                for item in result:
                    slot = item["slot"]
                    if "history_append" in item:
                        item["observation"]["history"] = self._histories[slot]+[item.pop("history_append")]
                    self._histories[slot] = item["observation"]["history"]
            timing = self.timings.setdefault(op, {"calls":0,"seconds":0.0})
            timing["calls"] += 1
            timing["seconds"] += time.perf_counter()-started
            return result

    def close(self):
        with self._lock:
            if self.process.stdin and not self.process.stdin.closed:
                self.process.stdin.close()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        if self.process.stdout:
            self.process.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
