#!/usr/bin/env python3
"""Start a bounded, detached compact CPU campaign with immutable inputs.

No scheduler or recurring automation is installed. A continuation uses a NEW
directory and a saved model; it is not an in-place replay of an interrupted run.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def prepare(directory, binary, model, seconds=3600, workers=6, seed=908202601,
            learning_profile="current"):
    directory, binary, model = (Path(p).resolve() for p in (directory, binary, model))
    if not 1 <= seconds <= 43200 or not 5 <= workers <= 8:
        raise ValueError("duration must be 1..43200 seconds and workers 5..8")
    if learning_profile not in ("current", "historical-gen3"):
        raise ValueError("unknown compact learning profile")
    if not binary.is_file() or not os.access(binary, os.X_OK) or not model.is_file():
        raise ValueError("executable and initial model must exist")
    directory.mkdir(parents=True, exist_ok=False)
    frozen = directory / "bin"
    frozen.mkdir()
    shutil.copy2(binary, frozen / "paisho-compact")
    shutil.copy2(model, directory / "initial-model.json")
    shutil.copy2(__file__, frozen / "campaign.py")
    command = [str(frozen / "paisho-compact"), "selfplay", "--budgets", "512,256,128,64,32",
               "--workers", str(workers), "--seconds", str(seconds), "--games", "1000000",
               "--decision-limit", "2048", "--samples", "128", "--learning-rate", "0.01",
               "--lambda", "0.5", "--seed", str(seed), "--model", str(directory / "initial-model.json"),
               "--output", str(directory / "training")]
    historical = learning_profile == "historical-gen3"
    explicit_learning = {"replay-capacity": "0" if historical else "65536",
                         "replay-ratio": "0" if historical else "4",
                         "repetition-cycles": "0" if historical else "4",
                         "reuse-search": "false" if historical else "true"}
    for flag, value in explicit_learning.items():
        command.extend(["--" + flag, value])
    manifest = {"schema": "paisho-compact-campaign-v1", "created_at": now(),
                "learning_profile": learning_profile,
                "explicit_learning_options": explicit_learning,
                "seconds": seconds, "workers": workers, "command": command,
                "files_sha256": {"bin/paisho-compact": digest(frozen / "paisho-compact"),
                                  "bin/campaign.py": digest(frozen / "campaign.py"),
                                  "initial-model.json": digest(directory / "initial-model.json")},
                "initial_model_source": str(model), "start_positions": "standard-only",
                "stop_policy": "elapsed budget or operational error; never wins or Elo milestones",
                "continuation": "new campaign directory from an explicitly chosen durable model",
                "evaluation_capacity": "reserve two CPU workers; separate bounded comparisons",
                "cutoffs_seconds": {"32": 8, "64": 17.5, "128": 25, "256": 34, "512": 58}}
    atomic_json(directory / "campaign.json", manifest)
    return manifest


def execute(directory):
    directory = Path(directory).resolve()
    # Exclusive receipt prevents two coordinators from running the same plan.
    with (directory / "started.json").open("x") as stream:
        json.dump({"pid": os.getpid(), "started_at": now()}, stream)
    status = {"status": "starting", "coordinator_pid": os.getpid(), "started_at": now()}
    child = None
    try:
        plan = json.loads((directory / "campaign.json").read_text())
        for name, expected in plan["files_sha256"].items():
            if digest(directory / name) != expected:
                raise ValueError(f"frozen input changed: {name}")
        with (directory / "training.log").open("ab", buffering=0) as log:
            child = subprocess.Popen(plan["command"], stdin=subprocess.DEVNULL, stdout=log, stderr=log)
            status.update(status="training", trainer_pid=child.pid)
            atomic_json(directory / "status.json", status)
            code = child.wait()
        status.update(status="completed" if code == 0 else "failed", exit_code=code, finished_at=now())
    except Exception as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        status.update(status="failed", error=str(error), finished_at=now())
    atomic_json(directory / "status.json", status)
    return 0 if status["status"] == "completed" else 1


def start(args):
    directory = args.directory.resolve()
    prepare(directory, args.binary, args.model, args.seconds, args.workers, args.seed,
            args.learning_profile)
    with (directory / "coordinator.log").open("ab", buffering=0) as log:
        process = subprocess.Popen([sys.executable, str(directory / "bin/campaign.py"), "_run",
                                    "--directory", str(directory)], start_new_session=True,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=log, close_fds=True)
    atomic_json(directory / "launch.json", {"coordinator_pid": process.pid, "launched_at": now()})
    return {"directory": str(directory), "coordinator_pid": process.pid, "seconds": args.seconds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("start")
    run.add_argument("--directory", type=Path, required=True)
    run.add_argument("--binary", type=Path, required=True)
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--seconds", type=int, default=3600)
    run.add_argument("--workers", type=int, default=6)
    run.add_argument("--seed", type=int, default=908202601)
    run.add_argument("--learning-profile", choices=("current", "historical-gen3"),
                     default="current")
    for name in ("_run", "status"):
        sub.add_parser(name).add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "_run":
        return execute(args.directory)
    if args.action == "start":
        print(json.dumps(start(args), ensure_ascii=False))
    else:
        result = {}
        for name in ("status.json", "training/progress.json", "training/summary.json"):
            path = args.directory / name
            if path.exists():
                result[name] = json.loads(path.read_text())
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
