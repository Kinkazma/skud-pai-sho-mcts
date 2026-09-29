#!/usr/bin/env python3
"""Exploratory symmetric batching-delay sweep; run with the campaign pause wrapper."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    binary = Path("target/release/examples/capacity_broker_smoke")
    service = Path("training-runs/live-curriculum-002/bin/paisho-mpsgraph-service")
    command = [str(binary.resolve()), "--service", str(service.resolve()),
               "--checkpoint", str(args.checkpoint.resolve()), "--workers", "20",
               "--actors", "80", "--warmup-decisions", "8", "--measured-decisions", "80",
               "--classes", "64:8,128:4,1024:4", "--wide-lanes", "1", "--seed", "20260905"]
    plan = {"command": command, "wait_microseconds": [5000, 1000, 0, 0, 1000, 5000],
            "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [binary, service, args.checkpoint]},
            "scope": "sampled positions, not complete live curriculum cycles"}
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for index, delay in enumerate(plan["wait_microseconds"]):
        result = subprocess.run(command + ["--wait-us", str(delay)],
                                text=True, capture_output=True)
        (args.output / f"{index}.log").write_text(result.stdout + result.stderr)
        print(f"run={index} wait_us={delay} exit={result.returncode}\n{result.stdout}", flush=True)
        result.check_returncode()


if __name__ == "__main__":
    main()
