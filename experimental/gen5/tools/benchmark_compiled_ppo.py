#!/usr/bin/env python3
"""Fixed-shape ABBA PPO experiment; invoke via benchmark_with_training_paused.py."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    common = [str(args.binary.resolve()), "--preset", "pure", "--batch", "64",
              "--actions", "1024", "--level", "1", "--mode", "terminal-ppo",
              "--warmup", "8", "--iterations", "128", "--seed", "1"]
    plan = {"command": common, "order": [False, True, True, False],
            "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
            "scope": "synthetic fixed batch; not an end-to-end campaign speedup"}
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for index, compiled in enumerate(plan["order"]):
        command = common + ["--compiled-ppo", str(compiled).lower()]
        result = subprocess.run(command, text=True, capture_output=True)
        (args.output / f"{index}.log").write_text(result.stdout + result.stderr)
        print(f"run={index} compiled={compiled} exit={result.returncode}", flush=True)
        print(result.stdout, flush=True)
        result.check_returncode()


if __name__ == "__main__":
    main()
