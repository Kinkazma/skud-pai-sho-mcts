#!/usr/bin/env python3
"""PPO padding sweep and forward/supervised reference; use the campaign pause wrapper."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gradients", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    binary = Path("apple/paisho-mpsgraph/.build/release/paisho-mpsgraph-bench").resolve()
    common = [str(binary), "--preset", "pure", "--batch", "64", "--level", "1",
              "--warmup", "8", "--iterations", "64", "--seed", "1", "--profile-ppo", "true"]
    cases = [("terminal-ppo", cap) for cap in [64, 128, 1024, 1024, 128, 64]]
    cases.append(("both", 1024))
    if args.gradients:
        cases = [(mode, 1024) for mode in ["terminal-loss", "terminal-gradients", "terminal-ppo",
                                          "terminal-ppo", "terminal-gradients", "terminal-loss"]]
    plan = {"common": common, "cases": cases,
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "scope": "synthetic; capacities >=17 preserve active input/target values; both is a different objective"}
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for index, (mode, capacity) in enumerate(cases):
        result = subprocess.run(common + ["--mode", mode, "--actions", str(capacity)],
                                capture_output=True, text=True)
        (args.output / f"{index}.log").write_text(result.stdout + result.stderr)
        print(f"case={index} mode={mode} capacity={capacity}\n{result.stdout}", flush=True)
        result.check_returncode()


if __name__ == "__main__":
    main()
