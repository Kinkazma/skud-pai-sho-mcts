#!/usr/bin/env python3
"""Two full matched first cycles. Invoke through the pause wrapper.

Arguments: CONTROL_CONFIG INITIAL_CHECKPOINT OUTPUT_ROOT [round-size]
Copies production options but never changes production configuration or archives.
"""
import json
from pathlib import Path
import subprocess
import sys

from benchmark_live_persistence import checkpoint_metadata, digest
from paisho_control import ControlConfig


def main():
    config_path, checkpoint, output = map(Path, sys.argv[1:4])
    experiment = sys.argv[4] if len(sys.argv) == 5 else "disk-ram"
    if experiment not in ("disk-ram", "round-size") or len(sys.argv) not in (4, 5):
        raise ValueError("expected config checkpoint output [round-size]")
    config = ControlConfig.load(config_path)
    checkpoint = checkpoint.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    generation = checkpoint_metadata(checkpoint)["progress"]["generation"] + 1
    executable = Path(__file__).resolve().parents[1] / "target/release/paisho-live"
    options = dict(zip(config.command[1::2], config.command[2::2]))
    options.update({"--initial-checkpoint": str(checkpoint), "--target-generation": str(generation),
                    "--evaluation-every": "1000000", "--promotion-every": "1000000"})
    commands = []
    modes = ("disk", "ram") if experiment == "disk-ram" else ("round-80", "round-512")
    for mode in modes:
        options.update({"--campaign-dir": str(output / mode),
                        "--curriculum-dir": str(output / mode / "curriculum"),
                        "--disk-replay": str(mode == "disk").lower()})
        if experiment == "round-size":
            options["--actor-round-size"] = mode.split("-")[1]
        commands.append([str(executable), *(v for pair in options.items() for v in pair)])
    (output / "plan.json").write_text(json.dumps({"commands": commands,
        "executable_sha256": digest(executable), "checkpoint_sha256": digest(checkpoint)}, indent=2))
    rows = []
    for mode, command in zip(modes, commands):
        print(f"starting={mode}", flush=True)
        with (output / (mode + ".log")).open("x") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        path = next((output / mode).glob("attempts/*/generation-*/timing.json"))
        rows.append({"mode": mode, "timing": json.loads(path.read_text()),
                     "shard_sha256": digest(path.parent / "shard.psrbuf"),
                     "learning": json.loads((path.parent / "learning.json").read_text())})
    result = {"runs": rows, "identical_replays": rows[0]["shard_sha256"] == rows[1]["shard_sha256"],
              "scope": f"single {experiment} pair; order and GPU variability limit total-time inference"}
    (output / "report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
