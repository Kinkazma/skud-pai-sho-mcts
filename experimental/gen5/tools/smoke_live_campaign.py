#!/usr/bin/env python3
"""Opt-in real Metal recovery smoke; invoke through benchmark_with_training_paused.py."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--full-load", action="store_true",
                        help="two full 512-game/256-update cycles instead of the recovery test")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    campaign = root / "campaign"
    command = [str(Path("target/release/paisho-live").resolve()),
               "--campaign-dir", str(campaign), "--initial-checkpoint",
               str(args.initial_checkpoint.resolve()), "--target-generation", "54",
               "--actor-games", "8", "--actor-max-attempts", "128",
               "--actor-decision-limit", "256", "--actors", "8",
               "--steps-per-generation", "1", "--durable-every", "5",
               "--promotion-every", "10", "--evaluation-every", "10",
               "--promotion-pairs-per-batch", "2", "--promotion-max-attempted", "4",
               "--promotion-max-eligible", "4", "--promotion-decision-limit", "256",
               "--curriculum-pairs-per-batch", "2", "--curriculum-max-attempted", "4",
               "--curriculum-max-eligible", "4", "--curriculum-decision-limit", "256"]
    if args.full_load:
        for flag, value in {"--target-generation": "49", "--actor-games": "512",
                            "--actor-max-attempts": "2048", "--actors": "80",
                            "--steps-per-generation": "256"}.items():
            command[command.index(flag) + 1] = value
    (root / "command.json").write_text(json.dumps(command, indent=2))
    if args.full_load:
        started = time.monotonic()
        with (root / "full-load.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        result = {"wall_seconds": time.monotonic() - started,
                  "status": read(campaign / "live-status.json"),
                  "cycles": [read(p) for p in sorted((campaign / "attempts").glob("*/generation-*/learning.json"))]}
        assert result["status"]["games"] == 1024, result
        assert result["status"]["examples"] == 32768, result
        (root / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        return
    # This fixture intentionally imports G47 and crosses the G50 assessment boundary.
    observed_pids = set()
    with (root / "interrupted.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                state = read(campaign / "live-status.json")
                if state.get("learner_pid"):
                    observed_pids.add(state["learner_pid"])
                if state.get("completed_generation", 0) >= 51:
                    assert state["durable_generation"] == 50, state
                    break
                if process.poll() is not None:
                    raise RuntimeError(f"live smoke exited early: {process.returncode}; {log.name}")
                time.sleep(0.05)
            else:
                raise TimeoutError("live smoke did not reach volatile generation 51")
        finally:
            # Only the disposable smoke process group, never the stable campaign.
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)
    preserved = list((campaign / "attempts").glob("*/generation-*/shard.psrbuf"))
    assert preserved
    hashes_before = {p: p.read_bytes() for p in preserved}
    with (root / "resumed.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
    for path, content in hashes_before.items():
        assert path.read_bytes() == content, f"old replay replaced: {path}"
    text = (root / "resumed.log").read_text()
    assert "generation=51 stage=complete" in text, text
    blocks = [read(p) for p in sorted((campaign / "blocks").glob("*.json"))]
    assert [b["generation"] for b in blocks] == [50, 54], blocks
    assert blocks[-1]["games"] == 56, blocks[-1]
    again = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
    assert "work_needed=false" in again.stdout, again.stdout
    result = {"recovered_from": 50, "recomputed_from": 51, "completed_generation": 54,
              "durable_generations": [b["generation"] for b in blocks],
              "preserved_shards": len(preserved), "learner_pids_before_interrupt": sorted(observed_pids),
              "games": blocks[-1]["games"], "idempotent_completed_restart": True}
    (root / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
