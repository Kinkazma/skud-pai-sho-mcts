#!/usr/bin/env python3
"""Controlled persistence A/B. Default is plan-only; --run invokes the pause wrapper.

A = persistent live services, B = same live loop plus per-cycle checkpoint/restart.
Both retain PSW transport. This is not a comparison with historical stable binaries.
Extra common paisho-live arguments follow --. No GPU work occurs in plan-only mode.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import statistics
import struct
import subprocess
import sys
import time

from paisho_control import ControlConfig


def checkpoint_metadata(path):
    with path.open("rb") as stream:
        if stream.read(15) != b"PAISHO-CKPT-V2\n":
            raise ValueError("expected Swift V2 checkpoint")
        length = struct.unpack("<I", stream.read(4))[0]
        return json.loads(stream.read(length))


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def all_shards_match(runs, cycles):
    corpora = [[cycle["replay_sha256"] for cycle in run["cycles"]] for run in runs]
    return bool(corpora) and all(len(corpus) == cycles and corpus == corpora[0] for corpus in corpora)


def stopped_group(config):
    state = json.loads(config.state_path.read_text())
    pid = state.get("process_pid")
    if not pid:
        raise RuntimeError("stable process PID unavailable; cannot verify OS pause")
    text = subprocess.check_output(["ps", "-axo", "pid=,pgid=,stat="], text=True)
    rows = [line.split() for line in text.splitlines() if line.strip()]
    leader = next((row for row in rows if int(row[0]) == pid), None)
    if leader is None:
        raise RuntimeError("stable process disappeared; cannot verify OS pause")
    members = [dict(pid=int(row[0]), pgid=int(row[1]), stat=row[2])
               for row in rows if row[1] == leader[1]]
    if any("T" not in row["stat"] and "Z" not in row["stat"] for row in members):
        raise RuntimeError(f"stable process group is not stopped: {members}")
    return members


def run_arm(command, log, config):
    before = stopped_group(config)
    started = time.monotonic()
    samples = 1
    with log.open("x") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            while process.poll() is None:
                stopped_group(config)
                samples += 1
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            if process.returncode:
                raise RuntimeError(f"benchmark exited {process.returncode}; see {log}")
            after = stopped_group(config)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    return dict(seconds=time.monotonic() - started, ps_before=before, ps_after=after,
                stopped_group_samples=samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["live", "service", "initial-checkpoint", "control-config", "output"]:
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--order", choices=["AB", "ABBA"], default="ABBA")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--inside-wrapper", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("common", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.cycles < 2:
        parser.error("at least two cycles are required")
    common = args.common[1:] if args.common[:1] == ["--"] else args.common
    controlled = {"--campaign-dir", "--initial-checkpoint", "--service", "--target-generation",
                  "--actor-games", "--steps-per-generation", "--durable-every",
                  "--restart-services-every-cycle", "--evaluation-every", "--promotion-every"}
    if any(value.split("=", 1)[0] in controlled for value in common):
        parser.error("common arguments cannot override benchmark-controlled fields")
    for name in ["live", "service", "initial_checkpoint", "control_config", "output"]:
        setattr(args, name, getattr(args, name).resolve())
    metadata = checkpoint_metadata(args.initial_checkpoint)
    start = metadata["progress"]["generation"]
    target = start + args.cycles
    commands = []
    for index, arm in enumerate(args.order):
        campaign = args.output / f"{index + 1}-{arm}"
        command = [str(args.live), *common, "--initial-checkpoint", str(args.initial_checkpoint),
                   "--service", str(args.service), "--campaign-dir", str(campaign),
                   "--target-generation", str(target), "--actor-games", "512",
                   "--steps-per-generation", "256", "--durable-every", "5",
                   "--promotion-every", str(target + 1), "--evaluation-every", str(target + 1),
                   "--restart-services-every-cycle", str(arm == "B").lower()]
        commands.append(dict(arm=arm, campaign=str(campaign), command=command))
    plan = dict(scope="persistence plus per-cycle durability, same PSW live loop",
                order=args.order, cycles=args.cycles, games_per_cycle=512, steps_per_cycle=256,
                initial_file_sha256=digest(args.initial_checkpoint), initial_step=metadata["trainingStep"],
                live_sha256=digest(args.live), service_sha256=digest(args.service), runs=commands)
    if not args.run:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.inside_wrapper:
        arguments = sys.argv[1:]
        separator = arguments.index("--") if "--" in arguments else len(arguments)
        arguments = arguments[:separator] + ["--inside-wrapper"] + arguments[separator:]
        wrapper = Path(__file__).with_name("benchmark_with_training_paused.py")
        return subprocess.call([sys.executable, str(wrapper), "--config", str(args.control_config),
                                "--", sys.executable, str(Path(__file__).resolve()), *arguments])
    config = ControlConfig.load(args.control_config)
    stopped_group(config)
    args.output.mkdir(parents=True, exist_ok=False)
    # Private executable copies keep concurrent parent rebuilds outside the comparison.
    binaries = args.output / "bin"
    binaries.mkdir()
    for source, name in [(args.live, "paisho-live"), (args.service, "paisho-mpsgraph-service")]:
        shutil.copy2(source, binaries / name)
    for item in commands:
        item["command"][0] = str(binaries / "paisho-live")
        service_index = item["command"].index("--service") + 1
        item["command"][service_index] = str(binaries / "paisho-mpsgraph-service")
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2))
    runs = []
    for index, item in enumerate(commands):
        print(f"starting arm={item['arm']} repetition={index + 1}", flush=True)
        result = run_arm(item["command"], args.output / f"{index + 1}-{item['arm']}.log", config)
        campaign = Path(item["campaign"])
        block = json.loads(sorted((campaign / "blocks").glob("block-*.json"))[-1].read_text())
        if (block["generation"] != target or block["training_step"] != metadata["trainingStep"] + 256 * args.cycles
                or block["games"] != 512 * args.cycles):
            raise RuntimeError("arm did not complete the specified workload")
        cycles = []
        for directory in sorted((campaign / "attempts").glob("*/generation-*")):
            cycles.append(dict(generation=directory.name,
                collection=json.loads((directory / "collection.json").read_text()),
                learning=json.loads((directory / "learning.json").read_text()),
                replay_sha256=digest(directory / "shard.psrbuf")))
        runs.append(dict(**item, **result, cycles=cycles))
        (args.output / "runs.json").write_text(json.dumps(runs, indent=2))
    means = {arm: statistics.mean(run["seconds"] for run in runs if run["arm"] == arm) for arm in "AB"}
    matched = all_shards_match(runs, args.cycles)
    report = dict(scope=plan["scope"], mean_seconds=means,
                  all_shards_identical=matched,
                  observed_restart_over_persistent_ratio=means["B"] / means["A"],
                  matched_restart_over_persistent_ratio=means["B"] / means["A"] if matched else None,
                  comparison_status="matched replay workload" if matched else "UNMATCHED: no matched speed claim",
                  first_cycle_producers=[run["cycles"][0]["collection"]["producer"] for run in runs],
                  corpus_digests=[[cycle["replay_sha256"] for cycle in run["cycles"]] for run in runs],
                  caveat="Same seeds and starting weights; Metal may diverge numerically. Inspect corpus/attempt counts.",
                  runs=runs)
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "runs"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
