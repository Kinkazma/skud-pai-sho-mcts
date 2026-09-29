#!/usr/bin/env python3
"""Bounded teacher bootstrap -> pure terminal-PPO live process (Python 3.11+).

  python3 tools/paisho_live_launch.py --bootstrap-campaign BOOTSTRAP_DIR \
    [--live-executable /absolute/path/paisho-live] -- \
    --campaign-dir NEW_LIVE_DIR --target-generation ABSOLUTE_N [live options]

Runs/resumes the saved bootstrap plan to its immutable result. The bounded
budget_exhausted result is NOT a passed gate: it continues pure PPO at Random,
without further teacher production or teacher replay ingestion. gate_passed
selects Site. No automatic endless retries; on a child failure, exit 1 and rerun
this identical launcher command after resolving the error.

The launcher owns --initial-checkpoint and --opponent: both are rejected in
forwarded options, including --flag=value syntax. All other --flag VALUE pairs
are forwarded verbatim. --campaign-dir and --target-generation are required
once. The live directory must be separate from the bootstrap directory.

paisho-live reads generation G from the selected checkpoint and starts at G+1
(or resumes its own journal); the target is absolute and is never rebased here.
No config migration or journal mutation is performed by this launcher.
Before os.execv, stderr receives one JSON handoff summary. Thereafter stdout,
stderr and exit status belong to paisho-live. Pre-exec failure: 1; usage: 2.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import paisho_teacher_bootstrap as bootstrap


def forwarded_options(arguments):
    if len(arguments) % 2:
        raise ValueError("live options require --flag VALUE pairs")
    for token in arguments:
        if token.split("=", 1)[0] in ("--initial-checkpoint", "--opponent"):
            raise ValueError("launcher owns --initial-checkpoint and --opponent; omit these overrides")
    options = {}
    for flag, value in zip(arguments[::2], arguments[1::2]):
        if not flag.startswith("--") or "=" in flag:
            raise ValueError("live options require --flag VALUE syntax")
        if flag in options:
            raise ValueError("duplicate live option: " + flag)
        options[flag] = value
    for flag in ("--campaign-dir", "--target-generation"):
        if flag not in options:
            raise ValueError("missing live option " + flag)
    if int(options["--target-generation"]) < 0:
        raise ValueError("target generation must be nonnegative")
    return options


def launch(campaign, executable, arguments):
    options = forwarded_options(arguments)
    campaign = Path(campaign).resolve(strict=True)
    executable = Path(executable).resolve(strict=True)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("live executable is not executable")
    live_directory = Path(options["--campaign-dir"]).resolve()
    if live_directory.is_relative_to(campaign) or campaign.is_relative_to(live_directory):
        raise ValueError("live and bootstrap directories must be separate, not nested")
    result = bootstrap.run(campaign)
    if result["status"] not in ("gate_passed", "budget_exhausted"):
        raise ValueError("bootstrap has no terminal result")
    passed = result["status"] == "gate_passed"
    if result["gate_passed"] is not passed or result["teacher_stopped_permanently"] is not True:
        raise ValueError("inconsistent bootstrap terminal status")
    plan = bootstrap.read_json(campaign / "plan.json")
    expected_pass = result["evaluation"]["conclusion"] == bootstrap.UPPER
    if plan.get("champion_only"):
        if options.get("--champion-only") != "true":
            raise ValueError("V5 bootstrap requires champion-only live selection")
        expected_pass = (expected_pass and result["cycles_completed"] >= plan["minimum_teacher_cycles"]
                         and result["selected_checkpoint"] != plan["initial_checkpoint"]["path"])
    if expected_pass != passed:
        raise ValueError("bootstrap status disagrees with independent Random evaluation")
    selected = Path(result["selected_checkpoint"]).resolve(strict=True)
    bootstrap.verify({"path": str(selected), "sha256": result["selected_checkpoint_sha256"]})
    generation = bootstrap.checkpoint(selected)["progress"]["generation"]
    if int(options["--target-generation"]) < generation:
        raise ValueError(f"target generation precedes selected checkpoint generation {generation}")
    opponent = "site" if passed else "random"
    argv = [str(executable), *arguments, "--initial-checkpoint", str(selected), "--opponent", opponent]
    print(json.dumps({"bootstrap_status": result["status"], "gate_passed": passed,
        "selected_checkpoint": str(selected), "checkpoint_generation": generation,
        "first_live_generation_if_new": generation + 1,
        "target_generation": int(options["--target-generation"]), "opponent": opponent,
        "teacher_stopped_permanently": True, "live_executable": str(executable)}, sort_keys=True), file=sys.stderr, flush=True)
    os.execv(str(executable), argv)


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--bootstrap-campaign", type=Path, required=True)
    cli.add_argument("--live-executable", type=Path, default=bootstrap.ROOT / "target/release/paisho-live")
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--help" in argv and "--" not in argv:
        cli.parse_args(argv)
    if "--" not in argv:
        cli.error("separate paisho-live options with --")
    split = argv.index("--")
    args = cli.parse_args(argv[:split])
    try:
        launch(args.bootstrap_campaign, args.live_executable, argv[split + 1:])
        return 0  # Only reached when execv is mocked in tests.
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"live-launch: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
