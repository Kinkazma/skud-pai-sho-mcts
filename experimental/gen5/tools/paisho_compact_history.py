#!/usr/bin/env python3
"""Bounded, resumable comparisons and provisional +100 Elo historical marks.

This companion never changes a learner, promotes a model, or stops training.
`run --once` executes one frozen sweep; `update` recalculates reports only.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from report_compact_progress import report, markdown, HISTORICAL_RULES

BUDGETS = (32, 64, 128, 256, 512)
SCHEMA = "paisho-compact-history-v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value, immutable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if immutable:
        with path.open("x") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def locked(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "history.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another history command is already running") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def jobs():
    # Own-budget references come first. The 512/512 comparison serves both roles.
    return [(b, b) for b in BUDGETS] + [(b, 512) for b in BUDGETS if b != 512]


def model_snapshot(training):
    progress = read(training / "progress.json")
    version = progress["published_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise ValueError("invalid durable learner version")
    source = training / "models" / f"version-{version:08d}.json"
    # The learner publishes this immutable file before its progress pointer.
    model = read(source)
    return version, source, model


def protocol(args):
    return {"pairs": args.pairs, "workers": args.workers,
            "seconds_per_comparison": args.seconds, "decision_limit": args.decision_limit,
            "budgets": list(BUDGETS), "minimum_milestone_pairs": 8,
            "starts": "standard", "move_time_limit": None,
            "training_cutoffs_apply_to_comparisons": False}


def prepare_sweep(args):
    training, output, binary = args.training.resolve(), args.output.resolve(), args.binary.resolve()
    version, source, model = model_snapshot(training)
    identity = {"training": str(training), "binary_sha256": digest(binary),
                "protocol": protocol(args)}
    config = output / "configuration.json"
    if config.exists():
        if read(config) != identity:
            raise ValueError("history configuration changed; use a new output directory")
    else:
        write(config, identity, immutable=True)
    existing = sorted((output / "sweeps").glob("sweep-*/manifest.json"))
    for manifest_path in existing:
        manifest = read(manifest_path)
        directory = manifest_path.parent
        if not (directory / "finished.json").exists():
            return directory, manifest
        if manifest["model_sha256"] == digest(source):
            return directory, manifest
    ordinal = len(existing)
    directory = output / "sweeps" / f"sweep-{ordinal:06d}"
    directory.mkdir(parents=True)
    snapshot = directory / "model.json"
    shutil.copyfile(source, snapshot)
    manifest = {"schema": SCHEMA, "created_at": now(), "ordinal": ordinal,
                **identity, "published_version": version, "model_source": str(source),
                "model_sha256": digest(snapshot), "training_steps": model.get("training_steps"),
                "first_pair": 1000000 + ordinal * 100000,
                "jobs": [{"candidate": candidate, "reference": reference,
                          "directory": f"mcts{candidate}-vs-legacy{reference}"}
                         for candidate, reference in jobs()]}
    if digest(source) != manifest["model_sha256"]:
        raise ValueError("durable model changed while taking snapshot")
    write(directory / "manifest.json", manifest, immutable=True)
    return directory, manifest


def milestone_thresholds(run, previous):
    identity = run["identity"]
    pairs = run["paired_results"]
    if (identity.get("rules", HISTORICAL_RULES) != HISTORICAL_RULES or
            identity.get("reference_kind") != "legacy-cpu-heuristic" or
            identity["candidate_budget"] != identity["reference_budget"]):
        return []
    if (run["failed"] or not run["complete_archive"] or
            pairs["eligible_pairs"] != pairs["scheduled_pairs"] or
            pairs["eligible_pairs"] < 8 or not run["regularized_pair_delta"]):
        return []
    regularized = next((item for item in run["regularized_pair_delta"]
                        if item["prior_sd"] == 300), None)
    if regularized is None:
        return []
    delta = regularized["posterior_mode"]
    if not math.isfinite(delta):
        raise ValueError("invalid milestone delta")
    return [threshold for threshold in range(100, 100 * math.floor(delta / 100) + 1, 100)
            if threshold not in previous]


def record_milestones(output, sweep, result):
    recorded = []
    for run in result["runs"]:
        budget = run["identity"]["candidate_budget"]
        parent = output / "milestones" / f"mcts{budget}"
        previous = {read(path)["threshold"] for path in parent.glob("plus-*/mark.json")}
        for threshold in milestone_thresholds(run, previous):
            target = parent / f"plus-{threshold:04d}"
            target.mkdir(parents=True, exist_ok=True)
            snapshot = target / "model.json"
            if snapshot.exists() and digest(snapshot) != digest(sweep / "model.json"):
                raise ValueError("unfinished milestone has a different model")
            shutil.copyfile(sweep / "model.json", snapshot)
            mark = {"schema": SCHEMA, "recorded_at": now(), "budget": budget,
                    "threshold": threshold, "sweep": str(sweep),
                    "model_sha256": digest(snapshot), "comparison": run,
                    "scale": "provisional regularized score-Elo delta against own frozen legacy",
                    "baseline": "old legacy at the same simulation budget = delta 0",
                    "canonical_davidson_rating": None,
                    "interpretation": "Historical observation, not established strength or promotion. Repeated checkpoint selection and small samples can exaggerate progress.",
                    "stops_training": False}
            write(target / "mark.json", mark, immutable=True)
            recorded.append(str(target))
    return recorded


def milestone_projections(output, sweep, result):
    projections = []
    snapshot_hash = digest(sweep / "model.json")
    for path in sorted((output / "milestones").glob("mcts*/plus-*/mark.json")):
        mark = read(path)
        if mark["model_sha256"] != snapshot_hash:
            continue
        bridges = [run for run in result["runs"]
                   if run["identity"].get("rules", HISTORICAL_RULES) == HISTORICAL_RULES and
                   mark["comparison"]["identity"].get("rules", HISTORICAL_RULES) == HISTORICAL_RULES and
                   run["identity"]["candidate_budget"] == mark["budget"] and
                   run["identity"]["reference_budget"] == 512 and
                   run["identity"].get("reference_kind") == "legacy-cpu-heuristic" and
                   run["human_projection"].get("status") == "conditional-provisional-bridge" and
                   run["identity"]["candidate_model_sha256"] == mark["model_sha256"]]
        if len(bridges) > 1:
            raise ValueError("multiple 512 bridges for the same milestone snapshot")
        bridge = bridges[0] if bridges else None
        value = {"schema": SCHEMA, "mark": str(path), "budget": mark["budget"],
                 "threshold": mark["threshold"], "model_sha256": mark["model_sha256"],
                 "source_sweep": str(sweep), "human_anchor": result.get("human_anchor"),
                 "bridge_run": bridge["run"] if bridge else None,
                 "bridge_files_sha256": bridge["files_sha256"] if bridge else None,
                 "human_projection": bridge["human_projection"] if bridge else
                    {"status": "pending-same-model-mcts512-comparison"},
                 "historical_mark_is_unchanged": True}
        raw = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        revision = hashlib.sha256(raw.encode()).hexdigest()
        archived = path.parent / "external-projections" / f"{revision}.json"
        if not archived.exists():
            write(archived, value, immutable=True)
        elif digest(archived) != revision:
            raise ValueError("milestone external projection revision changed")
        write(path.parent / "external-projection.json", {"sha256": revision, "revision": str(archived), **value})
        projections.append({"mark": str(path), "revision": str(archived), "sha256": revision, **value})
    return projections


def publish_report(sweep, result):
    # Preserve every published interpretation, including changes to human anchors.
    raw = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    revision = hashlib.sha256(raw.encode()).hexdigest()
    previous = sweep / "report.json"
    if previous.exists() and digest(previous) != revision:
        old_archive = sweep / "report-revisions" / digest(previous) / "report.json"
        if not old_archive.exists():
            publish_report(sweep, read(previous))
    archive = sweep / "report-revisions" / revision
    archive.mkdir(parents=True, exist_ok=True)
    json_path, md_path = archive / "report.json", archive / "report.md"
    if not json_path.exists():
        write(json_path, result, immutable=True)
    elif digest(json_path) != revision:
        raise ValueError("archived report revision changed")
    if not md_path.exists():
        with md_path.open("x") as stream:
            stream.write(markdown(result))
            stream.flush()
            os.fsync(stream.fileno())
    write(sweep / "report-revision.json", {"sha256": revision, "json": str(json_path),
                                         "markdown": str(md_path)})
    write(sweep / "report.json", result)
    temporary = sweep / "report.md.tmp"
    shutil.copyfile(md_path, temporary)
    temporary.replace(sweep / "report.md")


def update_sweep(args, sweep):
    paths = []
    manifest = read(sweep / "manifest.json")
    if digest(sweep / "model.json") != manifest["model_sha256"]:
        raise ValueError("sweep snapshot hash changed")
    for job in manifest["jobs"]:
        path = sweep / job["directory"]
        receipt_path = sweep / (job["directory"] + ".receipt.json")
        if not receipt_path.exists():
            continue
        receipt = read(receipt_path)
        if receipt["returncode"] == 0 and (path / "summary.json").exists():
            paths.append(path)
    result = report(paths, args.human_anchor, args.anchor_cohort)
    expected = {(job["candidate"], job["reference"]): manifest["first_pair"] + index * manifest["protocol"]["pairs"]
                for index, job in enumerate(manifest["jobs"])}
    for run in result["runs"]:
        identity = run["identity"]
        if (identity["candidate_model_sha256"] != manifest["model_sha256"] or
                (identity["candidate_budget"], identity["reference_budget"]) not in expected or
                identity["first_pair"] != expected.get((identity["candidate_budget"], identity["reference_budget"])) or
                identity["workers"] != manifest["protocol"]["workers"] or
                identity["decision_limit"] != manifest["protocol"]["decision_limit"] or
                identity["move_ms"] is not None or
                run["paired_results"]["scheduled_pairs"] != manifest["protocol"]["pairs"]):
            raise ValueError("comparison identity differs from the frozen sweep")
    record_milestones(args.output.resolve(), sweep, result)
    result["milestone_projections"] = milestone_projections(args.output.resolve(), sweep, result)
    publish_report(sweep, result)
    return result


def run_sweep(args):
    sweep, manifest = prepare_sweep(args)
    if (sweep / "finished.json").exists():
        update_sweep(args, sweep)
        return sweep
    snapshot = sweep / "model.json"
    for index, job in enumerate(manifest["jobs"]):
        summary_path = args.training / "summary.json"
        if ((getattr(args, "follow", False) or hasattr(args, "_training_identity")) and
                summary_path.exists() and read(summary_path).get("status") != "completed-bounded-pilot"):
            follow_status(args, "training-failed-no-final-sweep", sweep)
            return sweep
        if (getattr(args, "_training_identity", None) is not None and
                not (args.training / "summary.json").exists() and
                process_identity(args.training_pid) != args._training_identity):
            follow_status(args, "training-exited-without-summary", sweep)
            return sweep
        name = job["directory"]
        receipt_path = sweep / (name + ".receipt.json")
        if receipt_path.exists():
            continue  # Failed/partial jobs are retained, never silently retried.
        destination = sweep / name
        if destination.exists():
            write(receipt_path, {"returncode": None, "status": "interrupted-no-retry",
                                "recorded_at": now()}, immutable=True)
            continue
        if digest(args.binary.resolve()) != manifest["binary_sha256"] or digest(snapshot) != manifest["model_sha256"]:
            raise ValueError("frozen binary/model identity changed")
        command = [str(args.binary.resolve()), "compare", "--candidate", str(snapshot),
                   "--output", str(destination), "--pairs", str(args.pairs),
                   "--simulations", str(job["candidate"]), "--reference-simulations", str(job["reference"]),
                   "--workers", str(args.workers), "--seconds", str(args.seconds),
                   "--decision-limit", str(args.decision_limit),
                   "--first-pair", str(manifest["first_pair"] + index * args.pairs)]
        write(args.output / "status.json", {"schema": SCHEMA, "state": "comparing", "at": now(),
              "sweep": str(sweep), "job": name, "companion_pid": os.getpid()})
        started = time.monotonic()
        with (sweep / (name + ".log")).open("x") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                returncode = child.wait(timeout=args.seconds + 60)
            finally:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
        if digest(args.binary.resolve()) != manifest["binary_sha256"] or digest(snapshot) != manifest["model_sha256"]:
            raise ValueError("binary/model changed during comparison")
        write(receipt_path, {"command": command, "returncode": returncode,
              "elapsed_seconds": time.monotonic() - started, "finished_at": now(),
              "binary_sha256": manifest["binary_sha256"], "model_sha256": manifest["model_sha256"]}, immutable=True)
        update_sweep(args, sweep)
    write(sweep / "finished.json", {"finished_at": now(), "all_jobs_attempted": True}, immutable=True)
    update_sweep(args, sweep)
    write(args.output / "status.json", {"schema": SCHEMA, "state": "sweep-finished", "at": now(),
          "sweep": str(sweep), "companion_pid": os.getpid()})
    return sweep


def refresh_all(args):
    for path in sorted((args.output / "sweeps").glob("sweep-*/manifest.json")):
        update_sweep(args, path.parent)


def process_identity(pid):
    if pid is None or pid <= 0:
        return None
    result = subprocess.run(["ps", "-p", str(pid), "-o", "lstart=", "-o", "command="],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def follow_status(args, state, sweep):
    write(args.output / "status.json", {"schema": SCHEMA, "state": state,
          "at": now(), "sweep": str(sweep) if sweep else None,
          "companion_pid": os.getpid()})


def run_follow(args):
    identity = process_identity(args.training_pid)
    args._training_identity = identity
    sweep = None
    while True:
        summary_path = args.training / "summary.json"
        if summary_path.exists():
            summary = read(summary_path)
            if summary.get("status") != "completed-bounded-pilot":
                follow_status(args, "training-failed-no-final-sweep", sweep)
                return sweep
            final_hash = digest(model_snapshot(args.training)[1])
            # A resumed unfinished sweep may come first; complete it once, then
            # evaluate the final durable model once. No new learning is launched.
            for _ in range(2):
                if (sweep is not None and (sweep / "finished.json").exists() and
                        read(sweep / "manifest.json")["model_sha256"] == final_hash):
                    break
                sweep = run_sweep(args)
            if not (sweep / "finished.json").exists() or read(sweep / "manifest.json")["model_sha256"] != final_hash:
                raise ValueError("could not reach the final durable snapshot")
            follow_status(args, "final-sweep-finished", sweep)
            return sweep
        if not identity or process_identity(args.training_pid) != identity:
            follow_status(args, "training-exited-without-summary", sweep)
            return sweep
        if sweep is None or digest(model_snapshot(args.training)[1]) != read(sweep / "manifest.json")["model_sha256"]:
            sweep = run_sweep(args)
            continue
        follow_status(args, "waiting-for-new-checkpoint", sweep)
        time.sleep(5)


def interrupted(signum, _frame):
    raise InterruptedError(f"history interrupted by signal {signum}; training is unaffected")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "update", "status"))
    parser.add_argument("--training", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--decision-limit", type=int, default=2048)
    parser.add_argument("--human-anchor", type=Path)
    parser.add_argument("--anchor-cohort")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="one frozen sweep (the default)")
    mode.add_argument("--follow", action="store_true", help="new frozen sweeps while the specified training process remains alive")
    parser.add_argument("--training-pid", type=int)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.command == "status":
        path = args.output / "status.json"
        print(json.dumps(read(path) if path.exists() else {"state": "not-started"}, indent=2))
        return
    if args.follow and (args.command != "run" or args.training_pid is None):
        parser.error("--follow requires run and --training-pid")
    if args.command == "run" and (args.training is None or args.binary is None):
        parser.error("run requires --training and --binary")
    if not (1 <= args.pairs <= 10000 and 1 <= args.workers <= 64 and
            1 <= args.seconds <= 3600 and 1 <= args.decision_limit <= 8192):
        parser.error("comparison limits are outside native supported ranges")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    with locked(args.output):
        try:
            if args.command == "run":
                print(run_follow(args) if args.follow else run_sweep(args))
            else:
                refresh_all(args)
        except Exception as error:
            write(args.output / "status.json", {"schema": SCHEMA, "state": "failed", "at": now(),
                  "error": str(error), "companion_pid": os.getpid()})
            raise


if __name__ == "__main__":
    main()
