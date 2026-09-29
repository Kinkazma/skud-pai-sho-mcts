#!/usr/bin/env python3
"""Bounded CURRICULUM_V4 sidecar; never controls the stable campaign or starts PPO.

Create an immutable plan in a NEW directory, then execute/resume it:
  python3 tools/paisho_teacher_bootstrap.py plan --campaign NEW --initial-checkpoint CKPT \
      --source SNAPSHOT [--source SNAPSHOT] --service SERVICE
  python3 tools/paisho_teacher_bootstrap.py run --campaign NEW

Defaults: <=5 cycles, <=2048 teacher positions/cycle, 32 updates/cycle at B64.
32*64 is one sampler pass only if the selected corpus has exactly 2048 examples.
Cap --actions controls learner packing, not teacher selection. Evaluation is
independent RandomAgent play, with generation-curriculum defaults. Repeated tests
use the declared per-evaluation alpha/beta, not a campaign-wide error guarantee.
An upper-hypothesis result permanently ends this sidecar; an exhausted budget
also ends it without claiming a win. Only a subsequent parent decision can start
terminal PPO, which must not sample this teacher corpus.

On errors, invoke run again. Incomplete teacher directories are retained and a
fresh attempt is used. Learner/evaluator archives resume in place. result.json
is immutable and returned immediately on every later invocation (no subprocess).
Stdout is one final JSON object with status and selected_checkpoint. Child logs
and exact argv are retained under the campaign; failures go to stderr.
Python 3.11+; standard library only. Exit 0 means a plan or terminal result was
published/read, including budget_exhausted (NOT gate success). Exit 1 is an
operational/data error; exit 2 is an argparse usage error. run accepts only
--campaign; change experiment options by creating a new plan, not on resume.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
UPPER = "supports-upper-elo-hypothesis"
CONCLUSIONS = {UPPER, "supports-lower-elo-hypothesis", "supports-central-elo-window",
               "inconclusive-maximum-eligible-pairs", "inconclusive-maximum-attempted-pairs"}
EVAL_DEFAULTS = {"classes": "64:8,128:4,1024:4", "max-attempted-pairs": 800,
    "max-eligible-pairs": 400, "decision-limit": 2048, "start-horizon": 64,
    "start-source-limit": 16384, "start-source-attempts": 16,
    "sampling-temperature": 1.0, "sampling-uniform-mix": 0.05,
    "lower-elo": -100.0, "elo0": 0.0, "elo1": 100.0, "alpha": 0.025, "beta": 0.025,
    "wait-us": 5000}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": digest(path)}


def verify(record):
    if digest(record["path"]) != record["sha256"]:
        raise ValueError(f"provenance changed: {record['path']}")


def read_json(path):
    return json.loads(Path(path).read_text())


def publish(path, value):
    """Atomic no-overwrite JSON, including an fsync before publication."""
    path = Path(path)
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".publish-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink()


def checkpoint(path):
    # Read metadata and verify the existing V2 payload digest without numpy/GPU.
    path = Path(path)
    size = path.stat().st_size
    magic = b"PAISHO-CKPT-V2\n"
    with path.open("rb") as stream:
        if stream.read(len(magic)) != magic:
            raise ValueError("expected V2 checkpoint")
        encoded_length = stream.read(4)
        if len(encoded_length) != 4:
            raise ValueError("truncated checkpoint metadata length")
        length = struct.unpack("<I", encoded_length)[0]
        if length > 16 * 1024 * 1024 or length > size - len(magic) - 36:
            raise ValueError("invalid checkpoint metadata length")
        meta = json.loads(stream.read(length))
        stream.seek(0)
        remaining, hash_ = size - 32, hashlib.sha256()
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("truncated checkpoint")
            hash_.update(chunk)
            remaining -= len(chunk)
        if hash_.digest() != stream.read(32):
            raise ValueError("checkpoint checksum mismatch")
    if meta["formatVersion"] != 2 or meta["trainingStep"] != meta["progress"]["scheduler"]["completedSteps"]:
        raise ValueError("invalid checkpoint metadata")
    return meta


def source_identity(path):
    path = Path(path)
    path = (path / "snapshot.psrsnap" if path.is_dir() else path).resolve(strict=True)
    record = identity(path)
    text = path.read_text()
    if not text.startswith("PAISHO-REPLAY-SNAPSHOT\t1\n"):
        raise ValueError("expected replay snapshot V1")
    shards = []
    for line in text.splitlines():
        if line.startswith("shard\t"):
            fields = line.split("\t")
            if len(fields) != 6:
                raise ValueError("malformed replay shard reference")
            name = fields[2]
            if Path(name).name != name or name in (".", ".."):
                raise ValueError("invalid shard path")
            shards.append(identity(path.parent / name))
    if not shards:
        raise ValueError("empty source snapshot")
    record["shards"] = shards
    return record


def seed(master, domain, cycle):
    raw = hashlib.sha256(f"paisho-bootstrap-v1/{master}/{domain}/{cycle}".encode()).digest()
    value = int.from_bytes(raw[:8], "little") & ((1 << 63) - 1)
    # Evaluation and all teacher/sampler seeds occupy disjoint namespaces.
    return value | (1 << 63) if domain.startswith("evaluation") else value


def argv_options(options):
    return [part for key, value in options.items() for part in ("--" + key, str(value))]


def command(directory, argv):
    directory.mkdir(parents=True, exist_ok=True)
    index = 0
    while (directory / f"call-{index:04}.json").exists():
        index += 1
    prefix = directory / f"call-{index:04}"
    publish(prefix.with_suffix(".json"), {"argv": argv})
    print(f"bootstrap_stage={directory.name} program={Path(argv[0]).name} logs={prefix}",
          file=sys.stderr, flush=True)
    with prefix.with_suffix(".stdout").open("x") as out, prefix.with_suffix(".stderr").open("x") as err:
        completed = subprocess.run(argv, stdout=out, stderr=err, check=False)
    if completed.returncode:
        raise RuntimeError(f"{Path(argv[0]).name} exited {completed.returncode}; see {prefix}.stderr; resume with run")
    return prefix.with_suffix(".stdout").read_text()


def evaluate(root, plan, candidate, cycle):
    directory = root / f"evaluation-{cycle:03}"
    directory.mkdir(exist_ok=True)
    receipt = directory / "complete.json"
    if receipt.exists():
        result = read_json(receipt)
        verify(result["candidate"])
        verify(result["evidence"])
        if result["candidate"] != identity(candidate):
            raise ValueError("evaluation candidate changed")
        return result
    archive = directory / "archive"
    options = dict(plan["evaluation"])
    options.update({"output-dir": archive, "candidate-checkpoint": candidate,
                    "opponent": "random", "service": plan["binaries"]["service"]["path"],
                    "preset": plan["preset"], "level": plan["level"],
                    "start-seed": seed(plan["seed"], "evaluation-start", cycle),
                    "model-seed": seed(plan["seed"], "evaluation-model", cycle), "first-pair-id": 0})
    executable = plan["binaries"]["evaluate"]["path"]
    command(directory, [executable] + argv_options(options))
    command(directory, [executable, "--verify", str(archive)])
    evidence = archive / "result/result.json"
    conclusion = read_json(evidence)["analysis"]["conclusion"]
    if conclusion not in CONCLUSIONS:
        raise ValueError(f"unknown evaluation conclusion: {conclusion}")
    result = {"candidate": identity(candidate), "evidence": identity(evidence), "conclusion": conclusion}
    publish(receipt, result)
    return result


def teacher(root, plan, cycle):
    directory = root / f"cycle-{cycle:03}"
    directory.mkdir(exist_ok=True)
    receipt = directory / "teacher.json"
    if receipt.exists():
        result = read_json(receipt)
        for item in [result["snapshot"], result["provenance"], *result["snapshot"]["shards"]]:
            verify(item)
        return Path(result["snapshot"]["path"])
    index = 0
    while True:
        attempt = directory / f"teacher-attempt-{index:03}"
        corpus = attempt / "corpus"
        snapshot = corpus / "snapshot.psrsnap"
        if snapshot.exists():
            break  # Snapshot-last publication survived a sidecar interruption.
        if not attempt.exists():
            attempt.mkdir()
            options = {"positions": plan["positions"], "seed": seed(plan["seed"], "teacher", cycle),
                       "workers": plan["teacher_workers"], "output": corpus}
            if "teacher_simulations" in plan:
                options["simulations"] = plan["teacher_simulations"]
            sources = [part for source in plan["sources"] for part in ("--source", source["path"])]
            command(attempt, [plan["binaries"]["teacher"]["path"]] + sources + argv_options(options))
            if not snapshot.exists():
                raise ValueError("teacher succeeded without publishing snapshot")
            break
        index += 1  # Preserve incomplete output; never overwrite or retry there.
    provenance = corpus / "provenance.json"
    details = read_json(provenance)
    if details["seed"] != seed(plan["seed"], "teacher", cycle) or not 0 < details["selected_positions"] <= plan["positions"]:
        raise ValueError("teacher provenance mismatch")
    publish(receipt, {"snapshot": source_identity(snapshot), "provenance": identity(provenance)})
    return snapshot


def train(root, plan, candidate, snapshot, cycle):
    directory = root / f"cycle-{cycle:03}"
    receipt = directory / "learner.json"
    if receipt.exists():
        result = read_json(receipt)
        verify(result["checkpoint"])
        return Path(result["checkpoint"]["path"])
    meta = checkpoint(candidate)
    target = meta["trainingStep"] + plan["steps"]
    options = {"service": plan["binaries"]["service"]["path"], "snapshot": snapshot,
        "replay-dir": snapshot.parent, "run-dir": directory / "learner", "initial-checkpoint": candidate,
        "preset": plan["preset"], "level": plan["level"], "batch": plan["batch"],
        "actions": plan["actions"], "model-seed": seed(plan["seed"], "model", cycle),
        "sampler-seed": seed(plan["seed"], "sampler", cycle),
        "generation": max(meta["progress"]["generation"], plan.get("generation_floor", 0)) + 1,
        "learning-rate": plan["learning_rate"], "objective": "supervised",
        "target-step": target, "checkpoint-every": plan["checkpoint_every"]}
    # Same run directory and absolute target on retry: the existing learner owns
    # commit verification, orphan recovery, Adam restoration and skipped steps.
    output = command(directory, [plan["binaries"]["train"]["path"]] + argv_options(options))
    summary = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
    selected = Path(summary["latest_checkpoint"]).resolve(strict=True)
    if not selected.is_relative_to((directory / "learner").resolve()) or int(summary["completed_training_step"]) != target:
        raise ValueError("unexpected learner summary")
    if checkpoint(selected)["trainingStep"] != target:
        raise ValueError("learner checkpoint target mismatch")
    publish(receipt, {"checkpoint": identity(selected), "target_step": target, "summary": summary})
    return selected


def select_teacher_candidate(root, plan, incumbent, candidate, cycle):
    """V5: retain the incumbent unless independent paired play promotes the candidate."""
    directory = root / f"promotion-{cycle:03}"
    directory.mkdir(exist_ok=True)
    receipt = directory / "selection.json"
    if receipt.exists():
        result = read_json(receipt)
        for key in ("incumbent", "candidate", "selected"):
            verify(result[key])
        if result["incumbent"] != identity(incumbent) or result["candidate"] != identity(candidate):
            raise ValueError("teacher selection checkpoint mismatch")
        return Path(result["selected"]["path"])
    options = {key: value for key, value in plan["evaluation"].items()
               if key not in ("lower-elo", "elo0", "elo1")}
    options.update({"output-dir": directory / "archive", "candidate-checkpoint": candidate,
        "champion-checkpoint": incumbent, "service": plan["binaries"]["service"]["path"],
        "preset": plan["preset"], "level": plan["level"], "elo0": 0, "elo1": 10,
        "start-seed": seed(plan["seed"], "evaluation-promotion-start", cycle),
        "model-seed": seed(plan["seed"], "evaluation-promotion-model", cycle), "first-pair-id": 0})
    output = command(directory, [plan["binaries"]["promote"]["path"]] + argv_options(options))
    conclusions = [line.removeprefix("conclusion=") for line in output.splitlines() if line.startswith("conclusion=")]
    if len(conclusions) != 1 or conclusions[0] not in {
            "PromoteCandidate", "RejectCandidate", "InconclusiveMaximumEligiblePairs", "InconclusiveMaximumAttemptedPairs"}:
        raise ValueError("invalid teacher promotion conclusion")
    selected = candidate if conclusions[0] == "PromoteCandidate" else incumbent
    publish(receipt, {"incumbent": identity(incumbent), "candidate": identity(candidate),
                     "selected": identity(selected), "conclusion": conclusions[0]})
    return selected


def run(root):
    root = Path(root).resolve(strict=True)
    # No controller access. Serialize only this new campaign's sidecar.
    with (root / ".bootstrap.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "result.json").exists():
            return read_json(root / "result.json")
        binding = {"plan_sha256": digest(root / "plan.json")}
        if (root / "execution.json").exists():
            if read_json(root / "execution.json") != binding:
                raise ValueError("plan changed since execution began")
        else:
            publish(root / "execution.json", binding)
        plan = read_json(root / "plan.json")
        for record in [plan["initial_checkpoint"], plan["authority"], *plan["binaries"].values(),
                       *plan["sources"], *(s for source in plan["sources"] for s in source["shards"])]:
            verify(record)
        candidate = Path(plan["initial_checkpoint"]["path"])
        for cycle in range(plan["max_cycles"] + 1):
            evaluation = evaluate(root, plan, candidate, cycle)
            passed = evaluation["conclusion"] == UPPER
            if plan.get("champion_only"):
                passed = passed and cycle >= plan["minimum_teacher_cycles"] and candidate != Path(plan["initial_checkpoint"]["path"])
            if passed or cycle == plan["max_cycles"]:
                result = {"status": "gate_passed" if passed else "budget_exhausted",
                    "gate_passed": passed, "teacher_omitted": cycle == 0,
                    "cycles_completed": cycle, "selected_checkpoint": str(candidate),
                    "selected_checkpoint_sha256": digest(candidate), "evaluation": evaluation,
                    "plan_sha256": digest(root / "plan.json"), "teacher_stopped_permanently": True,
                    "next_action": "parent decides whether to start terminal PPO without teacher data"}
                publish(root / "result.json", result)
                return result
            snapshot = teacher(root, plan, cycle + 1)
            trained = train(root, plan, candidate, snapshot, cycle + 1)
            candidate = (select_teacher_candidate(root, plan, candidate, trained, cycle + 1)
                         if plan.get("champion_only") else trained)


def parser():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = cli.add_subparsers(dest="command", required=True)
    resume = sub.add_parser("run")
    resume.add_argument("--campaign", type=Path, required=True)
    create = sub.add_parser("plan", help="persist provenance; does not execute any child")
    create.add_argument("--campaign", type=Path, required=True)
    create.add_argument("--initial-checkpoint", type=Path, required=True)
    create.add_argument("--source", type=Path, action="append", required=True)
    create.add_argument("--service", type=Path, required=True)
    for name, executable in (("teacher", "paisho-teacher-relabel"), ("train", "paisho-train"), ("evaluate", "paisho-evaluate")):
        create.add_argument("--" + name + "-executable", type=Path, default=ROOT / "target/release" / executable)
    for name, default in (("max-cycles", 5), ("positions", 2048), ("steps", 32), ("batch", 64),
                          ("seed", 17), ("teacher-workers", os.cpu_count() or 1), ("checkpoint-every", 8)):
        create.add_argument("--" + name, type=int, default=default)
    create.add_argument("--champion-only", action="store_true")
    create.add_argument("--minimum-teacher-cycles", type=int, default=1)
    create.add_argument("--promote-executable", type=Path, default=ROOT / "target/release/paisho-promote")
    create.add_argument("--actions", type=int, choices=(128, 1024), default=1024)
    create.add_argument("--preset", choices=("pure", "micro"), default="pure")
    create.add_argument("--level", type=int, choices=(0, 1), default=1)
    create.add_argument("--learning-rate", type=float, default=1e-4)
    for name, default in {**EVAL_DEFAULTS, "workers": os.cpu_count() or 1}.items():
        create.add_argument("--eval-" + name, type=type(default), default=default)
    create.add_argument("--eval-pairs-per-batch", type=int,
                        help="defaults to the selected --eval-workers")
    return cli


def create_plan(args):
    meta = checkpoint(args.initial_checkpoint)
    for name in ("max_cycles", "positions", "steps", "batch", "teacher_workers", "checkpoint_every"):
        if getattr(args, name) <= 0:
            raise ValueError(name + " must be positive")
    if not 0 <= args.seed < 1 << 64 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("invalid seed or learning rate")
    evaluation = {key[5:].replace("_", "-"): value for key, value in vars(args).items() if key.startswith("eval_")}
    if evaluation["pairs-per-batch"] is None:
        evaluation["pairs-per-batch"] = evaluation["workers"]
    if not all(math.isfinite(evaluation[x]) for x in ("lower-elo", "elo0", "elo1")):
        raise ValueError("evaluation Elo hypotheses must be finite")
    if not evaluation["lower-elo"] < evaluation["elo0"] < evaluation["elo1"] or evaluation["elo0"] < 0:
        raise ValueError("evaluation needs lower < nonnegative center < upper")
    if not all(0 < evaluation[x] < 1 for x in ("alpha", "beta")):
        raise ValueError("alpha/beta must lie in (0,1)")
    for name in ("max-attempted-pairs", "max-eligible-pairs", "workers", "pairs-per-batch",
                 "decision-limit", "start-source-limit", "start-source-attempts"):
        if evaluation[name] <= 0:
            raise ValueError("evaluation " + name + " must be positive")
    if evaluation["max-attempted-pairs"] < evaluation["max-eligible-pairs"]:
        raise ValueError("attempted-pair budget must cover eligible-pair budget")
    if evaluation["wait-us"] < 0 or evaluation["start-horizon"] < 0:
        raise ValueError("evaluation wait and horizon must be nonnegative")
    if not math.isfinite(evaluation["sampling-temperature"]) or evaluation["sampling-temperature"] < 0 or not 0 <= evaluation["sampling-uniform-mix"] <= 1:
        raise ValueError("invalid evaluation sampling")
    for item in evaluation["classes"].split(","):
        parts = item.split(":")
        if len(parts) != 2 or any(int(x) <= 0 for x in parts):
            raise ValueError("evaluation classes require positive CAPACITY:BATCH")
    plan = {"version": 1, "initial_checkpoint": identity(args.initial_checkpoint),
            "initial_training_step": meta["trainingStep"], "initial_generation": meta["progress"]["generation"],
            "sources": [source_identity(p) for p in args.source],
            "orchestrator": identity(__file__),
            "authority": identity(ROOT / "docs/authority/CURRICULUM_V4.md"),
            "binaries": {name: identity(getattr(args, name + "_executable")) for name in ("teacher", "train", "evaluate")},
            "evaluation": evaluation, "teacher": "offline MCTS-8; policy weight 1; original terminal WDL",
            "seed_protocol": "SHA256 domain/cycle v1; evaluation high bit set, teacher/sampler high bit cleared"}
    if args.champion_only:
        if not 1 <= args.minimum_teacher_cycles <= args.max_cycles:
            raise ValueError("minimum teacher cycles must be within the positive cycle budget")
        plan.update({"version": 2, "champion_only": True,
                     "minimum_teacher_cycles": args.minimum_teacher_cycles,
                     "authority": identity(ROOT / "docs/authority/CURRICULUM_V5.md")})
        plan["binaries"]["promote"] = identity(args.promote_executable)
    plan["budgets"] = {"maximum_teacher_positions": args.max_cycles * args.positions,
                       "maximum_training_steps": args.max_cycles * args.steps,
                       "maximum_evaluations": args.max_cycles + 1}
    plan["binaries"]["service"] = identity(args.service)
    for name in ("max_cycles", "positions", "steps", "batch", "seed", "teacher_workers",
                 "checkpoint_every", "actions", "preset", "level", "learning_rate"):
        plan[name] = getattr(args, name)
    root = args.campaign.resolve()
    if any(Path(source["path"]).is_relative_to(root) for source in plan["sources"]) or Path(plan["initial_checkpoint"]["path"]).is_relative_to(root):
        raise ValueError("new campaign must not contain its source artifacts")
    root.mkdir()  # Explicitly NEW: never repurpose a stable/existing campaign.
    publish(root / "plan.json", plan)
    return {"status": "planned", "campaign": str(root), "plan_sha256": digest(root / "plan.json")}


def main(argv=None):
    """CLI entry point; returns 0 or 1. argparse raises SystemExit(2) on usage errors."""
    try:
        args = parser().parse_args(argv)
        print(json.dumps(create_plan(args) if args.command == "plan" else run(args.campaign), sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"bootstrap: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
