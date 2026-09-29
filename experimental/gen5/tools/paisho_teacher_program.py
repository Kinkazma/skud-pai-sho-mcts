#!/usr/bin/env python3
"""Persistent MCTS-32 teaching / PPO alternation until the independent Random goal.

Create a program with plan --campaign NEW --learner CKPT --champion CKPT
--source SNAPSHOT (repeatable) --bin-directory BIN --live-options JSON.
Then run --campaign NEW under paisho_control. No occurrence-count exit.
Immutable episode/chunk receipts own recovery; program-status.json is telemetry.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import tempfile

import paisho_teacher_bootstrap as b

INCONCLUSIVE = {"InconclusiveMaximumEligiblePairs", "InconclusiveMaximumAttemptedPairs"}


def teacher_disabled(root):
    path = Path(root) / "teacher-disabled.json"
    if not path.exists():
        return False
    value = b.read_json(path)
    if value.get("disabled") is not True:
        raise ValueError("invalid teacher disable instruction")
    return True


def write_status(root, plan, state, **updates):
    value = {"format": "paisho-teacher-program-status-v1", "phase": "starting",
        "teacher_disabled": teacher_disabled(root),
        "generation": state["generation"], "durable_generation": state["generation"],
        "checkpoint": state["learner"]["path"], "champion_checkpoint": state["champion"]["path"],
        "completed_step": state["training_step"], "teacher_steps_completed": state["teacher_steps"],
        "chunks_per_occurrence": plan["chunks"], "steps_per_chunk": plan["steps"],
        "active_live_directory": None, "learner_run_directory": None,
        "last_evaluation": state.get("last_evaluation"), "goal_passed": state.get("teacher_complete", False), "site_ready": state.get("site_ready", False), "site_entry_min_win_rate": 0.7, **updates}
    with tempfile.NamedTemporaryFile(dir=root, prefix=".status-", mode="w", delete=False) as stream:
        json.dump(value, stream); stream.flush(); os.fsync(stream.fileno()); temporary = stream.name
    os.replace(temporary, root / "program-status.json")


def achieved(wins, draws, games):
    # Integer arithmetic defines the user's exact boundary, including equality.
    return games > 0 and (100 * wins >= 80 * games or
                         (100 * wins >= 60 * games and 100 * draws >= 20 * games))


def verify_state(state):
    b.verify(state["learner"]); b.verify(state["champion"])
    return state


def measure(root, plan, candidate, round_id, chunk):
    """Fixed attempted-pair panel. Exclusions never improve the denominator."""
    root.mkdir(parents=True, exist_ok=True)
    receipt = root / "complete.json"
    if receipt.exists():
        result = b.read_json(receipt)
        if result["candidate"] != b.identity(candidate):
            raise ValueError("goal evaluation candidate changed")
        for record in result["evidence"]: b.verify(record)
        return result
    wins = draws = losses = attempted = 0
    evidence = []
    for batch, first in enumerate(range(0, plan["goal_pairs"], plan["goal_batch_pairs"])):
        count = min(plan["goal_batch_pairs"], plan["goal_pairs"] - first)
        directory = root / f"batch-{batch:03}"
        archive = directory / "archive"
        options = dict(plan["goal_evaluation"])
        options.update({"output-dir": archive, "candidate-checkpoint": candidate,
            "service": plan["binaries"]["service"]["path"], "preset": "pure", "level": 1,
            "opponent": "random", "pairs-per-batch": count, "max-attempted-pairs": count,
            "max-eligible-pairs": count, "first-pair-id": first,
            "model-seed": b.seed(plan["seed"], "evaluation-goal-model", round_id * 100000 + chunk * 100 + batch),
            "start-seed": b.seed(plan["seed"], "evaluation-goal-start", round_id * 100000 + chunk * 100 + batch)})
        result_file = archive / "result/result.json"
        # Existing CLI owns partial-archive recovery. Completed evidence is verified once.
        if not (directory / "verified.json").exists():
            executable = plan["binaries"]["evaluate"]["path"]
            b.command(directory, [executable] + b.argv_options(options))
            b.command(directory, [executable, "--verify", str(archive)])
            b.publish(directory / "verified.json", b.identity(result_file))
        record = b.read_json(directory / "verified.json"); b.verify(record)
        analysis = b.read_json(result_file)["analysis"]
        if analysis["attempted_pairs"] != count:
            raise ValueError("goal panel did not execute its fixed attempted-pair budget")
        attempted += count
        wins += analysis["candidate_wins"]; draws += analysis["draws"]; losses += analysis["candidate_losses"]
        evidence.append(record)
    games = attempted * 2
    if games != plan["goal_pairs"] * 2 or min(wins, draws, losses) < 0 or wins + draws + losses > games:
        raise ValueError("invalid goal panel counts")
    result = {"candidate": b.identity(candidate), "opponent": "random", "wins": wins,
        "draws": draws, "losses": losses, "games": games, "unresolved": games - wins - draws - losses,
        "win_rate": wins / games, "draw_rate": draws / games,
        "passed": achieved(wins, draws, games), "site_ready": 100 * wins >= 70 * games, "evidence": evidence}
    b.publish(receipt, result)
    return result


def measure_selected(root, plan, state, champion, occurrence, chunk):
    # An unchanged network against the same fixed Random protocol needs no new panel.
    previous = state.get("last_evaluation")
    if previous and previous.get("candidate") == b.identity(champion):
        for record in previous["evidence"]: b.verify(record)
        return previous
    return measure(root, plan, champion, occurrence, chunk)


def live_segment(root, plan, state, target):
    args = dict(plan["live_options"])
    args.update({"campaign-dir": str(root), "curriculum-dir": str(root / "curriculum"),
        "initial-checkpoint": state["learner"]["path"], "initial-champion-checkpoint": state["champion"]["path"],
        "target-generation": target, "initial-generation": state["generation"],
        "opponent": state.get("tier", "random") if state.get("site_ready") else "random",
        "hold-random": "false" if state.get("site_ready") else "true",
        "service": plan["binaries"]["service"]["path"], "promote-executable": plan["binaries"]["promote"]["path"],
        "evaluate-executable": plan["binaries"]["evaluate"]["path"],
        "champion-only": "true", "continue-inconclusive": "true", "mcts-ceiling": 32})
    b.command(root.parent / (root.name + "-calls"), [plan["binaries"]["live"]["path"]] + b.argv_options(args))
    block = b.read_json(root / "blocks" / f"block-{target:020}.json")
    assessment_file = root / "assessments" / f"assessment-{target:020}.json"
    assessment = b.read_json(assessment_file) if assessment_file.exists() else {}
    learner = block["checkpoint"] if assessment.get("promotion") in INCONCLUSIVE else assessment.get("selected", block["checkpoint"])
    meta = b.checkpoint(learner)
    sources = sorted(root.glob("attempts/*/generation-*/snapshot.psrsnap"), key=lambda p: (p.parent.name, p.parent.parent.name))
    # For a resumed generation, keep only the newest completed attempt's corpus.
    unique = {p.parent.name: p for p in sources}
    sources = list(unique.values())[-plan["source_generations"]:]
    if not sources: raise ValueError("PPO segment published no fresh source snapshots")
    return {**state, "learner": b.identity(learner), "champion": b.identity(assessment.get("champion", block["champion"])),
        "generation": target, "training_step": meta["trainingStep"],
        "sources": [b.source_identity(p) for p in sources], "tier": assessment.get("tier", block["tier"])}


def execute(root):
    root = Path(root).resolve(strict=True)
    with (root / ".program.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = b.read_json(root / "teacher-program-plan.json")
        binding = {"plan_sha256": b.digest(root / "teacher-program-plan.json")}
        if (root / "execution.json").exists():
            if b.read_json(root / "execution.json") != binding: raise ValueError("program plan changed")
        else: b.publish(root / "execution.json", binding)
        if (root / "complete.json").exists(): return verify_state(b.read_json(root / "complete.json"))
        for record in [plan["authority"], *plan["binaries"].values()]: b.verify(record)
        state = verify_state(plan["initial_state"])
        write_status(root, plan, state, phase="teacher-evaluation", occurrence=0, chunk=0)
        baseline = measure(root / "initial-goal", plan, Path(state["champion"]["path"]), 0, 0)
        state.update({"last_evaluation": baseline, "teacher_complete": baseline["passed"], "site_ready": baseline["site_ready"]})
        if not (state["teacher_complete"] and state["site_ready"]):
            warm_receipt = root / "warmup.json"
            if warm_receipt.exists(): state = verify_state(b.read_json(warm_receipt))
            else:
                write_status(root, plan, state, phase="ppo", occurrence=0,
                    active_live_directory=str(root / "warmup-ppo"))
                state = live_segment(root / "warmup-ppo", plan, state, state["generation"] + plan["warmup_generations"])
                b.publish(warm_receipt, state)
        occurrence = 1
        while not (state.get("teacher_complete") and state.get("site_ready")):
            episode = root / f"occurrence-{occurrence:06}"
            episode.mkdir(exist_ok=True)
            if (episode / "complete.json").exists():
                state = verify_state(b.read_json(episode / "complete.json"))
                if state.get("teacher_complete") and state.get("site_ready"): break
                occurrence += 1; continue
            if not (episode / "input.json").exists(): b.publish(episode / "input.json", state)
            elif b.read_json(episode / "input.json") != state: raise ValueError("occurrence input changed")
            teacher_plan = {**plan, "sources": state["sources"], "seed": b.seed(plan["seed"], "teacher-occurrence", occurrence)}
            for source in teacher_plan["sources"]:
                b.verify(source)
                for shard in source["shards"]: b.verify(shard)
            for chunk in range(1, plan["chunks"] + 1):
                receipt = episode / f"chunk-{chunk:03}.json"
                if receipt.exists():
                    state = verify_state(b.read_json(receipt))
                    if state.get("teacher_complete"): break
                    continue
                if teacher_disabled(root) or state.get("teacher_complete") or state.get("tier") == "self":
                    break
                status = {"occurrence": occurrence, "chunk": chunk}
                write_status(root, plan, state, phase="teacher-targets", **status)
                snapshot = b.teacher(episode, teacher_plan, chunk)
                provenance = b.read_json(snapshot.parent / "provenance.json")
                if provenance["teacher"]["protocol"] != "paisho-offline-mcts32-teacher-v1":
                    raise ValueError("this program requires the current engine's MCTS-32 teacher")
                write_status(root, plan, state, phase="teacher-learning",
                    learner_run_directory=str(episode / f"cycle-{chunk:03}" / "learner"), **status)
                trained = b.train(episode, {**teacher_plan, "generation_floor": state["generation"]}, Path(state["learner"]["path"]), snapshot, chunk)
                write_status(root, plan, state, phase="teacher-promotion", **status)
                champion = b.select_teacher_candidate(episode, teacher_plan, Path(state["champion"]["path"]), trained, chunk)
                write_status(root, plan, state, phase="teacher-evaluation", **status)
                evaluation = measure_selected(episode / f"goal-{chunk:03}", plan, state, champion, occurrence, chunk)
                meta = b.checkpoint(trained)
                state = {**state, "learner": b.identity(trained), "champion": b.identity(champion),
                    "generation": meta["progress"]["generation"], "training_step": meta["trainingStep"],
                    "teacher_steps": state["teacher_steps"] + plan["steps"],
                    "last_evaluation": evaluation, "teacher_complete": evaluation["passed"], "site_ready": evaluation["site_ready"], "goal_passed": evaluation["passed"]}
                b.publish(receipt, state)
                if state["goal_passed"]: break
            if not (state.get("teacher_complete") and state.get("site_ready")):
                target = state["generation"] + plan["ppo_generations"]
                write_status(root, plan, state, phase="ppo", occurrence=occurrence,
                    active_live_directory=str(episode / "ppo"))
                state = live_segment(episode / "ppo", plan, state, target)
                evaluation = measure_selected(episode / "goal-after-ppo", plan, state, Path(state["champion"]["path"]), occurrence, plan["chunks"] + 1)
                state["last_evaluation"] = evaluation
                state["teacher_complete"] = state.get("teacher_complete", False) or evaluation["passed"]
                state["site_ready"] = evaluation["site_ready"]
                state["goal_passed"] = state["teacher_complete"]
                if state["site_ready"] and state.get("tier") == "random": state["tier"] = "site"
            b.publish(episode / "complete.json", state)
            if state.get("teacher_complete") and state.get("site_ready"): break
            occurrence += 1
        if state.get("site_ready") and state.get("tier") == "random": state["tier"] = "site"
        # No G1000 exit before the goal. Afterwards preserve the existing PPO target.
        if not (root / "goal-achieved.json").exists(): b.publish(root / "goal-achieved.json", state)
        if state["generation"] < plan["post_goal_target"]:
            write_status(root, plan, state, phase="ppo", goal_passed=True,
                active_live_directory=str(root / "post-goal-ppo"))
            state = live_segment(root / "post-goal-ppo", plan, state, plan["post_goal_target"])
        write_status(root, plan, state, phase="goal-achieved", goal_passed=True)
        b.publish(root / "complete.json", state)
        return state


def create(args):
    root = args.campaign.resolve()
    learner, champion = b.identity(args.learner), b.identity(args.champion)
    meta = b.checkpoint(args.learner)
    live_options = b.read_json(args.live_options)
    for key in ("initial-checkpoint", "initial-champion-checkpoint", "campaign-dir", "target-generation", "opponent"):
        live_options.pop(key, None)
    live_options.update({"mixed-self-play":"true", "win-duration-reward":"true", "site-min-win-rate-70":"true", "curriculum-start-horizon":"0", "curriculum-decision-limit":"2048"})
    live_options.setdefault("start-horizon", "64")
    live_options.setdefault("actor-decision-limit", "256")
    binaries = {name: b.identity(args.bin_directory / executable) for name, executable in {
        "teacher": "paisho-teacher-relabel", "train": "paisho-train", "evaluate": "paisho-evaluate",
        "promote": "paisho-promote", "live": "paisho-live-v7", "service": "paisho-mpsgraph-service"}.items()}
    evaluation = {**b.EVAL_DEFAULTS, "workers": 20, "pairs-per-batch": 10,
                  "max-attempted-pairs": 256, "max-eligible-pairs": 128, "decision-limit": 256}
    goal = {**evaluation, "start-horizon": 0, "decision-limit": 2048}
    plan = {"format": "paisho-teacher-program-v1", "authority": b.identity(b.ROOT / "docs/authority/CURRICULUM_V8.md"),
        "binaries": binaries, "live_options": live_options, "evaluation": evaluation, "goal_evaluation": goal,
        "goal_pairs": 200, "goal_batch_pairs": 10, "seed": 70332026,
        "chunks": 10, "steps": 6400, "positions": 65536, "teacher_simulations": 32, "teacher_workers": 20,
        "batch": 64, "actions": 1024, "learning_rate": 0.0001, "checkpoint_every": 1280,
        "preset": "pure", "level": 1, "ppo_generations": 50, "source_generations": 20,
        "post_goal_target": 1000, "warmup_generations": 10,
        "initial_state": {"learner": learner, "champion": champion,
            "generation": meta["progress"]["generation"], "training_step": meta["trainingStep"],
            "teacher_steps": 0, "tier": args.tier, "sources": [b.source_identity(p) for p in args.source]}}
    root.mkdir()
    b.publish(root / "teacher-program-plan.json", plan)
    b.publish(root / "teacher-disabled.json", {"disabled": True, "authority": "CURRICULUM_V8"})
    write_status(root, plan, plan["initial_state"], phase="planned")
    return {"campaign": str(root), "plan_sha256": b.digest(root / "teacher-program-plan.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run"); run.add_argument("--campaign", type=Path, required=True)
    plan = sub.add_parser("plan")
    for name in ("campaign", "learner", "champion", "bin-directory", "live-options"):
        plan.add_argument("--" + name, type=Path, required=True)
    plan.add_argument("--source", type=Path, action="append", required=True)
    plan.add_argument("--tier", default="random", choices=("random", "site", "mcts:8", "mcts:32", "self"))
    args = parser.parse_args()
    try:
        result = execute(args.campaign) if args.command == "run" else create(args)
        print(json.dumps(result, sort_keys=True)); return 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"teacher-program: {error}", file=sys.stderr, flush=True); return 1


if __name__ == "__main__": sys.exit(main())
