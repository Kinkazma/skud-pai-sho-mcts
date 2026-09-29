#!/usr/bin/env python3
"""Counterfactual collection deadlines from saved compact-MCTS game traces.

No matches are run and no artifacts are modified. This extends the occupied-
worker denominator in benchmark_mcts_deadline_sweep.py: completed / sum(min(t,d)).
Right-censored traces shorter than d produce bounds, never invented completions
or synthetic draws. This diagnostic neither starts runs nor retunes fixed profiles.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct

LEGACY_COMPARISON_REFERENCE = "CpuMctsEvaluator with historical HeuristicWeights::default; no learned coefficients"


def comparison_reference(plan, directory):
    """Recognize archived legacy plans without treating learned512 as old512."""
    artifact = plan.get("reference_model")
    reuse = {key: plan.get(key, False) for key in ("candidate_reuse", "reference_reuse")}
    if any(not isinstance(value, bool) for value in reuse.values()):
        raise ValueError("comparison reuse options must be booleans")
    kind = plan.get("reference_kind")
    if kind is None:
        kind = ("compact-model" if artifact is not None else
                "legacy-cpu-heuristic" if plan.get("reference") == LEGACY_COMPARISON_REFERENCE else "unknown")
    if kind not in ("compact-model", "legacy-cpu-heuristic", "legacy-cpu-heuristic-retained", "unknown"):
        raise ValueError("unsupported comparison reference kind")
    if kind in ("legacy-cpu-heuristic", "legacy-cpu-heuristic-retained"):
        if (artifact is not None or plan.get("reference") != LEGACY_COMPARISON_REFERENCE
                or reuse["reference_reuse"] != (kind == "legacy-cpu-heuristic-retained")):
            raise ValueError("legacy comparison reference identity mismatch")
    result = {**reuse, "reference_kind": kind, "reference_model_sha256": None,
              "reference_weights_sha256": None, "reference_training_steps": None}
    if kind == "compact-model":
        if not isinstance(artifact, dict):
            raise ValueError("learned comparison reference needs a frozen model artifact")
        path = directory / artifact["snapshot"]
        if digest(path.read_bytes()) != artifact["sha256"]:
            raise ValueError("reference snapshot hash mismatch")
        result.update(reference_model_sha256=artifact["sha256"],
                      reference_weights_sha256=model_hash(path),
                      reference_training_steps=artifact.get("training_steps"))
    elif artifact is not None:
        raise ValueError("nonlearned comparison reference has a model artifact")
    return result


def comparison_evaluators(plan, reference, game=None):
    """New archives bind both frozen roles on every leg; old plans remain readable."""
    expected = {"candidate_artifact_sha256": plan["candidate"]["sha256"],
                "reference_kind": reference["reference_kind"],
                "reference_artifact_sha256": reference["reference_model_sha256"],
                "source_sha256": plan["build"]["source_sha256"]}
    # Old archives predate these opt-in flags; new ones bind search policy to
    # both roles as well as binding the immutable coefficient snapshots.
    for key in ("candidate_reuse", "reference_reuse"):
        if key in plan:
            expected[key] = reference[key]
    if plan.get("evaluators") is not None and plan["evaluators"] != expected:
        raise ValueError("plan evaluator identities disagree with frozen artifacts")
    if game is not None and (plan.get("evaluators") is not None or game.get("evaluators") is not None):
        if game.get("evaluators") != expected:
            raise ValueError("game evaluator identities disagree with frozen artifacts")
    return expected


def digest(data):
    return hashlib.sha256(data).hexdigest()


def finite_positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def count(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def model_hash(path):
    value = json.loads(path.read_text())
    weights = value["weights"]
    if len(weights) != 64 or any(not math.isfinite(weight) for weight in weights):
        raise ValueError("invalid compact-model weights")
    raw = value["feature_schema"].encode() + b"\0"
    return digest(raw + b"".join(struct.pack("<d", weight) for weight in weights))


def psr_decisions(path, expected_hash):
    raw = path.read_bytes()
    if digest(raw) != expected_hash:
        raise ValueError(f"PSR hash mismatch: {path}")
    lines = [line.strip() for line in raw.decode().splitlines() if line.strip()]
    if not lines or lines[0] != "PAISHO-RECORD 1" or "actions" not in lines:
        raise ValueError(f"invalid PSR header: {path}")
    return len(lines) - lines.index("actions") - 1


@dataclass(frozen=True)
class Observation:
    identity: str
    elapsed: float
    disposition: str  # terminal, time-censored, decision-limit, repetition-adjudicated, no-legal-actions, error
    decisions: int
    learner_roots: int
    archived_samples: int
    decision_limit: int
    termination: str
    outcome: str | None
    archived_sample_capacity: int = 32


def load_run(directory):
    directory = directory.resolve()
    plan_path = directory / "plan.json"
    plan = json.loads(plan_path.read_text())
    schema = plan.get("schema")
    if schema not in ("paisho-compact-selfplay-plan-v1", "paisho-compact-comparison-plan-v1"):
        raise ValueError(f"unsupported compact plan: {directory}")
    compare = schema == "paisho-compact-comparison-plan-v1"
    groups = defaultdict(list)
    skipped = []
    hashes = {"plan.json": digest(plan_path.read_bytes())}
    source = plan["build"]["source_sha256"] if compare else plan["build_source_sha256"]
    model_cache = {}
    if compare:
        reference = comparison_reference(plan, directory)
        comparison_evaluators(plan, reference)
        candidate_path = directory / plan["candidate"]["snapshot"]
        if digest(candidate_path.read_bytes()) != plan["candidate"]["sha256"]:
            raise ValueError("frozen comparison model hash mismatch")
        candidate_hash = model_hash(candidate_path)
        hashes[str(candidate_path.relative_to(directory))] = digest(candidate_path.read_bytes())
        if reference["reference_model_sha256"] is not None:
            hashes[plan["reference_model"]["snapshot"]] = reference["reference_model_sha256"]
    for path in sorted((directory / "games").glob("game-*.json")):
        if path.name.endswith(".learning.json"):
            continue
        game = json.loads(path.read_text())
        hashes[str(path.relative_to(directory))] = digest(path.read_bytes())
        expected_schema = "paisho-compact-comparison-game-v1" if compare else "paisho-compact-selfplay-game-v1"
        if game.get("schema") != expected_schema:
            raise ValueError(f"unsupported game metadata: {path}")
        if compare:
            comparison_evaluators(plan, reference, game)
            termination = game["termination"]["kind"]
            if termination == "not_played":
                if game["wall_seconds"] != 0 or game["decisions"] != 0 or game["record"] is not None:
                    raise ValueError("not-played game contains an executed trace")
                skipped.append({"file":path.name,"reason":"not_played"})
                continue
            elapsed = finite_positive(game["wall_seconds"], "game wall time")
            disposition = {"rules":"terminal", "wall_limit":"time-censored",
                           "decision_limit":"decision-limit", "no_legal_actions":"no-legal-actions", "error":"error"}.get(termination)
            outcome = game["termination"].get("outcome")
            if disposition == "terminal" and outcome not in ("host_win", "guest_win", "draw"):
                raise ValueError("rules termination needs a real terminal outcome")
            record_path = directory / game["record"]
            learned = count(game["candidate"]["completed_decisions"], "candidate decisions")
            if learned + count(game["reference"]["completed_decisions"], "reference decisions") != game["decisions"]:
                raise ValueError("comparison side decision totals disagree")
            saved_samples = 0  # Comparison PSRs contain no recorded root-Q/features.
            archived_sample_capacity = 0
            budget = count(plan["mcts"]["simulations"], "simulation budget")
            limit = count(plan["decision_limit"], "decision limit")
            group = {"run":str(directory), "kind":"comparison-traces", "budget":budget,
                     "weights_sha256":candidate_hash,
                     "mode":{"legacy-cpu-heuristic":"vs-legacy", "legacy-cpu-heuristic-retained":"vs-retained-heuristic", "compact-model":"vs-compact-reference"}.get(reference["reference_kind"],"vs-unknown-reference"),
                     **reference,
                     "learner_seat":"host" if game["candidate_host"] else "guest",
                     "learning_enabled":False,"source_sha256":source,
                     "decision_limit":limit,"move_ms":plan.get("move_ms"),
                     "reference_budget":plan.get("reference_simulations",budget),
                     "workers":plan["workers"],"search":plan["mcts"]}
        else:
            termination = game["termination"]
            outcome = game["outcome"]
            if outcome not in ("host","guest","draw","ongoing"):
                raise ValueError("unknown self-play outcome")
            terminal = outcome != "ongoing" and game.get("error") is None
            disposition = "terminal" if terminal else {
                "decision-limit":"decision-limit", "no-legal-actions":"no-legal-actions", "error":"error",
                "repetition-training-loss":"repetition-adjudicated",
                "campaign-deadline":"time-censored", "game-deadline":"time-censored",
                "cancelled":"time-censored"}.get(termination)
            if terminal != bool(game["eligible_for_terminal_training"]):
                raise ValueError("terminal training flag disagrees with outcome/error")
            if terminal and termination != "terminal":
                raise ValueError("terminal outcome has an inconsistent termination")
            elapsed = finite_positive(game["elapsed_seconds"], "game elapsed time")
            record_path = path.parent / game["record_path"]
            learned = count(game["eligible_learner_roots"], "eligible learner roots")
            saved_samples = len(game["sampled_roots"])
            archived_sample_capacity = count(plan["options"]["samples"], "producer sample cap")
            if saved_samples > learned:
                raise ValueError("more saved samples than eligible roots")
            if saved_samples > archived_sample_capacity:
                raise ValueError("more saved samples than producer sample capacity")
            budget = count(game["simulations_requested_per_decision"], "simulation budget")
            limit = count(plan["options"]["decision_limit"], "decision limit")
            version = game["snapshot_version"]
            if version not in model_cache:
                snapshot = directory / "models" / f"version-{version:08}.json"
                model_cache[version] = model_hash(snapshot)
                hashes[str(snapshot.relative_to(directory))] = digest(snapshot.read_bytes())
            if model_cache[version] != game["weights_sha256"]:
                raise ValueError("game weights disagree with its frozen model version")
            if game["build_source_sha256"] != source:
                raise ValueError("game build source disagrees with run plan")
            assignments=plan.get("worker_assignments",[])
            assigned=[item for item in assignments if item["worker"]==game.get("worker")]
            expected_budget=(assigned[0]["simulations"] if len(assigned)==1 else plan["options"]["simulations"])
            if plan["options"].get("budgets") and len(assigned)!=1:
                raise ValueError("heterogeneous game needs one matching worker assignment")
            if budget != expected_budget:
                raise ValueError("game simulation budget disagrees with plan")
            group = {"run":str(directory),"kind":"selfplay-collection","budget":budget,
                     "weights_sha256":game["weights_sha256"],
                     "mode":"self-play" if game["self_play"] else "vs-legacy",
                     "learner_seat":game["learner_seat"],
                     "learning_enabled":game.get("learning_enabled",plan["options"].get("learn",True)),
                     "repetition_cycles":plan["options"].get("repetition_cycles",0),
                     "source_sha256":source,"decision_limit":limit,"move_ms":None,
                     "workers":plan["workers"],"search":plan["search"]}
        if disposition is None:
            raise ValueError(f"unknown game termination: {termination}")
        if not budget or not limit:
            raise ValueError("simulation and decision limits must be positive")
        decisions = count(game["decisions"], "decisions")
        if learned > decisions or decisions > limit:
            raise ValueError("learner/game decision counts violate the fixed decision limit")
        if psr_decisions(record_path, game["record_sha256"]) != decisions:
            raise ValueError("PSR decision count disagrees with game metadata")
        hashes[str(record_path.relative_to(directory))] = game["record_sha256"]
        key = json.dumps(group, sort_keys=True, separators=(",",":"))
        groups[key].append(Observation(str(path),elapsed,disposition,decisions,learned,
                                      saved_samples,limit,termination,outcome if disposition=="terminal" else None,
                                      archived_sample_capacity))
    return groups, {"run":str(directory),"files_sha256":hashes,"not_played":skipped,
                    "verification":"PSR hashes and action counts checked; terminal facts are from saved engine metadata, not a new rules replay."}


def quantile(values, fraction):
    ordered = sorted(values)
    offset = (len(ordered)-1)*fraction
    low = math.floor(offset)
    high = math.ceil(offset)
    return ordered[low] + (offset-low)*(ordered[high]-ordered[low])


def propose_grid(rows, budget):
    if not rows:
        return []
    observed = [row.elapsed for row in rows]
    # Include all observed terminal thresholds for small samples, quantiles for
    # larger ones, and the historical 8s point only for the MCTS-32 budget.
    terminal = [row.elapsed for row in rows if row.disposition=="terminal"]
    values = [quantile(observed,q) for q in (0.1,0.25,0.5,0.75,0.9,1.0)]
    if len(terminal) <= 12:
        values.extend(terminal)
    else:
        values.extend(quantile(terminal,q) for q in (0.1,0.25,0.5,0.75,0.9,1.0))
    if budget == 32:
        values.append(8.0)
    return sorted(set(value for value in values if value>0))


def ratio(numerator, denominator):
    return numerator/denominator if denominator>0 else None


def estimate(rows, deadline, sample_cap=32):
    finite_positive(deadline,"deadline")
    if sample_cap < 1:
        raise ValueError("sample cap must be positive")
    accepted, unknown, known_rejected = [], [], []
    for row in rows:
        if row.disposition=="terminal" and row.elapsed<=deadline:
            accepted.append(row)
        elif row.disposition=="time-censored" and row.elapsed<deadline:
            unknown.append(row)
        else:
            known_rejected.append(row)
    lower_cost = sum(min(row.elapsed,deadline) for row in rows)
    upper_cost = lower_cost + sum(deadline-row.elapsed for row in unknown)
    rejected_cost = sum(min(row.elapsed,deadline) for row in known_rejected)
    unknown_upper_cost = len(unknown)*deadline
    identified = not unknown
    quantities = {
        "terminal_games":(len(accepted),len(accepted)+len(unknown)),
        "potential_terminal_examples_capped":(
            sum(min(sample_cap,row.learner_roots) for row in accepted),
            sum(min(sample_cap,row.learner_roots) for row in accepted)
                +sum(min(sample_cap,row.decision_limit) for row in unknown)),
        "retained_learner_roots":(sum(row.learner_roots for row in accepted),
            sum(row.learner_roots for row in accepted)+sum(row.decision_limit for row in unknown)),
        "retained_psr_decisions":(sum(row.decisions for row in accepted),
            sum(row.decisions for row in accepted)+sum(row.decision_limit for row in unknown)),
        "archived_root_samples":(sum(row.archived_samples for row in accepted),
            sum(row.archived_samples for row in accepted)
                +sum(min(row.archived_sample_capacity,row.decision_limit) for row in unknown))}
    rates = {name:{"known_retained":minimum,"possible_retained_upper":maximum,
                   "per_occupied_worker_second":ratio(minimum,lower_cost) if identified else None,
                   "rate_bounds":[ratio(minimum,upper_cost),ratio(maximum,lower_cost)]}
             for name,(minimum,maximum) in quantities.items()}
    def length_summary(attribute):
        values=[getattr(row,attribute) for row in accepted]
        return {"known_retained_mean":statistics.mean(values) if values else None,
                "known_retained_median":statistics.median(values) if values else None,
                "complete_for_this_deadline":identified}
    return {"deadline_seconds":deadline,"attempted_traces":len(rows),
            "known_completed_games":len(accepted),"unknown_beyond_censoring":len(unknown),
            "point_estimate_identified":identified,"error_traces":sum(row.disposition=="error" for row in rows),
            "acceptance":ratio(len(accepted),len(rows)) if identified else None,
            "acceptance_bounds":[ratio(len(accepted),len(rows)),ratio(len(accepted)+len(unknown),len(rows))],
            "occupied_worker_seconds":lower_cost if identified else None,
            "occupied_worker_seconds_bounds":[lower_cost,upper_cost],
            "known_rejected_worker_seconds":rejected_cost,
            "rejected_compute_fraction":ratio(rejected_cost,lower_cost) if identified else None,
            "rejected_compute_fraction_bounds":[ratio(rejected_cost,upper_cost),
                ratio(rejected_cost+unknown_upper_cost,upper_cost)],
            "rates":rates,"retained_psr_length":length_summary("decisions"),
            "retained_learner_length":length_summary("learner_roots")}


def pareto_front(estimates):
    eligible = [row for row in estimates if row["point_estimate_identified"]
                and not row["error_traces"] and row["known_completed_games"]>0]
    def objectives(row):
        return tuple(row["rates"][metric]["per_occupied_worker_second"] for metric in
                     ("terminal_games","potential_terminal_examples_capped","retained_learner_roots")) \
            +(-row["rejected_compute_fraction"],)
    front=[]
    for row in eligible:
        candidate=objectives(row)
        if not any(all(a>=b for a,b in zip(objectives(other),candidate))
                   and any(a>b for a,b in zip(objectives(other),candidate)) for other in eligible):
            front.append(row["deadline_seconds"])
    return front


def summarize_observed(rows):
    terminal=[row for row in rows if row.disposition=="terminal"]
    occupied=sum(row.elapsed for row in rows)
    lost=sum(row.elapsed for row in rows if row.disposition!="terminal")
    return {"attempted_games":len(rows),"terminal_games":len(terminal),
            "dispositions":dict(Counter(row.disposition for row in rows)),
            "termination_reasons":dict(Counter(row.termination for row in rows)),
            "occupied_worker_seconds":occupied,"nonterminal_worker_seconds":lost,
            "observed_nonterminal_compute_fraction":ratio(lost,occupied),
            "terminal_games_per_worker_second":ratio(len(terminal),occupied),
            "terminal_learner_roots":sum(row.learner_roots for row in terminal),
            "terminal_learner_roots_per_worker_second":ratio(sum(row.learner_roots for row in terminal),occupied),
            "terminal_psr_decisions":sum(row.decisions for row in terminal),
            "terminal_psr_decisions_per_worker_second":ratio(sum(row.decisions for row in terminal),occupied),
            "observed_mean_seconds":statistics.mean(row.elapsed for row in rows) if rows else None,
            "observed_median_seconds":statistics.median(row.elapsed for row in rows) if rows else None,
            "terminal_mean_psr_decisions":statistics.mean(row.decisions for row in terminal) if terminal else None,
            "terminal_median_psr_decisions":statistics.median(row.decisions for row in terminal) if terminal else None}


def analyze(directories, deadlines=None, sample_cap=32):
    groups, inputs = {}, []
    seen=set()
    for directory in directories:
        identity=directory.resolve()
        if identity in seen:
            continue
        seen.add(identity)
        loaded, provenance=load_run(directory)
        groups.update(loaded)
        inputs.append(provenance)
    def summarize_group(identity, rows):
        grid=deadlines if deadlines is not None else propose_grid(rows,identity["budget"])
        estimates=[estimate(rows,cutoff,sample_cap) for cutoff in sorted(set(grid))]
        return {"identity":identity,"observed":summarize_observed(rows),
            "grid_source":"explicit CLI deadlines" if deadlines is not None else "observed-duration quantiles and terminal times; historical 8s included for budget32 only",
            "estimates":estimates,"pareto_deadlines_seconds":pareto_front(estimates),
            "singleton_warning":len(rows)==1,
            "next_step":"Fresh matched frozen-model collection pilot, preserving every timeout and decision-limit rejection; no cutoff selected automatically."}
    result_groups=[]
    aggregate_rows=defaultdict(list)
    aggregate_modes=defaultdict(list)
    for key,rows in sorted(groups.items()):
        identity=json.loads(key)
        result_groups.append(summarize_group(identity,rows))
        aggregate_identity={**identity,"mode":"observed-run-mixture","learner_seat":"mixed"}
        aggregate_key=json.dumps(aggregate_identity,sort_keys=True,separators=(",",":"))
        aggregate_rows[aggregate_key].extend(rows)
        aggregate_modes[aggregate_key].append({"mode":identity["mode"],"learner_seat":identity["learner_seat"],
            "attempted_games":len(rows),"terminal_games":sum(row.disposition=="terminal" for row in rows)})
    aggregates=[]
    for key,rows in sorted(aggregate_rows.items()):
        result=summarize_group(json.loads(key),rows)
        result["mode_breakdown"]=aggregate_modes[key]
        result["aggregation_scope"]="Only this exact run, budget, model weights and search configuration; observed mode/seat mix, not an extrapolation to a different collection mixture."
        aggregates.append(result)
    return {"schema":"paisho-compact-deadline-analysis-v1","analyzer_sha256":digest(Path(__file__).read_bytes()),
        "inputs":inputs,"sample_cap":sample_cap,"run_aggregates":aggregates,"groups":result_groups,
        "pareto_objectives":["maximize terminal games / occupied worker second",
            "maximize potential terminal examples capped per game / occupied worker second",
            "maximize uncapped retained learner roots / occupied worker second",
            "minimize rejected compute fraction"],
        "limitations":[
            "Counterfactual ideal trace truncation only; no new matches or causal timing experiment were run.",
            "sum(min(observed_time,deadline)) is an occupied-worker estimate, not global wall time or pure CPU time. Learner, persistence, idle time and hardware contention are not included.",
            "A time-censored game observed for c<deadline has unknown completion and cost between c and deadline; only bounds are reported and it cannot nominate a Pareto point.",
            "A decision-limit rejection stays rejected for the SAME fixed decision-limit protocol, even at a longer time cutoff. It is never relabeled a draw.",
            "Explicit repetition training losses are final nonterminal stops under that protocol. This report counts rule-terminal targets only; repetition penalties can still train, so their nonterminal compute is not necessarily wasted learning work.",
            "Soft per-game or per-move MCTS deadlines may change the final search/action and overshoot. The trace estimate min(t,d) assumes ideal interruption; it is not a fresh measured throughput.",
            "Groups keep exact weights, simulation budget, source, mode, seat and run separate. Learning-run durations do not establish fixed-model strength or attribute differences to architecture.",
            "Capped examples are potential terminal-position samples, not demonstrated learning quality; comparison traces do not contain archived root-Q training examples.",
            "Do not apply a production-throughput cutoff to strength evaluations: dropping long games can bias outcomes and break reversed-seat pairs.",
            "COMPACT_MCTS_V5 fixes 32=8s, 64=17.5s, 128=25s, 256=34s, 512=58s after fresh user-requested measurement. This read-only diagnostic does not authorize automatic recalibration. Unobserved budgets have no estimate.",
            "Candidate cutoffs and their Pareto front are selected on these same traces and are not a universal optimum or independent validation."]}


def format_number(value):
    return "?" if value is None else f"{value:.4g}"


def report_markdown(result):
    lines=["# Seuils temporels MCTS : analyse contrefactuelle de collecte", "",
           "Ces résultats proposent des essais ; ils ne choisissent aucun seuil de production et ne modifient pas les évaluations de force.", "",
           "Une seconde de travailleur occupé est la durée d’une partie sur un acteur. La somme des durées peut dépasser le temps écoulé quand les acteurs travaillent ensemble.", ""]
    for group in result["run_aggregates"]+result["groups"]:
        identity,observed=group["identity"],group["observed"]
        lines.extend([f"## MCTS-{identity['budget']} · {identity['mode']} · siège {identity['learner_seat'] or 'les deux'}", "",
            f"Modèle `{identity['weights_sha256'][:16]}` ; source `{identity['source_sha256'][:12]}` ; `{identity['run']}`.", "",
            f"{observed['terminal_games']} fins réelles sur {observed['attempted_games']} traces ; coût sans fin exploitable observé : {format_number(100*observed['observed_nonterminal_compute_fraction'])} % des secondes occupées.", "",
            "| Seuil s | Fins connues | Censures au-delà | Parties/s acteur | Racines élève/s acteur | PSR décisions/s acteur | Exemples plafonnés/s acteur | Coût rejeté | Longueur PSR moyenne/médiane | Pareto |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"])
        for row in group["estimates"]:
            rate=lambda name:format_number(row["rates"][name]["per_occupied_worker_second"])
            length=row["retained_psr_length"]
            waste=row["rejected_compute_fraction"]
            lines.append(f"| {format_number(row['deadline_seconds'])} | {row['known_completed_games']} | {row['unknown_beyond_censoring']} | {rate('terminal_games')} | {rate('retained_learner_roots')} | {rate('retained_psr_decisions')} | {rate('potential_terminal_examples_capped')} | {format_number(None if waste is None else 100*waste)} % | {format_number(length['known_retained_mean'])}/{format_number(length['known_retained_median'])} | {'oui' if row['deadline_seconds'] in group['pareto_deadlines_seconds'] else ''} |")
        lines.extend(["", "`?` signifie non identifié à cause d’une censure, pas zéro. Les bornes explicites figurent dans le JSON.", ""])
        if "mode_breakdown" in group:
            description=" ; ".join(f"{row['mode']} {row['learner_seat'] or 'deux sièges'} : {row['terminal_games']}/{row['attempted_games']} fins" for row in group["mode_breakdown"])
            lines.extend([f"Agrégat du mélange réellement observé dans ce run : {description}. Même budget et mêmes poids ; le détail par mode suit.", ""])
        if identity["kind"]=="comparison-traces":
            lines.extend(["Traces d’évaluation utilisées uniquement pour explorer les durées : les exemples plafonnés sont reconstructibles depuis les PSR, sans Q enregistré. Ne pas couper ces évaluations pour améliorer ce tableau.", ""])
        if group["singleton_warning"]:
            lines.extend(["Une seule trace dans ce groupe : aucune distribution de durée ne peut en être déduite.", ""])
    lines.extend(["## Limites", ""]+[f"- {item}" for item in result["limitations"]])
    return "\n".join(lines)+"\n"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs",type=Path,nargs="+")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--report",type=Path)
    parser.add_argument("--deadlines",type=float,nargs="+")
    parser.add_argument("--sample-cap",type=int,default=32)
    args=parser.parse_args()
    report=args.report or args.output.with_suffix(".md")
    if args.output.exists() or report.exists() or args.output.resolve()==report.resolve():
        parser.error("choose new distinct JSON and Markdown output files")
    if args.sample_cap<1:
        parser.error("sample cap must be positive")
    result=analyze(args.runs,args.deadlines,args.sample_cap)
    for path in (args.output,report):
        path.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write("\n")
    with report.open("x") as stream:
        stream.write(report_markdown(result))
    print(json.dumps({"groups":len(result["groups"]),"json":str(args.output),"report":str(report)}))


if __name__=="__main__":
    main()
