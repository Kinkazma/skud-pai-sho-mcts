#!/usr/bin/env python3
"""Report frozen compact matches and a conditional legacy-only human Elo bridge.

No match, training, promotion, or rating update is executed. The raw score-Elo
equivalent is distinct from the canonical Davidson league rating and from the
prior-dependent projection onto the old human anchor.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

from analyze_compact_deadlines import (comparison_evaluators, comparison_reference,
                                       model_hash, psr_decisions)
from paisho_human_corpus import posterior, replay_record

HISTORICAL_RULES = "skud-pai-sho-2022-03-14"


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def nonnegative_integer(value, name):
    if isinstance(value,bool) or not isinstance(value,int) or value<0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def candidate_score(game):
    termination=game["termination"]
    if termination["kind"]!="rules":
        return None
    outcome=termination.get("outcome")
    if outcome=="draw":
        return 0.5
    if outcome not in ("host_win","guest_win"):
        raise ValueError("rules termination requires a terminal outcome")
    return float((outcome=="host_win")==game["candidate_host"])


def expected_score_elo(score):
    """Keep mathematical boundaries explicit; never serialize Infinity as JSON."""
    if score is None:
        return {"value":None,"boundary":"no-eligible-pairs"}
    if not math.isfinite(score) or not 0<=score<=1:
        raise ValueError("expected score must be in [0,1]")
    if score==0:
        return {"value":None,"boundary":"negative-infinity"}
    if score==1:
        return {"value":None,"boundary":"positive-infinity"}
    return {"value":400*math.log10(score/(1-score)),"boundary":None}


def sign_test(favorable,unfavorable):
    """Same exact paired sign test as paisho-ai::PairedComparison."""
    trials=favorable+unfavorable
    if not trials:
        return 1.0
    end=min(favorable,unfavorable)
    logs=[]
    for successes in range(end+1):
        log_coefficient=sum(math.log(trials+1-index)-math.log(index)
                            for index in range(1,successes+1))
        logs.append(log_coefficient-trials*math.log(2))
    peak=max(logs)
    return min(1.0,2*math.exp(peak)*sum(math.exp(value-peak) for value in logs))


def wdl(scores):
    values=Counter(score for score in scores if score is not None)
    count=sum(values.values())
    score=(values[1]+0.5*values[0.5])/count if count else None
    return {"wins":values[1],"draws":values[0.5],"losses":values[0],"games":count,
            "win_rate":values[1]/count if count else None,
            "draw_rate":values[0.5]/count if count else None,
            "loss_rate":values[0]/count if count else None,"score":score}


def lengths(games):
    decisions=[game["decisions"] for game in games if game.get("record") is not None]
    turns=[game["completed_turns"] for game in games if game.get("record") is not None
           and game.get("completed_turns") is not None]
    def summary(values):
        return {"count":len(values),"mean":statistics.mean(values) if values else None,
                "median":statistics.median(values) if values else None,
                "minimum":min(values) if values else None,"maximum":max(values) if values else None}
    return {"decisions":summary(decisions),"completed_turns":summary(turns),
            "turns_note":"Engine completed_turns, optionally independently replayed; never decisions/2. Harmony bonuses can retain the same player."}


def lengths_by_result(games):
    return {name:lengths([game for game in games if candidate_score(game)==score])
            for name,score in (("win",1),("draw",0.5),("loss",0))}


def pair_results(games, first_pair, scheduled_pairs):
    by_pair={}
    for game in games:
        pair=nonnegative_integer(game["pair_id"],"pair_id")
        leg=nonnegative_integer(game["leg"],"leg")
        if not first_pair<=pair<first_pair+scheduled_pairs or leg not in (0,1):
            raise ValueError("game lies outside the planned reversed-seat schedule")
        if game["candidate_host"]!=(leg==0):
            raise ValueError("candidate seat disagrees with reversed-seat leg")
        if game["game_index"]!=2*(pair-first_pair)+leg:
            raise ValueError("game index disagrees with pair/leg schedule")
        key=(pair,leg)
        if key in by_pair:
            raise ValueError("duplicate pair leg")
        by_pair[key]=game
    eligible=[]
    excluded=[]
    pentanomial=[0]*5
    rated_games=[]
    for pair in range(first_pair,first_pair+scheduled_pairs):
        legs=[by_pair.get((pair,leg)) for leg in (0,1)]
        if all(legs):
            for field in ("starting_flower","host_seed","guest_seed"):
                if legs[0].get(field) is None or legs[0].get(field)!=legs[1].get(field):
                    raise ValueError(f"paired games have different {field}")
        scores=[candidate_score(game) if game is not None else None for game in legs]
        if any(score is None for score in scores):
            excluded.append({"pair_id":pair,"known_candidate_scores":scores,
                "reasons":[game["termination"]["kind"] if game is not None else "missing-game-metadata" for game in legs]})
            continue
        half_points=round(2*sum(scores))
        pentanomial[half_points]+=1
        eligible.append({"pair_id":pair,"candidate_scores":scores,"pair_mean_score":sum(scores)/2})
        rated_games.extend(legs)
    favorable=pentanomial[3]+pentanomial[4]
    unfavorable=pentanomial[0]+pentanomial[1]
    raw_scores=[candidate_score(game) for game in games]
    known_score_sum=sum(score for score in raw_scores if score is not None)
    unknown_games=2*scheduled_pairs-sum(score is not None for score in raw_scores)
    whole_bounds=([known_score_sum/(2*scheduled_pairs),
                  (known_score_sum+unknown_games)/(2*scheduled_pairs)] if scheduled_pairs else [None,None])
    return {"scheduled_pairs":scheduled_pairs,"eligible_pairs":len(eligible),"excluded_pairs":len(excluded),
            "pentanomial":pentanomial,"favorable_pairs":favorable,"tied_pairs":pentanomial[2],
            "unfavorable_pairs":unfavorable,"exact_two_sided_sign_test_p_value":sign_test(favorable,unfavorable),
            "eligible_pair_details":eligible,"excluded_pair_details":excluded,
            "paired_wdl":wdl(candidate_score(game) for game in rated_games),
            "all_terminal_wdl":wdl(raw_scores),"scheduled_score_missing_outcome_bounds":whole_bounds,
            "scheduled_score_bounds_note":"Sensitivity only: unknown games range from losses to wins. They are not added as actual results.",
            "eligible_lengths":lengths(rated_games),"eligible_lengths_by_result":lengths_by_result(rated_games)}


def read_run(directory, verifier=None):
    directory=directory.resolve()
    plan_path=directory/"plan.json"
    plan=json.loads(plan_path.read_text())
    if plan.get("schema")!="paisho-compact-comparison-plan-v1":
        raise ValueError(f"expected a compact comparison plan: {directory}")
    rules=plan.get("rules",HISTORICAL_RULES)
    model_path=directory/plan["candidate"]["snapshot"]
    if sha256(model_path.read_bytes())!=plan["candidate"]["sha256"]:
        raise ValueError("candidate snapshot hash mismatch")
    reference=comparison_reference(plan,directory)
    comparison_evaluators(plan,reference)
    candidate_budget=nonnegative_integer(plan["mcts"]["simulations"],"candidate simulations")
    reference_budget=nonnegative_integer(plan.get("reference_simulations",candidate_budget),"reference simulations")
    scheduled_pairs=nonnegative_integer(plan["pairs"],"pairs")
    first_pair=nonnegative_integer(plan["first_pair"],"first pair")
    if not min(candidate_budget,reference_budget,scheduled_pairs):
        raise ValueError("budgets and scheduled pair count must be positive")
    hashes={"plan.json":sha256(plan_path.read_bytes()),str(model_path.relative_to(directory)):sha256(model_path.read_bytes())}
    if reference["reference_model_sha256"] is not None:
        hashes[plan["reference_model"]["snapshot"]]=reference["reference_model_sha256"]
    games=[]
    for path in sorted((directory/"games").glob("game-*.json")):
        game=json.loads(path.read_text())
        if game.get("schema")!="paisho-compact-comparison-game-v1":
            raise ValueError(f"unsupported comparison game: {path}")
        comparison_evaluators(plan,reference,game)
        hashes[str(path.relative_to(directory))]=sha256(path.read_bytes())
        kind=game["termination"]["kind"]
        if kind not in ("rules","decision_limit","wall_limit","no_legal_actions","error","not_played"):
            raise ValueError("unknown game termination")
        decisions=nonnegative_integer(game["decisions"],"decisions")
        if game.get("completed_turns") is not None:
            nonnegative_integer(game["completed_turns"],"completed turns")
        if game.get("record") is None:
            if kind!="not_played" or decisions!=0:
                raise ValueError("played comparison game has no PSR")
        else:
            record_path=directory/game["record"]
            if psr_decisions(record_path,game["record_sha256"])!=decisions:
                raise ValueError("PSR and metadata decision counts disagree")
            rule_lines=[line.strip()[6:] for line in record_path.read_text().splitlines()
                        if line.strip().startswith("rules ")]
            if rule_lines != [rules]:
                raise ValueError("PSR rules disagree with comparison plan")
            hashes[str(record_path.relative_to(directory))]=game["record_sha256"]
            if verifier is not None:
                _,facts=replay_record(record_path.read_text(),verifier)
                expected={"host_win":"host","guest_win":"guest","draw":"draw"}.get(game["termination"].get("outcome"),"ongoing")
                if facts["outcome"]!=expected or facts["decisions"]!=decisions:
                    raise ValueError("rules replay disagrees with comparison outcome or decisions")
                if game.get("completed_turns") is not None and game["completed_turns"]!=facts["completed_turns"]:
                    raise ValueError("completed turns disagree with rules replay")
                game["completed_turns"]=facts["completed_turns"]
        candidate_score(game)  # Validate terminal outcomes even in an excluded pair.
        games.append(game)
    pairs=pair_results(games,first_pair,scheduled_pairs)
    summary_path=directory/"summary.json"
    published=None
    if summary_path.exists():
        published=json.loads(summary_path.read_text())
        if published.get("schema")!="paisho-compact-comparison-summary-v1":
            raise ValueError("unsupported published comparison summary")
        if published.get("rules",HISTORICAL_RULES)!=rules:
            raise ValueError("published rules disagree with comparison plan")
        hashes["summary.json"]=sha256(summary_path.read_bytes())
        fields=published.get("paired_comparison",{})
        for field,expected in (("rated_pairs",pairs["eligible_pairs"]),("excluded",pairs["excluded_pairs"]),
                               ("wins",pairs["paired_wdl"]["wins"]),("draws",pairs["paired_wdl"]["draws"]),
                               ("losses",pairs["paired_wdl"]["losses"])):
            if field in fields and fields[field]!=expected:
                raise ValueError(f"published summary disagrees with game records: {field}")
        if "score" in fields:
            actual,expected=fields["score"],pairs["paired_wdl"]["score"]
            if (actual is None)!=(expected is None) or (actual is not None and not math.isclose(actual,expected,abs_tol=1e-12)):
                raise ValueError("published summary disagrees with game records: score")
    complete=len(games)==2*scheduled_pairs
    error_games=[game["game_index"] for game in games if game["termination"]["kind"]=="error"]
    failed=bool(error_games or (published and published.get("status")=="failed"))
    pair_scores=[item["pair_mean_score"] for item in pairs["eligible_pair_details"]]
    delta=expected_score_elo(pairs["paired_wdl"]["score"])
    # One fractional expected-score observation per independent reversed-seat
    # pair. This deliberately does not pretend the two games are independent.
    pseudo_games=[{"metadata":{"human_rating":0},"bot_score":score} for score in pair_scores]
    regularized=[posterior(pseudo_games,prior_mean=0,prior_sd=sd) for sd in (300,600)] if pair_scores else []
    terminal_games=[game for game in games if candidate_score(game) is not None]
    return {"run":str(directory),"identity":{
        **reference,
        "rules":rules,
        "candidate_budget":candidate_budget,"reference_budget":reference_budget,
        "reference_budget_explicit":("reference_simulations" in plan),
        "candidate_model_sha256":plan["candidate"]["sha256"],"candidate_weights_sha256":model_hash(model_path),
        "candidate_training_steps":plan["candidate"].get("training_steps"),
        "reference":plan["reference"],"source_sha256":plan["build"]["source_sha256"],
        "move_ms":plan.get("move_ms"),"decision_limit":plan["decision_limit"],
        "workers":plan["workers"],"first_pair":first_pair},
        "files_sha256":hashes,"complete_archive":complete,"failed":failed,"game_error_indices":error_games,
        "verification":"independent rules replay, PSR hashes and counts" if verifier else "PSR hashes/counts and engine metadata; no independent replay",
        "game_terminations":dict(Counter(game["termination"]["kind"] for game in games)),
        "paired_results":pairs,"all_terminal_lengths":lengths(terminal_games),
        "all_terminal_lengths_by_result":lengths_by_result(terminal_games),"all_played_lengths":lengths(games),
        "score_equivalent_elo_delta":delta,
        "delta_formula":"400*log10(score/(1-score)); score=(wins+draws/2)/eligible games. Not the canonical Davidson fit.",
        "regularized_pair_delta":regularized,
        "regularization_note":"Working Gaussian delta prior N(0,300²), with N(0,600²) sensitivity; one fractional Elo-score likelihood contribution per reversed-seat pair. Conditional generalized-posterior intervals, not distribution-free guarantees.",
        "playing_strength_change":("Direct comparison against the exact frozen learned reference artifact; no legacy or human result is transferred automatically."
            if reference["reference_kind"]=="compact-model" else
            "Comparison against the named frozen opponent only, not a measured change against a previous learned version."),
        "run_wall_seconds":published.get("run_wall_seconds") if published else None}


def load_anchor(path,cohort=None):
    anchor=json.loads(path.read_text())
    if anchor.get("schema")!="paisho-human-elo-anchor-v1":
        raise ValueError("unsupported human anchor schema")
    matches=[group for group in anchor["groups"] if group["agent_label"]=="mcts-512"
             and (cohort is None or group["bot_cohort"]==cohort)]
    if len(matches)!=1:
        raise ValueError("select one historical mcts-512 anchor with --anchor-cohort")
    group=matches[0]
    scenarios=[{"name":"independent-game-working-anchor","posterior":group["baseline"]}]
    grouped=group.get("one_observation_per_group_sensitivity")
    if grouped:
        scenarios.append({"name":"declared-human-groups-dependence-sensitivity","posterior":grouped["posterior"]})
    return {"path":str(path.resolve()),"sha256":sha256(path.read_bytes()),
            "bot_cohort":group["bot_cohort"],"agent_label":group["agent_label"],
            "human_games":group["unique_games"],"historical_bot_binary_identity":group["bot_binary_identity"],
            "record_sha256":group["record_sha256"],"scenarios":scenarios}


def project_human(run,anchor):
    if anchor is None:
        return {"status":"no-human-anchor-supplied"}
    if run["identity"].get("rules",HISTORICAL_RULES)!=HISTORICAL_RULES:
        return {"status":"no-cross-rules-human-bridge","reason":"The historical human anchor uses the V1 rules profile. Results under corrected or other rules remain separate; no old human Elo is transferred."}
    if run["identity"]["reference_budget"]!=512:
        return {"status":"no-direct-mcts512-bridge","reason":"This comparison does not face the human-anchored 512-budget reference; no global scale slope is inferred."}
    if run["identity"].get("reference_kind")!="legacy-cpu-heuristic":
        return {"status":"no-direct-legacy-human-bridge","reason":"The historical human games anchor only the recognized legacy heuristic. A learned or unidentified reference, including learned512, is not that anchor; no chained projection is inferred."}
    if not run["complete_archive"] or run["failed"]:
        return {"status":"withheld-incomplete-or-failed-archive"}
    if not run["regularized_pair_delta"]:
        return {"status":"no-eligible-pairs"}
    scenarios=[]
    for human in anchor["scenarios"]:
        base=human["posterior"]
        for delta in run["regularized_pair_delta"]:
            raw=run["score_equivalent_elo_delta"]["value"]
            scenarios.append({"anchor_scenario":human["name"],"anchor_mode":base["posterior_mode"],
                "delta_prior_sd":delta["prior_sd"],"delta_posterior_mode":delta["posterior_mode"],
                "working_site_elo":base["posterior_mode"]+delta["posterior_mode"],
                "site_elo_using_raw_score_delta":None if raw is None else base["posterior_mode"]+raw,
                "sum_of_marginal_95_bounds":[base["credible_95"][0]+delta["credible_95"][0],
                                             base["credible_95"][1]+delta["credible_95"][1]],
                "bounds_label":"Sum of two conditional marginal 95% ranges; NOT a calibrated joint 95% interval. This sensitivity envelope does not quantify omitted systematic errors."})
    return {"status":"conditional-provisional-bridge","historical_anchor_sha256":anchor["sha256"],
        "historical_human_games":anchor["human_games"],"new_human_games_for_this_model":0,"scenarios":scenarios,
        "assumptions":[
            "The currently frozen CpuMctsEvaluator at budget512 is treated as the old human-labelled mcts-512 reference; those human JSONs contain no binary hash, so exact version equivalence remains unverified.",
            "Score-equivalent Elo differences are added to this single anchor with scale400 (unit local slope). This is an explicit transfer assumption, not a globally fitted internal/site conversion.",
            "The shared historical human anchor is reused as a reference, never relabelled as new human games for this learned model.",
            "Draw frequency, seat effects, matchup-specific exploits, opponent selection and deadline censoring can invalidate transitivity or bias the bridge.",
            "Excluded pairs are omitted from the main score; scheduled missing-outcome score bounds are reported separately.",
            "All candidates share the same uncertain human anchor, so their projected human-rating uncertainties are correlated.",
            "A move-time cap, when present, can reduce actual search below512 simulations and weakens equivalence to the historical reference."]}


def report(runs,anchor_path=None,anchor_cohort=None,verifier=None):
    anchor=load_anchor(anchor_path,anchor_cohort) if anchor_path else None
    verifier=verifier.resolve() if verifier else None
    verifier_before=sha256(verifier.read_bytes()) if verifier else None
    outputs=[]
    for directory in sorted(set(path.resolve() for path in runs)):
        result=read_run(directory,verifier)
        result["human_projection"]=project_human(result,anchor)
        outputs.append(result)
    outputs.sort(key=lambda run:(run["identity"]["candidate_budget"],run["identity"]["reference_budget"],run["run"]))
    if verifier and sha256(verifier.read_bytes())!=verifier_before:
        raise ValueError("rules verifier changed during the report")
    return {"schema":"paisho-compact-progress-report-v1","reporter_sha256":sha256(Path(__file__).read_bytes()),
        "human_anchor":anchor,"verifier":None if verifier is None else {"path":str(verifier),"sha256":verifier_before},
        "runs":outputs,"notes":[
            "W/D/L and score use the candidate's perspective. A score of60% is not a60% win rate.",
            "Only pairs with two rule-terminal games enter the main Elo-equivalent score. Every unfinished/error/unplayed pair stays excluded.",
            "No raw Elo delta is finite at score0 or1; the prior-dependent estimate remains separate and explicitly labelled.",
            "The exact sign test operates on favorable/unfavorable pairs and is the project's existing method; tied pairs do not enter that test.",
            "Small and selected lots do not establish a stable human rating or convergence. No promotion, canonical rating update or automatic production change is made.",
            "Differences between these runs are descriptive unless their opponent, setup, seeds, budgets and timing protocol were deliberately matched."]}


def number(value,digits=1):
    return "?" if value is None else f"{value:.{digits}f}"


def delta_text(delta):
    if delta["value"] is not None:
        return f"{delta['value']:+.0f}"
    return {"positive-infinity":"+∞ (score1)","negative-infinity":"−∞ (score0)"}.get(delta["boundary"],"?")


def reference_label(identity):
    kind=identity.get("reference_kind")
    name={"legacy-cpu-heuristic":"ancien", "legacy-cpu-heuristic-retained":"heuristique avec rétention", "compact-model":"modèle figé"}.get(kind,"référence non identifiée")
    return f"{name} MCTS-{identity['reference_budget']}"


def markdown(result):
    lines=["# Progression des MCTS compacts : comparaisons et ancrage humain provisoire","",
        "Les anciens matchs humains restent attachés à l’ancien MCTS-512. Les valeurs projetées ci-dessous sont des hypothèses de transfert, pas de nouveaux résultats contre des humains.","",
        "| Candidat / référence | Paires retenues / exclues | V / N / D | V / N / D % | Score | Δ Elo brut du score | Décisions moyenne / médiane | Tours achevés moyenne / médiane | Projection site régularisée |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for run in result["runs"]:
        pair=run["paired_results"];wdl=pair["paired_wdl"];length=pair["eligible_lengths"]
        projection=run["human_projection"]
        projected=projection["scenarios"][0]["working_site_elo"] if projection.get("scenarios") else None
        percentages=" / ".join(number(None if wdl[field] is None else 100*wdl[field]) for field in ("win_rate","draw_rate","loss_rate"))
        lines.append(f"| {run['identity']['candidate_budget']} / {reference_label(run['identity'])} | {pair['eligible_pairs']} / {pair['excluded_pairs']} | {wdl['wins']} / {wdl['draws']} / {wdl['losses']} | {percentages} | {number(None if wdl['score'] is None else 100*wdl['score'])} % | {delta_text(run['score_equivalent_elo_delta'])} | {number(length['decisions']['mean'])} / {number(length['decisions']['median'])} | {number(length['completed_turns']['mean'])} / {number(length['completed_turns']['median'])} | {number(projected,0)} |")
    for run in result["runs"]:
        identity=run["identity"];pairs=run["paired_results"]
        lines.extend(["",f"## {identity['candidate_budget']} simulations contre {reference_label(identity)}","",
            f"Run `{run['run']}` ; règles `{identity.get('rules',HISTORICAL_RULES)}` ; modèle `{identity['candidate_weights_sha256'][:16]}` ; source `{identity['source_sha256'][:12]}`.","",
            f"Paires favorables / égales / défavorables : {pairs['favorable_pairs']} / {pairs['tied_pairs']} / {pairs['unfavorable_pairs']}. Test exact des signes : p={pairs['exact_two_sided_sign_test_p_value']:.4g}.","",
            "Le Δ brut est seulement l’équivalent Elo du score contre cet adversaire. Les frontières infinies signifient un lot sans résultat contraire, pas une force infinie.",""])
        if identity.get("reference_model_sha256") is not None:
            lines.extend([f"Référence apprise figée : artefact `{identity['reference_model_sha256']}`. Le résultat mesure directement cet affrontement entre modèles ; aucune ancienne partie humaine n’est réattribuée à la référence apprise.",""])
        if run["regularized_pair_delta"]:
            regularized=run["regularized_pair_delta"][0]
            lines.extend([f"Avec a priori Δ ~ N(0,300²) et une contribution par paire, Δ central={regularized['posterior_mode']:+.0f}, intervalle conditionnel95% [{regularized['credible_95'][0]:+.0f} ; {regularized['credible_95'][1]:+.0f}]. La sensibilité N(0,600²) est conservée dans le JSON.",""])
        projection=run["human_projection"]
        if projection.get("scenarios"):
            lines.extend(["| Hypothèse d’ancrage | A priori σ du Δ | Repère site régularisé | Somme des bornes marginales |","|---|---:|---:|---|"])
            for scenario in projection["scenarios"]:
                bounds=scenario["sum_of_marginal_95_bounds"]
                lines.append(f"| {scenario['anchor_scenario']} | {scenario['delta_prior_sd']} | {scenario['working_site_elo']:.0f} | [{bounds[0]:.0f} ; {bounds[1]:.0f}] |")
            lines.extend(["","Ces fourchettes additionnent deux intervalles conditionnels : **ce ne sont pas des intervalles de confiance joints à95%**. L’équivalence exacte du vieux bot humain avec notre référence actuelle et la transférabilité Elo restent supposées. **Aucune partie humaine n’a été jouée par ce nouveau modèle dans ce rapport.**",""])
        else:
            lines.extend([f"Projection humaine non fournie : `{projection['status']}`.",""])
        lines.append("Les longueurs du tableau principal concernent les paires retenues. Le JSON conserve séparément toutes les fins réelles et toutes les parties jouées, y compris interrompues. Les tours viennent du moteur ; les décisions ne sont jamais divisées par deux.")
        lines.extend(["","| Résultat candidat, paires retenues | Parties | Décisions moyenne / médiane | Tours moyenne / médiane |",
                      "|---|---:|---:|---:|"])
        for key,label in (("win","Victoire"),("draw","Nulle"),("loss","Défaite")):
            split=pairs["eligible_lengths_by_result"][key]
            lines.append(f"| {label} | {split['decisions']['count']} | {number(split['decisions']['mean'])} / {number(split['decisions']['median'])} | {number(split['completed_turns']['mean'])} / {number(split['completed_turns']['median'])} |")
    lines.extend(["","## Limites",""]+[f"- {note}" for note in result["notes"]])
    return "\n".join(lines)+"\n"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs",type=Path,nargs="+")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--report",type=Path)
    parser.add_argument("--human-anchor",type=Path)
    parser.add_argument("--anchor-cohort")
    parser.add_argument("--verifier",type=Path,help="optional paisho-core verify_record executable for independent replay")
    args=parser.parse_args()
    destination=args.report or args.output.with_suffix(".md")
    if args.output.exists() or destination.exists() or args.output.resolve()==destination.resolve():
        parser.error("choose new, distinct JSON and Markdown output files")
    result=report(args.runs,args.human_anchor,args.anchor_cohort,args.verifier)
    for path in (args.output,destination):path.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write("\n")
    with destination.open("x") as stream:stream.write(markdown(result))
    print(json.dumps({"runs":len(result["runs"]),"output":str(args.output),"report":str(destination)}))


if __name__=="__main__":
    main()
