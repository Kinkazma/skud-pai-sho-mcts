#!/usr/bin/env python3
"""Preserve human exports, verify PSRs and update an explicitly provisional Elo anchor.

Build the verifier with ``cargo build -p paisho-core --release --example verify_record``.
Imports never overwrite a directory. They retain rejected originals as well as
valid games. Calibration can combine imports, but pools only an explicitly named
bot cohort and label; a cohort does not certify the historical binary's identity.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "paisho-human-corpus-v1"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def metadata(data):
    if not isinstance(data, dict):
        raise ValueError("export must be a JSON object")
    label, rating, side = data.get("agent"), data.get("elo"), data.get("human_side")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("agent must be a nonempty label")
    if isinstance(rating, bool) or not isinstance(rating, (int, float)) or not math.isfinite(rating):
        raise ValueError("elo must be a finite number")
    if side not in ("host", "guest"):
        raise ValueError("human_side must be host or guest")
    human_id = data.get("human_id")
    if human_id is not None and (not isinstance(human_id, str) or not human_id):
        raise ValueError("human_id, when supplied, must be a nonempty string")
    return {"agent_label": label, "human_rating": rating,
            "human_side": side, "human_id": human_id}


def replay_record(text, verifier):
    if not isinstance(text, str):
        raise ValueError("moves must contain PSR text")
    with tempfile.TemporaryDirectory(prefix="paisho-human-replay-") as temp:
        source, canonical = Path(temp) / "input.psr", Path(temp) / "canonical.psr"
        source.write_text(text, encoding="utf-8")
        result = subprocess.run([str(verifier), str(source), str(canonical)],
                                capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise ValueError("engine rejected PSR: " + result.stderr.strip()[:2000])
        facts = json.loads(result.stdout)
        if facts.get("outcome") not in ("host", "guest", "draw", "ongoing"):
            raise ValueError("invalid verifier outcome")
        return canonical.read_bytes(), facts


def import_corpus(source, output, verifier, cohort):
    source, output, verifier = source.resolve(), output.resolve(), verifier.resolve()
    files = sorted(source.glob("*.json"))
    if not files:
        raise ValueError("source directory contains no JSON files")
    if not verifier.is_file():
        raise ValueError("build or supply the verify_record executable")
    verifier_hash = sha256(verifier.read_bytes())
    output.mkdir(parents=True, exist_ok=False)
    (output / "originals").mkdir()
    (output / "records").mkdir()
    sources, records = [], {}
    for file in files:
        raw = file.read_bytes()
        digest = sha256(raw)
        original = output / "originals" / (digest + ".json")
        if not original.exists():
            original.write_bytes(raw)
        entry = {"source_name": file.name, "source_path": str(file),
                 "sha256": digest, "stored_path": str(original.relative_to(output))}
        try:
            data = json.loads(raw)
            # Replay and retain a valid record even if its rating metadata is unusable.
            if not isinstance(data, dict):
                raise ValueError("export must be a JSON object")
            psr, facts = replay_record(data.get("moves"), verifier)
            record_hash = sha256(psr)
            entry["record_sha256"] = record_hash
            record = records.setdefault(record_hash, {
                "record_sha256": record_hash, "psr_path": f"records/{record_hash}.psr",
                "replay": facts, "source_indexes": [], "metadata_variants": [],
                "bot_cohort": cohort, "bot_binary_identity": None})
            if record["replay"] != facts:
                raise ValueError("verifier returned inconsistent facts for the same PSR")
            path = output / record["psr_path"]
            if not path.exists():
                path.write_bytes(psr)
            record["source_indexes"].append(len(sources))
            info = metadata(data)
            entry["metadata"] = info
            if info not in record["metadata_variants"]:
                record["metadata_variants"].append(info)
        except (ValueError, TypeError, UnicodeError, subprocess.TimeoutExpired) as error:
            entry["error"] = str(error)
        sources.append(entry)
    for record in records.values():
        reasons = []
        if record["replay"]["outcome"] == "ongoing":
            reasons.append("nonterminal record")
        if len(record["metadata_variants"]) != 1:
            reasons.append("missing or conflicting rating/side/agent/human metadata")
        if any("error" in sources[i] for i in record["source_indexes"]):
            reasons.append("one source alias has invalid metadata")
        record["calibration_eligible"] = not reasons
        record["calibration_exclusions"] = reasons
        if not reasons:
            record["metadata"] = record["metadata_variants"][0]
            outcome = record["replay"]["outcome"]
            record["bot_score"] = (0.5 if outcome == "draw" else
                                   float(outcome != record["metadata"]["human_side"]))
    if sha256(verifier.read_bytes()) != verifier_hash:
        raise RuntimeError("verifier binary changed during import; do not use incomplete output")
    manifest = {
        "schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "bot_cohort": cohort,
        "cohort_note": "An observational grouping, not proof of a shared historical binary.",
        "replay_engine": {"path": str(verifier), "sha256": verifier_hash},
        "importer_sha256": sha256(Path(__file__).read_bytes()),
        "sources": sources, "records": sorted(records.values(), key=lambda x: x["record_sha256"]),
        "counts": {"source_files": len(sources), "unique_original_bytes": len({s["sha256"] for s in sources}),
                   "unique_replayed_games": len(records),
                   "calibration_eligible_games": sum(r["calibration_eligible"] for r in records.values()),
                   "rejected_source_files": sum("error" in s for s in sources)},
        "limitations": ["No player identity is inferred from a rating or filename.",
                        "Unknown historical bot binary identity remains null.",
                        "Identical canonical PSRs count once, including across source aliases."]}
    write_json(output / "manifest.json", manifest)
    return manifest


def load_observations(manifests):
    """Check archived bytes; deduplicate imports without silently pooling versions."""
    games, exclusions = {}, set()
    for path in manifests:
        manifest = json.loads(path.read_text())
        if manifest.get("schema") != SCHEMA:
            raise ValueError(f"unsupported corpus schema: {path}")
        for source in manifest["sources"]:
            if sha256((path.parent / source["stored_path"]).read_bytes()) != source["sha256"]:
                raise ValueError(f"original hash mismatch in {path}")
        for game in manifest["records"]:
            if sha256((path.parent / game["psr_path"]).read_bytes()) != game["record_sha256"]:
                raise ValueError(f"PSR hash mismatch in {path}")
            if not game["calibration_eligible"]:
                exclusions.add(game["record_sha256"])
                continue
            key = game["record_sha256"]
            value = {name: game[name] for name in
                     ("record_sha256", "bot_cohort", "bot_binary_identity", "metadata", "bot_score")}
            if key in games and games[key] != value:
                raise ValueError("same PSR has conflicting cohort, metadata or outcome across imports")
            games[key] = value
    if exclusions.intersection(games):
        raise ValueError("same PSR is eligible in one import and excluded in another")
    return sorted(games.values(), key=lambda game: game["record_sha256"])


def posterior(observations, prior_mean=1000, prior_sd=300, evidence_weight=1.0):
    """Grid integration. Draws use an explicit half-score power likelihood."""
    if prior_sd <= 0 or not math.isfinite(prior_sd):
        raise ValueError("prior_sd must be finite and positive")
    if not 0 < evidence_weight <= 1:
        raise ValueError("evidence_weight must be in (0, 1]")
    lower = math.floor(prior_mean - 10 * prior_sd)
    upper = math.ceil(prior_mean + 10 * prior_sd)
    xs = range(lower, upper + 1)
    log_weights = []
    for rating in xs:
        log_weight = -0.5 * ((rating - prior_mean) / prior_sd) ** 2
        for game in observations:
            z = math.log(10) / 400 * (rating - game["metadata"]["human_rating"])
            # Stable log(p), log(1-p) without overflowing at large rating gaps.
            softplus = max(z, 0) + math.log1p(math.exp(-abs(z)))
            game_weight = game.get("observation_weight", 1.0)
            if not 0 < game_weight <= 1:
                raise ValueError("observation_weight must be in (0, 1]")
            log_weight += evidence_weight * game_weight * (game["bot_score"] * z - softplus)
        log_weights.append(log_weight)
    peak = max(log_weights)
    weights = [math.exp(weight - peak) for weight in log_weights]
    total = sum(weights)
    weights = [weight / total for weight in weights]
    cdf, quantiles = 0.0, {}
    for x, weight in zip(xs, weights):
        cdf += weight
        for q in (0.025, 0.5, 0.975):
            if q not in quantiles and cdf >= q:
                quantiles[q] = x
    return {"prior_mean": prior_mean, "prior_sd": prior_sd, "evidence_weight": evidence_weight,
            "posterior_mode": lower + log_weights.index(peak),
            "posterior_mean": sum(x * weight for x, weight in zip(xs, weights)),
            "posterior_median": quantiles[0.5],
            "credible_95": [quantiles[0.025], quantiles[0.975]]}


def calibration(manifests, player_groups_path=None):
    observations = load_observations(manifests)
    player_groups = None
    if player_groups_path is not None:
        player_groups = json.loads(player_groups_path.read_text())
        if player_groups.get("schema") != "paisho-human-player-groups-v1":
            raise ValueError("unsupported player-group schema")
        mapping = player_groups.get("record_groups", {})
        if set(mapping) != {game["record_sha256"] for game in observations}:
            raise ValueError("player groups must explicitly cover exactly the calibrated games")
        if any(not isinstance(group, str) or not group for group in mapping.values()):
            raise ValueError("player groups must have nonempty string labels")
    groups = defaultdict(list)
    for game in observations:
        groups[(game["bot_cohort"], game["metadata"]["agent_label"], game["bot_binary_identity"])].append(game)
    results = []
    for (cohort, label, identity), games in sorted(groups.items(), key=lambda item: str(item[0])):
        counts = Counter(game["bot_score"] for game in games)
        identities = {game["metadata"]["human_id"] for game in games if game["metadata"]["human_id"] is not None}
        sensitivity = [posterior(games, mean, sd)
                       for mean in (1000, 1093, 1176) for sd in (200, 300, 400)]
        result = {
            "bot_cohort": cohort, "agent_label": label, "bot_binary_identity": identity,
            "unique_games": len(games),
            "record_sha256": [game["record_sha256"] for game in games],
            "bot_wins_draws_losses": [counts[1], counts[0.5], counts[0]],
            "human_ratings": dict(sorted(Counter(str(game["metadata"]["human_rating"]) for game in games).items())),
            "human_sides": dict(Counter(game["metadata"]["human_side"] for game in games)),
            "explicit_human_ids": len(identities),
            "games_without_human_id": sum(game["metadata"]["human_id"] is None for game in games),
            "baseline": posterior(games), "prior_sensitivity": sensitivity,
            "dependence_sensitivity": [posterior(games, evidence_weight=weight)
                                       for weight in (0.5, 1 / len(games))]}
        if player_groups is not None:
            mapping = player_groups["record_groups"]
            sizes = Counter(mapping[game["record_sha256"]] for game in games)
            weighted_games = [{**game, "observation_weight": 1 / sizes[mapping[game["record_sha256"]]]}
                              for game in games]
            result["declared_player_groups"] = dict(sizes)
            result["one_observation_per_group_sensitivity"] = {
                "interpretation": "Each declared player group has total evidence weight one; a sensitivity scenario, not a fitted correlation model.",
                "record_weights": {game["record_sha256"]: game["observation_weight"] for game in weighted_games},
                "posterior": posterior(weighted_games)}
        results.append(result)
    return {
        "schema": "paisho-human-elo-anchor-v1",
        "input_manifests": [{"path": str(path.resolve()), "sha256": sha256(path.read_bytes())} for path in manifests],
        "calibrator_sha256": sha256(Path(__file__).read_bytes()),
        "player_group_provenance": None if player_groups_path is None else {
            "path": str(player_groups_path.resolve()), "sha256": sha256(player_groups_path.read_bytes()),
            "declaration": player_groups},
        "unique_eligible_games": len(observations), "groups": results,
        "model": "Gaussian rating prior, Elo expected-score power likelihood; draws have score 1/2.",
        "assumptions": ["Baseline treats unique games as conditionally independent.",
                        "Recorded human ratings are treated as fixed; seat effect is zero.",
                        "Fractional evidence weights are declared sensitivity scenarios, not fitted correlation estimates.",
                        "This is a provisional anchor, not the site's sequential Elo update or canonical internal Elo.",
                        "A comment about difficulty contributes no additional numerical observations.",
                        "At one observed bot level, the slope of an internal/site conversion is unidentified.",
                        "Unknown player identity, selection of submitted games and historical bot version can bias the anchor."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    importer = sub.add_parser("import")
    importer.add_argument("--source", type=Path, required=True)
    importer.add_argument("--output", type=Path, required=True)
    importer.add_argument("--verifier", type=Path, default=ROOT / "target/release/examples/verify_record")
    importer.add_argument("--bot-cohort", required=True)
    estimate = sub.add_parser("calibrate")
    estimate.add_argument("--manifest", type=Path, action="append", required=True)
    estimate.add_argument("--output", type=Path, required=True)
    estimate.add_argument("--player-groups", type=Path,
                          help="optional explicit user/source group assignments; never inferred from Elo")
    args = parser.parse_args()
    if args.command == "import":
        manifest = import_corpus(args.source, args.output, args.verifier, args.bot_cohort)
        print(json.dumps(manifest["counts"], indent=2))
    else:
        result = calibration(args.manifest, args.player_groups)
        write_json(args.output, result)
        print(json.dumps({"unique_eligible_games": result["unique_eligible_games"],
                          "groups": [{key: group[key] for key in
                                      ("agent_label", "unique_games", "bot_wins_draws_losses", "baseline")}
                                     for group in result["groups"]]}, indent=2))


if __name__ == "__main__":
    main()
