#!/usr/bin/env python3
"""Refresh immutable human-game evidence and provisional external Elo in one command.

This is an on-demand command, not a watcher. Human results refine the historical
external anchor; they never change internal match results or canonical league Elo.
Changed/conflicting exports are retained and quarantined, not silently replaced.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import uuid

import paisho_human_corpus as corpus
import report_compact_progress as progress


def digest_json(value):
    return corpus.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode())


def history_inputs(histories):
    evidence, comparisons, marks = {}, set(), []
    for history in histories:
        if not history.is_dir():
            raise ValueError(f"history directory does not exist: {history}")
        hashes = {}
        for receipt_path in sorted((history / "sweeps").glob("*/*.receipt.json")):
            receipt = json.loads(receipt_path.read_text())
            directory = receipt_path.with_name(receipt_path.name.removesuffix(".receipt.json"))
            if receipt.get("returncode") == 0 and (directory / "summary.json").is_file():
                comparisons.add(directory.resolve())
                hashes[str(receipt_path.relative_to(history))] = corpus.sha256(receipt_path.read_bytes())
        for path in sorted((history / "milestones").glob("mcts*/plus-*/mark.json")):
            raw = path.read_bytes()
            mark = json.loads(raw)
            model_path = path.parent / "model.json"
            model_hash = corpus.sha256(model_path.read_bytes())
            if model_hash != mark["model_sha256"]:
                raise ValueError(f"milestone snapshot hash mismatch: {path}")
            identity = mark["comparison"]["identity"]
            if (identity["candidate_model_sha256"] != model_hash or
                    identity["candidate_budget"] != mark["budget"]):
                raise ValueError(f"milestone comparison identity mismatch: {path}")
            hashes[str(path.relative_to(history))] = corpus.sha256(raw)
            hashes[str(model_path.relative_to(history))] = model_hash
            marks.append((path.resolve(), mark, corpus.sha256(raw)))
        evidence[str(history)] = hashes
    return evidence, comparisons, marks


def project_milestones(marks, report):
    projections = []
    for path, mark, mark_hash in marks:
        eligible = []
        for run in report["runs"]:
            identity = run["identity"]
            if (identity["candidate_budget"] == mark["budget"] and
                    identity["reference_budget"] == 512 and
                    identity.get("reference_kind") == "legacy-cpu-heuristic" and
                    run["human_projection"].get("status") == "conditional-provisional-bridge" and
                    identity["candidate_model_sha256"] == mark["model_sha256"] and
                    identity["candidate_weights_sha256"] == mark["comparison"]["identity"]["candidate_weights_sha256"] and
                    run["complete_archive"] and not run["failed"] and
                    run["paired_results"]["eligible_pairs"] > 0):
                eligible.append({"run": run["run"], "files_sha256": run["files_sha256"],
                                 "human_projection": run["human_projection"]})
        projections.append({"mark": str(path), "mark_sha256": mark_hash,
            "model_sha256": mark["model_sha256"], "budget": mark["budget"],
            "threshold": mark["threshold"], "internal_comparison": mark["comparison"],
            "historical_mark_is_unchanged": True, "bridges": eligible,
            "status": "conditional-provisional-bridges" if eligible else "pending-same-model-mcts512-comparison"})
    return {"schema": "paisho-milestone-external-refresh-v1", "human_anchor": report["human_anchor"],
            "milestones": projections,
            "policy": "Exact model hash, weights and budget required. Multiple bridges are shown separately, never selected by score or pooled. History files and internal results are unchanged."}


def snapshot_inputs(source, bases, verifier, comparisons, cohort, histories=()):
    files = sorted(source.glob("*.json"))
    if not source.is_dir() or not files:
        raise ValueError("source directory contains no JSON exports")
    raw = {file.name: file.read_bytes() for file in files}
    history_evidence, history_comparisons, _ = history_inputs(histories)
    comparisons = sorted(set(comparisons) | history_comparisons)
    evidence = {"source": str(source), "cohort": cohort,
                "files": {name: corpus.sha256(data) for name, data in raw.items()},
                "base_manifests": {str(path): corpus.sha256(path.read_bytes()) for path in bases},
                "verifier_sha256": corpus.sha256(verifier.read_bytes()),
                "tool_sha256": corpus.sha256(Path(__file__).read_bytes()),
                "importer_sha256": corpus.sha256(Path(corpus.__file__).read_bytes()),
                "reporter_sha256": corpus.sha256(Path(progress.__file__).read_bytes()),
                "comparisons": {}, "histories": history_evidence}
    for directory in comparisons:
        if not (directory / "summary.json").is_file():
            raise ValueError(f"comparison must be finished before refreshing: {directory}")
        evidence["comparisons"][str(directory)] = {
            str(path.relative_to(directory)): corpus.sha256(path.read_bytes())
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.suffix in (".json", ".psr")}
    return raw, evidence


def isolate_explicit_versions(manifest, root):
    """Keep explicit future-model identities separate from the unknown old bot."""
    for record in manifest["records"]:
        identities = set()
        try:
            for index in record["source_indexes"]:
                original = json.loads((root / manifest["sources"][index]["stored_path"]).read_bytes())
                fields = {key: original[key] for key in
                          ("bot_binary_identity", "bot_version", "model_sha256")
                          if original.get(key) is not None}
                if any(not isinstance(value, str) or not value.strip() for value in fields.values()):
                    raise ValueError("invalid explicit bot-version metadata")
                identities.add(json.dumps(fields, sort_keys=True) if fields else None)
            if len(identities) != 1:
                raise ValueError("aliases disagree on explicit bot-version metadata")
            identity = identities.pop()
            if identity is not None:
                record["bot_binary_identity"] = identity
                record["bot_cohort"] += "/explicit-" + corpus.sha256(identity.encode())[:16]
        except (ValueError, TypeError) as error:
            record["calibration_eligible"] = False
            record["calibration_exclusions"].append(str(error))
    manifest["counts"]["calibration_eligible_games"] = sum(
        record["calibration_eligible"] for record in manifest["records"])


def merge_manifests(paths, output):
    """Verify every byte and publish a single deduplicated, conflict-aware view."""
    output.mkdir()
    (output / "originals").mkdir()
    (output / "records").mkdir()
    sources, records, variants = [], {}, {}
    seen_sources = set()
    for path in paths:
        manifest = json.loads(path.read_text())
        if manifest.get("schema") != corpus.SCHEMA:
            raise ValueError(f"unsupported corpus schema: {path}")
        for source in manifest["sources"]:
            raw = (path.parent / source["stored_path"]).read_bytes()
            if corpus.sha256(raw) != source["sha256"]:
                raise ValueError(f"original hash mismatch: {path}")
            key = (source["sha256"], source["source_name"])
            if key in seen_sources:
                continue
            seen_sources.add(key)
            target = output / "originals" / (source["sha256"] + ".json")
            if not target.exists():
                target.write_bytes(raw)
            sources.append({**source, "stored_path": str(target.relative_to(output))})
        for record in manifest["records"]:
            key = record["record_sha256"]
            psr = (path.parent / record["psr_path"]).read_bytes()
            if corpus.sha256(psr) != key:
                raise ValueError(f"PSR hash mismatch: {path}")
            target = output / "records" / (key + ".psr")
            if not target.exists():
                target.write_bytes(psr)
            variants.setdefault(key, []).append({"manifest": str(path), "record": record})
            records.setdefault(key, deepcopy(record))
    conflicts = []
    semantic_fields = ("bot_cohort", "bot_binary_identity", "metadata", "bot_score", "replay")
    for key, record in records.items():
        alternatives = variants[key]
        semantic = {digest_json({name: item["record"].get(name) for name in semantic_fields})
                    for item in alternatives}
        reasons = sorted({reason for item in alternatives
                          for reason in item["record"].get("calibration_exclusions", [])})
        if len(semantic) > 1:
            reasons.append("conflicting metadata, cohort, identity or outcome across archived imports")
        if any(not item["record"]["calibration_eligible"] for item in alternatives):
            reasons.append("excluded in at least one archived import")
        record["source_indexes"] = [i for i, source in enumerate(sources)
                                    if source.get("record_sha256") == key]
        record["psr_path"] = f"records/{key}.psr"
        record["calibration_eligible"] = not reasons
        record["calibration_exclusions"] = reasons
        if reasons:
            conflicts.append({"record_sha256": key, "reasons": reasons,
                              "archived_variants": alternatives})
    merged = {"schema": corpus.SCHEMA, "sources": sources,
              "records": sorted(records.values(), key=lambda record: record["record_sha256"]),
              "input_manifests": [{"path": str(path), "sha256": corpus.sha256(path.read_bytes())}
                                  for path in paths],
              "counts": {"unique_replayed_games": len(records),
                         "calibration_eligible_games": sum(r["calibration_eligible"] for r in records.values()),
                         "quarantined_games": len(conflicts),
                         "rejected_source_files": sum("error" in s for s in sources)}}
    corpus.write_json(output / "manifest.json", merged)
    corpus.write_json(output / "quarantine.json", {"records": conflicts,
        "source_errors": [source for source in sources if "error" in source],
        "policy": "Conflicts stay excluded until explicitly resolved in a new audited import; no newest-file-wins rule."})
    return merged


def write_atomic(path, value):
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def update(source, output, bases, verifier, cohort, comparisons=(), histories=()):
    source, output, verifier = source.resolve(), output.resolve(), verifier.resolve()
    bases = sorted(set(path.resolve() for path in bases))
    comparisons = sorted(set(path.resolve() for path in comparisons))
    histories = sorted(set(path.resolve() for path in histories))
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".update.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        raw, evidence = snapshot_inputs(source, bases, verifier, comparisons, cohort, histories)
        fingerprint = digest_json(evidence)
        latest_path = output / "latest.json"
        previous = json.loads(latest_path.read_text()) if latest_path.exists() else None
        if previous:
            # Reopen published evidence even on a no-op, so corruption cannot pass silently.
            previous_dir = output / previous["revision"]
            complete = json.loads((previous_dir / "complete.json").read_text())
            for name, expected in complete["files_sha256"].items():
                if corpus.sha256((previous_dir / name).read_bytes()) != expected:
                    raise ValueError(f"published refresh was modified: {name}")
            if previous["fingerprint"] == fingerprint:
                return {"status": "unchanged", **previous}
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        revision = Path("revisions") / (stamp + "-" + fingerprint[:12])
        directory = output / revision
        directory.mkdir(parents=True)
        snapshot = directory / "source-snapshot"
        snapshot.mkdir()
        for name, data in raw.items():
            (snapshot / name).write_bytes(data)
        corpus.write_json(directory / "inputs.json", evidence)
        imported = corpus.import_corpus(snapshot, directory / "import", verifier, cohort)
        isolate_explicit_versions(imported, directory / "import")
        # importer manifest is preserved verbatim; this extra manifest adds explicit identity isolation.
        corpus.write_json(directory / "import" / "versioned-manifest.json", imported)
        manifests = list(bases)
        if previous:
            manifests.append(output / previous["revision"] / "merged" / "manifest.json")
        manifests.append(directory / "import" / "versioned-manifest.json")
        merged = merge_manifests(manifests, directory / "merged")
        anchor = corpus.calibration([directory / "merged" / "manifest.json"])
        corpus.write_json(directory / "calibration.json", anchor)
        effective_comparisons = [Path(path) for path in evidence["comparisons"]]
        projection = progress.report(effective_comparisons, directory / "calibration.json", cohort) if effective_comparisons or histories else None
        if projection:
            corpus.write_json(directory / "comparisons.json", projection)
            (directory / "comparisons.md").write_text(progress.markdown(projection))
        if histories:
            _, _, marks = history_inputs(histories)
            corpus.write_json(directory / "milestone_projections.json", project_milestones(marks, projection))
        if snapshot_inputs(source, bases, verifier, comparisons, cohort, histories)[1] != evidence:
            raise RuntimeError("inputs changed during refresh; prior latest remains valid, rerun the command")
        summary = {"schema": "paisho-elo-refresh-v1", "revision": str(revision),
                   "fingerprint": fingerprint, "source_files": len(raw),
                   **merged["counts"], "groups": anchor["groups"],
                   "comparison_count": len(effective_comparisons),
                   "history_count": len(histories),
                   "milestone_projections": str(revision / "milestone_projections.json") if histories else None,
                   "internal_elo_policy": "Existing internal results/deltas are unchanged. Human data updates only the historical external anchor and conditional projections.",
                   "external_elo_policy": "No global conversion slope is fitted. Unknown old identities and explicit future versions are separate cohorts.",
                   "identity_policy": "Equal human Elo is not used as a player identifier. Dependence sensitivities remain explicit."}
        corpus.write_json(directory / "summary.json", summary)
        hashes = {str(path.relative_to(directory)): corpus.sha256(path.read_bytes())
                  for path in sorted(directory.rglob("*")) if path.is_file()}
        corpus.write_json(directory / "complete.json", {"files_sha256": hashes})
        write_atomic(latest_path, summary)
        return {"status": "updated", **summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, action="append", default=[])
    parser.add_argument("--verifier", type=Path, default=corpus.ROOT / "target/release/examples/verify_record")
    parser.add_argument("--cohort", required=True,
                        help="Explicit observational old-bot cohort, not a certification of its binary")
    parser.add_argument("--comparison", type=Path, action="append", default=[])
    parser.add_argument("--history", type=Path, action="append", default=[],
                        help="Read completed sweep comparisons and immutable milestones; works while the history companion holds its lock")
    args = parser.parse_args()
    result = update(args.source, args.output, args.base_manifest, args.verifier,
                    args.cohort, args.comparison, args.history)
    print(json.dumps({key: value for key, value in result.items() if key != "groups"}, indent=2))
    for group in result["groups"]:
        print(json.dumps({key: group[key] for key in ("bot_cohort", "unique_games",
                         "bot_wins_draws_losses", "human_ratings", "baseline")}, indent=2))


if __name__ == "__main__":
    main()
