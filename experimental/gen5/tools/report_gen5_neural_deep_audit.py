#!/usr/bin/env python3
"""Aggregate the frozen native diagnostics; never starts a game or changes weights."""
import collections
import hashlib
import json
import statistics as stats
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks/results/gen5-neural-deep-audit-2026-09-11"


def read(name):
    return json.loads((OUT / name).read_text())


def value_summary(rows):
    return {
        str(target): {
            "n": len(group),
            "mse": stats.mean((x["value"] - target) ** 2 for x in group),
            "mean_value": stats.mean(x["value"] for x in group),
            "wrong_sign": sum(x["value"] * target <= 0 for x in group),
            "certified_top": sum(x.get("certified_top", False) for x in group),
        }
        for target in [-1, 0, 1]
        if (group := [x for x in rows if x["target"] == target])
    }


def main():
    fifo = read("fifo-results.json")
    grad = read("gradient-results.json")
    interference = read("interference-results.json")
    tree = read("proof-tree-results.json")
    admissions = read("proof-admissions-sample.json")
    rows = fifo["rows"]
    positives = [x for x in admissions if x["value"] == 1]
    excluded = [x for x in positives if not x["priority"] and not x["reanalysis"]]
    final = [x for x in grad["rows"] if x["model"] == 2]
    proof_conflicts = [x for x in rows if x["proof"] is not None and x["proof"] != x["value"]]
    summary = {
        "fifo": {
            "n": len(rows), "sources": fifo["source_hashes_checked"],
            "reasons": dict(collections.Counter(x["reason"] for x in rows)),
            "with_current_proof": sum(x["proof"] is not None for x in rows),
            "conflicts": proof_conflicts,
            "conflicts_with_earlier_proof": sum(x["proof_known_before"] for x in proof_conflicts),
            "action_alias_groups": sum(x["action_alias_groups"] for x in rows),
            "terminal_losses_with_policy": sum(x["reason"] == "rules-terminal-z" and x["value"] == -1 and x["policy_weight"] > 0 for x in rows),
        },
        "direct_recall": {"proof_examples": len(admissions), "wins": len(positives), "excluded_at_admission": len(excluded), "native_control": read("recall-catalogue-results.json")},
        "neural": {
            "finite_checks": len(grad["finite_differences"]),
            "maximum_derivative_error": max(abs(x["analytic"] - x["numeric"]) for x in grad["finite_differences"]),
            "evaluated_examples": len(final),
            "mse": stats.mean((x["value"] - x["target"]) ** 2 for x in final),
            "mse_without_deep": stats.mean((x["value_without_deep"] - x["target"]) ** 2 for x in final),
            "wrong_sign_saturated": sum(abs(x["value"]) > .99 and x["value"] * x["target"] <= 0 for x in final),
            "batches": grad["batches"],
        },
        "proof_tree": {
            "unique_nonterminal_positions": len(tree["positions"]),
            "descendants": len([x for x in tree["positions"] if x["depth"] > 0]),
            "unindexed_descendants": value_summary([x for x in tree["positions"] if x["depth"] > 0 and not x["root_catalogue_contains"]]),
        },
        "searches": [
            {"noise": noise, "seed": seed, "n": len(group),
             "proved_win": sum(x["proven"] == 1 for x in group),
             "historical_certified_action_selected": sum(x["selected_certified"] for x in group),
             "unproved": [x for x in group if x["proven"] is None]}
            for noise, seed in [(False, 0), (True, 37), (True, 913)]
            if (group := [x for x in tree["searches"] if x["noise"] == noise and x["seed"] == seed])
        ],
        "value_interventions": interference["value_trials"],
    }
    assert len(rows) == 4096 and len(positives) == 86 and len(excluded) == 43
    assert summary["neural"]["maximum_derivative_error"] < 2e-7
    assert summary["fifo"]["conflicts_with_earlier_proof"] == 0
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for key, label, color in [
        ("alias_fit_trials", "Quatre cas seuls", "#b64a38"),
        ("alias_mixed_trials", "50 % cas / 50 % anciennes preuves", "#167a80"),
    ]:
        trials = interference[key]
        steps = [t["step"] for t in trials[0]["trace"]]
        for axis, field in zip(axes, ["witnesses", "old_archive"]):
            values = np.array([[t[field]["certified_top"] for t in run["trace"]] for run in trials])
            axis.plot(steps, values.mean(axis=0), marker="o", markersize=4, label=label, color=color)
            axis.fill_between(steps, values.min(axis=0), values.max(axis=0), alpha=.16, color=color)
            axis.set_xscale("symlog", linthresh=64)
            axis.set_xlabel("Présentations d’exemples de politique")
            axis.grid(alpha=.18)
    axes[0].set_title("Acquisition : quatre victoires immédiates")
    axes[0].set_ylabel("Choix réussi, sur 4")
    axes[0].set_ylim(-.2, 4.3)
    axes[1].set_title("Conservation : 107 positions d’autres sources")
    axes[1].set_ylabel("Coup certifié choisi, sur 107")
    axes[1].axhline(55, linestyle=":", color="#777", linewidth=1)
    axes[1].set_ylim(0, 65)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("V5 — apprendre une correction peut dégrader d’autres choix", fontsize=15)
    fig.text(.5, .015, "Copies du modèle final · trois graines · rappel tiré d’autres sources que les 107 positions mesurées\nArchives déjà exposées à la campagne : diagnostic local, sans estimation Elo", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, .94))
    for suffix in ["png", "svg"]:
        fig.savefig(OUT / f"consolidation.{suffix}", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for sign, axis in zip([-1, 1], axes):
        for positive, label, color in [(False, "Quart équilibré", "#167a80"), (True, "Quart réservé aux victoires", "#b64a38")]:
            trials = [x for x in interference["value_trials"] if x["positive_only_quarter"] == positive]
            steps = [x["step"] for x in trials[0]["trace"]]
            values = np.array([[next(g["mse"] for g in t["validation"] if g["target"] == sign) for t in r["trace"]] for r in trials])
            axis.plot(steps, values.mean(axis=0), label=label, color=color)
            axis.fill_between(steps, values.min(axis=0), values.max(axis=0), color=color, alpha=.15)
        axis.set_title("23 défaites forcées" if sign == -1 else "107 victoires forcées")
        axis.set_ylabel("Erreur de valeur — plus bas = mieux")
        axis.set_xlabel("Lots de 64 exemples")
        axis.grid(alpha=.18)
    axes[0].legend(fontsize=8)
    fig.suptitle("L’équilibre du rappel déplace le compromis entre gains et pertes", fontsize=13)
    fig.text(.5, .02, "48 exemples communs et équilibrés ; les 16 derniers changent · valeur seule · trois graines\nDeux variantes en progrès sur les défaites : cette expérience ne reproduit pas l’oubli de la campagne", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, .94))
    for suffix in ["png", "svg"]:
        fig.savefig(OUT / f"rappel-valeur.{suffix}", dpi=170)
    plt.close(fig)

    production = read("production-hashes-before.json")
    after = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in production}
    assert production == after, "Production changed during audit: inspect before claiming preservation"
    (OUT / "production-hashes-after.json").write_text(json.dumps(after, indent=2) + "\n")
    print(json.dumps({"fifo": len(rows), "proof_admission_exclusions": len(excluded), "gradients": len(grad["finite_differences"]), "production_files_unchanged": len(after)}))


if __name__ == "__main__":
    main()
