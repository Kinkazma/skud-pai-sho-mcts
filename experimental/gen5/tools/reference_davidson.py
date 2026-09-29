#!/usr/bin/env python3
"""Independent SciPy check for the Rust Davidson fixture.

This is deliberately not production code: it uses a different zero-sum
parameterization and SciPy's numerical BFGS optimizer. It prints constants that
are sealed into the Rust cross-check fixture; it never writes repository files.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, t


ELO_PER_LOG_STRENGTH = 400.0 / math.log(10.0)


def read_games(path: Path) -> list[tuple[int, str, str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream, delimiter="\t")
        return [
            (int(row["pair_id"]), row["host"], row["guest"], row["outcome"])
            for row in rows
        ]


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: reference_davidson.py GAMES.tsv", file=sys.stderr)
        return 2
    games = read_games(Path(sys.argv[1]))
    agents = sorted({agent for game in games for agent in game[1:3]})
    indices = {agent: index for index, agent in enumerate(agents)}

    def unpack(parameters: np.ndarray) -> tuple[np.ndarray, float, float]:
        free = parameters[: len(agents) - 1]
        skills = np.concatenate((free, [-free.sum()]))
        return skills, parameters[-2], parameters[-1]

    def objective(parameters: np.ndarray) -> float:
        skills, host_advantage, draw_log_weight = unpack(parameters)
        total = 0.0
        for _, host, guest, outcome in games:
            difference = skills[indices[host]] - skills[indices[guest]] + host_advantage
            logits = np.array([difference / 2.0, draw_log_weight, -difference / 2.0])
            maximum = logits.max()
            log_denominator = maximum + np.log(np.exp(logits - maximum).sum())
            observed = {"H": 0, "D": 1, "G": 2}[outcome]
            total += log_denominator - logits[observed]
        return float(total)

    result = minimize(
        objective,
        np.zeros(len(agents) + 1),
        method="BFGS",
        jac="3-point",
        options={"gtol": 1e-9, "maxiter": 10_000},
    )
    if not result.success and np.linalg.norm(result.jac, ord=np.inf) > 2e-6:
        print(result.message, file=sys.stderr)
        return 1
    skills, host_advantage, draw_log_weight = unpack(result.x)
    parameter_count = len(result.x)
    information = np.zeros((parameter_count, parameter_count))
    scores_by_pair: dict[int, np.ndarray] = {}
    for pair_id, host, guest, outcome in games:
        contrast = np.zeros(parameter_count)
        for agent, sign in ((host, 1.0), (guest, -1.0)):
            index = indices[agent]
            if index < len(agents) - 1:
                contrast[index] += sign
            else:
                contrast[: len(agents) - 1] -= sign
        contrast[-2] = 1.0
        features = np.zeros((3, parameter_count))
        features[0] = 0.5 * contrast
        features[1, -1] = 1.0
        features[2] = -0.5 * contrast
        logits = features @ result.x
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        mean = probabilities @ features
        centered = features - mean
        information += sum(
            probability * np.outer(vector, vector)
            for probability, vector in zip(probabilities, centered)
        )
        observed = {"H": 0, "D": 1, "G": 2}[outcome]
        negative_score = mean - features[observed]
        scores_by_pair.setdefault(pair_id, np.zeros(parameter_count))
        scores_by_pair[pair_id] += negative_score

    model_covariance = np.linalg.inv(information)
    meat = sum(np.outer(score, score) for score in scores_by_pair.values())
    cluster_count = len(scores_by_pair)
    correction = (
        cluster_count
        / (cluster_count - 1)
        * (len(games) - 1)
        / (len(games) - parameter_count)
    )
    cluster_covariance = correction * model_covariance @ meat @ model_covariance
    model_quantile = norm.ppf(0.975)
    cluster_quantile = t.ppf(0.975, cluster_count - 1)

    print("PAISHO-DAVIDSON-INDEPENDENT-REFERENCE\t2")
    for agent, skill in zip(agents, skills):
        print(f"rating\t{agent}\t{1500.0 + skill * ELO_PER_LOG_STRENGTH:.9f}")
    print(f"host_advantage_elo\t{host_advantage * ELO_PER_LOG_STRENGTH:.9f}")
    print(f"draw_log_weight\t{draw_log_weight:.12f}")
    print(f"log_likelihood\t{-objective(result.x):.12f}")
    for index, agent in enumerate(agents):
        contrast = np.zeros(parameter_count)
        if index < len(agents) - 1:
            contrast[index] = ELO_PER_LOG_STRENGTH
        else:
            contrast[: len(agents) - 1] = -ELO_PER_LOG_STRENGTH
        print_uncertainty(
            f"rating:{agent}",
            1500.0 + skills[index] * ELO_PER_LOG_STRENGTH,
            contrast,
            model_covariance,
            cluster_covariance,
            model_quantile,
            cluster_quantile,
        )
    host_contrast = np.zeros(parameter_count)
    host_contrast[-2] = ELO_PER_LOG_STRENGTH
    print_uncertainty(
        "host_advantage_elo",
        host_advantage * ELO_PER_LOG_STRENGTH,
        host_contrast,
        model_covariance,
        cluster_covariance,
        model_quantile,
        cluster_quantile,
    )
    draw_contrast = np.zeros(parameter_count)
    draw_contrast[-1] = 1.0
    print_uncertainty(
        "draw_log_weight",
        draw_log_weight,
        draw_contrast,
        model_covariance,
        cluster_covariance,
        model_quantile,
        cluster_quantile,
    )
    return 0


def print_uncertainty(
    name: str,
    estimate: float,
    contrast: np.ndarray,
    model_covariance: np.ndarray,
    cluster_covariance: np.ndarray,
    model_quantile: float,
    cluster_quantile: float,
) -> None:
    model_se = math.sqrt(float(contrast @ model_covariance @ contrast))
    cluster_se = math.sqrt(float(contrast @ cluster_covariance @ contrast))
    values = (
        estimate,
        model_se,
        estimate - model_quantile * model_se,
        estimate + model_quantile * model_se,
        cluster_se,
        estimate - cluster_quantile * cluster_se,
        estimate + cluster_quantile * cluster_se,
    )
    print("uncertainty\t" + name + "\t" + "\t".join(f"{value:.9f}" for value in values))


if __name__ == "__main__":
    raise SystemExit(main())
