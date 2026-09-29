#!/usr/bin/env python3
"""Resident frozen duels and finite-population, anytime-valid paired analysis.

No learner/control mutation. The random permutation is fixed before the first
game. Every unit is one start with BOTH seats and all still-active arms. A
mixture of nonnegative betting martingales permits optional stopping. These are
confidence bounds for this enumerated population, not all possible Pai Sho games.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import time

LAMBDAS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.49)
BUDGETS = (32, 64, 128, 256)


def save(path, obj):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2) + "\n")
    temporary.replace(path)


def log_capital(values, population, mean, sign):
    """X in [-1,1]; E[X_t|past]=(N*mu-sum(past))/(N-t+1).

    Constant signed stakes <= .49 keep each factor >= .02 under a
    feasible mean. The equal mixture starts at one. Ville + union bound
    over 8 comparisons * 2 tails gives familywise 95% time-uniform coverage.
    """
    wealth = [0.0] * len(LAMBDAS)
    previous = 0.0
    for i, x in enumerate(values):
        conditional = (population * mean - previous) / (population - i)
        if not -1.000000001 <= conditional <= 1.000000001:
            raise ValueError("mean outside finite population feasibility")
        for k, stake in enumerate(LAMBDAS):
            wealth[k] += math.log1p(sign * stake * (x - conditional))
        previous += x
    largest = max(wealth)
    return largest + math.log(sum(math.exp(w - largest) for w in wealth) / len(wealth))


def confidence_interval(values, population, alpha_tail=0.05 / 16):
    if not values:
        return (-1.0, 1.0)
    n, total = len(values), sum(values)
    if n > population or any(not -1 <= x <= 1 for x in values):
        raise ValueError("invalid finite population sample")
    if n == population:
        return (total / n, total / n)
    low = max(-1.0, (total - (population - n)) / population)
    high = min(1.0, (total + (population - n)) / population)
    threshold = math.log(1 / alpha_tail)
    # Positive capital is decreasing in candidate mean; negative is increasing.
    lower = low
    if log_capital(values, population, low, +1) >= threshold:
        a, b = low, high
        for _ in range(42):
            m = (a + b) / 2
            if log_capital(values, population, m, +1) >= threshold:
                a = m
            else:
                b = m
        lower = a  # outward rounding
    upper = high
    if log_capital(values, population, high, -1) >= threshold:
        a, b = low, high
        for _ in range(42):
            m = (a + b) / 2
            if log_capital(values, population, m, -1) >= threshold:
                b = m
            else:
                a = m
        upper = b
    return lower, upper


def pair_difference(games, index, budget, floor):
    lookup = {(g["floor"], g["seat"]): g for g in games
              if g["start_index"] == index and g["budget"] == budget}
    if any((f, seat) not in lookup for f in (0, floor) for seat in ("H", "G")):
        raise ValueError("incomplete paired start")
    return sum(int(lookup[floor, s]["outcome"] == "win") -
               int(lookup[0, s]["outcome"] == "win") for s in ("H", "G")) / 2


def strip_times(value):
    if isinstance(value, dict):
        return {k: strip_times(v) for k, v in value.items()
                if k not in ("id", "seconds", "v5_seconds", "gen35_seconds")}
    if isinstance(value, list):
        return [strip_times(v) for v in value]
    return value


def request(process, command):
    process.stdin.write(json.dumps(command) + "\n")
    process.stdin.flush()
    line = process.stdout.readline()
    if not line:
        raise RuntimeError(f"native process stopped, return code {process.poll()}")
    return json.loads(line)


def counts(games):
    return {o: sum(g["outcome"] == o for g in games) for o in ("win", "loss", "draw", "unresolved")}


def run(binary, plan_path, out):
    out.mkdir()
    plan = json.loads(plan_path.read_text())
    native = out / "native"
    errors = (out / "native.log").open("w")
    process = subprocess.Popen([str(binary), str(plan_path), str(native)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=errors, text=True, bufsize=1)
    save(out / "process.json", {"pid": process.pid, "binary": str(binary),
                               "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()})
    ready = json.loads(process.stdout.readline())
    if not ready.get("ready"):
        raise RuntimeError("native preparation failed")
    preparation = json.loads((native / "population.json").read_text())
    allowed_exclusions = {"pilot opening excluded", "empty or terminal prefix", "already terminal", "duplicate exact prefix"}
    if any(e["reason"] not in allowed_exclusions for e in preparation["excluded"]):
        raise RuntimeError("unanticipated input exclusion; do not start games")
    population = preparation["starts"]
    order = [p["index"] for p in population if not p["origin"].get("diagnostic_only", False)]
    random.Random(plan["permutation_seed"]).shuffle(order)
    save(out / "order.json", order)
    save(out / "ready.json", ready)
    print(f"Ready: {len(order)} distinct starts, {len(order)*24} maximum games; models loaded once", flush=True)

    # Same real complete games, serial then parallel. Timing fields alone may vary.
    # This diagnostic sample is NOT included in confirmatory estimates.
    parity_index = next(p["index"] for p in population if p["origin"].get("diagnostic_only", False))
    parity_jobs = [{"id": f"serial-{b}-{f}-{int(s)}", "start": parity_index,
                    "budget": b, "floor": f, "guest": s}
                   for b in (32, 256) for f in (0, 16, 32) for s in (False, True)]
    a = request(process, {"serial": True, "jobs": parity_jobs})
    parallel_jobs = [dict(j, id=j["id"].replace("serial", "parallel")) for j in parity_jobs]
    b = request(process, {"jobs": parallel_jobs})
    for left, right in zip(a["games"], b["games"]):
        if strip_times(left) != strip_times(right):
            raise AssertionError("parallel game result changed")
        ld = [json.loads(x) for x in (native / "games" / left["id"] / "decisions.jsonl").read_text().splitlines()]
        rd = [json.loads(x) for x in (native / "games" / right["id"] / "decisions.jsonl").read_text().splitlines()]
        if strip_times(ld) != strip_times(rd):
            raise AssertionError("parallel decision/counter changed")
    pilot = Path(plan["pilot_results"])
    pilot_games = [json.loads(s) for s in (pilot/"games.jsonl").read_text().splitlines()]
    for game in a["games"]:
        previous = next(g for g in pilot_games if g["start"] == game["start"] and
                        g["budget"] == game["budget"] and g["floor"] == game["floor"] and g["seat"] == game["seat"])
        if any(previous[k] != game[k] for k in ("psr_sha256", "new_decisions", "outcome", "termination", "seed")):
            raise AssertionError("existing pilot trajectory changed")
    save(out / "parallel-parity.json", {"exact": True, "existing_pilot_exact": True, "games_each": len(parity_jobs),
                                       "serial_seconds": a["batch_seconds"], "parallel_seconds": b["batch_seconds"]})
    print("Serial/parallel parity: all 12 full games, actions and work counters exact", flush=True)

    active = {(budget, floor) for budget in BUDGETS for floor in (16, 32)}
    samples = {key: [] for key in active}
    history = {key: [-1.0, 1.0] for key in active}
    decisions = {}
    games = []
    started = time.monotonic()
    offset = 0
    while active and offset < len(order):
        if (out / "STOP").exists():
            break
        # Short batches keep progress durable and allow cooperative stops.
        chunk = order[offset:offset + plan["batch_starts"]]
        jobs = []
        for index in chunk:
            for budget in BUDGETS:
                floors = [f for b, f in sorted(active) if b == budget]
                if not floors:
                    continue
                floors = [0] + floors
                for guest in (False, True):
                    rotation = (index + int(guest)) % len(floors)
                    for floor in floors[rotation:] + floors[:rotation]:
                        jobs.append({"id": f"main-{index}-{budget}-{floor}-{int(guest)}",
                                     "start": index, "budget": budget, "floor": floor, "guest": guest})
        response = request(process, {"jobs": jobs})
        batch = response["games"]
        games.extend(batch)
        with (out / "games.jsonl").open("a") as journal:
            for game in batch:
                journal.write(json.dumps(game) + "\n")
        for key in sorted(active):
            budget, floor = key
            samples[key].extend(pair_difference(batch, index, budget, floor) for index in chunk)
        offset += len(chunk)
        analysis = {}
        for key in sorted(samples):
            values = samples[key]
            if key in active and (offset % plan["analysis_every"] == 0 or offset == len(order)):
                lo, hi = confidence_interval(values, len(order))
                history[key] = [max(history[key][0], lo), min(history[key][1], hi)]
                if len(values) >= plan["minimum_starts"]:
                    if history[key][1] < plan["minimum_useful_gain"]:
                        decisions[key] = "useful_gain_ruled_out"
                    elif history[key][0] > plan["minimum_useful_gain"]:
                        decisions[key] = "useful_gain_established"
                    elif len(values) == len(order):
                        decisions[key] = "census_at_decision_boundary"
                    if key in decisions:
                        active.remove(key)
            budget, floor = key
            # Arms can stop early: both counts use exactly that arm's same starts.
            ids = set(order[:len(values)])
            selected = [g for g in games if g["budget"] == budget and g["start_index"] in ids]
            analysis[f"{budget}/{floor}"] = {"starts": len(values), "gain": sum(values)/len(values),
                "simultaneous_interval": history[key], "decision": decisions.get(key, "continue"),
                "baseline": counts([g for g in selected if g["floor"] == 0]),
                "floor": counts([g for g in selected if g["floor"] == floor])}
        save(out / "analysis.json", {"complete": not active, "population": len(order),
            "visited_starts": offset, "games": len(games), "elapsed_seconds": time.monotonic()-started,
            "comparisons": analysis})
        summary = "; ".join(f"{key}: {v['gain']:+.3f} [{v['simultaneous_interval'][0]:+.3f},{v['simultaneous_interval'][1]:+.3f}] {v['decision']}"
                            for key, v in analysis.items())
        print(f"{offset} starts / {len(games)} games / {time.monotonic()-started:.1f}s | {summary}", flush=True)
    process.stdin.write('{"finish":true}\n');process.stdin.flush()
    final = json.loads(process.stdout.readline())
    code = process.wait()
    if code != 0 or not final.get("complete"):
        raise RuntimeError(f"native completion failed: {code}")
    save(out / "completion.json", {"native": final, "all_comparisons_decided": not active,
         "games": len(games), "remaining": sorted(active)})
    errors.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    run(args.binary.resolve(), args.plan.resolve(), args.out.resolve())
