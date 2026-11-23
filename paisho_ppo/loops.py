"""Bounded full-sequence repetition detection, adapted from the old detector.

Only exact-state matching is retained. No old search filters, loser adjudication,
training targets, replay or acceptance mechanisms are imported.
"""
import hashlib
import json


def position_key(obs):
    # Completed turns do not change Skud's legal choices or terminal conditions.
    # Player + relative ownership recover absolute ownership unambiguously.
    state = {k: obs[k] for k in ("board", "reserves", "player", "phase", "rules", "outcome")}
    return hashlib.sha256(json.dumps(state, separators=(",", ":")).encode()).hexdigest()


def history_capacity(config):
    return max(config.repetition_max_period*config.repetition_cycles,
               config.repetition_min_decisions+config.repetition_max_period-1)+1


def repeated_suffix(history, config):
    for period in range(2, config.repetition_max_period+1):
        cycles = max(config.repetition_cycles,
                     (config.repetition_min_decisions+period-1)//period)
        span = period*cycles
        if len(history) <= span:
            continue
        start = len(history)-span-1
        if all(history[i] == history[i-period] for i in range(start+period, len(history))):
            return {"period": period, "cycles": cycles, "decisions": span}
    return None


def observe(history, observation, config):
    history.append(position_key(observation))
    del history[:-history_capacity(config)]
    if observation["outcome"] != -2:
        return None
    return repeated_suffix(history, config)
