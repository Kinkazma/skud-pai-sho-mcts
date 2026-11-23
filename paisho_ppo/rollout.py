import torch
from .model import pack
from .ppo import terminal_return
from .loops import position_key, observe


def start(engine, config, generator):
    openings = torch.randint(6, (config.environments,), generator=generator).tolist()
    observations = engine.request("reset", rules=config.rules, openings=openings)
    return {"observations": observations, "trajectories": [[] for _ in observations],
            "done": [False]*len(observations), "outcomes": [-2]*len(observations),
            "terminations": [""]*len(observations),
            "decisions": [0]*len(observations), "loop_details": [None]*len(observations),
            "loop_history": [[position_key(obs)] for obs in observations]}


def advance(engine, model, pending, config, generator, device, reservoir_generator=None, predict=None):
    slots = [i for i, done in enumerate(pending["done"]) if not done]
    if not slots:
        return
    observations = [pending["observations"][i] for i in slots]
    model.eval()
    with torch.inference_mode():
        log_probs = (model(pack(observations, device, config.context_moves)).log_softmax(-1).cpu()
                     if predict is None else predict(observations,slots))
    moves = []
    staged_samples = {}
    for row, slot in enumerate(slots):
        index = int(torch.multinomial(log_probs[row].exp(), 1, generator=generator))
        moves.append({"slot": slot, "index": index})
        observation = observations[row]
        compact = {k: v for k, v in observation.items() if k != "history"}
        compact["history_length"] = len(observation.get("history", []))
        staged_samples[slot] = {"observation": compact,
                               "index": index, "logp": float(log_probs[row][index])}
    # Neither samples nor histories are committed before native batch validation.
    reply = engine.request("step", moves=moves)
    for item in reply:
        slot, obs = item["slot"], item["observation"]
        pending["decisions"][slot] += 1
        samples = pending["trajectories"][slot]
        sample = staged_samples[slot]
        if len(samples) < config.samples_per_game:
            samples.append(sample)
        else:
            # Uniform reservoir over the entire game, without stopping its play.
            rng = reservoir_generator if reservoir_generator is not None else generator
            index = int(torch.randint(pending["decisions"][slot], (1,), generator=rng))
            if index < config.samples_per_game:
                samples[index] = sample
        pending["observations"][slot] = obs
        cycle = observe(pending["loop_history"][slot], obs, config)
        if obs["outcome"] != -2:
            pending["done"][slot] = True
            pending["outcomes"][slot] = obs["outcome"]
            pending["terminations"][slot] = "rules-terminal"
        elif not obs["actions"]:
            pending["done"][slot] = True
            pending["terminations"][slot] = "no-legal-actions-unknown"
        elif cycle is not None:
            pending["done"][slot] = True
            pending["terminations"][slot] = "loop-unknown"
            pending["loop_details"][slot] = {**cycle, "last_decision": pending["decisions"][slot]}


def finished_samples(pending):
    if not all(pending["done"]):
        raise ValueError("finish collection before updating the player")
    samples = []
    for slot, (outcome, trajectory) in enumerate(zip(pending["outcomes"], pending["trajectories"])):
        if outcome == -2:
            continue
        for sample in trajectory:
            full_history = pending["observations"][slot].get("history", [])
            observation = {**sample["observation"], "history": full_history}
            samples.append({**sample, "observation": observation,
                            "return": terminal_return(outcome, sample["observation"]["player"]),
                            "weight": pending["decisions"][slot]/len(trajectory)})
    return samples


def statistics(pending):
    outcomes = pending["outcomes"]
    return {"started_games": len(outcomes), "finished_games": sum(o != -2 for o in outcomes),
            "host_wins": outcomes.count(0),
            "guest_wins": outcomes.count(1), "draws": outcomes.count(-1),
            "loop_stops": pending["terminations"].count("loop-unknown"),
            "blocked_stops": pending["terminations"].count("no-legal-actions-unknown"),
            "decisions_played": sum(pending["decisions"]),
            "retained_samples": sum(len(t) for t in pending["trajectories"]),
            "terminations": pending["terminations"], "loop_details": pending["loop_details"]}
