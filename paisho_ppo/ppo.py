import math
import time
import torch
from .model import pack, _StableGather
from .runtime import Lookahead, defaults
from .execution import Execution


def terminal_return(outcome, player):
    if player not in (0, 1):
        raise ValueError("invalid player")
    if outcome == -2:
        raise ValueError("an unfinished game has no terminal return")
    if outcome == -1:
        return 0.0
    if outcome not in (0, 1):
        raise ValueError("invalid outcome or player")
    return 1.0 if outcome == player else -1.0


def clipped_loss(logp, old_logp, advantages, epsilon, weights=None):
    ratio = (logp - old_logp).exp()
    terms = -torch.minimum(ratio * advantages,
                           ratio.clamp(1-epsilon, 1+epsilon) * advantages)
    return (terms if weights is None else terms*weights).mean()


def update(model, optimizer, samples, config, baseline, generator, device, deadline=None,
           should_stop=None, progress=None, max_steps_per_call=None, prefetch=None, execution=None):
    enabled = defaults()["prefetch"] if prefetch is None else prefetch
    ahead = Lookahead(enabled)
    execution = Execution(device, enabled, execution)
    try:
        return _update(model, optimizer, samples, config, baseline, generator, device,
                       deadline, should_stop, progress, max_steps_per_call, ahead, execution)
    finally:
        ahead.close()


def _update(model, optimizer, samples, config, baseline, generator, device, deadline,
            should_stop, progress, max_steps_per_call, ahead, execution):
    if not samples:
        return {"updated": False, "complete": True, "optimizer_steps": 0,
                "unique_decisions_trained": 0, "reason": "no finished-game decisions"}, baseline
    progress = {} if progress is None else progress
    if not progress:
        progress.update(epoch=0, order=[], offset=0, steps=0, unique_decisions=0,
                        baseline=baseline, seconds=0.0, complete=False,
                        sums={"loss": 0.0, "entropy": 0.0, "kl": 0.0,
                              "clip_fraction": 0.0, "grad_norm": 0.0})
    baseline = progress["baseline"]
    returns = torch.tensor([s["return"] for s in samples], dtype=torch.float32)
    represented = torch.tensor([s.get("weight", 1.0) for s in samples], dtype=torch.float32)
    weights = represented/represented.mean()
    # A constant baseline from PREVIOUS rollouts; no state-value prediction,
    # no self-labelled targets, and no within-batch mean that erases uniform wins.
    advantages = returns - baseline
    scale = max(float(advantages.square().mean().sqrt()), 1.0)
    advantages /= scale
    return_mean = float((returns*weights).mean())
    next_baseline = config.baseline_decay * baseline + (1-config.baseline_decay) * return_mean
    if not torch.any(advantages != 0):
        progress["complete"] = True
        return {"updated": False, "complete": True, "optimizer_steps": 0,
                "unique_decisions_trained": 0, "reason": "zero outcome signal", "samples": len(samples)}, next_baseline
    started = time.monotonic()
    model.train()
    sums = progress["sums"]
    call_steps, stopped, budget_stop = 0, False, False
    while progress["epoch"] < config.epochs and not progress["complete"]:
        if (deadline is not None and time.monotonic() >= deadline) or (should_stop is not None and should_stop()):
            budget_stop = True
            break
        if max_steps_per_call is not None and call_steps >= max_steps_per_call:
            break
        if not progress["order"]:
            progress["order"] = torch.randperm(len(samples), generator=generator).tolist()
        order = progress["order"]
        if progress["offset"] < len(order):
            indices = order[progress["offset"]:progress["offset"]+config.minibatch]
            selected = [samples[i] for i in indices]
            key = (progress["epoch"], tuple(indices))
            prepared = ahead.get(key, pack, [s["observation"] for s in selected], "cpu", config.context_moves)
            following = order[progress["offset"]+len(indices):progress["offset"]+len(indices)+config.minibatch]
            if following:
                ahead.submit((progress["epoch"], tuple(following)), pack,
                             [samples[i]["observation"] for i in following], "cpu", config.context_moves)
            logits = model(execution.move(prepared))
            distribution = torch.distributions.Categorical(logits=logits)
            actions = torch.tensor([s["index"] for s in selected], device=device)
            old = torch.tensor([s["logp"] for s in selected], device=device)
            if distribution.logits.device.type == "mps":
                # Categorical.log_prob's default gather backward uses MPS
                # atomic scatter too, even with only one chosen move per row.
                logp = _StableGather.apply(distribution.logits[:, :, None], actions[:, None],
                                          distribution.logits.shape[1]).reshape(-1)
            else:
                logp = distribution.log_prob(actions)
            log_ratio = logp-old
            ratio = log_ratio.exp()
            kl = float(((ratio-1)-log_ratio).mean().detach())
            if not math.isfinite(kl):
                raise FloatingPointError("non-finite PPO divergence")
            if kl > config.target_kl:
                stopped = True
                progress["complete"] = True
                break
            signal = advantages[indices].to(device)
            entropy = distribution.entropy().mean()
            loss = clipped_loss(logp, old, signal, config.clip, weights[indices].to(device)) - config.entropy_weight*entropy
            grouped_metrics = distribution.logits.device.type == "mps" and ahead.executor is not None
            if not grouped_metrics and not torch.isfinite(loss):
                raise FloatingPointError("non-finite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = execution.clip(model.parameters(), config.grad_clip,
                                  error_if_nonfinite=not grouped_metrics)
            if grouped_metrics:
                observed = torch.stack((loss.detach(),entropy.detach(),
                    ((ratio-1).abs()>config.clip).float().mean().detach(),norm)).detach().cpu().tolist()
                if not all(math.isfinite(x) for x in observed):
                    raise FloatingPointError("non-finite PPO loss, gradient or metric")
            execution.step(optimizer)
            call_steps += 1
            progress["steps"] += 1
            progress["offset"] += len(indices)
            if progress["epoch"] == 0:
                progress["unique_decisions"] += len(indices)
            if grouped_metrics:
                values = dict(zip(("loss","entropy","clip_fraction","grad_norm"),observed))
                values["kl"] = kl
            else:
                values = {"loss": float(loss.detach()), "entropy": float(entropy.detach()),
                          "kl": kl, "clip_fraction": float(((ratio-1).abs()>config.clip).float().mean().detach()),
                          "grad_norm": float(norm)}
            for name, value in values.items():
                sums[name] += value
        if progress["offset"] >= len(order):
            progress["epoch"] += 1
            progress["offset"] = 0
            progress["order"] = []
    if progress["epoch"] >= config.epochs:
        progress["complete"] = True
    progress["seconds"] += time.monotonic()-started
    steps = progress["steps"]
    result = {name: value/max(steps, 1) for name, value in sums.items()}
    result.update(updated=steps>0, complete=progress["complete"], optimizer_steps=steps,
                  unique_decisions_trained=progress["unique_decisions"], samples=len(samples),
                  kl_stop=stopped, baseline=baseline, return_mean=return_mean,
                  represented_decisions=round(float(represented.sum())),
                  budget_stop=budget_stop, seconds=progress["seconds"])
    return result, next_baseline
