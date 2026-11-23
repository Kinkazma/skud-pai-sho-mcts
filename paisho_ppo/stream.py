"""Completion-driven terminal PPO, with bounded active game behavior snapshots."""
import math
import statistics
import torch
from . import rollout
from .loops import position_key
from .model import pack
from .ppo import update
from . import ensemble


def record_length(state,length):
    state['seen'] += 1
    state['lengths'].append(length)
    state['lengths'] = state['lengths'][-4096:]
    ordered = sorted(state['lengths'])
    state['median'] = statistics.median(ordered)
    state['p99'] = ordered[math.ceil(.99*len(ordered))-1]


def length_threshold(state,config):
    if state['seen'] < 100:
        return None
    return max(state['p99'],state['median']*config.length_median_factor)


def may_censor(length,state,counters,config):
    threshold = length_threshold(state,config)
    stopped = counters['length_outlier_stops']
    # Prospective resolved-game fraction is STRICTLY less than 1%. Starts and
    # active slots cannot artificially enlarge the abandonment allowance.
    within_budget = (stopped+1)*100 < counters['finished_games']+stopped+1
    return threshold is not None and length > threshold and within_budget


def snapshot(trainer):
    version = trainer.counters['optimizer_steps']
    if version not in trainer.behavior:
        trainer.behavior[version] = {name:p.detach().clone() for name,p in trainer.model.named_parameters()}
    return version


def initialize(trainer):
    if trainer.pending is None:
        trainer.pending = rollout.start(trainer.engine,trainer.config,trainer.sampling_rng)
        trainer.counters['started_games'] += trainer.config.environments
    if 'versions' not in trainer.pending:
        version = snapshot(trainer)
        trainer.pending['versions'] = [version]*len(trainer.pending['done'])


def predict(trainer,observations,slots):
    # Length buckets stop one exceptional history padding every short game to
    # 12,250 tokens. All operations in a bucket are batched on the one GPU.
    groups = {}
    for row,obs in enumerate(observations):
        length = 250+min(len(obs.get('history',[])),trainer.config.context_moves)
        groups.setdefault((length-1).bit_length(),[]).append(row)
    outputs = []
    for bucket in sorted(groups):
        rows = groups[bucket]
        batch = pack([observations[r] for r in rows],trainer.device,trainer.config.context_moves)
        versions = [trainer.pending['versions'][slots[r]] for r in rows]
        # Per-game parameters share one native batched attention call. vmap's
        # MPS attention fallback was measured and rejected: it ran games serially.
        params = ensemble.stack([trainer.behavior[v] for v in versions],trainer.model)
        logits = ensemble.forward(trainer.model,params,batch)
        outputs.append((rows,logits.log_softmax(-1)))
    result = [None]*len(observations)
    # Enqueue all independent GPU groups before copying any result back.
    for rows,values in outputs:
        values = values.cpu()
        for row,value in zip(rows,values):
            result[row] = value
    return result


def replace_closed(trainer,slots):
    openings = torch.randint(6,(len(slots),),generator=trainer.sampling_rng).tolist()
    replies = trainer.engine.request('reset_slots',rules=trainer.config.rules,
                                    openings=[{'slot':s,'opening':o} for s,o in zip(slots,openings)])
    version = snapshot(trainer)
    pending = trainer.pending
    for item in replies:
        slot,obs = item['slot'],item['observation']
        for key,value in {'observations':obs,'trajectories':[],'done':False,'outcomes':-2,
                          'terminations':'','decisions':0,'loop_details':None,
                          'loop_history':[position_key(obs)],'versions':version}.items():
            pending[key][slot] = value
    active = set(pending['versions'])
    trainer.behavior = {v:p for v,p in trainer.behavior.items() if v in active}
    trainer.counters['started_games'] += len(slots)


def collect(trainer):
    initialize(trainer)
    pending = trainer.pending
    previous = pending['done'].copy()
    count = sum(not value for value in previous)
    trainer.timings.call('selfplay_round',rollout.advance,trainer.engine,trainer.model,pending,
                         trainer.config,trainer.sampling_rng,trainer.device,trainer.reservoir_rng,
                         predict=lambda observations,slots:predict(trainer,observations,slots))
    trainer.counters['decisions_played'] += count
    # Rule-terminal outcomes and genuine repetition detection take precedence.
    for slot,done in enumerate(pending['done']):
        if not done and may_censor(pending['decisions'][slot],trainer.outliers,trainer.counters,trainer.config):
            pending['done'][slot] = True
            pending['terminations'][slot] = 'length-outlier-unknown'
            pending['loop_details'][slot] = {'length':pending['decisions'][slot],
                                           'threshold':length_threshold(trainer.outliers,trainer.config)}
            trainer.outliers['last_censor'] = {**pending['loop_details'][slot],
                                             'median':trainer.outliers['median'],'p99':trainer.outliers['p99']}
            # Reserve the allowance immediately: simultaneous slots cannot all
            # claim the same remaining <1% budget.
            trainer.counters['length_outlier_stops'] += 1
    trainer.record_closed_games(previous)
    closed = [i for i,(before,after) in enumerate(zip(previous,pending['done'])) if not before and after]
    for slot in closed:
        if pending['outcomes'][slot] != -2:
            record_length(trainer.outliers,pending['decisions'][slot])
            game = {k:[pending[k][slot]] for k in ('observations','trajectories','done','outcomes',
                                                  'terminations','decisions','loop_details')}
            trainer.ready.append({'game':game,
                                  'statistics':rollout.statistics(game),'progress':{}})
    if closed:
        replace_closed(trainer,closed)


def optimize_ready(trainer,deadline=None,should_stop=None,prefetch=None,max_steps=8):
    if not trainer.ready:
        return None
    job = trainer.ready[0]
    progress = job['progress']
    previous_steps = progress.get('steps',0)
    previous_unique = progress.get('unique_decisions',0)
    metrics,baseline = trainer.timings.call('ppo_update',update,trainer.model,trainer.optimizer,
        rollout.finished_samples(job['game']),trainer.config,trainer.baseline,trainer.shuffle_rng,trainer.device,
        deadline,should_stop,progress,max_steps_per_call=max_steps,prefetch=prefetch)
    trainer.counters['optimizer_steps'] += metrics['optimizer_steps']-previous_steps
    trainer.counters['trained_decisions'] += metrics['unique_decisions_trained']-previous_unique
    if metrics['complete']:
        trainer.baseline = baseline
        trainer.counters['rollouts'] += 1
        trainer.counters['updates'] += int(metrics['updated'])
        trainer.account_time()
        rate = trainer.counters['finished_games']/max(trainer.counters['elapsed_seconds'],1e-9)
        trainer.history.append({'rollout':trainer.counters['rollouts'],**job['statistics'],
                                'ppo':metrics,'finished_games_per_second':rate})
        trainer.history = trainer.history[-trainer.config.history_limit:]
        trainer.ready.pop(0)
    return metrics


def tick(trainer,deadline=None,should_stop=None,prefetch=None):
    metrics = optimize_ready(trainer,deadline,should_stop,prefetch)
    if metrics is not None and metrics.get('budget_stop'):
        return metrics
    # Explicit backpressure bounds the number of finished trajectories in RAM.
    # A long active game NEVER blocks the consumption of completed games.
    if len(trainer.ready) < trainer.config.environments:
        collect(trainer)
    return metrics
