"""Completion-driven terminal PPO, with resident per-decision behavior weights."""
import math
import statistics
from dataclasses import replace
import torch
from . import rollout
from paisho_ppo.loops import position_key
from paisho_ppo.model import pack
from paisho_ppo.ppo import update
from paisho_ppo import ensemble
from .cache import stacked
from .spatial import forward_spatial
from .geometry import geometry,forward_spatial as forward_geometry
from torch.nn import functional as F
from paisho_ppo.spatial import StableRelativeBias


def spatial_biases(model,params):
    for i in range(len(model.blocks)):
        key=f'_board_bias.{i}'
        if key not in params:
            bias=StableRelativeBias.apply(params[f'relative_bias.{i}'],
                model._relative_pairs,model._relative_inverse)
            params[key]=F.pad(bias,(1,0,1,0))


def record_length(state,length):
    state['seen'] += 1
    state['lengths'].append(length)
    state['lengths'] = state['lengths'][-4096:]
    ordered = sorted(state['lengths'])
    state['median'] = statistics.median(ordered)
    state['p99'] = ordered[math.ceil(.99*len(ordered))-1]


def snapshot(trainer):
    version = trainer.counters['optimizer_steps']
    if version not in trainer.behavior:
        trainer.behavior[version] = {name:p.detach().clone() for name,p in trainer.model.named_parameters()}
    return version


def initialize(trainer):
    if trainer.pending is None:
        trainer.pending = rollout.start(trainer.engine,trainer.config,trainer.sampling_rng)
        trainer.counters['started_games'] += trainer.config.environments
        if getattr(trainer,'population',None) is not None:
            trainer.pending['matches']=[trainer.population.launch() for _ in trainer.pending['done']]
            trainer.pending['learner_decisions']=[0]*len(trainer.pending['done'])
            for slot,match in enumerate(trainer.pending['matches']):
                if match['kind']=='mcts':trainer.opponent_pool.reset(slot,match)
    if 'versions' not in trainer.pending:
        version = snapshot(trainer)
        trainer.pending['versions'] = [version]*len(trainer.pending['done'])


def predict(trainer,observations,slots):
    if trainer.runtime.get('live_behavior',False):
        # A running game must use newly learned weights on its NEXT decision.
        # Trajectories already record each actual behavior log-probability;
        # freezing the actor for a whole long game unnecessarily ages them.
        version=snapshot(trainer)
        for slot in slots:trainer.pending['versions'][slot]=version
        active=set(trainer.pending['versions'])
        trainer.behavior={v:p for v,p in trainer.behavior.items() if v in active}
        trainer.policy_cache={k:v for k,v in getattr(trainer,'policy_cache',{}).items()
            if all(version in active for version in k)}
    # Length buckets stop one exceptional history padding every short game to
    # 12,250 tokens. All operations in a bucket are batched on the one GPU.
    groups = {}
    for row,obs in enumerate(observations):
        length = 250+min(len(obs.get('history',[])),trainer.config.context_moves)
        groups.setdefault((length-1).bit_length(),[]).append(row)
    outputs = []
    used_keys=set()
    for bucket in sorted(groups):
        rows = groups[bucket]
        if trainer.runtime.get('skip_forced_groups',False) and all(len(observations[r]['actions'])==1 for r in rows):
            # A singleton categorical distribution is exactly [log(1)] = [0].
            # Keep every row and the later CPU multinomial call in RNG order.
            outputs.append((rows,torch.zeros((len(rows),1),dtype=torch.float32)))
            continue
        from .inputs import prepare
        batch = trainer.timings.call('policy_pack',prepare,trainer,[observations[r] for r in rows])
        versions = [trainer.pending['versions'][slots[r]] for r in rows]
        used_keys.add(tuple(versions))
        # Per-game parameters share one native batched attention call. vmap's
        # MPS attention fallback was measured and rejected: it ran games serially.
        params = trainer.timings.call('behavior_stack',stacked,trainer,versions)
        if trainer.runtime.get('cache_spatial_bias',False):
            trainer.timings.call('behavior_bias',spatial_biases,trainer.model,params)
        if trainer.runtime.get('cache_geometry',False):
            trainer.timings.call('behavior_geometry',geometry,trainer.model,params)
            if trainer.runtime.get('cache_chronology',False):
                from .geometry import chronology
                trainer.timings.call('behavior_chronology',chronology,trainer.model,params,
                    max(len(observations[r].get('history',[])) for r in rows))
            if trainer.runtime.get('cache_layer_biases',False):
                from .geometry import layer_biases
                trainer.timings.call('behavior_layer_biases',layer_biases,trainer.model,params)
            if trainer.runtime.get('cache_history_inputs',False):
                from .geometry import history_inputs
                trainer.timings.call('behavior_history_inputs',history_inputs,trainer.model,params)
            logits=trainer.timings.call('policy_forward',forward_geometry,trainer.model,batch,params=params,
                canonical=trainer.runtime.get('cpu_canonical_inputs',False))
        elif trainer.runtime.get('cache_spatial_bias',False):
            logits=trainer.timings.call('policy_forward',forward_spatial,trainer.model,batch,params=params)
        else:
            logits=trainer.timings.call('policy_forward',ensemble.forward,trainer.model,params,batch)
        outputs.append((rows,logits.log_softmax(-1)))
    result = [None]*len(observations)
    # Enqueue all independent GPU groups before copying any result back.
    for rows,values in outputs:
        values = trainer.timings.call('policy_transfer',values.cpu)
        for row,value in zip(rows,values):
            result[row] = value
    cache=getattr(trainer,'policy_cache',{})
    if trainer.runtime.get('retain_workspaces',False):
        active=set(trainer.pending['versions'])
        cache={k:v for k,v in cache.items() if all(version in active for version in k)}
        for key in sorted(used_keys):
            if key in cache:cache[key]=cache.pop(key)
        while len(cache)>8:del cache[next(iter(cache))]
        trainer.policy_cache=cache
    else:trainer.policy_cache={k:v for k,v in cache.items() if k in used_keys}
    return result


def replace_closed(trainer,slots):
    replies = trainer.engine.request('reset_empty_slots',rules=trainer.config.rules,slots=slots)
    version = snapshot(trainer)
    pending = trainer.pending
    for item in replies:
        slot,obs = item['slot'],item['observation']
        for key,value in {'observations':obs,'trajectories':[],'done':False,'outcomes':-2,
                          'terminations':'','decisions':0,'loop_details':None,
                          'loop_history':[position_key(obs)],'versions':version}.items():
            pending[key][slot] = value
        if getattr(trainer,'population',None) is not None:
            match=trainer.population.launch()
            pending['matches'][slot]=match;pending['learner_decisions'][slot]=0
            if match['kind']=='mcts':trainer.opponent_pool.reset(slot,match)
    active = set(pending['versions'])
    trainer.behavior = {v:p for v,p in trainer.behavior.items() if v in active}
    trainer.counters['started_games'] += len(slots)


def collect(trainer):
    initialize(trainer)
    pending = trainer.pending
    previous = pending['done'].copy()
    if getattr(trainer,'population',None) is not None:
        from .mixed import advance
        count=trainer.timings.call('selfplay_round',advance,trainer)
    else:
        count = sum(not value for value in previous)
        trainer.timings.call('selfplay_round',rollout.advance,trainer.engine,trainer.model,pending,
                             trainer.config,trainer.sampling_rng,trainer.device,trainer.reservoir_rng,
                             predict=lambda observations,slots:predict(trainer,observations,slots))
        trainer.flow_decisions['selfplay']+=count
    trainer.counters['decisions_played'] += count
    trainer.record_closed_games(previous)
    closed = [i for i,(before,after) in enumerate(zip(previous,pending['done'])) if not before and after]
    for slot in closed:
        if pending['outcomes'][slot] != -2:
            record_length(trainer.outliers,pending['decisions'][slot])
            game = {k:[pending[k][slot]] for k in ('observations','trajectories','done','outcomes',
                                                  'terminations','decisions','loop_details')}
            game['reward_profile']=trainer.reward_profile
            match={'kind':'selfplay'}
            if getattr(trainer,'population',None) is not None:
                game['learner_decisions']=[pending['learner_decisions'][slot]]
                match={k:v for k,v in pending['matches'][slot].items() if k!='choice'}
            trainer.ready.append({'game':game,
                                  'statistics':rollout.statistics(game),'progress':{},
                                  'match':match,'entropy_weight':trainer.config.entropy_weight})
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
        rollout.finished_samples(job['game']),replace(trainer.config,entropy_weight=job.get('entropy_weight',trainer.config.entropy_weight)),trainer.baseline,trainer.shuffle_rng,trainer.device,
        deadline,should_stop,progress,max_steps_per_call=max_steps,prefetch=prefetch,execution=trainer.execution_options)
    trainer.counters['optimizer_steps'] += metrics['optimizer_steps']-previous_steps
    trainer.counters['trained_decisions'] += metrics['unique_decisions_trained']-previous_unique
    if getattr(trainer,'population',None) is not None:
        trainer.population.state['optimizer_steps'][job.get('match',{'kind':'selfplay'})['kind']]+=metrics['optimizer_steps']-previous_steps
    if metrics['complete']:
        trainer.baseline = baseline
        trainer.counters['rollouts'] += 1
        trainer.counters['updates'] += int(metrics['updated'])
        trainer.account_time()
        rate = trainer.counters['finished_games']/max(trainer.counters['elapsed_seconds'],1e-9)
        trainer.history.append({'rollout':trainer.counters['rollouts'],**job['statistics'],
                                'ppo':metrics,'finished_games_per_second':rate,
                                'match':job.get('match',{'kind':'selfplay'})})
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
