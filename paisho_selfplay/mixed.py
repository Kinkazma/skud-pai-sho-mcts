"""Nonblocking mixed collection. Only Transformer decisions enter PPO."""
import torch
from paisho_ppo.loops import observe
from .league import LIMIT
from .opponents import action_text


def advance(trainer):
    pending=trainer.pending;matches=pending['matches'];pool=trainer.opponent_pool
    active=[]
    for slot,done in enumerate(pending['done']):
        if done:continue
        if pending['decisions'][slot]>=LIMIT:
            pending['done'][slot]=True;pending['terminations'][slot]='move-limit-unknown'
        else:active.append(slot)
    neural=[];search=[]
    for slot in active:
        match=matches[slot]
        if match['kind']=='selfplay' or pending['observations'][slot]['player']==match['transformer_player']:
            neural.append(slot)
        else:
            search.append(slot)
            if match['choice'] is None:pool.submit(slot)
    moves=[];samples={}
    if neural:
        from .stream import predict
        observations=[pending['observations'][slot] for slot in neural]
        trainer.model.eval()
        with torch.inference_mode():log_probs=predict(trainer,observations,neural)
        for row,slot in enumerate(neural):
            index=int(torch.multinomial(log_probs[row].exp(),1,generator=trainer.sampling_rng))
            obs=pending['observations'][slot]
            compact={k:v for k,v in obs.items() if k!='history'}
            compact['history_length']=len(obs['history'])
            samples[slot]={'observation':compact,'index':index,'logp':float(log_probs[row][index])}
            moves.append({'slot':slot,'index':index})
    pool.harvest(matches,block=not moves)
    for slot in search:
        choice=matches[slot]['choice']
        if choice is None:continue
        legal=pending['observations'][slot]['actions']
        index=next((i for i,a in enumerate(legal) if action_text(a)==choice['action']),None)
        if index is None:raise ValueError('MCTS proposed a move outside the current legal actions')
        if not 0<=choice['simulations']<=matches[slot]['budget']:raise ValueError('MCTS exceeded its budget')
        moves.append({'slot':slot,'index':index})
    if not moves:return 0
    actions={m['slot']:action_text(pending['observations'][m['slot']]['actions'][m['index']]) for m in moves}
    reply=trainer.engine.request('step',moves=moves)
    for item in reply:
        slot,obs=item['slot'],item['observation'];match=matches[slot]
        pending['decisions'][slot]+=1
        if hasattr(trainer,'flow_decisions'):
            trainer.flow_decisions[match['kind']]+=1
        if slot in samples:
            pending['learner_decisions'][slot]+=1
            retained=pending['trajectories'][slot];sample=samples[slot]
            if len(retained)<trainer.config.samples_per_game:retained.append(sample)
            else:
                index=int(torch.randint(pending['learner_decisions'][slot],(1,),generator=trainer.reservoir_rng))
                if index<trainer.config.samples_per_game:retained[index]=sample
        pending['observations'][slot]=obs
        cycle=observe(pending['loop_history'][slot],obs,trainer.config)
        if obs['outcome']!=-2:
            pending['done'][slot]=True;pending['outcomes'][slot]=obs['outcome'];pending['terminations'][slot]='rules-terminal'
        elif not obs['actions']:
            pending['done'][slot]=True;pending['terminations'][slot]='no-legal-actions-unknown'
        elif pending['decisions'][slot]>=LIMIT:
            pending['done'][slot]=True;pending['terminations'][slot]='move-limit-unknown'
        elif cycle is not None and pending['loop_details'][slot] is None:
            pending['loop_details'][slot]={**cycle,'first_observed_at':pending['decisions'][slot]}
        if match['kind']=='mcts':match['choice']=None
    # Own per-slot pipes have no unrelated searches ahead of these applies.
    for slot,action in actions.items():
        if matches[slot]['kind']=='mcts':pool.apply(slot,action)
    return len(moves)
