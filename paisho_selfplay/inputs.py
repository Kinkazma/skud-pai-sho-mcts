"""Exact CPU input construction and asynchronous copies for resident inference."""
import numpy as np
import torch
from paisho_ppo.model import pack as reference_pack


def numpy_pack(observations,device,context_moves=12000,non_blocking=False):
    if type(context_moves) is not int or not 1<=context_moves<=12000:
        raise ValueError('context_moves must be in 1..12000')
    if not observations or any(not o['actions'] for o in observations):
        raise ValueError('model inputs must be nonterminal positions with legal actions')
    batch=len(observations);maximum=max(len(o['actions']) for o in observations)
    actions=np.zeros((batch,maximum,4),dtype=np.int64)
    mask=np.zeros((batch,maximum),dtype=np.bool_)
    for i,o in enumerate(observations):
        actions[i,:len(o['actions'])]=o['actions'];mask[i,:len(o['actions'])]=True
    def tensor(value,dtype):return torch.from_numpy(np.asarray(value,dtype=dtype))
    result={'board':tensor([o['board'] for o in observations],np.int64),
            'phase':tensor([o['phase'] for o in observations],np.int64),
            'player':tensor([o['player'] for o in observations],np.int64),
            'context':tensor([o['reserves']+[o['turns']] for o in observations],np.float32),
            'actions':torch.from_numpy(actions),'mask':torch.from_numpy(mask),
            'setup':tensor([o.get('setup',[0]+[1]*8) for o in observations],np.int64)}
    histories=[o.get('history',[]) for o in observations]
    lengths=[o.get('history_length',len(h)) for o,h in zip(observations,histories)]
    if any(type(n) is not int or n<0 or n>len(h) for n,h in zip(lengths,histories)):
        raise ValueError('invalid chronological history prefix')
    starts=[max(0,n-context_moves) for n in lengths]
    first_players=[h[0][4] if n else 1 for h,n in zip(histories,lengths)]
    if any(p not in (0,1) for p in first_players):raise ValueError('invalid first player in observed history')
    result['first_player']=tensor(first_players,np.int64)
    visible_lengths=[n-start for n,start in zip(lengths,starts)]
    maximum_history=max(visible_lengths)
    history=np.zeros((batch,maximum_history,5),dtype=np.int64)
    history_mask=np.zeros((batch,maximum_history),dtype=np.bool_)
    # Keep Torch's float32 arange and the exact original normalization equations.
    history_positions=torch.zeros((batch,maximum_history),dtype=torch.float32)
    for i,(h,n,start,visible) in enumerate(zip(histories,lengths,starts,visible_lengths)):
        if visible:
            history[i,:visible]=h[start:n];history_mask[i,:visible]=True
            history_positions[i,:visible]=torch.arange(start,n,dtype=torch.float32)
    result.update(history=torch.from_numpy(history),history_mask=torch.from_numpy(history_mask),history_positions=history_positions)
    result['context'][:,:24]/=3.0
    result['context'][:,24]=torch.log1p(result['context'][:,24])/8.0
    return {k:v.to(device,non_blocking=non_blocking) for k,v in result.items()}


def prepare(trainer,observations):
    if trainer.runtime.get('cpu_canonical_inputs',False) and trainer.runtime.get('cache_geometry',False):
        from types import SimpleNamespace
        from paisho_ppo.spatial import canonicalize
        if not hasattr(trainer.model,'_selfplay_cpu_geometry'):
            trainer.model._selfplay_cpu_geometry=SimpleNamespace(_canonical_index=trainer.model._canonical_index.cpu())
        batch=(numpy_pack if trainer.runtime.get('numpy_inputs',False) else reference_pack)(
            observations,'cpu',trainer.config.context_moves)
        batch=canonicalize(trainer.model._selfplay_cpu_geometry,batch)
        return {k:v.to(trainer.device,non_blocking=trainer.runtime.get('async_inputs',False)) for k,v in batch.items()}
    if trainer.runtime.get('numpy_inputs',False):
        return numpy_pack(observations,trainer.device,trainer.config.context_moves,
                          non_blocking=trainer.runtime.get('async_inputs',False))
    if trainer.runtime.get('async_inputs',False):
        batch=reference_pack(observations,'cpu',trainer.config.context_moves)
        return {k:v.to(trainer.device,non_blocking=True) for k,v in batch.items()}
    return reference_pack(observations,trainer.device,trainer.config.context_moves)
