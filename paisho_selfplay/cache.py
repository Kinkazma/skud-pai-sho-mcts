"""Immutable behavior-parameter workspace, reused only while versions match."""
from paisho_ppo import ensemble


def stacked(trainer,versions):
    key=tuple(versions)
    cache=getattr(trainer,'policy_cache',{})
    if trainer.runtime.get('cache_parameters',False) and key in cache:return cache[key]
    if trainer.runtime.get('broadcast_identical',False):
        raise ValueError('Expanded parameter views failed byte identity and cannot be enabled')
    params=ensemble.stack([trainer.behavior[v] for v in versions],trainer.model)
    if trainer.runtime.get('cache_parameters',False):
        cache[key]=params;trainer.policy_cache=cache
    return params
