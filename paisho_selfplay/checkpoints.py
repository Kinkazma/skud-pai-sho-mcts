"""One immutable checkpoint snapshot and one bounded background disk writer."""
from concurrent.futures import ThreadPoolExecutor
import errno
import json
import time
import torch
from paisho_ppo import storage,budget


def freeze(value):
    if type(value) in (int,float,str,bool,bytes,type(None)):return value
    if isinstance(value,torch.Tensor):return value.detach().to('cpu',copy=True)
    if isinstance(value,dict):return {k:freeze(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(freeze(v) for v in value)
    return value


def write_snapshot(directory,payload,summary,limit_mb):
    start=time.perf_counter()
    try:
        result=storage.save(directory,payload,summary,limit_mb)
        status={'saved':True,'generation':result['generation'],'storage_reclamation':result['storage_reclamation']}
    except (budget.SpaceUnavailable,OSError) as error:
        if isinstance(error,OSError) and error.errno not in (errno.ENOSPC,errno.EDQUOT):raise
        status={'saved':False,'deferred':True,'reason':str(error)}
    return status,time.perf_counter()-start


class PeriodicSaver:
    def __init__(self):
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='skud-checkpoint')
        self.future=None

    def poll(self,trainer,wait=False):
        if self.future is None or (not wait and not self.future.done()):return False
        status,seconds=self.future.result();self.future=None
        trainer.checkpoint_status=status
        measured=trainer.timings.values.setdefault('checkpoint_disk',{'calls':0,'seconds':0.})
        measured['calls']+=1;measured['seconds']+=seconds
        if status.get('deferred'):print(json.dumps({'checkpoint':status}),flush=True)
        return True

    def submit(self,trainer,directory):
        self.poll(trainer)
        if self.future is not None:return False
        # All mutable state is copied on the learner thread. The writer never
        # touches GPU tensors, Adam state, RNGs, games or the live population.
        start=time.perf_counter();payload=trainer.payload();summary=trainer.summary()
        payload=freeze(payload);summary=freeze(summary)
        measured=trainer.timings.values.setdefault('checkpoint_capture',{'calls':0,'seconds':0.})
        measured['calls']+=1;measured['seconds']+=time.perf_counter()-start
        self.future=self.executor.submit(write_snapshot,directory,payload,summary,trainer.config.storage_limit_mb)
        return True

    def close(self,trainer):
        try:self.poll(trainer,wait=True)
        finally:self.executor.shutdown(wait=True)
