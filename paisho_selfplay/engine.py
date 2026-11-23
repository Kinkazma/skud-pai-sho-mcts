"""Resident rules-only process with multi-slot empty starts."""
import os
import subprocess
import threading
from paisho_ppo.engine import BINARY, Engine


class SelfplayEngine(Engine):
    def __init__(self, workers=4, compact=True, json_backend='orjson'):
        if not BINARY.is_file(): raise RuntimeError('Build the selfplay rules adapter first')
        if type(workers) is not int or not 1<=workers<=64: raise ValueError('Invalid workers')
        self.workers,self.compact,self.json_backend=workers,compact,json_backend
        self.timings={};self._histories=[];self._lock=threading.Lock()
        self.process=subprocess.Popen([str(BINARY)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            env={**os.environ,'GPT_SKUD_ENGINE_WORKERS':str(workers)})
        self.description=self.request('describe')

    def request(self, op, **data):
        result=super().request(op,**data)
        if op=='reset_empty': self._histories=[o['history'] for o in result]
        elif op=='reset_empty_slots':
            for item in result: self._histories[item['slot']]=item['observation']['history']
        return result
