"""Resident native opponents, asynchronously searched by four CPU workers."""
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import json
import hashlib
import os
from pathlib import Path
import subprocess
import threading

ROOT=Path(__file__).resolve().parents[1]
BINARY=ROOT/'native/target/release/paisho-skud-league'
SPECS=ROOT/'training/opponents.json'
TILES=('', 'R3','R4','R5','W3','W4','W5','L','O','R','W','K','B')


def assets_root():
    value = os.environ.get('GPT_SKUD_MCTS_ROOT')
    if not value:
        raise ValueError('Supply --mcts-root with a checkout of the Gen3 repository')
    return Path(value).resolve(strict=True)


def identity():
    # One startup check only; never reread artifacts between decisions or games.
    digest=hashlib.sha256(SPECS.read_bytes())
    root = assets_root()
    for row in json.loads(SPECS.read_text()):
        actual=hashlib.sha256((root/row['model']).read_bytes()).hexdigest()
        if actual!=row['sha256']:raise ValueError('Published model hash changed')
        digest.update(actual.encode())
    digest.update(hashlib.sha256(BINARY.read_bytes()).digest())
    return digest.hexdigest()


def action_text(fields):
    family,tile,source,destination=fields
    def point(index):return f'{index%17-8},{8-index//17}'
    if family==0:return f'plant {TILES[tile]} {point(destination)}'
    if family==1:return f'arrange {point(source)} {point(destination)}'
    if family==2:return 'skip-bonus'
    if family==3:return f'accent {TILES[tile]} at {point(destination)}'
    if family==4:return f'accent {TILES[tile]} move {point(source)} {point(destination)}'
    if family==5:return f'plant-special {TILES[tile]} {point(destination)}'
    if family==6:return f'bonus-plant {TILES[tile]} {point(destination)}'
    raise ValueError('Unknown action family')


class Resident:
    def __init__(self):
        self.lock=threading.Lock()
        self.process=subprocess.Popen([str(BINARY),str(SPECS)],stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,text=True,cwd=assets_root(),env={**os.environ,'RAYON_NUM_THREADS':'1'})
        try:
            startup = self.process.stdout.readline()
            if not startup:
                raise ValueError('Could not load the Gen3 models. Check the core-memory assets.')
            ready=json.loads(startup)
            expected=json.loads(SPECS.read_text())
            if not ready['ready'] or ready['opponents']!=expected or ready['model_loads']!=5:
                raise ValueError('Published opponent identities differ')
        except BaseException:
            self.close()
            raise

    def request(self,op,**kwargs):
        with self.lock:
            self.process.stdin.write(json.dumps({'op':op,**kwargs})+'\n');self.process.stdin.flush()
            line=self.process.stdout.readline()
            if not line:raise RuntimeError(f'MCTS process exited: {self.process.poll()}')
            result=json.loads(line)
            if not result['ok']:raise ValueError(result['error'])
            return result['data']

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:self.process.stdin.close()
        self.process.wait(timeout=10)
        self.process.stdout.close()


class Pool:
    def __init__(self,environments,workers=4):
        self.residents=[];self.futures={};self.executor=ThreadPoolExecutor(max_workers=workers)
        try:
            # Each slot retains its five small published models for the entire campaign.
            # Private pipes prevent one slot's search blocking another slot's apply.
            for _ in range(environments):self.residents.append(Resident())
        except BaseException:self.close();raise

    def reset(self,slot,match):
        self.residents[slot].request('reset',generation=match['generation'],budget=match['budget'],
            seed=match['seed'],mcts_player=1-match['transformer_player'])

    def restore(self,matches,games):
        futures=[]
        for slot,match in enumerate(matches):
            if match['kind']=='mcts':
                futures.append(self.executor.submit(self.residents[slot].request,'restore',
                    generation=match['generation'],budget=match['budget'],seed=match['seed'],
                    mcts_player=1-match['transformer_player'],record=games[slot]['record'],choice=match['choice']))
        for future in futures:future.result()

    def submit(self,slot):
        if slot not in self.futures:self.futures[slot]=self.executor.submit(self.residents[slot].request,'choose')

    def harvest(self,matches,block=False):
        if block and self.futures:wait(self.futures.values(),timeout=.05,return_when=FIRST_COMPLETED)
        for slot,future in list(self.futures.items()):
            if future.done():matches[slot]['choice']=future.result();del self.futures[slot]

    def drain(self,matches):
        for slot,future in self.futures.items():matches[slot]['choice']=future.result()
        self.futures.clear()

    def apply(self,slot,action):self.residents[slot].request('apply',action=action)

    def close(self):
        self.executor.shutdown(wait=True)
        for resident in self.residents:resident.close()
