#!/usr/bin/env python3
"""Choose a reproducible initial bank from existing terminal receipts and pinned humans."""
from pathlib import Path
import json,hashlib,heapq
import argparse
p=argparse.ArgumentParser()
p.add_argument('--dataset',type=Path,required=True);p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();out=a.output.resolve();out.mkdir(exist_ok=False,parents=True)
data=json.load(a.dataset.open())
humans=[]
for g in data['games']:
 original=g['originals'][0]
 humans.append(dict(path=original['path'],sha256=original['sha256'],source='human/'+g['game_sha256'],human=True,held_out=g['held_out'],external=g.get('external_outcome',{}).get('outcome'),metadata={k:v for k,v in g.items() if k!='examples'}))
assert len(humans)<50000, "pinned humans leave no generated slots"
seen={x['sha256'] for x in humans};heap=[];n=50000-len(humans);scanned=0
# Reservoir hash selection across completed games; originals untouched. Native replay verifies every retained result.
for log in sorted(a.runs.resolve().glob('micro-gen5-*/training.log')):
 try:
  with log.open() as stream:
   for line in stream:
    try:r=json.loads(line)
    except ValueError:continue
    if r.get('rules')!='skud-pai-sho-gen5-v1' or r.get('outcome') not in ('Win(Host)','Win(Guest)','Draw') or r.get('error') or not r.get('psr_sha256'):continue
    h=r['psr_sha256']
    if h in seen:continue
    seen.add(h);scanned+=1
    path=log.parent/'training/games'/f"game-{r['id']:07d}.psr"
    if not path.exists():continue
    priority=int(h[:16],16)
    row=dict(path=str(path),sha256=h,source=str(log.parent/'training')+'/'+str(r['id']),human=False,held_out=False,external=None,metadata={k:r.get(k) for k in ('id','lane','outcome','collector','rules','case','seconds','decisions')})
    item=(-priority,h,row)
    if len(heap)<n:heapq.heappush(heap,item)
    elif item>heap[0]:heapq.heapreplace(heap,item)
 except FileNotFoundError:continue
rows=humans+[x[2] for x in sorted(heap,key=lambda x:x[1])]
(out/'sources.json').write_text(json.dumps(rows))
(out/'sources-small.json').write_text(json.dumps(humans[:40]+[x[2] for x in heap[:160]]))
print(json.dumps(dict(human=len(humans),selected=len(rows),scanned_unique_terminal=scanned)),flush=True)
