#!/usr/bin/env python3
"""Read-only V33 campaign census. Receipt mtimes date archive writes, not search starts."""
import collections as C
import hashlib
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'training-runs/micro-gen5-ui-20260912-131332-99610d'
OUT = ROOT / 'benchmarks/results/gen5-v33-campaign-review-2026-09-12'
OUT.mkdir(parents=True, exist_ok=True)
def read(p): return json.loads(Path(p).read_text())
def save(name, obj): (OUT/name).write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clock(t): return datetime.fromtimestamp(t, ZoneInfo('Europe/Paris')).isoformat(timespec='seconds')
start = read(RUN/'started.json')['unix_seconds']
status = read(RUN/'status.json')
initial = read(RUN/'resume-progress.json')
final = read(RUN/'training/progress.json')
retention = read(RUN/'training/checkpoint-retention.json')
protected = ['status.json','config.json','plan.json','initial-model.json','resume-progress.json',
 'publication-guard.json','training/progress.json','training/durable-progress.json',
 'training/model.json','training/replay-final.index.json','training/checkpoint-retention.json']
hashes={str(RUN/p):sha(RUN/p) for p in protected}
if (OUT/'input-hashes.json').exists():assert read(OUT/'input-hashes.json')==hashes
else:save('input-hashes.json',hashes)
save('status.json', status)
rows=[]; ignored=[]; seen=set()
with (RUN/'training.log').open() as f, (OUT/'receipts.jsonl').open('w') as dst:
 for line in f:
  try: j=json.loads(line)
  except json.JSONDecodeError:
   ignored.append(line[:200]); continue
  if 'id' not in j: ignored.append(j); continue
  assert j['id'] not in seen; seen.add(j['id'])
  case=j['case'] or {}
  t=(RUN/f"training/games/game-{j['id']:07}.json").stat().st_mtime
  r={k:j.get(k) for k in ['id','actor_version','collector','collector_version','model_version','updates',
   'reanalysis','measurement','frozen_evaluation','candidate_seat','outcome','termination','campaign_censored',
   'seconds','pool_wait_seconds','search_seconds','sample_seconds','maintenance_seconds','durable_save_seconds',
   'fresh_used','replay_used','human_used','durable_draws','proved_winning_recall_used','learned_batches',
   'eligible_examples','continuation_decisions','prefix_decisions','fresh_evidence','error','fully_learned',
   'reference_budget','reference_identity','inference_evaluations','inference_cache_hits','simulations',
   'policy_coverage_sum','policy_searches','psr_sha256','targets_file','targets_sha256']}
  r.update(time=t,local=clock(t),relative=t-start,group=case.get('opponent_generation') or 'unattributed',
   kind=case.get('kind'),source=case.get('human_source'),case=case.get('case'),
   first_attempt=case.get('before',{}).get('attempts')==0)
  r['result']='U' if r['outcome']=='Ongoing' else ('D' if r['outcome']=='Draw' else
   ('W' if r['outcome']==f"Win({r['candidate_seat']})" else 'L'))
  rows.append(r);dst.write(json.dumps(r)+'\n')
assert len(rows)==final['completed']-initial['completed']
assert sum(r['learned_batches'] for r in rows)==final['updates']-initial['updates']
def stat(rs):
 fresh=[r for r in rs if not r['reanalysis']]
 teachers=C.Counter();players=C.Counter()
 for r in rs:
  teachers.update(r['fresh_evidence'].get('teachers',{}));players.update(r['fresh_evidence'].get('players',{}))
 return dict(receipts=len(rs),fresh_games=len(fresh),terminal=sum(r['result']!='U' for r in fresh),
  reanalyses=len(rs)-len(fresh),results=dict(C.Counter(r['result'] for r in fresh)),
  terminations=dict(C.Counter(r['termination'] for r in fresh)),initial=sum(r['prefix_decisions']==0 for r in fresh),
  measurements=sum(bool(r['measurement']) for r in fresh),sources=len(set(r['source'] for r in fresh)),
  unique_psrs=len(set(r['psr_sha256'] for r in fresh)),teachers=dict(teachers),players=dict(players),
  observed=sum(r['fresh_evidence'].get('observed',0) for r in rs),
  value_only=sum(r['fresh_evidence'].get('value_only',0) for r in rs),
  policy_only=sum(r['fresh_evidence'].get('policy_only',0) for r in rs),
  sums={k:sum(r[k] or 0 for r in rs) for k in ['fresh_used','replay_used','human_used','durable_draws',
   'proved_winning_recall_used','learned_batches','eligible_examples','continuation_decisions','seconds',
   'pool_wait_seconds','sample_seconds','maintenance_seconds','durable_save_seconds',
   'inference_evaluations','inference_cache_hits','simulations','policy_searches','policy_coverage_sum']},
  search_seconds=[sum(r['search_seconds'][i] for r in rs) for i in range(2)])
summary=stat(rows)
summary.update(start=clock(start),end=clock(status['ended_unix_seconds']),
 wall_seconds=status['ended_unix_seconds']-start,deadline_seconds=status['end_unix_seconds']-start,
 first_receipt=clock(min(r['time'] for r in rows)),last_receipt=clock(max(r['time'] for r in rows)),
 first_receipt_seconds=min(r['relative'] for r in rows),
 counter_deltas={k:final[k]-initial.get(k,0) for k in ['completed','version','updates','elapsed_seconds',
  'learner_seconds','archive_seconds','publish_seconds','fresh_terminal_used','human_used','replay_evicted']},
 session_timers={k:final[k] for k in ['consolidation_seconds','durable_save_seconds']},
 groups={g:stat([r for r in rows if r['group']==g]) for g in sorted(set(r['group'] for r in rows))},
 errors=[r['id'] for r in rows if r['error']],ignored_log_lines=ignored,
 guard=final['publication_guard'],
 bilateral_terminal_mismatches=[r['id'] for r in rows if not r['reanalysis'] and r['result']!='U'
  and r['eligible_examples']!=r['continuation_decisions']],
 archived_time_monotonic_violations=sum(b['time']<a['time'] for a,b in zip(rows,rows[1:])))
summary['bins']=[]
for left in range(0,int(summary['wall_seconds'])+1,600):
 rs=[r for r in rows if left<=r['relative']<left+600]
 width=min(600,summary['wall_seconds']-left)
 s=stat(rs);s.update(start=left,end=left+width,local=clock(start+left),width=width)
 summary['bins'].append(s)
# Accepted snapshots are identified by actual non-measurement fresh play; frozen lots use older actors.
actors=[]
for identity in dict.fromkeys(r['collector'] for r in rows if not r['reanalysis'] and not r['measurement']):
 rs=[r for r in rows if r['collector']==identity and not r['reanalysis'] and not r['measurement']]
 actors.append(dict(identity=identity,version=rs[0]['collector_version'],first=clock(min(r['time'] for r in rs)),
  first_unix=min(r['time'] for r in rs),last=clock(max(r['time'] for r in rs)),games=len(rs)))
summary['actors']=actors
summary['publication_transitions']=[]
previous=None
for r in rows:
 if r['actor_version']!=previous:
  summary['publication_transitions'].append(dict(version=r['actor_version'],local=r['local'],
   time=r['time'],learner_version=r['model_version'],updates=r['updates'],id=r['id']))
  previous=r['actor_version']
# Rebuild every frozen slot from archive receipts, independently of the progress totals.
lots=[]
for g in ['3.1','3.2','3.3','3.4','3.5']:
 allr=[r for r in rows if r['group']==g and r['measurement']]
 for n in sorted(set(r['frozen_evaluation']['lot'] for r in allr)):
  rs=[r for r in allr if r['frozen_evaluation']['lot']==n]
  eligible=[r for r in rs if not r['campaign_censored'] and not r['error']]
  slots=[r['frozen_evaluation']['slot'] for r in eligible]
  assert len(set(slots))==len(slots)
  assert all(r['collector']==r['frozen_evaluation']['model'] for r in rs)
  assert all(r['candidate_seat']==('Host' if r['frozen_evaluation']['slot']%2==0 else 'Guest') for r in rs)
  results=C.Counter(r['result'] for r in eligible)
  l=dict(group=g,lot=n,version=rs[0]['collector_version'],identity=rs[0]['collector'],budget=rs[0]['reference_budget'],
   results=dict(results),games=len(eligible),complete=len(eligible)==100,promoted=len(eligible)==100 and results['W']>=60,
   first=clock(min(r['time'] for r in rs)),last=clock(max(r['time'] for r in rs)),
   first_unix=min(r['time'] for r in rs),last_unix=max(r['time'] for r in rs),
   prefixes=len(set(r['frozen_evaluation']['prefix'] for r in eligible)),
   sources=len(set(r['source'] for r in eligible)),seats=dict(C.Counter(r['candidate_seat'] for r in eligible)),
   paired_sources=len(set(sl//2 for sl in slots if (sl^1) in slots)),
   slots={r['frozen_evaluation']['slot']:dict(result=r['result'],prefix=r['frozen_evaluation']['prefix'],source=r['source'],id=r['id']) for r in eligible})
  if l['complete']:
   batch=next(b for b in final['opponent_ladders'][g]['batches'] if b.get('frozen',{}).get('lot')==n)
   assert (batch['wins'],batch['draws'],batch['unknown'],batch['promoted'])==(results['W'],results['D'],results['U'],l['promoted'])
  else:
   f=final['frozen_evaluations'][int(g[-1])-1]
   assert f['lot']==n
   assert {i:z for i,z in enumerate(f['results']) if z is not None}=={r['frozen_evaluation']['slot']:{'W':1,'L':-1,'D':0,'U':2}[r['result']] for r in eligible}
  lots.append(l)
summary['lots']=lots
# Paired bootstrap groups both seats from the same human source and preserves cross-reference dependence.
rng=np.random.default_rng(20260912)
def contrast(a,b,common_only=False):
 slots=sorted(set(a['slots'])&set(b['slots'])); pairs=sorted({sl//2 for sl in slots if (sl^1) in slots})
 assert all(a['slots'][sl]['prefix']==b['slots'][sl]['prefix'] for sl in slots)
 delta=np.array([sum((b['slots'][2*p+s]['result']=='W')-(a['slots'][2*p+s]['result']=='W') for s in (0,1))/2 for p in pairs])
 boots=delta[rng.integers(0,len(delta),(20000,len(delta)))].mean(axis=1)
 clusters=C.defaultdict(list)
 for p,d in zip(pairs,delta):clusters[a['slots'][2*p]['prefix']].append(float(d))
 values=np.array([sum(v) for v in clusters.values()]); weights=np.array([len(v) for v in clusters.values()])
 draws=rng.integers(0,len(values),(20000,len(values)))
 cluster_boot=values[draws].sum(axis=1)/weights[draws].sum(axis=1)
 return dict(group=a['group'],from_lot=a['lot'],to_lot=b['lot'],budget=a['budget'],pairs=len(pairs),
  win_delta=float(delta.mean()),paired_95=np.quantile(boots,[.025,.975]).tolist(),
  distinct_prefixes=len(clusters),prefix_clustered_95=np.quantile(cluster_boot,[.025,.975]).tolist())
summary['paired_changes']=[]
for g in ['3.1','3.2','3.3','3.4','3.5']:
 ls=[l for l in lots if l['group']==g]
 for a,b in zip(ls,ls[1:]):
  if a['budget']==b['budget']:summary['paired_changes'].append(contrast(a,b))
# Enumerate retained files and record their immutable source provenance.
checkpoints=[]
for m in retention['models']:
 p=Path(m['path']);a=read(p)
 assert len(a['parameters'])==93071
 checkpoints.append(dict(**m,sha256=sha(p),updates=a['updates'],provenance=a['provenance'],
  mtime=p.stat().st_mtime,bytes=p.stat().st_size))
summary['checkpoint_coverage']=dict(count=len(checkpoints),pruned=final['checkpoint_models_pruned'],
 maximum_observation_gap_seconds=max(np.diff(sorted(m['elapsed_millis']/1000 for m in checkpoints))),
 scored=sum(bool(m['results']) for m in checkpoints),bytes=sum(m['bytes'] for m in checkpoints))
save('checkpoints.json',checkpoints);save('summary.json',summary)
print(json.dumps({k:v for k,v in summary.items() if k not in ('groups','lots','bins','ignored_log_lines')},indent=2))
