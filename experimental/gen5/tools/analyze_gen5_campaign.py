#!/usr/bin/env python3
"""Read-only campaign journal analysis. Writes only the requested evidence directory."""
import argparse,collections,datetime as dt,hashlib,json,math
from pathlib import Path
import numpy as np

def read(p):return json.loads(Path(p).read_text())
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def stats(rows):
 w=sum(r['result']=='W' for r in rows);d=sum(r['result']=='D' for r in rows);l=sum(r['result']=='L' for r in rows);u=len(rows)-w-d-l
 return dict(n=len(rows),wins=w,draws=d,losses=l,unknown=u,win_rate=w/len(rows) if rows else None,terminal_score=(w+d/2)/(w+d+l) if w+d+l else None,unique_cases=len({r['case'] for r in rows}),unique_sources=len({r['source'] for r in rows}),first_attempts=sum(r['attempt']==0 for r in rows))
def paired(rows,mid):
 cells=collections.defaultdict(lambda:[[],[]]);source={}
 for r in rows:
  key=(r['case'],r['seat'],r['budget']);cells[key][int(r['time']>=mid)].append(float(r['result']=='W'));source[key]=r['source']
 diffs=[(source[k],float(np.mean(v[1])-np.mean(v[0])),len(v[0]),len(v[1])) for k,v in cells.items() if v[0] and v[1]]
 if not diffs:return dict(cells=0,sources=0,delta=None,interval=None)
 groups=collections.defaultdict(list)
 for src,v,_,_ in diffs:groups[src].append(v)
 totals=np.array([sum(v) for v in groups.values()]);counts=np.array([len(v) for v in groups.values()]);rng=np.random.default_rng(921)
 ids=rng.integers(len(groups),size=(4000,len(groups)));boot=totals[ids].sum(axis=1)/counts[ids].sum(axis=1)
 return dict(cells=len(diffs),sources=len(groups),delta=float(np.mean([v[1] for v in diffs])),interval=np.quantile(boot,[.025,.975]).tolist(),early_attempts=sum(v[2] for v in diffs),late_attempts=sum(v[3] for v in diffs))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--single-run',action='store_true',help='Analyze only this run, preserving its resume baseline');a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 runs=[];r=a.campaign.resolve()
 while True:
  o=read(r/'config.json');p=read(r/'training/progress.json');prior=read(o['resume_progress']) if o.get('resume_progress') else {};s=read(r/'status.json');runs.append((r,o,p,prior,s))
  if a.single_run or '20260911-183405' in r.name or not prior.get('previous_campaign'):break
  r=Path(prior['previous_campaign'])
 runs.reverse();allrows=[];summaries=[];hashes={};ladders={};checkpoints=[]
 for phase,(r,o,p,prior,s) in enumerate(runs):
  hashes.update({str(r/x):digest(r/x) for x in ['config.json','status.json','training/progress.json','training/model.json','training.log']})
  local=[];errors=[];learned=recall=proof=0;first_stamp=None;meta_rows=0
  f=(a.output/f'receipts-phase-{phase}.jsonl').open('w')
  with (r/'training.log').open() as stream:
   for raw in stream:
    if '"lane":' not in raw:continue
    row=json.loads(raw);meta_rows+=1
    if row.get('error'):errors.append((row['id'],row['error']))
    stamp=(r/'training/games'/f"game-{row['id']:07}.json").stat().st_mtime
    if first_stamp is None:first_stamp=stamp
    c=row.get('case') or {};before=c.get('before') or {};out=row['outcome'];seat=row.get('candidate_seat');terminal=out in ['Win(Host)','Win(Guest)','Draw']
    result='U' if not terminal else 'D' if out=='Draw' else 'W' if out==f'Win({seat})' else 'L'
    recall+=row.get('durable_draws',0);proof+=row.get('proved_winning_recall_used',0);learned+=row.get('fresh_used',0)+row.get('replay_used',0)+row.get('human_used',0)
    v=dict(id=row['id'],phase=phase,time=stamp,version=row['collector_version'],after_version=row['model_version'],updates=row['updates'],lane=row['lane'],reanalysis=bool(row.get('reanalysis')),generation=row.get('opponent') if row['lane']=='Historical' else 'selfplay',reference=row.get('reference_identity'),budget=row.get('reference_budget'),seat=seat,result=result,termination=row['termination'],error=row.get('error'),case=c.get('case'),source=c.get('human_source'),zone=c.get('zone'),actor=c.get('actor'),ticket=before.get('ticket'),attempt=before.get('attempts'),focus=before.get('focus'),rotate=(c.get('after') or {}).get('rotate_reason'),prefix=row.get('prefix_decisions',0),decisions=row.get('continuation_decisions',row['decisions']),seconds=row['seconds'],fresh=row.get('fresh_used',0),replay=row.get('replay_used',0),recall=row.get('durable_draws',0),proof_recall=row.get('proved_winning_recall_used',0),learned=row.get('fresh_used',0)+row.get('replay_used',0)+row.get('human_used',0),fully_learned=row.get('fully_learned'),censored=row.get('campaign_censored'),psr=row.get('psr_sha256'),search_seconds=sum(row.get('search_seconds',[])),pool_wait=row.get('pool_wait_seconds',0),simulations=row.get('simulations',0))
    f.write(json.dumps(v,separators=(',',':'))+'\n');local.append(v)
  f.close();expected=p['completed']-prior.get('completed',0);assert len(local)==expected,(r,len(local),expected)
  assert local[-1]['after_version']==p['version'];assert local[-1]['updates']==p['updates']
  fresh=[x for x in local if not x['reanalysis']];ref=[x for x in fresh if x['lane']=='Historical'];sp=[x for x in fresh if x['lane']=='Selfplay']
  summaries.append(dict(run=str(r),phase=phase,start=s['started_unix_seconds'],end=s['ended_unix_seconds'],first_receipt=first_stamp,receipts=len(local),continuations=len(fresh),selfplay=len(sp),reference=len(ref),reanalyses=len(local)-len(fresh),terminals=sum(x['result']!='U' for x in fresh),errors=errors,version=[prior.get('version'),p['version']],updates=[prior.get('updates'),p['updates']],learned=learned,recall=recall,proof_recall=proof,reference_results=stats(ref),by_generation={g:stats([x for x in ref if x['generation']==g]) for g in sorted({x['generation'] for x in ref})},last_lanes=p['lanes'],initial_lanes=prior.get('lanes',{})))
  for k,l in p.get('opponent_ladders',{'3.1':p.get('legacy_ladder',{})}).items():
   old=prior.get('opponent_ladders',{}).get(k,prior.get('legacy_ladder',{}) if k=='3.1' else {});ladders[f'{phase}:{k}']={'stage':l.get('stage'),'new_batches':l.get('batches',[])[len(old.get('batches',[])):],'initial_current':old.get('current',[]),'current':l.get('current',[])}
  allrows.extend(local)
  for file in sorted((r/'training/models').glob('model-*.json')):
   model=read(file);checkpoints.append(dict(path=str(file),phase=phase,version=int(file.stem.split('-')[1]),updates=model['updates'],time=file.stat().st_mtime,sha256=digest(file)))
  print('phase',phase,'receipts',len(local),'references',len(ref),flush=True)
 fresh=[x for x in allrows if not x['reanalysis']];latest=[x for x in fresh if x['phase']==len(runs)-1];mid=(summaries[-1]['first_receipt']+runs[-1][4]['end_unix_seconds'])/2
 trends={}
 for generation in sorted({r['generation'] for r in latest if r['lane']=='Historical'}):
  rows=[r for r in latest if r['generation']==generation];early=[r for r in rows if r['time']<mid];late=[r for r in rows if r['time']>=mid]
  first=[r for r in rows if r['attempt']==0]
  trends[generation]=dict(all=stats(rows),early=stats(early),late=stats(late),first=stats(first),first_early=stats([r for r in first if r['time']<mid]),first_late=stats([r for r in first if r['time']>=mid]),matched=paired(rows,mid),unique_trajectories=len({r['psr'] for r in rows}),unknown_reasons=dict(collections.Counter(r['termination'] for r in rows if r['result']=='U')))
 buckets=collections.defaultdict(list)
 for r in fresh:buckets[(r['phase'],int(r['time']//600)*600,r['generation'])].append(r)
 bins=[dict(phase=k[0],start=k[1],generation=k[2],**stats(rows),mean_new_decisions=float(np.mean([r['decisions'] for r in rows])),median_seconds=float(np.median([r['seconds'] for r in rows]))) for k,rows in sorted(buckets.items())]
 memory=read(runs[-1][0]/'training/checkpoint-retention.json');quality=[m for m in memory['models'] if any(all(n>=2 for n in res['attempts']) for res in m.get('results',{}).values())]
 summary=dict(phases=summaries,trends=trends,comparison_midpoint=mid,bins=bins,checkpoints=checkpoints,retention=dict(retained=len(memory['models']),quality_eligible=len(quality),any_results=sum(bool(m.get('results')) for m in memory['models'])),ladders=ladders)
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(a.output/'input-hashes.json').write_text(json.dumps(hashes,indent=2)+'\n');print('complete',len(allrows),flush=True)
if __name__=='__main__':main()
