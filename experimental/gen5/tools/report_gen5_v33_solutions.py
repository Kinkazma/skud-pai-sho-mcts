#!/usr/bin/env python3
"""Summarize prototype evidence without modifying campaign state."""
import hashlib
import json
import statistics as S
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'benchmarks/results/gen5-v33-solutions-2026-09-12'
REVIEW=ROOT/'benchmarks/results/gen5-v33-campaign-review-2026-09-12'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,v):(OUT/name).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
r=read(OUT/'results.json');fits=r['teaching']['fits'];assert len(fits)==18
teaching=[]
for mi in [0,4]:
 for mode in ['legacy_raw','corrected_raw','coupled_loss']:
  xs=[x for x in fits if x['model']==mi and x['mode']==mode];assert len(xs)==3
  assert all(x['value_parameters_exact'] for x in xs)
  teaching.append(dict(model=mi,mode=mode,
   initial=xs[0]['history'][0],mean_final={part:{k:S.mean(x['history'][-1][part][k] for x in xs) for k in xs[0]['history'][-1][part]} for part in ['train','test']},
   proof_initial=xs[0]['proof_retention_before'],proof_final=[x['proof_retention_after'] for x in xs],
   seeds=[x['seed'] for x in xs],mean_fit_seconds=S.mean(x['seconds'] for x in xs)))
projections=[]
for x in r['repair']['projected_updates']['runs']:
 end=x['finite_corrections'][-1];ordinary=next(t for t in x['trials'] if t['kind']=='ordinary' and t['rate']==.02)
 assert all(a<=b+1e-9 for a,b in zip(end['reference_losses'],x['before']))
 assert x['verified_choices_lost']==0
 projections.append(dict(model=x['model'],reference_before=x['before'],reference_after=end['reference_losses'],
  initial_fresh_loss=x['fresh_before'],ordinary_fresh_loss=ordinary['fresh_loss'],protected_fresh_loss=end['fresh_loss'],
  preserved_fresh_loss_improvement=(x['fresh_before']-end['fresh_loss'])/(x['fresh_before']-ordinary['fresh_loss']),
  finite_correction_iterations=end['iteration'],seconds=x['finite_correction_seconds'],
  verified_before=x['old_verified_choices'],verified_after=x['new_verified_choices']))
cache=r['teaching']['cache'];assert cache['request_outputs_exact']
baseline=S.mean(x['seconds'] for x in cache['abba'] if not x['cached']);cached=S.mean(x['seconds'] for x in cache['abba'] if x['cached'])
p=read(OUT/'branch-panel-results.json');m=read(OUT/'branch-panel-manifest.json')
guard_sources=set(read(REVIEW/'guard-source-exclusion.json')['guard_sources'])
indices=[i for i,x in enumerate(p['positions']) if m['positions'][i]['panel']=='historical' and x['source'] not in guard_sources and x.get('target')==1.]
assert len(indices)==271
by={(x['model'],x['position']):x for x in p['results']};branches=[]
for i,model in enumerate(m['models']):
 scores={}
 for prefix in ['', 'coupled_']:
  before=[by[0,j][prefix+'certified_selected'] or by[0,j][prefix+'immediate_selected'] for j in indices]
  after=[by[i,j][prefix+'certified_selected'] or by[i,j][prefix+'immediate_selected'] for j in indices]
  scores[prefix or 'raw']=dict(before=sum(before),after=sum(after),gains=sum(not a and b for a,b in zip(before,after)),lost_verified_choices=sum(a and not b for a,b in zip(before,after)))
 branches.append(dict(label=model['label'],path=model['path'],scores=scores))
guards=[]
for path in sorted(OUT.glob('native-guard-*/verification.json')):
 x=read(path);guards.append(dict(directory=path.parent.name,state=x['state'],accepted_score=x['accepted']))
assert len(guards)==8
for g in guards:
 if g['directory'] in ['native-guard-accepted-policy-repaired-value','native-guard-improved-policy-repaired-value']:
  assert g['state']['last_decision']=='projected' and g['state']['accepted_fraction']==.5
 else:
  assert g['state']['last_decision']=='accepted' and g['state']['accepted_fraction']==1. and not g['state']['reasons']
summary=dict(teaching=teaching,protected_updates=projections,recall=read(OUT/'recall-cache-contracts.json'),
 branch_candidates=branches,guards=guards,repairs=r['repair']['repairs'],
 cache=dict(baseline_seconds=baseline,cached_seconds=cached,reduction=1-cached/baseline,requests_per_run=32,actual_searches_per_baseline=32,actual_searches_per_cached=16,scope=cache['scope']),
 limitation='Diagnostic policies/values only; no new matches or promotion. Historical source-disjoint panels were used to select/check prototypes, not independent global-strength validation.')
save('summary.json',summary)
production=read(REVIEW/'input-hashes.json');sources=read(ROOT/'benchmarks/results/gen5-loop-integration-2026-09-12/candidate-source-hashes.json')
assert all(sha(p)==h for p,h in production.items())
assert all(sha(ROOT/p)==h for p,h in sources.items())
paths=[*ROOT.glob('crates/paisho-train/examples/gen5_v33_solution_probe/*.rs'),ROOT/'crates/paisho-train/examples/gen5_v33_solution_probe.rs',ROOT/'crates/paisho-train/examples/gen5_v33_branch_compose.rs',ROOT/'tools/probe_gen5_v33_recall_cache.py',Path(__file__).resolve(),OUT/'results.json',OUT/'summary.json',OUT/'recall-cache-contracts.json',OUT/'branch-panel-manifest.json',OUT/'branch-panel-results.json']
save('verification.json',dict(production_files_unchanged=len(production),production_sources_unchanged=len(sources),native_fits=len(fits),native_branch_panel_evaluations=len(p['results']),new_matches=0,production_writes=0,hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}))
print(json.dumps({'preserved_fresh_learning':[x['preserved_fresh_loss_improvement'] for x in projections], 'cache':summary['cache'],'branches':[(x['label'],x['scores']) for x in branches],'production_unchanged':True},indent=2))
