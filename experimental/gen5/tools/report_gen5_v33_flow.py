#!/usr/bin/env python3
"""Reconcile native probes, frozen artifacts and the entire V33 receipt census."""
import collections as C
import gzip
import hashlib
import json
import math
import statistics as S
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'benchmarks/results/gen5-v33-flow-audit-2026-09-12'
REVIEW = ROOT / 'benchmarks/results/gen5-v33-campaign-review-2026-09-12'
def read(p): return json.loads(Path(p).read_text())
def save(name, x): (OUT/name).write_text(json.dumps(x, indent=2, ensure_ascii=False)+'\n')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def best(x): return max(range(len(x)), key=x.__getitem__)
def softmax(x):
    v=[math.exp(a-max(x)) for a in x]
    return [a/sum(v) for a in v]
manifest=read(OUT/'manifest.json')
native=read(OUT/'results.json')
teachers={(x['game'], x['decision']):x for x in native['teacher_rows']}
all_teachers=[]
for path in manifest['bundles']:
    assert sha(path)==Path(path).name.removesuffix('.json.gz')
    b=json.loads(gzip.decompress(Path(path).read_bytes()))
    for l in b['lessons']:
        e=l['evidence']
        if e['policy_source']!='full-search-estimate': continue
        p=e['target_prior']; q=e['completed_action_values']; mask=e['excluded_actions']
        assert len(p)==len(q)==len(mask)
        allowed=[i for i in range(len(p)) if not mask[i]]
        t=softmax([math.log(max(p[i],1e-300))+q[i] for i in allowed])
        ti=allowed[best(t)]; pi=max(allowed, key=p.__getitem__)
        allowed_q=sum(p[i]*q[i] for i in allowed)/sum(p[i] for i in allowed)
        row=dict(game=b['game_id'],decision=l['decision'],
                 same_argmax_allowed=ti==pi,
                 q_regret=max(q[i] for i in allowed)-q[ti])
        all_teachers.append(row)
        key=(b['game_id'],l['decision'])
        if key in teachers:
            n=teachers[key]
            assert ti==n['teacher_argmax']
            n['coupled_argmax_allowed']=pi
            n['expected_search_q']['coupled_before_allowed']=allowed_q
assert len(all_teachers)==3360 and len(teachers)==3162
assert all(x['coupled_prior_max_error']==0. for x in teachers.values())
assert max(x['coordinate_correction_max_error'] for x in teachers.values())<2e-15
def difference(a,b):
    ds=[x['expected_search_q'][a]-x['expected_search_q'][b] for x in teachers.values()]
    return dict(n=len(ds), negative=sum(x< -1e-12 for x in ds),
                positive=sum(x>1e-12 for x in ds), mean=S.mean(ds),
                median=S.median(ds), minimum=min(ds), maximum=max(ds))
changes={f'{a}_minus_{b}':difference(a,b) for a,b in [
    ('teacher','coupled_before'),('teacher','coupled_before_allowed'),
    ('deployed_after_exact_raw_fit','coupled_before'),
    ('deployed_after_exact_raw_fit','coupled_before_allowed'),
    ('deployed_after_corrected_raw_fit','teacher')]}
assert changes['teacher_minus_coupled_before_allowed']['negative']==0
assert changes['deployed_after_corrected_raw_fit_minus_teacher']['negative']==0
assert changes['deployed_after_corrected_raw_fit_minus_teacher']['positive']==0
control=native['native_teacher_control']
assert control['coupled_teacher'][1]>control['coupled_before'][1]
assert control['deployed_after_exact_raw_fit'][1]<control['coupled_before'][1]
assert abs(control['deployed_after_corrected_raw_fit'][1]-control['coupled_teacher'][1])<1e-15
assert all(a[0]<b[0] for a,b in zip(control['raw_policy_sequence'],control['raw_policy_sequence'][1:]))

receipts=[json.loads(s) for s in (REVIEW/'receipts.jsonl').open()]
fresh=sum(r['fresh_used'] for r in receipts)
lanes={}
for name in ['reanalysis','archive-reanalysis']:
    rs=[r for r in receipts if r['reanalysis'] and r['kind']==name]
    lanes[name]=dict(receipts=len(rs),fresh=sum(r['fresh_used'] for r in rs))
ages=dict(total_fresh=fresh,
          collector_lags_learner_over_10000_positions=sum(r['fresh_used'] for r in receipts if r['model_version']-r['collector_version']>10000),
          collector_equals_current_actor_positions=sum(r['fresh_used'] for r in receipts if r['collector_version']==r['actor_version']),
          lanes=lanes)

panel=read(REVIEW/'panel-results.json'); pm=read(REVIEW/'panel-manifest.json')
guard_sources=set(read(REVIEW/'guard-source-exclusion.json')['guard_sources'])
outside={i for i,x in enumerate(panel['positions']) if x['source'] not in guard_sources and pm['positions'][i]['panel']=='historical' and x.get('target')==1.}
guard={i for i,x in enumerate(panel['positions']) if pm['positions'][i]['panel']=='guard' and x.get('target')==1.}
assert len(outside)==271 and len(guard)==32
panel_stats=[]
for i,m in enumerate(pm['models']):
    rs=[r for r in panel['results'] if r['model']==i]
    def count(indices):
        v=[r for r in rs if r['position'] in indices]
        return dict(certified=sum(r['certified_selected'] for r in v),
                    certified_or_immediate=sum(r['certified_selected'] or r['immediate_selected'] for r in v),
                    coupled_certified_or_immediate=sum(bool(r.get('coupled_certified_selected')) or bool(r.get('coupled_immediate_selected')) for r in v) if m['coupled'] else None)
    panel_stats.append(dict(version=m['version'],guard=count(guard),outside=count(outside)))
checks=[x for g in native['gradients'] for x in g['results']['directional_checks']]
assert max(abs(x['actual_delta']-x['first_order_delta']) for x in checks)<2e-8
assert all(g['results']['batches_clipped']==0 for g in native['gradients'])
for g in native['gradients']:
    relations={(x['step_source'],x['test_loss']):x for x in g['results']['relations']}
    assert relations['fresh_policy','outside_any_win_mass']['gradient_dot']<0
    assert relations['fresh_value','outside_win_value']['gradient_dot']<0
    assert relations['guard_policy','outside_any_win_mass']['gradient_dot']>0
    assert relations['fresh_policy','fresh_value']['gradient_dot']==0

summary=dict(teacher_count=len(all_teachers), exact_collector_count=len(teachers),
    same_argmax_allowed=sum(x['same_argmax_allowed'] for x in all_teachers),
    q_regret_over_point_one=sum(x['q_regret']>.1 for x in all_teachers),
    counterfactual_expected_q=changes,collector_age=ages,models=panel_stats,
    native_teacher_control=native['native_teacher_control'],
    gradient_checks=len(checks),gradient_max_directional_error=max(abs(x['actual_delta']-x['first_order_delta']) for x in checks),
    caveat='Q values are search estimates, exact policy fits are counterfactual operators, old proof panels are not held-out general-strength evaluations.')
save('summary.json',summary)
save('teacher-counterfactuals.json',list(teachers.values()))
protected=read(REVIEW/'input-hashes.json')
assert all(sha(p)==h for p,h in protected.items())
sources=read(ROOT/'benchmarks/results/gen5-loop-integration-2026-09-12/candidate-source-hashes.json')
assert all(sha(ROOT/p)==h for p,h in sources.items())
assert all(sha(x['path'])==x['sha256'] for x in manifest['positions'])
verification=dict(production_files_unchanged=len(protected),production_sources_unchanged=len(sources),
    bundles_verified=len(manifest['bundles']),positions_verified=len(manifest['positions']),
    native_models=len(manifest['models']),coupled_priors_bit_exact=len(teachers),
    directional_checks=len(checks),new_campaigns=0,new_matches=0,production_parameter_writes=0,
    diagnostic_files={str(p.relative_to(ROOT)):sha(p) for p in [
        OUT/'manifest.json',OUT/'results.json',OUT/'summary.json',OUT/'duplicate-verification.json',
        ROOT/'crates/paisho-train/examples/gen5_v33_flow_probe.rs',
        ROOT/'crates/paisho-train/examples/gen5_v33_flow_probe/gradients.rs',
        ROOT/'tools/audit_gen5_v33_duplicates.py',ROOT/'tools/report_gen5_v33_flow.py']})
save('verification.json',verification)
print(json.dumps({k:v for k,v in summary.items() if k not in ['models']},indent=2))

# Standalone scientific figure; no browser process or interactive benchmark.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
c=native['native_teacher_control']; labels=['Avant','Cible du MCTS','Après apprentissage\nidéal + couplage','Coordonnées\ncorrigées']
v=[100*c[k][1] for k in ['coupled_before','coupled_teacher','deployed_after_exact_raw_fit','deployed_after_corrected_raw_fit']]
axes[0].bar(range(4),v,color=['#64748b','#0d9488','#dc2626','#0d9488'])
axes[0].set_xticks(range(4),labels,fontsize=9);axes[0].set_ylabel('Probabilité de l’action au meilleur Q (%)')
axes[0].set_title('Contrôle : la cible progresse, sa réutilisation régresse',fontsize=11)
for i,y in enumerate(v): axes[0].text(i,y+.02,f'{y:.4f} %',ha='center',fontsize=9)
axes[0].set_ylim(0,1.22)
chosen=[x for x in panel_stats if x['version'] in [967090,971556,976070,1005390,1029160,1050557]]
chosen.sort(key=lambda x:x['version']);versions=[str(x['version']) for x in chosen]
axes[1].plot(versions,[100*x['guard']['certified_or_immediate']/32 for x in chosen],marker='o',label='Contrôle de publication : 32 victoires')
axes[1].plot(versions,[100*x['outside']['certified_or_immediate']/271 for x in chosen],marker='o',label='Hors de ses sources : 271 victoires')
axes[1].set_ylabel('Premier coup vérifié gagnant (%)');axes[1].set_xlabel('Versions conservées (dernière : apprenant non publié)')
axes[1].set_ylim(30,100);axes[1].tick_params(axis='x',labelsize=8);axes[1].legend(fontsize=8,loc='lower right')
axes[1].set_title('Le contrôle ciblé ne mesure pas toute la conservation',fontsize=11)
fig.suptitle('V5 — audit du flux recherche → politique → jeu',fontsize=15)
fig.text(.5,-.025,'À gauche : Q contrôlés, apprentissage exact hypothétique. À droite : positions historiques connues, pas une mesure Elo.',ha='center',fontsize=9)
fig.savefig(OUT/'flow-audit.png',dpi=180,bbox_inches='tight')
fig.savefig(OUT/'flow-audit.svg',bbox_inches='tight')
