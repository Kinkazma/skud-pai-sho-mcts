#!/usr/bin/env python3
"""Summarize a frozen campaign and native checkpoint panel; no production writes."""
import collections as C
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as md

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'benchmarks/results/gen5-v33-campaign-review-2026-09-12'
RUN=ROOT/'training-runs/micro-gen5-ui-20260912-131332-99610d'
def read(p):return json.loads(Path(p).read_text())
def save(n,j):(OUT/n).write_text(json.dumps(j,indent=2,ensure_ascii=False)+'\n')
s=read(OUT/'summary.json');manifest=read(OUT/'panel-manifest.json');panel=read(OUT/'panel-results.json')
rows=[json.loads(line) for line in (OUT/'receipts.jsonl').open()]
lookup={(r['model'],r['position']):r for r in panel['results']}
guard=read(RUN/'publication-guard.json')['rows'];sources=set();prefixes=set()
for row in guard:
 h=hashlib.sha256(row['prefix'].encode()).hexdigest();prefixes.add(h)
 proof=read(ROOT/f'data/learning/gen5-durable-v1/proofs/{h}.json');sources.add(proof['human_source'])
control=[i for i,p in enumerate(panel['positions'][:390]) if p['source'] not in sources and p['prefix_sha256'] not in prefixes]
sets={'historical':list(range(390)),'guard':list(range(390,447)),'outside_guard_sources':control}
scores=[]
for mi,m in enumerate(manifest['models']):
 d={'version':m['version'],'identity':panel['identities'][mi]['identity'],'coupled':m['coupled'],'path':m['path']}
 for name,ids in sets.items():
  wins=[i for i in ids if panel['positions'][i]['target']==1]
  class_ids={z:[i for i in ids if panel['positions'][i]['target']==z] for z in [-1,0,1]}
  errors={z:sum((lookup[mi,i]['value']-z)**2 for i in ix)/len(ix) for z,ix in class_ids.items() if ix}
  d[name]=dict(positions=len(ids),wins=len(wins),sources=len(set(panel['positions'][i]['source'] for i in ids)),
   raw=sum(lookup[mi,i]['certified_selected'] for i in wins),pure_raw=sum(lookup[mi,i]['pure_certified_selected'] for i in wins),
   coupled=sum(lookup[mi,i]['coupled_certified_selected'] for i in wins) if m['coupled'] else None,
   mass=sum(lookup[mi,i]['certified_mass'] for i in wins)/len(wins),
   mse=errors,class_counts={z:len(ix) for z,ix in class_ids.items()},balanced_mse=sum(errors.values())/len(errors),
   raw_gained=sum(not lookup[0,i]['certified_selected'] and lookup[mi,i]['certified_selected'] for i in wins),
   raw_lost=sum(lookup[0,i]['certified_selected'] and not lookup[mi,i]['certified_selected'] for i in wins),
   coupled_gained=sum(not lookup[0,i]['coupled_certified_selected'] and lookup[mi,i]['coupled_certified_selected'] for i in wins) if m['coupled'] else None,
   coupled_lost=sum(lookup[0,i]['coupled_certified_selected'] and not lookup[mi,i]['coupled_certified_selected'] for i in wins) if m['coupled'] else None)
 scores.append(d)
initial=np.array(read(RUN/'initial-model.json')['parameters'])
for d in scores:
 p=np.array(read(d['path'])['parameters']);delta=p-initial
 d['parameter_drift']=dict(changed=int((p!=initial).sum()),rms=float(np.sqrt(np.mean(delta*delta))),max=float(np.abs(delta).max()))
save('panel-summary.json',scores)
accepted_index=next(i for i,m in enumerate(manifest['models']) if m['version']==1029160)
final_index=next(i for i,m in enumerate(manifest['models']) if m['version']==1050557)
lost_guard=[i for i in sets['guard'] if (lookup[accepted_index,i]['certified_selected'] and not lookup[final_index,i]['certified_selected']) or
 (lookup[accepted_index,i]['coupled_certified_selected'] and not lookup[final_index,i]['coupled_certified_selected'])]
fresh=[r for r in rows if not r['reanalysis']];references=[r for r in fresh if r['reference_budget'] is not None]
presentations=sum(s['sums'][k] for k in ['fresh_used','replay_used','human_used'])
analysis=dict(terminal_per_second=s['terminal']/s['wall_seconds'],receipt_per_second=s['receipts']/s['wall_seconds'],
 terminal_per_second_after_first_receipt=s['terminal']/(s['wall_seconds']-s['first_receipt_seconds']),
 new_positions_per_second=s['sums']['fresh_used']/s['wall_seconds'],updates_per_second=s['sums']['learned_batches']/s['wall_seconds'],
 learned_presentations=presentations,recall_presentations=s['sums']['durable_draws'],
 recall_fraction=s['sums']['durable_draws']/presentations,
 reference_fresh_games=len(references),reference_new_positions=sum(r['fresh_used'] for r in references),
 all_game_new_positions=sum(r['fresh_used'] for r in fresh),
 reference_game_positions_fraction=sum(r['fresh_used'] for r in references)/sum(r['fresh_used'] for r in fresh),
 reference_all_new_positions_fraction=sum(r['fresh_used'] for r in references)/s['sums']['fresh_used'],
 guard_overlap_positions=sum(p['prefix_sha256'] in prefixes for p in panel['positions'][:390]),
 outside_guard_sources_positions=len(control),outside_guard_sources_wins=scores[0]['outside_guard_sources']['wins'],
 final_vs_accepted_guard_lost=lost_guard,
 final_vs_accepted_guard_gained=[i for i in sets['guard'] if not lookup[accepted_index,i]['certified_selected'] and lookup[final_index,i]['certified_selected']],
 fully_learned=sum(r['fully_learned'] for r in rows),not_fully_learned=sum(not r['fully_learned'] for r in rows),
 fresh_not_consumed=s['sums']['eligible_examples']-s['sums']['fresh_used'],
 accepted_idle_seconds=read(RUN/'status.json')['end_unix_seconds']-s['publication_transitions'][-1]['time'],
 learner_updates_after_last_publication=4818997-s['publication_transitions'][-1]['updates'])
analysis['openings']={}
for g in ['3.1','3.2','3.3','3.4','3.5','selfplay']:
 rs=[r for r in fresh if r['prefix_decisions']==0 and r['group']==g]
 analysis['openings'][g]=dict(games=len(rs),results=dict(C.Counter(r['result'] for r in rs)),
  budgets=dict(C.Counter(r['reference_budget'] for r in rs)),initial_states=len(set(r['case'] for r in rs)))
# Time-varying costs describe the recorded concurrent workload; sums are not a serial wall-time decomposition.
analysis['cost_bins']=[]
for b in s['bins']:
 rs=[r for r in rows if b['start']<=r['relative']<b['end']]
 f=[r for r in rs if not r['reanalysis']]
 analysis['cost_bins'].append(dict(local=b['local'],fresh=len(f),new_decisions=sum(r['continuation_decisions'] for r in f),
  evaluations=b['sums']['inference_evaluations'],search_seconds=b['search_seconds'],pool_wait=b['sums']['pool_wait_seconds'],
  maintenance=b['sums']['maintenance_seconds'],seconds=b['sums']['seconds']))
save('analysis.json',analysis)
# Reconciliations and frozen-source assertions.
assert presentations==read(RUN/'training/progress.json')['recall_quotas']['consumed_examples']-read(RUN/'resume-progress.json')['recall_quotas']['consumed_examples']
assert s['sums']['durable_draws']==read(RUN/'training/progress.json')['recall_quotas']['consumed_recall']-read(RUN/'resume-progress.json')['recall_quotas']['consumed_recall']
for m,d in zip(manifest['models'],scores):
 if 'identity' in m:assert m['identity']==d['identity']
assert len(lost_guard)>0
assert scores[final_index]['guard']['balanced_mse']>scores[accepted_index]['guard']['balanced_mse']
replays=[json.loads(line) for line in (OUT/'verified-replays.jsonl').open()]
assert len(replays)==len((OUT/'replay-paths.txt').read_text().splitlines())
verification=dict(receipts_reconciled=len(rows),new_updates_reconciled=s['sums']['learned_batches'],
 bilateral_terminal_counts_verified=s['terminal'],native_replayed_games=len(replays),cycles_verified=sum(r['cycle_verified'] for r in replays),
 compact=read(OUT/'compact-verification.json'),dense=read(OUT/'dense-verification.json'),
 models=len(scores),positions=447,model_positions=len(panel['results']),certificates_verified=441,
 shared_bank=panel['shared_bank'],no_new_matches=True,no_training=True,
 source_hash_changes=[p for p,h in read(ROOT/'benchmarks/results/gen5-loop-integration-2026-09-12/candidate-source-hashes.json').items()
  if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h],
 production_hash_changes=[p for p,h in read(OUT/'input-hashes.json').items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h])
verification['position_hash_changes']=[p['path'] for p in manifest['positions'] if 'sha256' in p and hashlib.sha256(Path(p['path']).read_bytes()).hexdigest()!=p['sha256']]
assert not verification['source_hash_changes'] and not verification['production_hash_changes']
assert not verification['position_hash_changes']
save('verification.json',verification)

tz=ZoneInfo('Europe/Paris');date=lambda t:datetime.fromtimestamp(t,tz)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
 'axes.titleweight':'bold','axes.labelcolor':'#24334a','text.color':'#172b45'})
fig,axs=plt.subplots(2,2,figsize=(14,9.5));fig.patch.set_facecolor('#f9fafc')
ax=axs[0,0]
sample=rows[::200]+[rows[-1]]
ax.plot([date(r['time']) for r in sample],[r['model_version'] for r in sample],color='#97a2b3',label='Modèle entraîné',lw=2)
trans=s['publication_transitions'];end=read(RUN/'status.json')['end_unix_seconds']
ax.step([date(r['time']) for r in trans]+[date(end)],[r['version'] for r in trans]+[trans[-1]['version']],where='post',color='#157c75',lw=2.2,label='Modèle accepté pour jouer')
ax.set_title('Les poids continuent d’évoluer ; les joueurs avancent par étapes',fontsize=11)
ax.set_ylabel('Version des poids');ax.ticklabel_format(style='plain',axis='y');ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M',tz=tz));ax.legend(fontsize=9)

ax=axs[0,1];colors=['#146c94','#7256a8','#df7539','#248475','#b65068']
for g,color in zip(['3.1','3.2','3.3','3.4','3.5'],colors):
 ls=[l for l in s['lots'] if l['group']==g]
 for budget in [8,32]:
  bs=[l for l in ls if l['budget']==budget];xs=[l['lot']+(int(g[-1])-3)*.035 for l in bs];ys=[100*l['results'].get('W',0)/l['games'] for l in bs]
  ax.plot(xs,ys,color=color,lw=1.4,ls='--' if budget==8 else '-',marker='o',mfc='white' if budget==8 else color,label=f'Gen{g}' if budget==32 else None)
ax.axhline(60,color='#8290a2',lw=1,ls=':');ax.set_ylim(20,90);ax.set_xlim(.8,4.25)
ax.set_xticks([1,2,3,4],['967090','976070','1005390','1029160\n(lots incomplets)'])
ax.set_ylabel('Victoires / parties du lot (%)');ax.set_xlabel('Checkpoint figé de la V5, à MCTS 512')
ax.set_title('Résultats figés : adversaires à MCTS 8 ○ puis 32 ●',fontsize=11);ax.legend(ncol=3,fontsize=8)

ax=axs[1,0];available=[d for d in scores if d['coupled'] and d['version']!=1050557]
when={t['version']:date(t['time']) for t in trans}
for name,label,color in [('guard','32 victoires protégées','#157c75'),('outside_guard_sources','271 victoires hors sources du contrôle','#b65b46')]:
 ax.plot([when[d['version']] for d in available],[100*d[name]['raw']/d[name]['wins'] for d in available],marker='o',color=color,lw=2,label=label)
 ax.scatter([date(end)],[100*scores[final_index][name]['raw']/scores[final_index][name]['wins']],marker='x',color=color,s=70)
ax.set_ylim(30,100);ax.set_ylabel('Choix de politique certifiés gagnants (%)');ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M',tz=tz))
ax.set_title('Politique : fort gain ciblé, recul hors du contrôle',fontsize=11)
ax.legend(fontsize=8);ax.text(.03,.03,'Traits : modèles acceptés conservés · × : dernier modèle entraîné\nPanels connus de la campagne, sans garantie de force générale',transform=ax.transAxes,fontsize=8)

ax=axs[1,1];xs=[date(datetime.fromisoformat(b['local']).timestamp()+b['width']/2) for b in s['bins']]
ax.bar(xs,[b['terminal']/b['width'] for b in s['bins']],width=np.array([b['width'] for b in s['bins']])/86400*.85,color='#4d79a7',alpha=.85)
ax.axhline(analysis['terminal_per_second'],color='#26354a',ls='--',label=f"Moyenne : {analysis['terminal_per_second']:.2f} parties/s")
ax.set_ylim(0,4.2);ax.set_ylabel('Parties terminées / seconde');ax.set_title('Débit réel par tranches de dix minutes',fontsize=11)
ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M',tz=tz));ax.legend(fontsize=9)
ax.text(.03,.94,'Réanalyses exclues · première tranche avec chargement',transform=ax.transAxes,va='top',fontsize=8)
fig.suptitle('Gen5 V33 — bilan du 12 septembre 2026, 13:13–15:00',fontsize=17,fontweight='bold',y=.99)
fig.tight_layout(rect=(0,0,1,.965),h_pad=2,w_pad=2)
fig.savefig(OUT/'bilan.png',dpi=180);fig.savefig(OUT/'bilan.svg');plt.close(fig)

checkpoints=read(OUT/'checkpoints.json');time_by_version={r['model_version']:r['time'] for r in rows}
with (OUT/'checkpoints.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=['version','identity','updates','time_of_version','retention_observed_seconds','guard_raw','historical_raw','outside_guard_sources_raw','kind','path','sha256']);writer.writeheader()
 byidentity={d['identity']:d for d in scores}
 for c in checkpoints:
  d=byidentity[c['identity']]
  writer.writerow(dict(version=c['version'],identity=c['identity'],updates=c['updates'],
   time_of_version=date(time_by_version.get(c['version'],trans[0]['time'])).isoformat(),retention_observed_seconds=c['elapsed_millis']/1000,
   guard_raw=d['guard']['raw'],historical_raw=d['historical']['raw'],outside_guard_sources_raw=d['outside_guard_sources']['raw'],
   kind=c['provenance']['kind'],path=c['path'],sha256=c['sha256']))
print(json.dumps(analysis,indent=2));print(json.dumps(verification,indent=2))
