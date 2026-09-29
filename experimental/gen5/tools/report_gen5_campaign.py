#!/usr/bin/env python3
"""Summarize frozen diagnostic outputs; no campaign/model modifications."""
import argparse
import collections
import datetime as dt
import hashlib
import json
import struct
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as md
import matplotlib.pyplot as plt
import numpy as np

TZ = ZoneInfo('Europe/Paris')
def when(t): return dt.datetime.fromtimestamp(t, TZ)
def mean(v): return float(np.mean(v)) if len(v) else None
def clustered(values, sources):
    groups = collections.defaultdict(list)
    for v, src in zip(values, sources): groups[src].append(v)
    sums = np.array([sum(v) for v in groups.values()])
    ns = np.array([len(v) for v in groups.values()])
    ix = np.random.default_rng(921).integers(len(ns), size=(4000, len(ns)))
    boot = sums[ix].sum(axis=1) / ns[ix].sum(axis=1)
    return dict(mean=mean(values), sources=len(ns), interval=np.quantile(boot, [.025,.975]).tolist())

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('directory', type=Path); out=ap.parse_args().directory
    load=lambda f: json.loads((out/f).read_text())
    s=load('summary.json'); manifest=load('fixed-panel-manifest.json'); native=load('fixed-panel-results.json')
    def digest(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            for block in iter(lambda:f.read(1<<20),b''): h.update(block)
        return h.hexdigest()
    campaign_hashes=load('input-hashes.json')
    for path, expected in campaign_hashes.items(): assert digest(path)==expected
    for entry in manifest['models']+manifest['positions']: assert digest(entry['path'])==entry['sha256']
    for entry in manifest['positions']:
        if entry['kind']=='certificate':
            proof=json.loads(Path(entry['path']).read_text())
            assert hashlib.sha256(proof['prefix'].encode()).hexdigest()==Path(entry['path']).stem
            assert proof['rules']=='skud-pai-sho-gen5-v1'
    final_path=Path(s['phases'][2]['run'])/'training/model.json'
    final=json.loads(final_path.read_text()); last=json.loads(Path(manifest['models'][-1]['path']).read_text())
    assert {k:v for k,v in final.items() if k!='provenance'}=={k:v for k,v in last.items() if k!='provenance'}
    assert all(struct.pack('d',a)==struct.pack('d',b) for a,b in zip(final['parameters'],last['parameters']))
    (out/'verification.json').write_text(json.dumps(dict(unchanged_campaign_files=len(campaign_hashes),
        panel_input_hashes=len(manifest['models'])+len(manifest['positions']),proof_filename_bindings=384,
        final_parameter_bits_equal=True,final_model_sha256=digest(final_path),new_training_games=0,
        new_training_updates=0,final_updates=final['updates']),indent=2)+'\n')
    specs=manifest['models']; pos=native['positions']; rows=collections.defaultdict(dict)
    for r in native['results']: rows[r['model']][r['position']]=r
    assert len(native['results'])==len(specs)*len(pos)
    assert all(v['updates']==m['updates'] for v,m in zip(native['identities'],specs))
    groups={g:[i for i,p in enumerate(manifest['positions']) if p['group']==g] for g in sorted({p['group'] for p in manifest['positions']})}
    groups['all_certificates']=[i for i,p in enumerate(manifest['positions']) if p['kind']=='certificate']
    curves=[]
    for mi, model in enumerate(specs):
        for group, ids in groups.items():
            r=rows[mi]; known=[i for i in ids if pos[i]['target'] is not None]
            wins=[i for i in ids if pos[i]['certified_actions']]; immediate=[i for i in ids if pos[i]['immediate_actions']]
            by_outcome={str(z):dict(n=len(j),value=mean([r[i]['value'] for i in j]),mse=mean([(r[i]['value']-z)**2 for i in j]),wrong_sign=sum(r[i]['value']*z<0 for i in j)) for z in [-1,0,1] for j in [[i for i in known if pos[i]['target']==z]]}
            curves.append(dict(model=mi,version=model['version'],time=model['time'],group=group,n=len(ids),
                mse=mean([(r[i]['value']-pos[i]['target'])**2 for i in known]),by_outcome=by_outcome,
                certified_n=len(wins),certified_correct=sum(r[i]['certified_selected'] for i in wins),
                certified_mass=mean([r[i]['certified_mass'] for i in wins]),
                immediate_n=len(immediate),immediate_correct=sum(r[i]['immediate_selected'] for i in immediate),
                source_count=len({pos[i]['source'] for i in ids})))
    initial_ensemble=next(i for i,m in enumerate(specs) if m['phase']==2 and m['version']==s['phases'][2]['version'][0])
    comparisons={}
    for a,b,label in [(0,len(specs)-1,'whole_evening'),(initial_ensemble,len(specs)-1,'ensemble')]:
        groups_out={}
        for group,ids in groups.items():
            known=[i for i in ids if pos[i]['target'] is not None]; wins=[i for i in ids if pos[i]['certified_actions']]; imm=[i for i in ids if pos[i]['immediate_actions']]
            def delta(indices, metric):
                return clustered([metric(rows[b][i],i)-metric(rows[a][i],i) for i in indices],[pos[i]['source'] for i in indices]) if indices else None
            def transition(indices,field):
                return dict(retained=sum(rows[a][i][field] and rows[b][i][field] for i in indices),
                    gained=sum(not rows[a][i][field] and rows[b][i][field] for i in indices),
                    lost=sum(rows[a][i][field] and not rows[b][i][field] for i in indices),
                    still_missed=sum(not rows[a][i][field] and not rows[b][i][field] for i in indices))
            groups_out[group]=dict(value_mse_delta=delta(known,lambda r,i:(r['value']-pos[i]['target'])**2),
                certified_mass_delta=delta(wins,lambda r,i:r['certified_mass']),
                certified_choice_delta=delta(wins,lambda r,i:float(r['certified_selected'])),
                immediate_choice_delta=delta(imm,lambda r,i:float(r['immediate_selected'])),
                immediate_transitions=transition(imm,'immediate_selected'),certified_transitions=transition(wins,'certified_selected'))
        comparisons[label]=dict(initial=a,final=b,groups=groups_out)
    receipts=[json.loads(line) for ph in range(3) for line in (out/f'receipts-phase-{ph}.jsonl').open()]
    fresh=[r for r in receipts if not r['reanalysis']]
    phase_stats=[]
    for ph in s['phases']:
        q=[r for r in receipts if r['phase']==ph['phase']]; games=[r for r in q if not r['reanalysis']]
        first=[r for r in games if r['lane']=='Historical' and r['attempt']==0]
        phase_stats.append(dict(phase=ph['phase'],fresh_examples=sum(r['fresh'] for r in q),
            updates=ph['updates'][1]-ph['updates'][0],terminals=ph['terminals'],
            terminal_per_second=ph['terminals']/(ph['end']-ph['start']),
            terminal_per_second_after_first_receipt=ph['terminals']/(max(r['time'] for r in q)-ph['first_receipt']),
            first_reference_results=dict(collections.Counter(r['result'] for r in first)),
            reference_rotations=dict(collections.Counter(r['rotate'] for r in games if r['lane']=='Historical' and r['rotate'])),
            unknowns=dict(collections.Counter(r['termination'] for r in games if r['result']=='U'))))
    # Replay result/hash bindings were checked independently, not inferred from counters.
    selected=load('receipt-verification/selected.json'); replays=[json.loads(l) for l in (out/'receipt-verification/core-replay.jsonl').open()]
    assert len(selected)==len(replays)
    for a,b in zip(selected,replays):
        expected='U' if b['outcome']=='Ongoing' else 'D' if b['outcome']=='Draw' else 'W' if b['outcome']==f"Win({a['seat']})" else 'L'
        assert expected==a['result']
    checkpoints=[m for m in specs if m['phase']==2 and 'models/model-' in m['path']]
    gap=max(zip(checkpoints,checkpoints[1:]),key=lambda pair:pair[1]['time']-pair[0]['time'])
    # Streaming replay of the actual no-score retention rule. It loses the middle.
    entries=[]
    for ordinal in range(252):
        entries.append(ordinal)
        if len(entries)<=64: continue
        chosen=list(range(len(entries)-32,len(entries)))+[0]
        while len(chosen)<64:
            chosen.append(max((i for i in range(len(entries)) if i not in chosen),key=lambda i:(min(abs(entries[i]-entries[j]) for j in chosen),-i)))
        entries=[e for i,e in enumerate(entries) if i in chosen]
    actual=json.loads((Path(s['phases'][2]['run'])/'training/checkpoint-retention.json').read_text())
    assert entries==[m['ordinal'] for m in actual['models']]
    final_phase=[r for r in curves if r['group']=='all_certificates' and specs[r['model']]['phase']==2 and 'models/model-' in specs[r['model']]['path']]
    windows={}
    for label, cohort in [('first_32_checkpoints',final_phase[:32]),('last_32_checkpoints',final_phase[-32:])]:
        windows[label]=dict(start=cohort[0]['time'],end=cohort[-1]['time'],n=len(cohort),
            median_mse=float(np.median([r['mse'] for r in cohort])),
            median_win_mse=float(np.median([r['by_outcome']['1']['mse'] for r in cohort])),
            median_loss_mse=float(np.median([r['by_outcome']['-1']['mse'] for r in cohort])),
            median_certified_mass=float(np.median([r['certified_mass'] for r in cohort])),
            median_certified_choices=float(np.median([r['certified_correct'] for r in cohort])),
            median_immediate_choices=float(np.median([r['immediate_correct'] for r in cohort])))
    # Resume copies with equal model identities must have exactly equal diagnostics.
    duplicates=[]
    for mi in range(len(specs)):
        prior=next((j for j in range(mi) if native['identities'][j]['identity']==native['identities'][mi]['identity']),None)
        if prior is not None:
            for i in range(len(pos)):
                assert {k:v for k,v in rows[mi][i].items() if k!='model'}=={k:v for k,v in rows[prior][i].items() if k!='model'}
            duplicates.append([prior,mi])
    (out/'panel-summary.json').write_text(json.dumps(dict(curves=curves,comparisons=comparisons,phase_stats=phase_stats,
        windows=windows,
        verification=dict(replayed=len(replays),dense=load('receipt-verification/verified-targets.json'),certificates=384,identical_resume_pairs=duplicates),
        checkpoint_gap=dict(start=gap[0]['time'],end=gap[1]['time'],seconds=gap[1]['time']-gap[0]['time'],streaming_rule_reproduced=True)),indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.grid':True,'grid.alpha':.2})
    c=[r for r in curves if r['group']=='all_certificates']; times=[when(r['time']) for r in c]
    fig,axes=plt.subplots(3,1,figsize=(13,10),sharex=True,gridspec_kw={'height_ratios':[2,2,1]})
    def segments(ax, ys, label, color):
        # Never draw invented trajectories across gaps in surviving checkpoints.
        start=0; first=True
        for j in range(1,len(times)+1):
            if j==len(times) or (times[j]-times[j-1]).total_seconds()>180:
                ax.plot(times[start:j],ys[start:j],'.-',lw=1,ms=3,color=color,label=label if first else None)
                first=False; start=j
    segments(axes[0],[r['by_outcome']['1']['mse'] for r in c],'302 positions gagnantes','#23789c')
    segments(axes[0],[r['by_outcome']['-1']['mse'] for r in c],'79 positions perdantes','#d35841')
    axes[0].set_ylabel('Erreur quadratique de valeur\nPlus bas = mieux');axes[0].legend(loc='upper right')
    segments(axes[1],[100*r['certified_mass'] for r in c],'Probabilité du coup certifié (302 positions)','#7b4da3')
    segments(axes[1],[100*r['immediate_correct']/r['immediate_n'] for r in c],'Victoire immédiate choisie en premier (208 positions)','#247d53')
    axes[1].set_ylabel('Pourcentage');axes[1].set_ylim(0,100);axes[1].legend(loc='lower right')
    axes[2].scatter(times,[m['phase'] for m in specs],marker='|',s=140,color='#425267')
    axes[2].set_yticks([0,1,2],['Phase initiale','Optimisée','Mélange 80/20']);axes[2].set_ylabel('Checkpoints\nconsultables')
    for ax in axes:
        ax.axvline(when(s['phases'][2]['start']),color='black',ls='--',lw=1)
        ax.axvspan(when(gap[0]['time']),when(gap[1]['time']),color='#e3aa4f',alpha=.13)
    axes[2].xaxis.set_major_formatter(md.DateFormatter('%H:%M',tz=TZ));axes[2].set_xlabel('11 septembre 2026 — heure de Paris')
    fig.suptitle('V5 : apprentissage réel, mais progrès irrégulier du choix des coups',fontsize=15,y=.985)
    fig.text(.075,.945,'170 checkpoints × 390 positions : 384 issues de l’archive d’apprentissage, 6 témoins diagnostiques. Pas une mesure Elo.',fontsize=10)
    fig.text(.075,.02,'Trait noir : passage au mélange 80/20 et au rappel direct de victoires. Zone beige : 95,7 min sans checkpoint conservé.',fontsize=10)
    fig.tight_layout(rect=[0,.035,1,.935]);fig.savefig(out/'apprentissage.png',dpi=160);fig.savefig(out/'apprentissage.svg');plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(13,8),sharex=True)
    colors=['#167895','#8d55ab','#dd8642','#287f54','#c34251']
    for gen,color in zip([f'Gen3.{i}' for i in range(1,6)],colors):
        buckets=collections.defaultdict(list)
        for r in fresh:
            if r['generation']==gen:buckets[(r['phase'],int(r['time']//1200)*1200)].append(r)
        points=[(when(mean([r['time'] for r in q])),100*sum(r['result']=='W' for r in q)/len(q)) for (ph,t),q in sorted(buckets.items()) if len(q)>=20]
        axes[0].plot([p[0] for p in points],[p[1] for p in points],'o',ms=5,label=gen,color=color)
    axes[0].set_ylabel('Victoires / tentatives (%)');axes[0].set_ylim(0,25);axes[0].legend(ncol=5,loc='upper right')
    for ph,color in [(1,'#167895'),(2,'#287f54')]:
        buckets=collections.defaultdict(list)
        for r in fresh:
            if r['phase']==ph and r['result']!='U':buckets[int(r['time']//600)*600].append(r)
        full=[(t,q) for t,q in sorted(buckets.items()) if t>=s['phases'][ph]['first_receipt'] and t+600<=s['phases'][ph]['end']]
        axes[1].plot([when(t+300) for t,q in full],[len(q)/600 for t,q in full],'.-',color=color,label='Gen3.1 uniquement' if ph==1 else '80 % auto-jeu / 20 % références')
    axes[1].set_ylabel('Parties terminées / seconde');axes[1].set_ylim(0,5);axes[1].legend(loc='upper right')
    for ax in axes:ax.axvline(when(s['phases'][2]['start']),color='black',ls='--',lw=1)
    axes[1].xaxis.set_major_formatter(md.DateFormatter('%H:%M',tz=TZ));axes[1].set_xlabel('11 septembre 2026 — heure de Paris')
    fig.suptitle('V5 : résultats pendant l’entraînement et débit réel',fontsize=15,y=.985)
    fig.text(.075,.94,'Références figées à MCTS 8 ; V5 à 256/512. Positions humaines, sièges adaptatifs et répétitions : pas une évaluation équilibrée.',fontsize=10)
    fig.text(.075,.015,'Haut : tranches de 20 min, ≥20 tentatives, horodatage moyen. Bas : fenêtres complètes de 10 min ; réanalyses exclues.',fontsize=10)
    fig.tight_layout(rect=[0,.035,1,.93]);fig.savefig(out/'resultats.png',dpi=160);fig.savefig(out/'resultats.svg');plt.close(fig)
    print(json.dumps(dict(phase_stats=phase_stats,whole=comparisons['whole_evening']['groups']['all_certificates'],ensemble=comparisons['ensemble']['groups']['all_certificates']),indent=2))

if __name__=='__main__': main()
