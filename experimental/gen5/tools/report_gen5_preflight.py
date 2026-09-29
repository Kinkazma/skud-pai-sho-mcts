#!/usr/bin/env python3
"""Audit Gen5 receipts and draw frozen cutoff curves without generating games."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import statistics as st
import subprocess


def read(path):
    return json.loads(path.read_text())


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def terminal(row):
    return row['termination'] == 'rules-terminal' and row.get('error') is None


def rows(run):
    return [read(path) for path in sorted((run/'games').glob('game-*.json')) if '.targets.' not in path.name]


def summary(run, selected=None):
    result = read(run/'report.json')
    games = rows(run) if selected is None else selected
    times = [r['seconds'] for r in games if terminal(r)]
    return {'run': str(run), 'attempts': len(games), 'terminations': dict(Counter(r['termination'] for r in games)),
            'terminal': len(times), 'mean_terminal_seconds': st.mean(times) if times else None,
            'median_terminal_seconds': st.median(times) if times else None,
            'p95_terminal_seconds': sorted(times)[min(len(times)-1,int(.95*len(times)))] if times else None,
            'wall_seconds': result['elapsed_seconds'],
            'terminal_per_wall_second': len(times)/result['elapsed_seconds'],
            'terminal_per_worker_second': len(times)/sum(r['seconds'] for r in games),
            'eligible_terminal_positions': sum(r['eligible_examples'] for r in games if terminal(r)),
            'fresh_terminal_used': sum(r['fresh_used'] for r in games if terminal(r)),
            'archive_seconds': result['archive_seconds'], 'learner_seconds': result['learner_seconds'],
            'publish_seconds': result['publish_seconds'],
            'search_seconds': sum(sum(r['search_seconds']) for r in games),
            'maintenance_seconds': sum(r['maintenance_seconds'] for r in games),
            'pool_wait_seconds': sum(r['pool_wait_seconds'] for r in games)}


def parity(left, right):
    """Compare frozen trajectories and every numeric label across runtimes."""
    a={r['id']:r for r in rows(left)}; b={r['id']:r for r in rows(right)}
    if not a or a.keys()!=b.keys():
        raise ValueError('different or empty game cohorts')
    examples=0
    for key,x in a.items():
        y=b[key]
        if x['psr_sha256']!=y['psr_sha256'] or x['termination']!=y['termination']:
            raise ValueError(f'trajectory differs for game {key}')
        targets=[]
        for run,row in [(left,x),(right,y)]:
            raw=(run/'games'/row['targets_file']).read_bytes()
            if digest(raw)!=row['targets_sha256']:
                raise ValueError('target hash mismatch')
            data=json.loads(gzip.decompress(raw))
            for ex in data:
                ex.pop('source_run',None)
            targets.append(data)
        if targets[0]!=targets[1]:
            raise ValueError(f'training targets differ for game {key}')
        examples+=len(targets[0])
    return {'games':len(a),'examples':examples,'exact_trajectories_and_targets':True,
            'ignored_field':'source_run only'}


def curve(games, cutoff):
    cohort = [r for r in games if not r['campaign_censored'] and r.get('error') is None]
    # All games truncated by their own wall limit remain in the denominator.
    # Extrapolating past their observation limit would fabricate information.
    if any(r['termination']=='wall-limit' and r['cap_seconds'] is not None and r['cap_seconds']+1e-9 < cutoff for r in cohort):
        raise ValueError('cutoff beyond a censored observation')
    retained = [r for r in cohort if terminal(r) and r['seconds'] <= cutoff]
    occupied = sum(min(r['seconds'], cutoff) for r in cohort)
    accepted_time = sum(r['seconds'] for r in retained)
    positions = sum(r['eligible_examples'] for r in retained)
    return {'cutoff_seconds': cutoff, 'attempts': len(cohort), 'terminal_games': len(retained),
            'terminal_positions': positions, 'occupied_worker_seconds': occupied,
            'terminal_games_per_worker_second': len(retained)/occupied if occupied else 0,
            'terminal_positions_per_worker_second': positions/occupied if occupied else 0,
            'nonterminal_compute_fraction': 1-accepted_time/occupied if occupied else 0}


def cutoff_report(cohorts, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True,exist_ok=True)
    results = {}
    for name, spec in cohorts.items():
        run=Path(spec['run'] if isinstance(spec,dict) else spec); plan=read(run/'plan.json')
        if plan['options']['learn']:
            raise ValueError('a changing model cannot calibrate a frozen-model curve')
        games=rows(run)
        if isinstance(spec,dict):
            games=[g for g in games if all(g.get(k)==v for k,v in spec.items() if k!='run')]
        if not games:raise ValueError(f'empty cohort: {name}')
        finite_caps=[r['cap_seconds'] for r in games if r['cap_seconds'] is not None]
        cap=min(finite_caps) if finite_caps else max(r['seconds'] for r in games)
        grid=sorted(set([i/4 for i in range(1,int(min(20,cap)*4)+1)] + [i for i in range(20,int(cap)+1)] + [cap]))
        points=[curve(games,c) for c in grid]
        best=max(points,key=lambda x:x['terminal_games_per_worker_second'])
        best_positions=max(points,key=lambda x:x['terminal_positions_per_worker_second'])
        results[name]={'summary':summary(run,games),'selection':spec,'parent':plan['parent'],'rules':plan['rules'],
                       'options':plan['options'],'curves':points,'best_games':best,
                       'best_positions':best_positions,'campaign_censored':sum(r['campaign_censored'] for r in games)}
    for metric, label, filename in [
        ('terminal_games_per_worker_second','Parties terminées / seconde de travailleur','debit-parties'),
        ('terminal_positions_per_worker_second','Positions terminales / seconde de travailleur','debit-positions'),
        ('terminal_games','Nombre de parties terminées conservées','parties-conservees')]:
        fig, ax=plt.subplots(figsize=(11,6))
        for name,result in results.items():
            points=result['curves']; ax.plot([p['cutoff_seconds'] for p in points],[p[metric] for p in points],label=name)
            best=max(points,key=lambda p:p[metric]);ax.scatter(best['cutoff_seconds'],best[metric],s=25)
        ax.set_xlabel('Coupure choisie (secondes)');ax.set_ylabel(label)
        ax.set_title('Gen5 — effets de la coupure, modèles et profils figés')
        ax.grid(alpha=.2);ax.legend(ncol=3);ax.set_xlim(left=0)
        fig.text(.5,.02,'Courbes contrefactuelles : hors redémarrage, apprentissage et écritures. Confirmation réelle séparée.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.05,1,1))
        for ext in ['png','svg']:fig.savefig(output/f'{filename}.{ext}',dpi=150)
        plt.close(fig)
    (output/'cutoffs.json').write_text(json.dumps({'schema':'gen5-cutoff-curves-v1','cohorts':results,'objective':'terminal games / occupied worker-second; actual wall and learned-position rates checked separately','automatic_retuning':False},indent=2)+'\n')
    print(json.dumps({name:r['best_games'] for name,r in results.items()},indent=2))


def audit(root, verifier, run_names=None):
    cache={};facts=[];target_count=0
    paths=sorted(p for directory in ([root/name for name in run_names] if run_names else [root]) for p in directory.rglob('*.psr'))
    unique={}
    for path in paths:unique.setdefault(digest(path.read_bytes()),path)
    native=subprocess.run([str(verifier),'--batch'],input=''.join(str(p)+'\n' for p in unique.values()),capture_output=True,text=True,check=True)
    verified=[json.loads(line) for line in native.stdout.splitlines()]
    if len(verified)!=len(unique):raise ValueError('incomplete native batch verification')
    cache=dict(zip(unique,verified))
    for path in paths:
        raw=path.read_bytes();sha=digest(raw)
        meta=path.with_suffix('.json')
        if not meta.exists():continue
        r=read(meta);v=cache[sha]
        if r['psr_sha256']!=sha or r['decisions']!=v['decisions']:raise ValueError(f'receipt mismatch {path}')
        outcome={'host':'Win(Host)','guest':'Win(Guest)','draw':'Draw','ongoing':'Ongoing'}[v['outcome']]
        if 'outcome' in r and r['outcome']!=outcome:raise ValueError(f'outcome mismatch {path}')
        if 'score' in r and r.get('error') is None:
            expected=None if v['outcome']=='ongoing' else .5 if v['outcome']=='draw' else float((v['outcome']=='host')==(r['candidate_seat']=='H'))
            if r['score']!=expected:raise ValueError(f'score mismatch {path}')
        if 'targets_file' in r:
            raw_targets=(path.parent/r['targets_file']).read_bytes()
            if digest(raw_targets)!=r['targets_sha256']:raise ValueError(f'target hash mismatch {path}')
            examples=json.loads(gzip.decompress(raw_targets))
            if len(examples)!=r['eligible_examples']:raise ValueError('target count mismatch')
            if not terminal(r) and r['termination']!='repetition-training-loss' and examples:raise ValueError('fabricated terminal target')
            for ex in examples:
                if ex['rules']!=r['rules'] or not 0<ex['decision']<=r['decisions']:raise ValueError('target provenance mismatch')
                if not -1 <= ex['value'] <= 1:raise ValueError('target value out of range')
                if ex['policy'] and abs(sum(ex['policy'])-1)>1e-8:raise ValueError('policy not normalized')
                if len(ex['actions'])!=len(ex['policy']) or len(ex['action_features'])!=len(ex['policy']):raise ValueError('action alignment mismatch')
                if r['termination']=='repetition-training-loss' and (ex['policy'] or ex['value']!=-1):raise ValueError('bad repetition penalty')
            target_count+=len(examples)
        facts.append({'path':str(path.relative_to(root)),'sha256':sha,**v})
    report={'records':len(facts),'unique_psrs':len(cache),'targets':target_count,'facts':facts}
    (root/('replay-verification-final.json' if run_names else 'replay-verification.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='facts'}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['summary','cutoffs','audit','parity'])
    p.add_argument('path',type=Path)
    p.add_argument('--other',type=Path)
    p.add_argument('--runs',nargs='+',help='Audit only these archived subdirectories, without replaying already verified experiments')
    p.add_argument('--output',type=Path)
    p.add_argument('--verifier',type=Path,default=Path('target/release/examples/verify_record'))
    a=p.parse_args()
    if a.mode=='summary':print(json.dumps(summary(a.path),indent=2))
    elif a.mode=='cutoffs':cutoff_report(read(a.path),a.output or a.path.parent/'analysis')
    elif a.mode=='parity':
        if a.other is None:p.error('parity requires --other')
        print(json.dumps(parity(a.path,a.other),indent=2))
    else:audit(a.path,a.verifier.resolve(),a.runs)


if __name__=='__main__':main()
