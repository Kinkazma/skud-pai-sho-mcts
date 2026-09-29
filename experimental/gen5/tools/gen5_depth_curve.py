#!/usr/bin/env python3
"""Complement lower depth floors, reusing exact prior frozen-game controls.

Compare every floor with the unchanged search, and 6/8/12 directly with 5.
Use a newly randomized permutation of the same enumerated population; old
16/32 results remain available, not recomputed or claimed as new observations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import time

from gen5_depth_confirmation import confidence_interval, counts, request, save, strip_times


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired(games, start, budget, floor, reference):
    lookup = {(g['floor'], g['seat']): g for g in games if
              g['start_index'] == start and g['budget'] == budget}
    return sum(int(lookup[floor, seat]['outcome'] == 'win') -
               int(lookup[reference, seat]['outcome'] == 'win') for seat in ('H','G')) / 2


def classify(lo, hi, n, population, minimum):
    if n == population:
        return 'positive_census' if lo > 0 else 'negative_census' if hi < 0 else 'equal_census'
    if n < minimum:
        return None
    if hi-lo <= .05:
        if lo > 0:
            return 'positive_precise'
        if hi < 0:
            return 'negative_precise'
    if lo > -.02 and hi < .02:
        return 'bounded_within_two_percentage_points'
    return None


def validate_population(current, previous):
    if current != previous:
        raise RuntimeError('population/prefix/seed changed; cached controls cannot be reused')


def copy_cached(old_native, native, game):
    source = old_native/'games'/game['id']
    if digest(source/'game.psr') != game['psr_sha256']:
        raise RuntimeError('cached PSR changed')
    if json.loads((source/'result.json').read_text()) != game:
        raise RuntimeError('cached result changed')
    destination = native/'games'/game['id']
    destination.mkdir()
    for name in ('game.psr','result.json','decisions.jsonl'):
        shutil.copy2(source/name, destination/name)
    save(destination/'reused.json', {'source':str(source),'psr_sha256':game['psr_sha256'],
                                   'timing_origin':'previous eight-worker trial, not newly measured'})
    return game


def parity(process, population, native, old_native, floors, out):
    index = next(p['index'] for p in population['starts'] if p['origin'].get('diagnostic_only'))
    jobs = [dict(id=f'serial-{b}-{f}-{int(s)}',start=index,budget=b,floor=f,guest=s)
            for b in (32,256) for f in (0,*floors) for s in (False,True)]
    a = request(process, {'serial':True,'jobs':jobs})
    b = request(process, {'jobs':[dict(j,id=j['id'].replace('serial','parallel')) for j in jobs]})
    for x,y in zip(a['games'],b['games']):
        assert strip_times(x)==strip_times(y), 'serial/parallel game changed'
        left=[json.loads(s) for s in (native/'games'/x['id']/'decisions.jsonl').read_text().splitlines()]
        right=[json.loads(s) for s in (native/'games'/y['id']/'decisions.jsonl').read_text().splitlines()]
        assert strip_times(left)==strip_times(right), 'serial/parallel decision/counter changed'
        if x['floor']==0:
            old=json.loads((old_native/'games'/x['id']/'result.json').read_text())
            assert strip_times(x)==strip_times(old), 'previous baseline changed'
    save(out/'parallel-parity.json', {'exact':True,'games_each':len(jobs),'old_baselines_exact':4,
        'serial_seconds':a['batch_seconds'],'parallel_seconds':b['batch_seconds']})


def run(binary, plan_path, out):
    out.mkdir()
    plan=json.loads(plan_path.read_text())
    old_run=Path(plan['reuse_run']);old_native=old_run/'native'
    old_plan=json.loads((old_native/'plan.json').read_text())
    for field in ('model_sha256','config_sha256','opponent','maximum_new_decisions','starts'):
        if plan[field]!=old_plan[field]:
            raise RuntimeError(f'cached control protocol differs: {field}')
    if digest(old_run/'games.jsonl')!=plan['reuse_journal_sha256']:
        raise RuntimeError('cached journal changed')
    cache={g['id']:g for g in map(json.loads,(old_run/'games.jsonl').read_text().splitlines()) if g['floor']==0}
    errors=(out/'native.log').open('w');native=out/'native'
    process=subprocess.Popen([str(binary),str(plan_path),str(native)],stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,stderr=errors,text=True,bufsize=1)
    save(out/'process.json',{'pid':process.pid,'binary':str(binary),'binary_sha256':digest(binary)})
    ready=json.loads(process.stdout.readline());assert ready['ready']
    population=json.loads((native/'population.json').read_text())
    validate_population(population,json.loads((old_native/'population.json').read_text()))
    order=[p['index'] for p in population['starts'] if not p['origin'].get('diagnostic_only')]
    random.Random(plan['permutation_seed']).shuffle(order);save(out/'order.json',order);save(out/'ready.json',ready)
    floors=plan['floors'];budgets=plan['budgets']
    comparisons={(b,f,0) for b in budgets for f in floors}|{(b,f,5) for b in budgets for f in floors if f!=5}
    alpha_tail=.05/(2*len(comparisons))
    save(out/'family.json',{'comparisons':sorted(comparisons),'alpha_tail':alpha_tail,
        'population':len(order),'rule':'95% simultaneous for this 28-comparison family; old 16/32 family separate'})
    print(f'Ready {len(order)} starts, {len(comparisons)} contrasts, {len(cache)} cached baseline games available',flush=True)
    parity(process,population,native,old_native,floors,out)
    print('Twenty serial/parallel games exact, including four prior baselines',flush=True)
    active=set(comparisons);values={k:[] for k in active};bounds={k:[-1.,1.] for k in active};decisions={}
    games=[];offset=0;reused=0;fresh=0;started=time.monotonic()
    while active and offset<len(order):
        if (out/'STOP').exists():break
        chunk=order[offset:offset+plan['batch_starts']];jobs=[];batch=[]
        for index in chunk:
            for budget in budgets:
                needed=sorted({f for b,a,z in active if b==budget for f in (a,z)})
                for guest in (False,True):
                    rotation=(index+int(guest))%max(1,len(needed))
                    for floor in needed[rotation:]+needed[:rotation]:
                        identifier=f'main-{index}-{budget}-{floor}-{int(guest)}'
                        if identifier in cache:
                            batch.append(copy_cached(old_native,native,cache[identifier]));reused+=1
                        else:jobs.append(dict(id=identifier,start=index,budget=budget,floor=floor,guest=guest))
        response=request(process,{'jobs':jobs});batch.extend(response['games']);fresh+=len(jobs);games.extend(batch)
        with (out/'games.jsonl').open('a') as journal:
            for game in batch:journal.write(json.dumps(game)+'\n')
        for key in sorted(active):
            b,f,ref=key;values[key].extend(paired(batch,index,b,f,ref) for index in chunk)
        offset+=len(chunk);analysis={}
        for key in sorted(comparisons):
            xs=values[key];b,f,ref=key
            if key in active and (offset%plan['analysis_every']==0 or offset==len(order)):
                lo,hi=confidence_interval(xs,len(order),alpha_tail)
                bounds[key]=[max(bounds[key][0],lo),min(bounds[key][1],hi)]
                decision=classify(*bounds[key],len(xs),len(order),plan['minimum_starts'])
                if decision:decisions[key]=decision;active.remove(key)
            ids=set(order[:len(xs)])
            selected=[g for g in games if g['budget']==b and g['start_index'] in ids]
            analysis[f'{b}/{f}-vs-{ref}']={'starts':len(xs),'gain':sum(xs)/len(xs),
                'simultaneous_interval':bounds[key],'decision':decisions.get(key,'continue'),
                'reference':counts([g for g in selected if g['floor']==ref]),
                'floor':counts([g for g in selected if g['floor']==f])}
        save(out/'analysis.json',{'complete':not active,'population':len(order),'visited_starts':offset,
            'games':len(games),'new_games':fresh,'reused_baselines':reused,'elapsed_seconds':time.monotonic()-started,
            'comparisons':analysis})
        print(f'{offset} starts, {fresh} new games, {reused} exact reused controls, {len(decisions)}/{len(comparisons)} decided',flush=True)
    process.stdin.write('{"finish":true}\n');process.stdin.flush();final=json.loads(process.stdout.readline())
    code=process.wait();assert code==0 and final['complete']
    save(out/'completion.json',{'native':final,'all_comparisons_decided':not active,'new_games':fresh,
        'reused_baselines':reused,'games':len(games),'remaining':sorted(active)})
    errors.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('binary',type=Path);p.add_argument('plan',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();run(a.binary.resolve(),a.plan.resolve(),a.out.resolve())
