#!/usr/bin/env python3
"""Select a fixed cutoff using terminal games / occupied worker seconds only.

Read-only empirical optimization; no matches, training or automatic retuning.
Input is a fixed-game-count collection with frozen weights. Incomplete cohorts
require explicit opt-in and retain missing slots as unknown cost/outcome bounds.
"""
import argparse
import json
import math
from pathlib import Path

from analyze_compact_deadlines import Observation, load_run, estimate, digest


def optimize(rows, maximum=30.0, step=0.5, conservative=False):
    if not math.isfinite(maximum) or not math.isfinite(step) or maximum<=0 or step<=0:
        raise ValueError('positive finite maximum and step required')
    grid=sorted({min(maximum, step*i) for i in range(1, math.ceil(maximum/step)+1)})
    table=[]
    previous=None
    for cutoff in grid:
        point=estimate(rows,cutoff,128)
        rate=point['rates']['terminal_games']['per_occupied_worker_second']
        bounds=point['rates']['terminal_games']['rate_bounds']
        selection_rate=bounds[0] if conservative else rate
        cost=point['occupied_worker_seconds']
        finished=point['known_completed_games']
        marginal=None
        if previous and cost is not None and previous['cost'] is not None and cost>previous['cost']:
            marginal=(finished-previous['finished'])/(cost-previous['cost'])
        table.append(dict(seconds=cutoff,finished=finished,attempts=len(rows),
                          completion_fraction=point['acceptance'],cost=cost,rate=rate,
                          rate_bounds=bounds,selection_rate=selection_rate,
                          marginal_games_per_extra_worker_second=marginal,
                          unknown=point['unknown_beyond_censoring'],
                          rejected_compute_fraction=point['rejected_compute_fraction']))
        previous=table[-1]
    eligible=[r for r in table if r['selection_rate'] is not None and r['selection_rate']>0]
    # A later peak can beat an earlier one: never stop at the first zero slope.
    best=max(eligible,key=lambda r:(r['selection_rate'],-r['seconds'])) if eligible else None
    return dict(best=best,table=table,objective='terminal_games_per_occupied_worker_second',
                formula='count(terminal and duration<=t) / sum(min(observed_duration,t))',
                constraints='time search domain only; no completion/retention percentage or sample-volume constraint',
                selection='maximum conservative rate lower bound' if conservative else 'maximum identified empirical rate')


def analyze_run(path, maximum=30.0, allow_incomplete=False):
    plan=json.loads((path/'plan.json').read_text())
    summary_path=path/'summary.json'
    summary=json.loads(summary_path.read_text()) if summary_path.exists() else None
    options=plan['options']
    if plan.get('schema')!='paisho-compact-selfplay-plan-v1' or options.get('learn',True):
        raise ValueError('a frozen selfplay collection is required')
    groups,provenance=load_run(path)
    rows=[row for group in groups.values() for row in group]
    if any(row.disposition=='error' for row in rows):
        raise ValueError('failed games cannot be silently excluded')
    missing=[]
    if allow_incomplete and summary is None:
        ids=[json.loads(Path(row.identity).read_text())['game_id'] for row in rows]
        if len(set(ids))!=len(ids) or not set(ids)<=set(range(options['games'])):
            raise ValueError('invalid game IDs in incomplete cohort')
        missing=sorted(set(range(options['games']))-set(ids))
        # No saved elapsed time or outcome exists for these slots. Bound their
        # possible cost by [0,t] and finishes by [0,1]; do not drop them.
        rows.extend(Observation(f'missing-game-{i}',0.0,'time-censored',0,0,0,
                                options['decision_limit'],'missing-record',None)
                    for i in missing)
    elif (summary is None or summary['progress']['errors'] or len(rows)!=options['games']
            or summary['progress']['consumed_games']!=options['games']):
        raise ValueError('complete fixed-game-count cohort required; do not drop in-flight or failed games')
    weights={json.loads(key)['weights_sha256'] for key in groups}
    if len(weights)!=1:
        raise ValueError('one frozen model required')
    result=optimize(rows,maximum,conservative=bool(missing))
    result.update(budget=options['simulations'],workers=options['workers'],
                  model_weights_sha256=next(iter(weights)),provenance=provenance,
                  missing_game_ids=missing,
                  observed_run_wall_seconds=summary['elapsed_seconds'] if summary else None,
                  observed_run_terminal_games=sum(r.disposition=='terminal' for r in rows),
                  selected_cutoff_status='fixed empirical setting; no automatic recalibration',
                  limitations='Small-sample ideal trace truncation, on a 0.5s grid. Soft deadline behavior and timing noise can change future games; this is not a universal optimum or measured whole-machine throughput.')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-seconds',type=float,default=30)
    parser.add_argument('--allow-incomplete',action='store_true',
                        help='explicit conservative bounds for missing records in an interrupted frozen cohort')
    args=parser.parse_args()
    if args.output.exists(): parser.error('choose a new output')
    results=sorted([analyze_run(p,args.max_seconds,args.allow_incomplete) for p in args.runs],key=lambda r:r['budget'])
    report=dict(schema='compact-terminal-game-rate-selection-v1',runs=results,
                tool_sha256=digest(Path(__file__).read_bytes()),automatic_recalibration=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream: json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps([dict(budget=r['budget'],best=r['best']) for r in results]))


if __name__=='__main__': main()
