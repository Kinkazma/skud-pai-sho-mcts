#!/usr/bin/env python3
"""Matched MCTS-32 CPU deadline experiments; use benchmark_with_training_paused.py."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


def summarize(rows, workers, wall):
    accepted = [r for r in rows if r['status'] == 'accepted']
    occupied = sum(r['elapsed_seconds'] for r in rows)
    rate = workers * len(accepted) / occupied if occupied else 0
    decisions = sorted(r['decisions'] for r in accepted)
    return {'attempts': len(rows), 'accepted': len(accepted),
            'timeouts': sum(r['status'] == 'timeout' for r in rows),
            'unfinished': sum(r['status'] == 'unfinished' for r in rows),
            'acceptance': len(accepted)/len(rows) if rows else 0,
            'occupied_worker_seconds': occupied, 'wall_seconds': wall,
            'saturated_games_per_second': rate,
            'observed_games_per_second': len(accepted)/wall if wall else 0,
            'estimated_minutes_per_10000': 10000/rate/60 if rate else None,
            'unique_psrs': len({r['psr_sha256'] for r in accepted}),
            'mean_decisions': sum(decisions)/len(decisions) if decisions else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--thresholds', type=float, nargs='+', default=[8, 1, 50, 5])
    p.add_argument('--attempts', type=int, default=80)
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--first-id', type=int, default=30000)
    a = p.parse_args()
    if min(a.attempts, a.workers) < 1 or any(not 0 < t < float('inf') for t in a.thresholds):
        p.error('positive finite limits required')
    a.output.mkdir(parents=True, exist_ok=False)
    binary = Path('target/release/paisho-mcts-corpus')
    frozen = (a.output / binary.name).resolve()
    shutil.copy2(binary, frozen)
    shutil.copy2(__file__, a.output/'runner.py')
    plan = {'simulations': 32, 'backend': 'cpu', 'workers': a.workers,
            'attempts_per_arm': a.attempts, 'first_id': a.first_id, 'thresholds_seconds': a.thresholds,
            'binary_sha256': hashlib.sha256(frozen.read_bytes()).hexdigest(),
            'scope': 'matched fresh seeds; hard subprocess deadlines, completed PSRs saved; no training mutation',
            'estimator': 'workers * accepted / sum(attempt wall times); observed wall throughput also reported'}
    (a.output/'plan.json').write_text(json.dumps(plan, indent=2))
    results=[]
    for index, limit in enumerate(a.thresholds):
        directory=a.output/f'arm-{index:02d}-{limit:g}s'
        directory.mkdir()
        def job(identity):
            dest=(directory/f'{identity:08d}').resolve()
            started=time.monotonic()
            try:
                result=subprocess.run([str(frozen),'game','32',str(identity),'cpu',str(dest),'16384'],
                                      stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=limit)
                elapsed=time.monotonic()-started
                if result.returncode:
                    raise RuntimeError(f'worker {identity}: '+result.stderr.decode(errors='replace'))
                report=json.loads((dest/'result.json').read_text())
                status='accepted' if report['terminal'] else 'unfinished'
                if elapsed>limit: status='timeout'
                row={'ordinal':identity,'status':status,'elapsed_seconds':elapsed,
                     'decisions':report['decisions'],'outcome':report['outcome']}
                if status=='accepted':
                    row['psr_sha256']=hashlib.sha256((dest/'game.psr').read_bytes()).hexdigest()
                    row['cuts']=len(report['cuts'])
            except subprocess.TimeoutExpired:
                # subprocess.run kills/reaps this exact worker, including blocked searches.
                row={'ordinal':identity,'status':'timeout','elapsed_seconds':time.monotonic()-started}
            (directory/f'{identity:08d}.json').write_text(json.dumps(row))
            return row
        begin=time.monotonic()
        rows=[]
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures=[pool.submit(job,a.first_id+i) for i in range(a.attempts)]
            for future in as_completed(futures):
                rows.append(future.result())
                if len(rows)%20==0:
                    print(f'limit={limit:g}s attempts={len(rows)}/{a.attempts} accepted={sum(r["status"]=="accepted" for r in rows)}',flush=True)
        summary={'threshold_seconds':limit,**summarize(rows,a.workers,time.monotonic()-begin)}
        (directory/'summary.json').write_text(json.dumps({**summary,'rows':sorted(rows,key=lambda r:r['ordinal'])},indent=2))
        results.append(summary)
        (a.output/'results.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
