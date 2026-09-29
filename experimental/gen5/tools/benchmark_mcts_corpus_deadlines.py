#!/usr/bin/env python3
"""CPU corpus pilot with hard per-process deadlines. Use the training pause wrapper."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--attempts', type=int, default=40)
    parser.add_argument('--workers', type=int, default=10)
    args = parser.parse_args()
    if args.attempts < 1 or args.workers < 1:
        parser.error('counts must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    binary = Path('target/release/examples/start_source_bench').resolve()
    frozen = (args.output / 'start_source_bench').resolve()
    shutil.copy2(binary, frozen)
    shutil.copy2('crates/paisho-train/examples/start_source_bench.rs', args.output / 'source.rs')
    shutil.copy2(__file__, args.output / 'runner.py')
    plan = {'attempts_per_mode': args.attempts, 'workers': args.workers,
            'threads_per_game': 1, 'deadlines_seconds': {'mcts8': 20, 'mcts32': 50},
            'first_seed_ordinal': 10000, 'binary_sha256': hashlib.sha256(frozen.read_bytes()).hexdigest(),
            'scope': 'pilot to estimate 10000 accepted terminal games per mode; no production training change'}
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2))
    summaries = []
    for mode, deadline in plan['deadlines_seconds'].items():
        directory = args.output / mode
        directory.mkdir()
        started = time.monotonic()
        def job(ordinal):
            destination = (directory / f'attempt-{ordinal:05}').resolve()
            command = [str(frozen), mode, '1', '1', str(destination), str(ordinal)]
            begin = time.monotonic()
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=deadline)
                elapsed = time.monotonic() - begin
                (destination / 'stderr.txt').write_text(result.stderr)
                if result.returncode:
                    status, decisions = 'error', None
                else:
                    report = json.loads((destination / 'result.json').read_text())
                    status = 'accepted' if report['valid_starts'] == 1 else 'unfinished'
                    decisions = report['decisions']
            except subprocess.TimeoutExpired:
                # subprocess.run kills and reaps the Rust child before returning.
                elapsed = time.monotonic() - begin
                status, decisions = 'timeout', None
            row = {'ordinal': ordinal, 'status': status, 'elapsed_seconds': elapsed, 'decisions': decisions}
            (directory / f'attempt-{ordinal:05}.json').write_text(json.dumps(row))
            return row
        rows = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(job, 10000+i) for i in range(args.attempts)]
            for future in as_completed(futures):
                rows.append(future.result())
                if len(rows) % 10 == 0:
                    print(f'{mode} completed={len(rows)}/{args.attempts} accepted={sum(r["status"] == "accepted" for r in rows)} wall={time.monotonic()-started:.2f}', flush=True)
        wall = time.monotonic() - started
        accepted = sum(r['status'] == 'accepted' for r in rows)
        total = sum(r['elapsed_seconds'] for r in rows)
        summary = {'mode': mode, 'attempts': len(rows), 'accepted': accepted,
                   'timeouts': sum(r['status'] == 'timeout' for r in rows),
                   'unfinished': sum(r['status'] == 'unfinished' for r in rows),
                   'errors': sum(r['status'] == 'error' for r in rows),
                   'wall_seconds': wall, 'occupied_worker_seconds': total,
                   'estimated_seconds_per_10000_accepted': total / args.workers / accepted * 10000 if accepted else None,
                   'finite_batch_seconds_per_10000_accepted': wall / accepted * 10000 if accepted else None,
                   'rows': sorted(rows, key=lambda r: r['ordinal'])}
        (directory / 'summary.json').write_text(json.dumps(summary, indent=2))
        summaries.append(summary)
        print(json.dumps({k:v for k,v in summary.items() if k != 'rows'}), flush=True)
    (args.output / 'results.json').write_text(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
