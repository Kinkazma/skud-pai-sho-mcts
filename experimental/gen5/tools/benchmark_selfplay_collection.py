#!/usr/bin/env python3
"""Frozen collection comparisons. Invoke through benchmark_with_training_paused.py.

Each case is NAME:GAMES:EXTERNAL:HORIZON:LIMIT:BATCH:WORKERS:IN_FLIGHT:WAIT_US.
No learner updates or production writes. Partial logs survive interruption.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
from paisho_teacher_bootstrap import identity, publish


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--service', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('cases', nargs='+')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    executable = args.output / 'mixed_collection_bench'
    shutil.copy2('target/release/examples/mixed_collection_bench', executable)
    commands = []
    for case in args.cases:
        name, games, external, horizon, limit, batch, workers, flights, wait = case.split(':')
        if not name.replace('-', '').replace('_', '').isalnum():
            parser.error('case names must be simple labels')
        commands.append((name, [str(executable.resolve()), str(args.service.resolve()),
            str(args.checkpoint.resolve()), f'1024:{batch}', workers, wait, games, 'false',
            flights, external, horizon, limit]))
    publish(args.output / 'plan.json', {'cases': commands,
        'identities': [identity(p) for p in [executable, args.service, args.checkpoint]],
        'scope': 'cold-service frozen collection; no strength or full-cycle speed claim'})
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    rows = []
    for index, (name, command) in enumerate(commands):
        path = args.output / f'{index}-{name}'
        with path.with_suffix('.stdout').open('w') as out, path.with_suffix('.stderr').open('w') as err:
            proc = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
            try:
                code = proc.wait()
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
        stdout = path.with_suffix('.stdout').read_text()
        row = {'case': name, 'exit_code': code}
        if stdout.strip():
            row.update(json.loads(stdout))
        rows.append(row)
        print(json.dumps(row), flush=True)
    publish(args.output / 'results.json', rows)
    return int(any(row['exit_code'] for row in rows))


if __name__ == '__main__':
    raise SystemExit(main())
