#!/usr/bin/env python3
"""One bounded Gen4 process, frozen inputs, durable status; no recurring automation."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def command(output, seconds, threads, with_replay=False):
    result = [str(output / 'bin/paisho-micro'), 'selfplay',
            '--model', str(output / 'initial-model.json'),
            '--output', str(output / 'training'), '--seconds', str(seconds),
            '--games', '10000000', '--threads', str(threads), '--workers', str(threads),
            '--game-seconds', '30', '--decision-limit', '600',
            '--low-budget', '64', '--high-budget', '256', '--high-fraction', '0.2',
            '--replay-capacity', '4096', '--replay-ratio', '4', '--rate', '0.02',
            '--history-interval', '1800']
    if with_replay:
        result.extend(['--replay', str(output / 'initial-replay.json')])
    return result


def execute(output):
    plan = json.loads((output / 'plan.json').read_text())
    for name, expected in plan['hashes'].items():
        if digest(output / name) != expected:
            raise ValueError('frozen input changed: ' + name)
    # Exclusive launch receipt prevents accidental duplicate execution.
    with (output / 'started.json').open('x') as f:
        json.dump({'started_at': now(), 'coordinator_pid': os.getpid()}, f)
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    start = time.time()
    status = {'state': 'starting', 'started_at': now(), 'coordinator_pid': os.getpid(),
              'estimated_end_utc': dt.datetime.fromtimestamp(start + plan['seconds'], dt.timezone.utc).isoformat(),
              'automatic_resume': False}
    write(output / 'status.json', status)
    try:
        with (output / 'training.log').open('xb') as log:
            child = subprocess.Popen(plan['command'], stdout=log, stderr=subprocess.STDOUT)
            status.update(state='running', training_pid=child.pid)
            write(output / 'status.json', status)
            code = child.wait()
        status.update(state='completed' if code == 0 else 'failed', returncode=code,
                      ended_at=now(), elapsed_seconds=time.time()-start)
        write(output / 'status.json', status)
        return code
    except BaseException as exc:
        if 'child' in locals() and child.poll() is None:
            child.terminate()
            child.wait()
        status.update(state='failed', error=str(exc), ended_at=now())
        write(output / 'status.json', status)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['start', 'run', 'status'])
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--binary', type=Path)
    p.add_argument('--model', type=Path)
    p.add_argument('--replay', type=Path)
    p.add_argument('--seconds', type=int, default=36000)
    p.add_argument('--threads', type=int, default=10)
    a = p.parse_args()
    out = a.output.resolve()
    if a.mode == 'run':
        return execute(out)
    if a.mode == 'status':
        print((out / 'status.json').read_text())
        progress = out / 'training/progress.json'
        if progress.exists():
            print(progress.read_text())
        return 0
    if not 0 < a.seconds <= 43200 or not 0 < a.threads <= (os.cpu_count() or 1):
        p.error('bounded positive seconds and available CPU threads required')
    if not all(x and x.is_file() for x in [a.binary, a.model]) or (a.replay and not a.replay.is_file()):
        p.error('binary and model files required; optional replay must exist and use current rules')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'bin').mkdir()
    inputs = [(a.binary, 'bin/paisho-micro'), (a.model, 'initial-model.json'),
              (Path(__file__), 'bin/campaign.py')]
    if a.replay:
        inputs.append((a.replay, 'initial-replay.json'))
    for source, name in inputs:
        shutil.copy2(source, out / name)
    names = [name for _, name in inputs]
    revision = subprocess.run(['git', '-C', str(Path(__file__).resolve().parents[1]), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    plan = {'created_at': now(), 'source_revision': revision, 'seconds': a.seconds, 'threads': a.threads,
            'command': command(out, a.seconds, a.threads, with_replay=bool(a.replay)),
            'replay_import': 'explicit current-rule targets' if a.replay else 'empty replay; weights-only warm start',
            'hashes': {name: digest(out / name) for name in names},
            'milestones': 'each +100 provisional internal score-Elo vs frozen starting Gen4, by budget; no site conversion',
            'deadline': 'native elapsed training budget; bounded actor/evaluation drain afterward',
            'automatic_resume': False}
    write(out / 'plan.json', plan)
    with (out / 'coordinator.log').open('xb') as log:
        process = subprocess.Popen([sys.executable, str(out / 'bin/campaign.py'), 'run', '--output', str(out)],
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({'output': str(out), 'coordinator_pid': process.pid, 'seconds': a.seconds}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
