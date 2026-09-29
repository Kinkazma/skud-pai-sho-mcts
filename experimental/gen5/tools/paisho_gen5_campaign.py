#!/usr/bin/env python3
"""Freeze and run one Gen5 campaign until an explicit absolute deadline."""
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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def execute(out):
    plan = json.loads((out / 'plan.json').read_text())
    for name, expected in plan['hashes'].items():
        if digest(out / name) != expected:
            raise ValueError('frozen input changed: ' + name)
    if time.time() >= plan['end_unix_seconds']:
        raise ValueError('deadline already passed; no automatic continuation')
    with (out / 'started.json').open('x') as f:
        json.dump({'coordinator_pid': os.getpid(), 'unix_seconds': time.time()}, f)
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    status = {'state': 'starting', 'coordinator_pid': os.getpid(),
              'started_unix_seconds': time.time(), 'end_unix_seconds': plan['end_unix_seconds'],
              'automatic_resume': False}
    write(out / 'status.json', status)
    child = None
    awake = None
    try:
        with (out / 'training.log').open('xb') as log:
            child = subprocess.Popen(plan['command'], stdout=log, stderr=subprocess.STDOUT)
            if sys.platform == 'darwin':
                awake = subprocess.Popen(['caffeinate', '-i', '-w', str(child.pid)])
            status.update(state='running', training_pid=child.pid)
            write(out / 'status.json', status)
            code = child.wait()
        # The dashboard may have extended the deadline while preserving native RAM.
        status['end_unix_seconds']=json.loads((out/'status.json').read_text()).get('end_unix_seconds',status['end_unix_seconds'])
        status.update(state='completed' if code == 0 else 'failed', termination='user-stop' if code == 0 and (out/'training/stop-request.json').exists() else 'deadline-or-limit', returncode=code,
                      ended_unix_seconds=time.time())
        write(out / 'status.json', status)
        return code
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait()
        status.update(state='failed', error=str(exc), ended_unix_seconds=time.time())
        write(out / 'status.json', status)
        raise
    finally:
        if awake is not None and awake.poll() is None:
            awake.terminate()


def start(config, binary, out, end, history_source=None):
    end_time = dt.datetime.fromisoformat(end)
    if end_time.tzinfo is None:
        raise ValueError('deadline needs an explicit UTC offset')
    deadline = end_time.timestamp()
    if not 0 < deadline-time.time() <= 86400:
        raise ValueError('deadline must be within the next twenty-four hours')
    options = json.loads(config.read_text())
    # User correction (2026-09-10): no per-game clock for any Gen3.1
    # collection budget, including resumes inheriting the old [32] exception.
    # The absolute campaign deadline and CPU admission budget still apply.
    options['historical_unlimited_budgets'] = [32, 64, 128]
    if any(not 1 <= b <= 2048 for b, _ in options.get('budgets', [[64, .8], [256, .2]])):
        raise ValueError('training simulations must be at most 2048')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'bin').mkdir()
    shutil.copy2(binary, out / 'bin/paisho-gen5')
    shutil.copy2(Path(__file__), out / 'bin/campaign.py')
    names = ['bin/paisho-gen5', 'bin/campaign.py']
    for i, opponent in enumerate(options.get('opponents', [])):
        name = f'opponent-gen3-{i+1}.json'
        source = Path(opponent['model'])
        if digest(source) != opponent['sha256']:
            raise ValueError('frozen opponent changed: ' + opponent['generation'])
        shutil.copy2(source, out / name)
        opponent['model'] = str(out / name)
        names.append(name)
    for key, name in [('model', 'initial-model.json'), ('reference', 'reference-gen3-1.json'),
                      ('evaluation_anchor', 'evaluation-anchor.json'), ('resume_progress', 'resume-progress.json'),
                      ('human_dataset', 'human-dataset.json'), ('replay_index', 'initial-replay.index.json'),
                      ('publication_guard', 'publication-guard.json'),
                      ('publication_validation', 'publication-validation.json')]:
        if options.get(key):
            shutil.copy2(Path(options[key]), out / name)
            options[key] = str(out / name)
            names.append(name)
    options.update(output=str(out / 'training'), end_unix_seconds=deadline,
                   seconds=86400, games=10000000, learn=True)
    write(out / 'config.json', options)
    names.append('config.json')
    command = [str(out / 'bin/paisho-gen5'), 'run', str(out / 'config.json')]
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    write(out / 'plan.json', {'source_revision': revision, 'end_unix_seconds': deadline,
          'end_local': end_time.isoformat(), 'command': command,
          'hashes': {name: digest(out / name) for name in names},
          'automatic_resume': False, 'analysis_after_end': 'wait for user message',
          'deadline_semantics': 'no new search or learning after deadline; durable archive drain may finish afterward'})
    if history_source is not None and Path(history_source).exists():
        shutil.copytree(history_source, out / 'training/history')
    with (out / 'coordinator.log').open('xb') as log:
        child = subprocess.Popen([sys.executable, str(out / 'bin/campaign.py'), 'run', '--output', str(out)],
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return {'output': str(out), 'coordinator_pid': child.pid, 'end_local': end_time.isoformat()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['start', 'run', 'status'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path)
    p.add_argument('--binary', type=Path)
    p.add_argument('--end')
    a = p.parse_args()
    out = a.output.resolve()
    if a.mode == 'run':
        return execute(out)
    if a.mode == 'status':
        print((out / 'status.json').read_text())
        path = out / 'training/progress.json'
        if path.exists():
            print(path.read_text())
        return 0
    if not a.config or not a.binary or not a.end:
        p.error('start requires --config, --binary and --end')
    print(json.dumps(start(a.config, a.binary, out, a.end)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
