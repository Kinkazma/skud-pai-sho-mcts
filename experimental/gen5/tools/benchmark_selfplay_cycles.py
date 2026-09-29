#!/usr/bin/env python3
"""Compare complete frozen-start live cycles, preserving the user's training pause.

Invoke via benchmark_with_training_paused.py. Variants are a JSON object mapping
names to live option overrides. Every arm starts from the same durable block.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from paisho_teacher_bootstrap import identity, publish, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--program-plan', required=True, type=Path)
    parser.add_argument('--block', required=True, type=Path)
    parser.add_argument('--variants', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cycles', type=int, default=1)
    parser.add_argument('arms', nargs='+')
    a = parser.parse_args()
    if a.cycles < 1:
        parser.error('cycles must be positive')
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    plan, state, variants = map(read_json, [a.program_plan, a.block, a.variants])
    bindir = out / 'bin'
    bindir.mkdir()
    for source, name in [(Path('target/release/paisho-live'), 'live'),
                         (Path(plan['binaries']['service']['path']), 'service')]:
        shutil.copy2(source, bindir / name)
    options = {**plan['live_options'], 'initial-checkpoint': state['checkpoint'],
        'initial-generation': str(state['generation']), 'initial-champion-checkpoint': state['champion'],
        'target-generation': str(state['generation'] + a.cycles), 'opponent': state['tier'],
        'evaluation-every': '1000000', 'promotion-every': '1000000', 'service': str(bindir / 'service')}
    commands = []
    for index, arm in enumerate(a.arms):
        campaign = out / f'{index:02d}-{arm}'
        opts = {**options, **variants[arm], 'campaign-dir': str(campaign),
                'curriculum-dir': str(campaign / 'curriculum')}
        command = [str(bindir / 'live'), *(x for k, v in opts.items() for x in ['--' + k, str(v)])]
        commands.append({'arm': arm, 'campaign': str(campaign), 'command': command})
    publish(out / 'plan.json', {'commands': commands, 'cycles': a.cycles,
        'identities': [identity(p) for p in [bindir / 'live', bindir / 'service',
            Path(state['checkpoint']), Path(state['champion']), a.program_plan, a.block, a.variants]],
        'scope': 'complete ordinary cycles including final durable checkpoint; scheduled assessments excluded'})
    def interrupt(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    rows = []
    for item in commands:
        campaign = Path(item['campaign'])
        start = time.monotonic()
        print('starting=' + campaign.name, flush=True)
        with (out / (campaign.name + '.log')).open('x') as log:
            proc = subprocess.Popen(item['command'], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
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
        generations = []
        for path in sorted(campaign.glob('attempts/*/generation-*/timing.json')):
            generations.append({'generation': int(path.parent.name.split('-')[1]),
                'timing': read_json(path), 'collection': read_json(path.parent / 'collection.json'),
                'learning': read_json(path.parent / 'learning.json'),
                'shard': identity(path.parent / 'shard.psrbuf')})
        row = {'arm': item['arm'], 'exit_code': code, 'wall_seconds': time.monotonic()-start,
               'generations': generations}
        publish(out / (campaign.name + '-result.json'), row)
        rows.append(row)
        print(json.dumps({'arm': item['arm'], 'exit_code': code, 'cycles': [
            {'generation': g['generation'], 'seconds': g['timing']['cycle_seconds'],
             'fresh_decisions': g['collection'].get('retained_neural_decisions')} for g in generations]}), flush=True)
        if code:
            raise RuntimeError('failed cycle: ' + str(campaign))
    publish(out / 'results.json', rows)


if __name__ == '__main__':
    main()
