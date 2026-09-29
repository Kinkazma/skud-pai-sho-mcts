#!/usr/bin/env python3
"""Prepare a frozen V8 continuation and a separate controller config; never resume it.

The active controller must be paused with no training PID. Only preparing files
is performed: installing the returned controller config remains a separate action.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import urllib.request

import paisho_teacher_bootstrap as b
import paisho_teacher_program as program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source-program', 'block', 'profile', 'new-program', 'bin-directory',
                 'control-config', 'output-control', 'evidence']:
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    source = a.source_program.resolve(strict=True)
    a.block.resolve(strict=True).relative_to(source)
    old_control = b.read_json(a.control_config)
    with urllib.request.urlopen(f"http://{old_control['host']}:{old_control['port']}/api/status", timeout=10) as response:
        control_state = json.load(response)
    if control_state['desired'] != 'paused' or control_state['observed'] != 'paused' or control_state.get('pid'):
        raise ValueError('preserve a settled user pause before preparing the continuation')
    if not program.teacher_disabled(source):
        raise ValueError('this migration requires the already-disabled teacher')
    old_plan = b.read_json(source / 'teacher-program-plan.json')
    block = b.read_json(a.block)
    latest = max(int(p.stem.split('-')[1]) for p in source.glob('**/blocks/block-*.json'))
    if block['generation'] != latest:
        raise ValueError('requested block is not the newest durable block')
    meta = b.checkpoint(block['checkpoint'])
    with Path(block['checkpoint']).open('rb') as stream:
        stream.seek(-32, 2)
        content_digest = stream.read(32).hex()
    if (meta['progress']['generation'] != block['generation'] or meta['trainingStep'] != block['training_step']
            or content_digest != block['checkpoint_sha256']):
        raise ValueError('durable block and checkpoint disagree')
    b.checkpoint(block['champion'])
    for record in old_plan['binaries'].values():
        b.verify(record)
    overrides = b.read_json(a.profile)
    if any(str(overrides.get(k)) != v for k, v in {
            'actor-games':'256', 'external-games':'16', 'start-horizon':'128',
            'actor-decision-limit':'2048', 'actor-max-attempts':'1024'}.items()):
        raise ValueError('profile does not match the measured V8 collection contract')
    if set(overrides) - {'actor-games','external-games','start-horizon','actor-decision-limit',
            'actor-max-attempts','classes','actor-workers','actor-in-flight','wait-us'}:
        raise ValueError('this migration only changes measured collection options')
    root, binaries = a.new_program.resolve(), a.bin_directory.resolve()
    if root.exists() or binaries.exists() or a.output_control.exists():
        raise FileExistsError('use new output paths; previous plans are immutable')
    binaries.mkdir(parents=True)
    frozen = {}
    for name, old in old_plan['binaries'].items():
        original = b.ROOT / 'target/release/paisho-live' if name == 'live' else Path(old['path'])
        destination = binaries / Path(old['path']).name
        shutil.copy2(original, destination)
        frozen[name] = b.identity(destination)
    for name in ['paisho_teacher_bootstrap.py', 'paisho_teacher_program.py']:
        script = (b.ROOT / 'tools' / name).read_text()
        script = script.replace('ROOT = Path(__file__).resolve().parents[1]', f'ROOT = Path({str(b.ROOT)!r})')
        (binaries / name).write_text(script)
    plan = copy.deepcopy(old_plan)
    plan['authority'] = b.identity(b.ROOT / 'docs/authority/CURRICULUM_V8.md')
    plan['binaries'] = frozen
    plan['live_options'].update(overrides)
    # These options are also overridden by live_segment; remove stale paths from the plan.
    plan['live_options'].update({'service': frozen['service']['path'],
        'evaluate-executable': frozen['evaluate']['path'], 'promote-executable': frozen['promote']['path']})
    state = plan['initial_state']
    state.update({'learner': b.identity(block['checkpoint']), 'champion': b.identity(block['champion']),
        'generation': block['generation'], 'training_step': block['training_step'], 'tier': block['tier'],
        'teacher_steps': b.read_json(source / 'program-status.json')['teacher_steps_completed'],
        'sources': [b.source_identity(Path(block['checkpoint']).parent / 'snapshot.psrsnap')]})
    plan['migration'] = {'source_plan': b.identity(source / 'teacher-program-plan.json'),
        'durable_block': b.identity(a.block), 'benchmark': b.identity(a.evidence)}
    root.mkdir(parents=True)
    b.publish(root / 'teacher-program-plan.json', plan)
    b.publish(root / 'teacher-disabled.json', {'disabled': True, 'authority': 'CURRICULUM_V8'})
    panel = b.read_json(source / 'initial-goal/complete.json')
    if panel['candidate'] == state['champion']:
        for evidence in panel['evidence']:
            b.verify(evidence)
        (root / 'initial-goal').mkdir()
        b.publish(root / 'initial-goal/complete.json', panel)
    program.write_status(root, plan, state, phase='planned')
    config = copy.deepcopy(old_control)
    config.update({'name': 'Pai Sho — auto-jeu V8',
        'command': [old_control['command'][0], str(binaries / 'paisho_teacher_program.py'), 'run', '--campaign', str(root)],
        'progress_directory': str(root), 'state_path': str(root / 'control-state.json'), 'log_path': str(root / 'control.log')})
    b.publish(a.output_control, config)
    print(json.dumps({'program': str(root), 'controller_config': str(a.output_control.resolve()),
        'generation': state['generation'], 'training_step': state['training_step'],
        'installed': False, 'resumed': False}))


if __name__ == '__main__':
    main()
