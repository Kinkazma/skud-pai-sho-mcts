#!/usr/bin/env python3
"""Restore the published durable Gen5 state into a separate writable experiment.

The installed archive is never opened for writing. Native archive writes replace
files atomically, so hard links share immutable bytes until a revision changes.
Run from any directory; all recorded paths are relative to the repository root.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = Path('assets/gen5-history')
STRING = re.compile(rb'"(?:[^"\\]|\\.)*"')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def rewrite_strings(data, mapping):
    """Change path strings only: no round trip of weights or counters."""
    def replace(match):
        value = json.loads(match[0])
        for old, new in mapping:
            if value == old or value.startswith(old + '/'):
                return json.dumps(new + value[len(old):], ensure_ascii=False).encode()
        return match[0]
    return STRING.sub(replace, data)


def link_archive(source, destination):
    destination.mkdir(parents=True)
    count = 0
    for parent, dirs, files in os.walk(source):
        relative = Path(parent).relative_to(source)
        for name in dirs:
            if (Path(parent)/name).is_symlink():
                raise ValueError('Archive directory must not be a symlink')
            (destination/relative/name).mkdir()
        for name in files:
            path = Path(parent)/name
            if path.is_symlink():
                raise ValueError('Archive member must not be a symlink')
            os.link(path, destination/relative/name)
            count += 1
            if count % 100000 == 0:
                print(json.dumps({'linked_archive_files': count}), flush=True)
    return count


def prepare(root, output, mode, seconds):
    root = root.resolve()
    output = output.resolve()
    if not output.is_relative_to(root/'runs') or output == root/'runs':
        raise ValueError('Select a new directory below this repository\'s runs/')
    if output.exists():
        raise ValueError('Output already exists; it will not be overwritten')
    if mode not in {'last-campaign', 'prepared'} or not 0 < seconds <= 7*86400:
        raise ValueError('Select an explicit algorithm and positive bounded duration')
    installed = root/BASE
    original = json.loads((installed/'state'/f'config.{mode}.json').read_text())
    original_progress = json.loads((installed/'state/resume-progress.json').read_text())
    if original_progress['replay_positions'] != 327680:
        raise ValueError('Unexpected historical FIFO; install all historical parts')
    output.mkdir(parents=True)
    work = output.relative_to(root)
    archives = work/'archives'
    # Keep archive names/order, including the empty gen5-loop-v2 directory.
    current = Path(original['case_curriculum']['archive'])
    count = link_archive(root/current, root/archives/current.name)
    for archive in original['recall_archive_sources']:
        source = root/archive
        source.mkdir(parents=True, exist_ok=True)  # Empty archive has no TAR member.
        dest = root/archives/source.name
        dest.symlink_to(os.path.relpath(source, dest.parent), target_is_directory=True)
    mapping = [(str(BASE/'archives'), str(archives))]
    local = output/'initial-state'
    local.mkdir()
    raw_progress = (installed/'state/resume-progress.json').read_bytes()
    replacements = list(mapping)
    for lane in ['ordinary', 'proofs']:
        path = installed/'state'/f'coverage-{lane}.json'
        data = rewrite_strings(path.read_bytes(), mapping)
        target = local/path.name
        target.write_bytes(data)
        old = original_progress['durable_recall']['coverage'][lane]['cursor']
        replacements += [(old['manifest'], str(target.relative_to(root))),
                         (old['sha256'], sha(data))]
    # Numeric model files remain untouched; native identity stays stable.
    progress_path = local/'resume-progress.json'
    progress_path.write_bytes(rewrite_strings(raw_progress, replacements))
    progress = json.loads(progress_path.read_bytes())
    config = json.loads(rewrite_strings(json.dumps(original).encode(), mapping))
    config.update(output=str(work/'run'), seconds=seconds, end_unix_seconds=None,
                  resume_progress=str(progress_path.relative_to(root)))
    config_path = output/'config.json'
    config_path.write_text(json.dumps(config, indent=2)+'\n')
    receipt = {'schema': 'paisho-gen5-historical-continuation-v1', 'algorithm': mode,
               'source_progress_sha256': sha((installed/'state/resume-progress.json').read_bytes()),
               'source_learner_sha256': sha((installed/'state/learner.json').read_bytes()),
               'source_actor_sha256': sha((installed/'state/accepted.json').read_bytes()),
               'linked_archive_files': count, 'replay_positions': progress['replay_positions'],
               'updates': progress['updates'], 'new_duration_seconds': seconds,
               'historical_remaining_seconds': original_progress['remaining_seconds'],
               'in_flight_worker_state_restored': False, 'training_started': False}
    (output/'restoration.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return config_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'run', 'verify'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['last-campaign', 'prepared'], default='last-campaign')
    p.add_argument('--seconds', type=float)
    a = p.parse_args()
    output = (ROOT/a.output).resolve()
    if a.command == 'prepare':
        if a.seconds is None:
            p.error('--seconds is required; the original campaign is not restarted automatically')
        print(prepare(ROOT, output, a.mode, a.seconds))
        return
    receipt = json.loads((output/'restoration.json').read_text())
    engine = (ROOT/'experimental/gen5' if receipt['algorithm'] == 'prepared'
              else ROOT/'portable-models/gen5-history-engine')
    config = output/'config.json'
    if a.command == 'verify':
        command = [engine/'target/release/examples/gen5_portable_recovery', 'verify', config, output/'verification']
    else:
        if (output/'run').exists():
            raise ValueError('Run already started; never restart from stale initial state')
        command = [engine/'target/release/paisho-gen5', 'run', config]
    subprocess.run(list(map(str, command)), cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
