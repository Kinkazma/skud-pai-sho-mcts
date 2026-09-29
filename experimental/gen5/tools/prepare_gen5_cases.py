#!/usr/bin/env python3
"""Prepare a human-case Gen5 continuation. Never signal or launch training."""
import argparse
import hashlib
import json
from pathlib import Path
from prepare_gen5_resume import prepare


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure(campaign, recovery, output, archive):
    campaign, recovery, output, archive = map(lambda p: Path(p).resolve(), (campaign, recovery, output, archive))
    old = json.loads((campaign / 'config.json').read_text())
    state = json.loads((campaign / 'training/progress.json').read_text())
    receipt = json.loads((recovery / 'receipt.json').read_text())
    resumed = json.loads((recovery / 'resume-progress.json').read_text())
    if any(state[k] != receipt[k] for k in ('version', 'updates', 'completed')):
        raise ValueError('recovery does not match the completed campaign')
    if digest(receipt['model']) != receipt['model_sha256']:
        raise ValueError('recovered model changed')
    if any(resumed[k] != state[k] for k in ('version', 'updates', 'completed', 'replay_positions')):
        raise ValueError('resume state differs from final progress')
    if not (recovery / 'replay.index.json').is_file():
        raise ValueError('durable replay is missing')
    # Preserve the original anchor, reference, human split and RAM replay.
    old.update(model=receipt['model'], resume_progress=str(recovery / 'resume-progress.json'),
               replay_index=str(recovery / 'replay.index.json'), seconds=3600,
               output=str(output.parent / 'training'), games=10000000, learn=True,
               budgets=[[256, .5], [512, .5]], policy_min_budget=256,
               checkpoint_fraction=0., decision_limit=800,
               historical_unlimited_budgets=[32, 64, 128],
               case_curriculum=dict(archive=str(archive), reversals=3, losses=10,
                                    unresolved_attempts=10, reanalysis_positions=4,
                                    reanalysis_budget=512, durable_fraction=.1))
    old.pop('end_unix_seconds', None)
    old.setdefault('evaluation_anchor', str(campaign / 'initial-model.json'))
    if not old.get('human_dataset'):
        raise ValueError('human training dataset required')
    if archive == campaign or campaign in archive.parents:
        raise ValueError('durable archive must live outside disposable campaigns')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(old, f, indent=2); f.write('\n')
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--recovery', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--reuse-verified-recovery', action='store_true')
    a = p.parse_args()
    status = json.loads((a.campaign / 'status.json').read_text())
    if status.get('state') != 'completed' or status.get('returncode') != 0:
        p.error('prepare from a completed campaign; this tool never pauses a process')
    if not a.reuse_verified_recovery:
        prepare(a.campaign, a.recovery)
    print(configure(a.campaign, a.recovery, a.output, a.archive))


if __name__ == '__main__':
    main()
