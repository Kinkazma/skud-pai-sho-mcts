#!/usr/bin/env python3
"""Frozen real-game capacity sweep. Run through benchmark_with_training_paused.py."""
import argparse
import json
from pathlib import Path
import subprocess
from paisho_teacher_bootstrap import identity, publish, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--games', type=int, default=128)
    parser.add_argument('--modes', default='baseline,wait0,wide8,wide16,prefetch,baseline')
    args = parser.parse_args()
    plan = read_json(args.plan)
    modes = {
        'baseline': ('64:8,128:4,1024:4',20,5000,False),
        'wait0': ('64:8,128:4,1024:4',20,0,False),
        'wide8': ('64:8,128:8,1024:8',40,1000,False),
        'wide16': ('64:16,128:16,1024:16',80,1000,False),
        'pipe8': ('1024:8',80,1000,False),
        'pipe16': ('1024:16',160,1000,False),
        'pipe4x3': ('1024:4',80,1000,False),
        'pipe8x3': ('1024:8',120,1000,False),
        'pipe4': ('1024:4',40,1000,False),
        'pipe16x3': ('1024:16',160,1000,False),
        'single8': ('1024:8',40,1000,False),
        'single16': ('1024:16',80,1000,False),
        'single32': ('1024:32',160,1000,False),
        'wide32': ('64:8,128:4,1024:32',160,1000,False),
        'wide64': ('64:8,128:4,1024:64',256,1000,False),
        'small2': ('64:2,128:1,1024:16',80,1000,False),
        'prefetch': ('64:8,128:4,1024:4',20,5000,True),
    }
    executable = Path('target/release/examples/mixed_collection_bench').resolve()
    base = [str(executable),plan['binaries']['service']['path'],plan['initial_state']['learner']['path']]
    commands = [(mode,base+[str(v).lower() for v in (*modes[mode][:3],args.games,modes[mode][3])]) for mode in args.modes.split(',')]
    commands = [(mode,command+[str(3 if mode.endswith('x3') else 2 if mode.startswith('pipe') else 1)]) for mode,command in commands]
    args.output.mkdir(parents=True,exist_ok=False)
    publish(args.output/'plan.json',{'commands':commands,'identities':[identity(p) for p in base],
        'scope':'fixed checkpoint, seeds, terminal mixed games; exploratory cold-service sweep'})
    results=[]
    for index,(mode,command) in enumerate(commands):
        result=subprocess.run(command,text=True,capture_output=True)
        (args.output/f'{index}-{mode}.stdout').write_text(result.stdout)
        (args.output/f'{index}-{mode}.stderr').write_text(result.stderr)
        result.check_returncode()
        row={'mode':mode,**json.loads(result.stdout)}
        results.append(row)
        print(json.dumps(row),flush=True)
    publish(args.output/'results.json',results)

if __name__=='__main__': main()
