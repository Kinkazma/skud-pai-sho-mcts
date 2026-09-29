#!/usr/bin/env python3
"""Execute a finite, sequential list of Gen5 experiments and archive receipts."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(manifest):
    data = json.loads(manifest.read_text())
    binary = Path(data['binary']).resolve()
    expected = data['binary_sha256']
    if digest(binary) != expected:
        raise ValueError('frozen executable changed')
    receipts = []
    for job in data['jobs']:
        config = Path(job['config']).resolve()
        options = json.loads(config.read_text())
        output = Path(options['output'])
        if output.exists():
            raise ValueError('refusing to repeat an existing experiment: ' + str(output))
        log = config.with_suffix('.log')
        started = time.monotonic()
        with log.open('xb') as stream:
            code = subprocess.call([str(binary), job['command'], str(config)], stdout=stream, stderr=subprocess.STDOUT)
        receipt = {'config': str(config), 'config_sha256': digest(config),
                   'command': job['command'], 'returncode': code,
                   'wall_seconds': time.monotonic() - started, 'output': str(output)}
        report = output / 'report.json'
        if report.exists():
            receipt['report_sha256'] = digest(report)
            result = json.loads(report.read_text())
            receipt['summary'] = {k: v for k, v in result.items() if k not in ('games',)}
        receipts.append(receipt)
        print(json.dumps(receipt), flush=True)
        manifest.with_suffix('.receipts.json').write_text(json.dumps({
            'binary_sha256': expected, 'updated_at': dt.datetime.now(dt.timezone.utc).isoformat(),
            'experiments': receipts}, indent=2) + '\n')
        if code:
            raise RuntimeError(f'experiment failed ({code}), see {log}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    a = p.parse_args()
    run(a.manifest)


if __name__ == '__main__':
    main()
