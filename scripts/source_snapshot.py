#!/usr/bin/env python3
"""Refresh the manifest of tracked public files, or verify an existing snapshot."""
import argparse,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--refresh',action='store_true');args=p.parse_args();manifest=ROOT/'source-manifest.json'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
if args.refresh:
 names=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
 entries={n:sha(ROOT/n) for n in sorted(names) if n and n!='source-manifest.json'}
 manifest.write_text(json.dumps({'description':'SHA-256 of tracked public files; large data uses data/manifests separately.','files':entries},indent=2)+'\n')
else:
 entries=json.loads(manifest.read_text())['files'];bad=[n for n,h in entries.items() if not (ROOT/n).is_file() or sha(ROOT/n)!=h]
 if bad:raise SystemExit('Changed or missing: '+', '.join(bad))
 print(f'Verified {len(entries)} tracked source files')
