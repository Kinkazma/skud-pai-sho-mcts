#!/usr/bin/env python3
"""Index a completed historical export; private source paths never enter Git."""
import argparse
import gzip
import hashlib
import json
import sqlite3
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build(root, journal, shard_size=10000):
    base = root/'assets/gen5-history'
    state = base/'state'
    summary = json.loads((base/'archive-summary.json').read_text())
    if not (state/'finalization.json').is_file():
        raise ValueError('Native state finalization must pass before indexing')
    out = root/'data/manifests/gen5-history'
    if out.exists():
        raise ValueError('Manifest output exists; inspect it before replacing')
    out.mkdir(parents=True)
    shards, batch = [], {}
    count = total = 0

    def flush():
        nonlocal batch
        if not batch:
            return
        body = gzip.compress(json.dumps(batch,separators=(',',':'),sort_keys=True).encode(),compresslevel=6,mtime=0)
        path = out/f'{len(shards)+1:04d}.json.gz'
        path.write_bytes(body)
        shards.append({'path':str(path.relative_to(root)),'sha256':sha(body),'files':len(batch),'bytes':len(body)})
        batch = {}

    def add(path, digest, length):
        nonlocal count,total
        path = Path(path)
        path = path.relative_to(root) if path.is_absolute() else path
        if not path.is_relative_to('assets/gen5-history') or '..' in path.parts:
            raise ValueError('Export path is outside the historical group')
        if not (root/path).is_file() or (root/path).stat().st_size != length:
            raise ValueError('Export member missing or changed: '+str(path))
        if str(path) in batch:
            raise ValueError('Duplicate export destination')
        batch[str(path)]={'sha256':digest,'bytes':length}
        count += 1
        total += length
        if len(batch)>=shard_size:
            flush()

    db=sqlite3.connect('file:'+str(journal.resolve())+'?mode=ro',uri=True)
    if db.execute('select count(*) from files').fetchone()[0] != summary['files']:
        raise ValueError('Archive journal is not complete')
    for path,digest,length in db.execute('select destination,sha,bytes from files order by destination'):
        add(path,digest,length)
    db.close()
    fifo=json.loads(journal.with_suffix('.fifo.json').read_text())
    seen=set()
    for spec in fifo.values():
        path=Path(spec['path'])
        if path in seen:
            continue
        seen.add(path)
        add(path,spec['sha256'],path.stat().st_size)
    # These modest files can change during native identity reconciliation.
    inputs=json.loads((state/'export-inputs.json').read_text())
    for spec in inputs['files']:
        data=(root/spec['path']).read_bytes()
        spec.update(sha256=sha(data),bytes=len(data))
    (state/'export-inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
    for path in sorted(state.rglob('*')):
        if path.is_file():
            if 'private' in path.name or path.suffix in {'.tmp','.partial'}:
                raise ValueError('Private or unfinished export state file')
            data=path.read_bytes()
            add(path,sha(data),len(data))
    data=(base/'archive-summary.json').read_bytes()
    add(base/'archive-summary.json',sha(data),len(data))
    flush()
    manifest={'group':'gen5-history','optional':True,'files':{},'shards':shards,'file_count':count,'payload_bytes':total}
    (out.parent/'gen5-history.json').write_text(json.dumps(manifest,indent=2)+'\n')
    catalog_path=root/'data/release-assets.json'
    catalog=json.loads(catalog_path.read_text())
    catalog['groups']['gen5-history']={'files':count,'bytes':total,'manifest':'data/manifests/gen5-history.json',
        'optional':True,'status':'indexed-not-packaged','parts':[],
        'required_sources':['experimental/gen5/manage.py'],
        'description':'Complete historical durable state; not required for play or a new training pilot.'}
    catalog_path.write_text(json.dumps(catalog,indent=2)+'\n')
    print(json.dumps({'files':count,'payload_bytes':total,'manifest_shards':len(shards)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--journal',type=Path,required=True)
    args=parser.parse_args()
    build(args.root.resolve(),args.journal)
