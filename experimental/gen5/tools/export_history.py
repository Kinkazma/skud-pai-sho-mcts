#!/usr/bin/env python3
"""Export a paused Gen5 recovery tree; never run or modify the source campaign.

The destination contains neutral relative paths. Bundle names preserve original
sort keys. Numeric JSON tokens are untouched; source exclusion hashes are explicit.
A private SQLite journal supports interruption/resume and is NEVER a release asset.
"""
import argparse, concurrent.futures, gzip, hashlib, json, os, re, shutil, sqlite3, time
from pathlib import Path

SOURCE_PAIR = re.compile(rb'("source_run"\s*:\s*)"([^"\\]*)"(\s*,\s*"game_id"\s*:\s*(?:"([^"\\]*)"|(\d+)))')
PRIVATE_PATH = re.compile(rb'/Users/[^"\\\n]+')

def sha(b): return hashlib.sha256(b).hexdigest()

def sanitize(b):
    """Keep all numerical tokens and sequence-bank exclusion identities exact."""
    count = 0
    def source(m):
        nonlocal count
        count += 1
        run = m[2]; game = m[4] if m[4] is not None else m[5]
        digest = sha(run + b'/' + game)
        return m[1] + b'"portable-source/' + digest.encode() + b'"' + m[3]
    b = SOURCE_PAIR.sub(source, b)
    b = PRIVATE_PATH.sub(lambda m: b'source-artifact:' + sha(m[0]).encode(), b)
    return b, count

def atomic(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_bytes(body); tmp.replace(path)

def export_one(item):
    src, dest, kind, expected = item
    before = src.stat(); raw = src.read_bytes()
    if expected and sha(raw) != expected: raise ValueError('Source hash mismatch: ' + str(src))
    old = sha(raw)
    body = gzip.decompress(raw) if kind == 'bundle' or kind == 'fifo' else raw
    clean, source_keys = sanitize(body)
    if kind in ('bundle', 'fifo', 'revisions'):
        result = raw if clean == body else gzip.compress(clean, compresslevel=1, mtime=0)
    else: result = clean
    new = sha(result)
    if kind == 'revisions':
        # Dense annotations compress well; the native reader supports both forms.
        result = gzip.compress(clean, compresslevel=1, mtime=0)
        new = sha(result); dest = dest.with_suffix('.json.gz')
    if kind == 'bundle':
        if old != src.name.removesuffix('.json.gz'): raise ValueError('Invalid source bundle hash')
        dest = dest.with_name(old + '__' + new + '.json.gz') if old != new else dest
    if before.st_size != src.stat().st_size or before.st_mtime_ns != src.stat().st_mtime_ns:
        raise ValueError('Source changed during export')
    if not dest.exists() or sha(dest.read_bytes()) != new: atomic(dest, result)
    return str(src), str(dest), old, new, len(result), source_keys

def export_archives(config, destination, journal, workers):
    c = json.loads(config.read_text())
    roots = [Path(c['case_curriculum']['archive'])] + list(map(Path, c['recall_archive_sources']))
    db = sqlite3.connect(journal)
    db.execute('create table if not exists files (source text primary key, destination text, old_sha text, sha text, bytes integer, source_keys integer)')
    completed = {r[0] for r in db.execute('select source from files')}
    start = time.monotonic(); total = len(completed); count = 0
    def jobs():
        for root in roots:
            # Only the archive's actual runtime stores; never follow symlinks.
            for sub in ['', 'proofs', 'revisions']:
                directory = root / sub
                if not directory.exists(): continue
                out = destination / 'archives' / root.name / sub
                out.mkdir(parents=True, exist_ok=True)
                for entry in os.scandir(directory):
                    if entry.name.startswith('.') or entry.name.endswith('.tmp'): continue
                    if sub == '' and not entry.name.endswith('.json.gz'): continue
                    if sub and not entry.name.endswith('.json'): continue
                    if entry.is_symlink(): raise ValueError('Archive symlink')
                    src = Path(entry.path)
                    if str(src) not in completed:
                        yield src, out / entry.name, 'bundle' if sub == '' else sub, None
    # Bounded submission keeps millions of records out of the executor queue.
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        it = iter(jobs())
        while True:
            chunk = []
            for _ in range(256):
                try: chunk.append(next(it))
                except StopIteration: break
            if not chunk: break
            if shutil.disk_usage(destination).free < 20 * 1024**3:
                raise RuntimeError('Less than 20 GiB free; export safely stopped before next batch')
            for row in pool.map(export_one, chunk):
                db.execute('insert into files values (?,?,?,?,?,?)', row)
                count += 1
            db.commit()
            if count % 4096 == 0:
                print(json.dumps({'new_files':count,'total_files':total+count,'seconds':round(time.monotonic()-start,1)}),flush=True)
    summary = dict(files=db.execute('select count(*) from files').fetchone()[0],
                   bytes=db.execute('select sum(bytes) from files').fetchone()[0],
                   preserved_source_keys=db.execute('select sum(source_keys) from files').fetchone()[0])
    db.close(); print(json.dumps(summary),flush=True)
    atomic(destination / 'archive-summary.json', (json.dumps(summary,indent=2)+'\n').encode())

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path);p.add_argument('destination',type=Path)
    p.add_argument('--journal',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();a.destination.mkdir(parents=True,exist_ok=True)
    export_archives(a.config,a.destination,a.journal,a.workers)
