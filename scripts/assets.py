#!/usr/bin/env python3
"""List, install, verify or package release resources. Never uploads anything."""
import argparse
import hashlib
import gzip
import json
import tarfile
from pathlib import Path, PurePosixPath
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
MAX_ARCHIVE_BYTES = 1_800_000_000  # Decimal bytes, including TAR headers/padding.
PROFILES = {
    'gen3': ['core-memory'],  # The portable launcher prepares the entire suite.
    'gen35-replay': ['core-memory', 'gen35-replay'],
    'human': ['human'],
    'gen4': [],
    'gen5-play': ['core-memory', 'gen5-memory'],
    'gen5': ['core-memory', 'human', 'gen5-memory', 'gen5-learning'],
    'gen5-legacy-bundle': ['core-memory', 'human', 'gen5'],
    'gen5-history': ['core-memory', 'human', 'gen5-memory', 'gen5-learning', 'gen5-history-index', 'gen5-history'],
    'apple': ['apple'],
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def entries_for(root, group):
    path = root / 'data/manifests' / (group + '.json')
    if Path(group).name != group or not path.is_file():
        raise ValueError('Unknown resource group; see data/ASSETS.md')
    manifest = json.loads(path.read_text())
    entries = dict(manifest.get('files', {}))
    for shard in manifest.get('shards', []):
        name = shard['path']
        rel = PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or not name.startswith('data/manifests/'):
            raise ValueError('Invalid manifest shard path')
        source = root / name
        if not source.is_file():
            raise ValueError('Missing optional manifest index: ' + name
                             + '; install the complete gen5-history profile first.')
        if source.is_symlink() or digest(source) != shard['sha256']:
            raise ValueError('Manifest shard hash mismatch: ' + name)
        rows = json.loads(gzip.decompress(source.read_bytes()))
        if entries.keys() & rows.keys():
            raise ValueError('Duplicate resource across manifest shards')
        entries.update(rows)
    for name in entries:
        rel = PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or str(rel) != name:
            raise ValueError('Invalid asset path in manifest: ' + name)
    return entries


def require_groups(groups, root=ROOT):
    """Fast preflight; full SHA-256 verification is an explicit install step."""
    missing = []
    for group in groups:
        entries = entries_for(root, group)
        if any(not (root / n).is_file() or (root / n).stat().st_size != e['bytes']
               for n, e in entries.items()):
            missing.append(group)
    if missing:
        raise ValueError('Missing or incomplete release resources: ' + ', '.join(missing)
                         + '. Install ALL parts of these groups; see data/ASSETS.md.')


def verify_inputs(root, entries):
    for name, entry in entries.items():
        path = root / name
        if (not path.is_file() or path.is_symlink()
                or path.stat().st_size != entry['bytes']
                or digest(path) != entry['sha256']):
            raise ValueError('Missing or altered asset: ' + name)


def header(name, size):
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ''
    info.mtime = 0
    return info


def member_bytes(name, size):
    return len(header(name, size).tobuf(format=tarfile.PAX_FORMAT)) + (size + 511) // 512 * 512


def archive_bytes(payload):
    # Two end blocks, then the tarfile module pads to a full record.
    record = tarfile.RECORDSIZE
    return ((payload + 1024 + record - 1) // record) * record


def plan_parts(entries, limit=MAX_ARCHIVE_BYTES):
    parts, batch, size = [], [], 0
    for name, entry in entries.items():
        cost = member_bytes(name, entry['bytes'])
        if archive_bytes(cost) > limit:
            raise ValueError('Single asset exceeds archive limit: ' + name)
        if batch and archive_bytes(size + cost) > limit:
            parts.append(batch)
            batch, size = [], 0
        batch.append(name)
        size += cost
    if batch:
        parts.append(batch)
    return parts


def pack(root, group, out, limit=MAX_ARCHIVE_BYTES):
    entries = entries_for(root, group)
    parts = plan_parts(entries, limit)
    paths = [out / f'{group}-{i + 1:02d}.tar' for i in range(len(parts))]
    index = out / (group + '.json')
    if index.exists() or any(p.exists() or p.with_suffix('.tar.partial').exists() for p in paths):
        raise ValueError('Pack output exists; choose a new output folder')
    verify_inputs(root, entries)
    out.mkdir(parents=True, exist_ok=True)
    result = []
    for path, names in zip(paths, parts):
        tmp = path.with_suffix('.tar.partial')
        try:
            with tarfile.open(tmp, 'w', format=tarfile.PAX_FORMAT) as archive:
                for name in names:
                    with (root / name).open('rb') as source:
                        archive.addfile(header(name, entries[name]['bytes']), source)
            expected = archive_bytes(sum(member_bytes(n, entries[n]['bytes']) for n in names))
            if tmp.stat().st_size != expected or expected > limit:
                raise ValueError('Unexpected archive size: ' + path.name)
            tmp.rename(path)
        finally:
            tmp.unlink(missing_ok=True)
        result.append({'file': path.name, 'bytes': path.stat().st_size,
                       'sha256': digest(path), 'files': len(names)})
    manifest = {'group': group, 'max_archive_bytes': limit, 'parts': result}
    index.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def verify_packs(root, group, out):
    """Read every TAR member against the source manifest without extracting it."""
    entries = entries_for(root, group)
    index = json.loads((out / (group + '.json')).read_text())
    seen = set()
    for part in index['parts']:
        name = part['file']
        if Path(name).name != name:
            raise ValueError('Invalid archive filename')
        path = out / name
        if (path.stat().st_size != part['bytes'] or part['bytes'] > MAX_ARCHIVE_BYTES
                or digest(path) != part['sha256']):
            raise ValueError('Archive size or SHA-256 mismatch: ' + name)
        with tarfile.open(path, 'r|') as archive:
            for member in archive:
                entry = entries.get(member.name)
                if (entry is None or member.name in seen or not member.isfile()
                        or member.size != entry['bytes']):
                    raise ValueError('Unexpected, duplicated or altered member: ' + member.name)
                h = hashlib.sha256()
                with archive.extractfile(member) as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        h.update(block)
                if h.hexdigest() != entry['sha256']:
                    raise ValueError('Member SHA-256 mismatch: ' + member.name)
                seen.add(member.name)
    if seen != set(entries):
        raise ValueError('Incomplete archive set: ' + group)
    return len(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['list', 'install', 'check', 'verify', 'pack', 'verify-packs'])
    parser.add_argument('group', nargs='?')
    parser.add_argument('--profile', choices=PROFILES)
    parser.add_argument('--output', default='packs')
    parser.add_argument('--from-dir', help='Directory containing downloaded release TAR files')
    parser.add_argument('--base-url', help='HTTPS release asset directory; loopback HTTP for tests')
    parser.add_argument('--repair', action='store_true', help='Restore altered manifest-listed resource files')
    args = parser.parse_args()
    try:
        groups = PROFILES[args.profile] if args.profile else ([args.group] if args.group else [])
        if args.command == 'list':
            catalog = json.loads((ROOT / 'data/release-assets.json').read_text())
            print(json.dumps({g: catalog['groups'][g] for g in (groups if args.profile or args.group else catalog['groups'])}, indent=2))
            return
        if not groups and args.profile:
            print('This profile needs no additional release resource group.')
            return
        if not groups:
            parser.error('Choose a group or --profile')
        if args.command == 'install':
            from scripts.asset_install import install
            install(groups, args.from_dir, args.base_url, args.repair)
            return
        if args.command == 'check':
            require_groups(groups)
            print('Required resource files are present (run verify for SHA-256 checks).')
            return
        for group in groups:
            if args.command == 'verify':
                entries = entries_for(ROOT, group)
                verify_inputs(ROOT, entries)
                print(f'Verified {len(entries)} files in {group}')
            elif args.command == 'pack':
                print(json.dumps(pack(ROOT, group, (ROOT / args.output).resolve()), indent=2))
            else:
                count = verify_packs(ROOT, group, (ROOT / args.output).resolve())
                print(f'Verified all {count} members of {group} archives')
    except (ValueError, OSError, KeyError, tarfile.TarError) as error:
        raise SystemExit(str(error)) from error


if __name__ == '__main__':
    main()
