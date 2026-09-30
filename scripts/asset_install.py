"""Install only manifest-listed resources after complete archive verification."""
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from .assets import ROOT, MAX_ARCHIVE_BYTES, entries_for
from .asset_download import download, matches


def destination(root, name):
    rel = PurePosixPath(name)
    if rel.is_absolute() or not rel.parts or '..' in rel.parts or str(rel) != name:
        raise ValueError('Invalid resource path: ' + name)
    path = root
    for piece in rel.parts:
        path = path / piece
        if path.is_symlink():
            raise ValueError('Refusing symbolic link at resource destination: ' + name)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Resource path escapes checkout: ' + name)
    return path


def stage_archives(root, group, entries, archives, repair=False):
    """No destination is written until the group's members all pass their hashes."""
    with tempfile.TemporaryDirectory(prefix='.asset-install-', dir=root) as temporary:
        stage = Path(temporary)
        seen = set()
        for path in archives:
            with tarfile.open(path, 'r|') as archive:
                for member in archive:
                    name = member.name
                    spec = entries.get(name)
                    if (spec is None or name in seen or not member.isfile()
                            or member.size != spec['bytes']):
                        raise ValueError('Unexpected, duplicate or altered archive member: ' + name)
                    destination(root, name)
                    output = stage / name
                    output.parent.mkdir(parents=True, exist_ok=True)
                    h = hashlib.sha256()
                    with archive.extractfile(member) as source, output.open('xb') as target:
                        for block in iter(lambda: source.read(1024 * 1024), b''):
                            h.update(block)
                            target.write(block)
                    if h.hexdigest() != spec['sha256']:
                        raise ValueError('Archive member SHA-256 mismatch: ' + name)
                    seen.add(name)
        if seen != set(entries):
            raise ValueError('Incomplete archive set: ' + group)
        # Check the entire commit set first, including parent paths.
        changed = []
        for name, spec in entries.items():
            target = destination(root, name)
            if matches(target, spec):
                continue
            if target.exists() and (not repair or not target.is_file()):
                raise ValueError('Existing resource differs: ' + name + '; use --repair to restore release data.')
            parent = target.parent
            while parent != root:
                if parent.exists() and not parent.is_dir():
                    raise ValueError('Resource parent is not a directory: ' + name)
                parent = parent.parent
            changed.append(name)
        for name in changed:
            target = destination(root, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            (stage / name).replace(target)
        return len(changed)


def install(groups, source_dir=None, base_url=None, repair=False, root=ROOT):
    root = root.resolve()
    catalog = json.loads((root / 'data/release-assets.json').read_text())
    if source_dir is not None and base_url is not None:
        raise ValueError('Choose --from-dir OR --base-url, not both.')
    source_dir = Path(source_dir).resolve() if source_dir is not None else None
    base_url = base_url or catalog.get('release_url')
    results = []
    # Check the whole profile before downloading any earlier shared group.
    for group in groups:
        spec = catalog['groups'][group]
        for required in spec.get('required_sources', []):
            if not destination(root, required).is_file():
                raise ValueError('Missing generation source: ' + required + '; switch to its documented branch first.')
    for group in groups:
        spec = catalog['groups'][group]
        entries = entries_for(root, group)
        valid = True
        for name, entry in entries.items():
            target = destination(root, name)
            if not matches(target, entry):
                valid = False
                if target.exists() and not repair:
                    raise ValueError('Existing resource differs: ' + name + '; use --repair to restore release data.')
        if valid:
            results.append({'group': group, 'status': 'already-verified', 'installed_files': 0})
            print(group + ': already verified; no download or rewrite.', flush=True)
            continue
        if not source_dir and not base_url:
            raise ValueError('No published release URL yet. Supply --from-dir with the prepared archives, '
                             'or --base-url for a trusted release. See docs/INSTALL.md.')
        print(group + ': verifying all required archive parts.', flush=True)
        archives = []
        for part in spec['parts']:
            if Path(part['file']).name != part['file'] or not 0 < part['bytes'] <= MAX_ARCHIVE_BYTES:
                raise ValueError('Invalid archive specification')
            if source_dir:
                path = source_dir / part['file']
            else:
                path = download(base_url, part, destination(root, '.cache/release-assets'))
            if not matches(path, part):
                raise ValueError('Missing or altered archive: ' + part['file'] + '; installation was not changed.')
            archives.append(path)
        count = stage_archives(root, group, entries, archives, repair)
        results.append({'group': group, 'status': 'installed', 'installed_files': count})
        print(f'{group}: verified and installed {count} files.', flush=True)
    return results
