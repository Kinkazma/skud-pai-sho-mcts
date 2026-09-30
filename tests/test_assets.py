import hashlib
import gzip
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from scripts import assets


class ResourcePacks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'data/manifests').mkdir(parents=True)

    def fixture(self, values):
        entries = {}
        for name, value in values.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
            entries[name] = {'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()}
        (self.root / 'data/manifests/test.json').write_text(json.dumps({'files': entries}))
        return entries

    def test_inclusive_archive_bound_and_roundtrip_with_long_unicode_path(self):
        entries = self.fixture({'assets/a': b'a' * 8000,
                                'assets/' + 'x' * 130 + '/é': b'b' * 7000})
        out = self.root / 'out'
        manifest = assets.pack(self.root, 'test', out, limit=10240)
        self.assertEqual(len(manifest['parts']), 2)
        self.assertEqual([p['bytes'] for p in manifest['parts']], [10240, 10240])
        self.assertEqual(assets.verify_packs(self.root, 'test', out), len(entries))
        with tarfile.open(out / manifest['parts'][0]['file']) as archive:
            m = archive.next()
            self.assertEqual((m.uid, m.gid, m.mtime, m.uname, m.gname), (0, 0, 0, '', ''))

    def test_one_oversized_member_rejected_before_output(self):
        self.fixture({'assets/a': b'a' * 10000})
        out = self.root / 'out'
        with self.assertRaisesRegex(ValueError, 'Single asset'):
            assets.pack(self.root, 'test', out, limit=10240)
        self.assertFalse(out.exists())

    def test_missing_group_explains_installation_and_changed_bytes_fail(self):
        self.fixture({'assets/a': b'original'})
        (self.root / 'assets/a').unlink()
        with self.assertRaisesRegex(ValueError, 'Install ALL parts'):
            assets.require_groups(['test'], self.root)
        (self.root / 'assets/a').write_bytes(b'changed!')
        with self.assertRaisesRegex(ValueError, 'altered asset'):
            assets.pack(self.root, 'test', self.root / 'out')

    def test_incomplete_set_is_rejected_even_with_valid_individual_archives(self):
        self.fixture({'assets/a': b'a' * 8000, 'assets/b': b'b' * 8000})
        out = self.root / 'out'
        manifest = assets.pack(self.root, 'test', out, limit=10240)
        manifest['parts'].pop()
        (out / 'test.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            assets.verify_packs(self.root, 'test', out)

    def test_determinism_and_existing_outputs_preserved(self):
        self.fixture({'assets/a': b'a' * 7000})
        first = assets.pack(self.root, 'test', self.root / 'one')
        second = assets.pack(self.root, 'test', self.root / 'two')
        self.assertEqual(first, second)
        with self.assertRaisesRegex(ValueError, 'output exists'):
            assets.pack(self.root, 'test', self.root / 'one')

    def test_tampered_archive_is_rejected(self):
        self.fixture({'assets/a': b'data'})
        out = self.root / 'out'
        manifest = assets.pack(self.root, 'test', out)
        path = out / manifest['parts'][0]['file']
        with path.open('r+b') as stream:
            stream.seek(520)
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            assets.verify_packs(self.root, 'test', out)

    def test_sharded_manifest_keeps_multipart_contents_and_detects_corruption(self):
        entries = self.fixture({'assets/a': b'a'*8000,'assets/b': b'b'*8000})
        shards = []
        for i,(name,entry) in enumerate(entries.items()):
            path = self.root/f'data/manifests/part-{i}.json.gz'
            body = gzip.compress(json.dumps({name:entry}).encode(),mtime=0)
            path.write_bytes(body)
            shards.append({'path':str(path.relative_to(self.root)),'sha256':hashlib.sha256(body).hexdigest()})
        manifest = self.root/'data/manifests/test.json'
        manifest.write_text(json.dumps({'files':{},'shards':shards}))
        out = self.root/'out'
        result = assets.pack(self.root,'test',out,limit=10240)
        self.assertEqual(len(result['parts']),2)
        self.assertEqual(assets.verify_packs(self.root,'test',out),2)
        (self.root/shards[0]['path']).write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'Manifest shard hash mismatch'):
            assets.entries_for(self.root,'test')

    def test_sharded_manifest_rejects_duplicate_members_and_parent_paths(self):
        entries = self.fixture({'assets/a':b'a'})
        path = self.root/'data/manifests/shard.json.gz'
        body = gzip.compress(json.dumps(entries).encode(),mtime=0)
        path.write_bytes(body)
        spec = {'path':'data/manifests/shard.json.gz','sha256':hashlib.sha256(body).hexdigest()}
        manifest = self.root/'data/manifests/test.json'
        manifest.write_text(json.dumps({'files':entries,'shards':[spec]}))
        with self.assertRaisesRegex(ValueError,'Duplicate resource'):
            assets.entries_for(self.root,'test')
        spec['path']='data/manifests/../../outside.gz'
        manifest.write_text(json.dumps({'shards':[spec]}))
        with self.assertRaisesRegex(ValueError,'Invalid manifest shard path'):
            assets.entries_for(self.root,'test')


if __name__ == '__main__':
    unittest.main()
