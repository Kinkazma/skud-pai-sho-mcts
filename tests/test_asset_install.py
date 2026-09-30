import contextlib
import hashlib
import gzip
import http.server
import io
import json
import socket
import tarfile
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.assets import pack
from scripts.asset_download import download, validate_url
from scripts.asset_install import install


@contextlib.contextmanager
def serve(folder, range_mode='honor'):
    requests = []
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            requested = self.headers.get('Range')
            requests.append(requested)
            value = (folder / self.path.lstrip('/')).read_bytes()
            start = int(requested.split('=')[1].split('-')[0]) if requested else 0
            if requested and range_mode == 'honor':
                self.send_response(206)
                self.send_header('Content-Range', f'bytes {start}-{len(value)-1}/{len(value)}')
                body = value[start:]
            else:
                self.send_response(200)
                body = value
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            if range_mode == 'interrupt':
                self.wfile.write(body[:2000])
                self.wfile.flush()
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
            else:
                self.wfile.write(body)
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class Installer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.source, self.root, self.archives = base/'source', base/'checkout', base/'packs'
        entries = {}
        for name, value in {'assets/a': b'a'*7000, 'assets/b': b'b'*7000}.items():
            path = self.source/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
            entries[name] = {'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()}
        for root in [self.source, self.root]:
            (root/'data/manifests').mkdir(parents=True)
            (root/'data/manifests/demo.json').write_text(json.dumps({'files': entries}))
        self.group = pack(self.source, 'demo', self.archives, limit=10240)
        self.catalog = {'release_url': None, 'groups': {'demo': self.group}}
        self.save_catalog()

    def save_catalog(self):
        (self.root/'data/release-assets.json').write_text(json.dumps(self.catalog))

    def install(self, **kw):
        return install(['demo'], root=self.root, **kw)

    def test_local_install_and_idempotent_without_any_source(self):
        self.install(source_dir=self.archives)
        before = [(self.root/n).stat().st_mtime_ns for n in ['assets/a', 'assets/b']]
        with patch('scripts.asset_install.download', side_effect=AssertionError('download')):
            result = self.install()
        self.assertEqual(result[0]['status'], 'already-verified')
        self.assertEqual(before, [(self.root/n).stat().st_mtime_ns for n in ['assets/a', 'assets/b']])

    def test_missing_part_does_not_install_first_part(self):
        (self.archives/self.group['parts'][1]['file']).unlink()
        with self.assertRaisesRegex(ValueError, 'Missing or altered archive'):
            self.install(source_dir=self.archives)
        self.assertFalse((self.root/'assets').exists())

    def test_wrong_archive_hash_prevents_writes(self):
        (self.archives/self.group['parts'][0]['file']).write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError, 'Missing or altered archive'):
            self.install(source_dir=self.archives)
        self.assertFalse((self.root/'assets').exists())

    def test_unexpected_path_even_with_matching_archive_digest_is_rejected(self):
        path = self.archives/self.group['parts'][0]['file']
        with tarfile.open(path, 'w') as tar:
            info = tarfile.TarInfo('../escape'); info.size = 1
            tar.addfile(info, io.BytesIO(b'x'))
        self.group['parts'][0].update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        self.save_catalog()
        with self.assertRaisesRegex(ValueError, 'Unexpected'):
            self.install(source_dir=self.archives)
        self.assertFalse((self.root.parent/'escape').exists())
        self.assertFalse((self.root/'assets').exists())

    def test_resource_directory_symlink_is_refused(self):
        (self.root/'assets').symlink_to(self.source/'assets', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symbolic link'):
            self.install(source_dir=self.archives)

    def test_partial_install_and_explicit_repair_preserve_unrelated_files(self):
        (self.root/'assets').mkdir()
        (self.root/'assets/a').write_bytes(b'a'*7000)
        (self.root/'assets/other').write_bytes(b'keep')
        self.assertEqual(self.install(source_dir=self.archives)[0]['installed_files'], 1)
        (self.root/'assets/a').write_bytes(b'altered')
        with self.assertRaisesRegex(ValueError, '--repair'):
            self.install(source_dir=self.archives)
        self.assertEqual(self.install(source_dir=self.archives, repair=True)[0]['installed_files'], 1)
        self.assertEqual((self.root/'assets/other').read_bytes(), b'keep')

    def test_generation_sources_must_be_installed_first(self):
        self.catalog['groups']['demo']['required_sources'] = ['experimental/launcher.py']
        self.save_catalog()
        with self.assertRaisesRegex(ValueError, 'switch to its documented branch'):
            self.install(source_dir=self.archives)
        self.assertFalse((self.root/'assets').exists())

    def test_later_generation_requirement_prevents_earlier_downloads(self):
        self.catalog['groups']['later'] = {'required_sources': ['experimental/missing.py'], 'parts': []}
        self.save_catalog()
        with patch('scripts.asset_install.download', side_effect=AssertionError('download')):
            with self.assertRaisesRegex(ValueError, 'switch to its documented branch'):
                install(['demo', 'later'], base_url='https://example.com/release', root=self.root)
        self.assertFalse((self.root/'assets').exists())

    def test_no_fake_release_url(self):
        with self.assertRaisesRegex(ValueError, 'No published release URL'):
            self.install()
        self.assertFalse((self.root/'assets').exists())

    def test_optional_index_is_installed_before_history_and_reused(self):
        manifest = self.source/'data/manifests/demo.json'
        rows = json.loads(manifest.read_text())['files']
        shard = self.source/'data/manifests/demo-index/0001.json.gz'
        shard.parent.mkdir()
        shard.write_bytes(gzip.compress(json.dumps(rows).encode(), mtime=0))
        name = str(shard.relative_to(self.source))
        entry = {'bytes': shard.stat().st_size,
                 'sha256': hashlib.sha256(shard.read_bytes()).hexdigest()}
        for root in [self.source, self.root]:
            (root/'data/manifests/demo-index.json').write_text(json.dumps({'files': {name: entry}}))
            (root/'data/manifests/demo.json').write_text(json.dumps({'shards': [{'path': name, **entry}]}))
        index = pack(self.source, 'demo-index', self.archives)
        self.catalog['groups']['demo-index'] = index
        self.save_catalog()
        with self.assertRaisesRegex(ValueError, 'Missing optional manifest index'):
            self.install(source_dir=self.archives)
        result = install(['demo-index', 'demo'], source_dir=self.archives, root=self.root)
        self.assertEqual([r['installed_files'] for r in result], [1, 2])
        with patch('scripts.asset_install.download', side_effect=AssertionError('download')):
            result = install(['demo-index', 'demo'], root=self.root)
        self.assertTrue(all(r['status'] == 'already-verified' for r in result))

    def test_http_download_then_reuse_verified_cache(self):
        with serve(self.archives) as (url, requests):
            self.install(base_url=url)
            self.assertEqual(len(requests), 2)
            self.install(base_url=url)
            self.assertEqual(len(requests), 2)
            (self.root/'assets/a').unlink()
            self.install(base_url=url)
            self.assertEqual(len(requests), 2)

    def test_resume_range_and_server_ignoring_range(self):
        for mode in ['honor', 'ignore']:
            cache = self.root/mode
            cache.mkdir()
            part = self.group['parts'][0]
            value = (self.archives/part['file']).read_bytes()
            (cache/(part['file']+'.partial')).write_bytes(value[:2000])
            with serve(self.archives, mode) as (url, requests):
                path = download(url, part, cache)
                self.assertEqual(path.read_bytes(), value)
                self.assertEqual(requests, ['bytes=2000-'])

    def test_interruption_then_resume(self):
        part = self.group['parts'][0]
        cache = self.root/'cache'
        with serve(self.archives, 'interrupt') as (url, requests):
            with self.assertRaises((ValueError, OSError)):
                download(url, part, cache)
        self.assertFalse((cache/part['file']).exists())
        partial = cache/(part['file']+'.partial')
        self.assertEqual(partial.stat().st_size, 2000)
        with serve(self.archives) as (url, requests):
            download(url, part, cache)
            self.assertEqual(requests, ['bytes=2000-'])

    def test_url_protocols_and_credentials(self):
        for url in ['http://example.com/p', 'file:///p', 'https://user:pass@example.com/p']:
            with self.assertRaises(ValueError):
                validate_url(url)
        self.assertEqual(validate_url('https://example.com/p'), 'https://example.com/p')


if __name__ == '__main__':
    unittest.main()
