"""Verified release downloads; no account, external SDK or remote execution."""
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .assets import digest


def validate_url(url):
    parsed = urllib.parse.urlsplit(url)
    local = parsed.hostname in {'localhost', '127.0.0.1', '::1'}
    if (parsed.scheme != 'https' and not (parsed.scheme == 'http' and local)):
        raise ValueError('Resource URLs require HTTPS (HTTP is allowed only for loopback tests).')
    if not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
        raise ValueError('Invalid resource URL; do not put credentials in URLs.')
    return url


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_url(newurl)
        if req.full_url.startswith('https:') and not newurl.startswith('https:'):
            raise ValueError('Refusing HTTPS downgrade redirect')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def matches(path, spec):
    return (path.is_file() and not path.is_symlink()
            and path.stat().st_size == spec['bytes'] and digest(path) == spec['sha256'])


def download(base_url, part, cache):
    """Keep interrupted bytes for Range resume; always verify the final full hash."""
    name = part['file']
    if Path(name).name != name:
        raise ValueError('Invalid archive filename')
    url = validate_url(base_url.rstrip('/') + '/' + urllib.parse.quote(name))
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / name
    if matches(path, part):
        return path
    partial = cache / (name + '.partial')
    if path.is_symlink() or partial.is_symlink():
        raise ValueError('Refusing symbolic link in resource cache')
    if matches(partial, part):
        partial.replace(path)
        return path
    offset = partial.stat().st_size if partial.exists() else 0
    if offset >= part['bytes']:
        partial.unlink()
        offset = 0
    headers = {'User-Agent': 'PaiShoResourceInstaller/1', 'Accept-Encoding': 'identity'}
    if offset:
        headers['Range'] = f'bytes={offset}-'
    request = urllib.request.Request(url, headers=headers)
    opener = urllib.request.build_opener(SafeRedirect())
    with opener.open(request, timeout=30) as response:
        if response.status == 206:
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
            if (not match or tuple(map(int, match.groups())) !=
                    (offset, part['bytes'] - 1, part['bytes'])):
                raise ValueError('Unexpected download range for ' + name)
        elif response.status == 200:
            offset = 0  # The host ignored Range: restart rather than append.
        else:
            raise ValueError('Unexpected download status for ' + name)
        encoding = response.headers.get('Content-Encoding', 'identity')
        if encoding != 'identity':
            raise ValueError('Unexpected content encoding for ' + name)
        with partial.open('ab' if offset else 'wb') as stream:
            size = offset
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                if size > part['bytes']:
                    raise ValueError('Download exceeds declared size: ' + name)
                stream.write(block)
    if not matches(partial, part):
        raise ValueError('Incomplete or altered download: ' + name + '; installation was not changed.')
    partial.replace(path)
    return path
