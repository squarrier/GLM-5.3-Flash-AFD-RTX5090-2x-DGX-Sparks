#!/usr/bin/env python3
"""Bounded source-overlay reconstruction, not a build or release approval."""
from pathlib import Path, PurePosixPath
import argparse, concurrent.futures, hashlib, json, os, re, stat, subprocess, urllib.request

HERE = Path(__file__).resolve().parent
MAX_FILE = 2 * 1024 * 1024
MAX_FILES = 140

def digest(data):
    return hashlib.sha256(data).hexdigest()

def safe_path(value):
    p = PurePosixPath(value)
    if p.is_absolute() or not p.parts or any(x in ('..', '.') for x in p.parts) or str(p) != value:
        raise ValueError('unsafe or noncanonical member path')
    return value

def reconstruct(lock_path, patch_path, output, cache, fetch):
    lock = json.loads(lock_path.read_bytes())
    patch = patch_path.read_bytes()
    if digest(patch) != lock['patch_sha256']:
        raise ValueError('patch digest mismatch')
    files = lock['files']
    if not 0 < len(files) <= MAX_FILES:
        raise ValueError('file budget exceeded')
    paths = [safe_path(x['path']) for x in files]
    if len(set(paths)) != len(paths):
        raise ValueError('duplicate target paths')
    clean_env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    clean_env.update(GIT_CONFIG_GLOBAL='/dev/null', GIT_CONFIG_NOSYSTEM='1', GIT_OPTIONAL_LOCKS='0')
    listing = subprocess.run(['git', 'apply', '--numstat', '-z', str(patch_path.resolve())],
                             env=clean_env, capture_output=True, timeout=30)
    if listing.returncode:
        raise ValueError('patch inventory refused')
    patch_paths = [safe_path(x.decode().split('\t', 2)[2]) for x in listing.stdout.split(b'\0') if x]
    source_paths = {x['path'] for x in files if 'grant' not in x}
    if len(set(patch_paths)) != len(patch_paths) or set(patch_paths) != source_paths:
        raise ValueError('patch path allowlist mismatch')
    for row in files:
        if row['mode'] not in ('100644', '100755'):
            raise ValueError('unsupported file mode')
        for source in [row.get('base'), row.get('grant')]:
            if source is None:
                continue
            url = source['url']
            if not re.fullmatch(r'https://raw\.githubusercontent\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/[0-9a-f]{40}/[^?#]+', url):
                raise ValueError('source URL must use an exact public GitHub commit')
            if not re.fullmatch('[0-9a-f]{64}', source['sha256']) or not 0 < source['bytes'] <= MAX_FILE:
                raise ValueError('invalid source size or digest')
    if output.exists():
        raise ValueError('output must not exist')
    if cache.is_symlink():
        raise ValueError('cache must not be a symlink')
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    sources = {x['sha256']: x for row in files for x in (row.get('base'), row.get('grant')) if x is not None}
    if sum(x['bytes'] for x in sources.values()) > 16 * 1024 * 1024:
        raise ValueError('input byte budget exceeded')

    def obtain(item):
        key, source = item
        target = cache / key
        if target.is_symlink():
            raise ValueError('cached blob must not be a symlink')
        if target.exists():
            data = target.read_bytes()
        else:
            if not fetch:
                raise ValueError('missing offline input ' + key)
            request = urllib.request.Request(source['url'], headers={'User-Agent': 'bounded-source-reconstruction'})
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.url != source['url']:
                    raise ValueError('unexpected input redirect')
                data = response.read(MAX_FILE + 1)
        if len(data) != source['bytes'] or digest(data) != key:
            raise ValueError('input digest or size mismatch')
        if not target.exists():
            target.write_bytes(data)
            target.chmod(0o600)
        return key, data

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        blobs = dict(pool.map(obtain, sources.items()))
    output.mkdir(parents=True, mode=0o700)
    for row in files:
        if 'base' not in row:
            continue
        target = output / row['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blobs[row['base']['sha256']])
        target.chmod(int(row['base']['mode'], 8) & 0o777)
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    env.update(GIT_CONFIG_GLOBAL='/dev/null', GIT_CONFIG_NOSYSTEM='1', GIT_OPTIONAL_LOCKS='0')
    applies = []
    for extra in (['--check'], []):
        q = subprocess.run(['git', 'apply', *extra, str(patch_path.resolve())], cwd=output,
                           env=env, capture_output=True, timeout=30)
        applies.append({'check_only': bool(extra), 'exit': q.returncode,
                        'stdout': q.stdout.decode(), 'stderr': q.stderr.decode()})
        if q.returncode:
            raise RuntimeError('patch apply refused: ' + q.stderr.decode())
    for row in files:
        if 'grant' in row:
            target = output / row['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blobs[row['grant']['sha256']])
            target.chmod(int(row['mode'], 8) & 0o777)
    expected = set(paths)
    actual = set()
    for p in output.rglob('*'):
        if p.is_symlink():
            raise ValueError('unexpected output symlink')
        if p.is_file():
            actual.add(p.relative_to(output).as_posix())
    if actual != expected:
        raise ValueError('output membership mismatch')
    checked = []
    for row in files:
        p = output / row['path']
        data = p.read_bytes()
        if len(data) != row['bytes'] or digest(data) != row['sha256']:
            raise ValueError('output content mismatch: ' + row['path'])
        mode = stat.S_IMODE(p.stat().st_mode)
        if mode != (int(row['mode'], 8) & 0o777):
            raise ValueError('output mode mismatch: ' + row['path'])
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if blob != row['blob']:
            raise ValueError('output Git blob mismatch: ' + row['path'])
        checked.append({'path': row['path'], 'sha256': digest(data), 'blob': blob, 'mode': row['mode']})
    return {'status': 'RECONSTRUCTION PASS; NOT A BUILD', 'scope': '129 implementation paths plus four grant/notice aliases; not full TensorFold or a build',
            'lock_sha256': digest(lock_path.read_bytes()), 'patch_sha256': digest(patch),
            'files': len(checked), 'apply': applies, 'checked': checked}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--cache', required=True, type=Path)
    ap.add_argument('--fetch', action='store_true', help='Fetch only exact public GitHub inputs from the lock')
    ap.add_argument('--receipt', required=True, type=Path)
    ap.add_argument('--lock', type=Path, default=HERE / 'source-lock.json')
    ap.add_argument('--patch', type=Path, default=HERE / 'implementation.patch')
    a = ap.parse_args()
    result = reconstruct(a.lock, a.patch, a.output, a.cache, a.fetch)
    a.receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'checked'}, indent=2))

if __name__ == '__main__':
    main()
