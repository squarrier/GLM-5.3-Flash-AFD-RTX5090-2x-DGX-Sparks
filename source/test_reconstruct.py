#!/usr/bin/env python3
"""Offline reconstruction guard tests. Requires the already verified public input cache."""
from pathlib import Path
import argparse, copy, importlib.util, json, tempfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('source_reconstruct', HERE / 'reconstruct.py')
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache', required=True, type=Path)
    ap.add_argument('--scratch', required=True, type=Path)
    ap.add_argument('--receipt', required=True, type=Path)
    a = ap.parse_args()
    original = json.loads((HERE / 'source-lock.json').read_bytes())
    results = []
    cases = ['positive', 'changed_patch', 'traversal', 'duplicate', 'moving_input_ref',
             'unsupported_mode', 'actual_mode_mismatch', 'patch_path_omission', 'corrupt_cache']
    for case in cases:
        with tempfile.TemporaryDirectory(prefix='source-test-', dir=a.scratch) as tmp:
            p = Path(tmp)
            lock = copy.deepcopy(original)
            patch = HERE / 'implementation.patch'
            cache = a.cache
            if case == 'changed_patch':
                patch = p / 'changed.patch'
                patch.write_bytes((HERE / 'implementation.patch').read_bytes() + b'\n')
            elif case == 'traversal':
                lock['files'][0]['path'] = '../escape'
            elif case == 'duplicate':
                lock['files'].append(copy.deepcopy(lock['files'][0]))
            elif case == 'moving_input_ref':
                r = next(x for x in lock['files'] if 'base' in x)
                r['base']['url'] = r['base']['url'].replace(lock['base_commit'], 'main')
            elif case == 'unsupported_mode':
                lock['files'][0]['mode'] = '120000'
            elif case == 'actual_mode_mismatch':
                lock['files'][0]['mode'] = '100755'
            elif case == 'patch_path_omission':
                lock['files'].pop(0)
            elif case == 'corrupt_cache':
                cache = p / 'bad-cache'
                cache.mkdir()
                for src in (x.get('base') or x.get('grant') for x in lock['files']):
                    if src:
                        target = cache / src['sha256']
                        if not target.exists():
                            target.write_bytes((a.cache / src['sha256']).read_bytes())
                first = next(x['base'] for x in lock['files'] if 'base' in x)
                (cache / first['sha256']).write_bytes(b'not-the-pinned-source')
            lp = p / 'lock.json'
            lp.write_text(json.dumps(lock))
            try:
                result = m.reconstruct(lp, patch, p / 'out', cache, False)
                assert case == 'positive', 'negative control unexpectedly passed: ' + case
                assert result['files'] == 133
                results.append({'case': case, 'pass': True, 'files': result['files'], 'apply': result['apply']})
            except ValueError as error:
                assert case != 'positive', str(error)
                results.append({'case': case, 'pass': True, 'refusal': str(error)})
    a.receipt.write_text(json.dumps({'status': 'PASS', 'tests': len(results), 'results': results}, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'tests': len(results), 'cases': [x['case'] for x in results]}, indent=2))

if __name__ == '__main__':
    main()
