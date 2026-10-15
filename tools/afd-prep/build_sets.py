#!/usr/bin/env python3
"""CPU AOT preparation using TensorFold's compiler/row helpers.
Apache-2.0. GPT-6.1 Sol via Hermes. AOT helpers: MiaAI-Lab and Ash Hart;
FP8 _fmm: MiaAI-Lab, Apache-2.0; compiler: Triton, MIT.
Run in the pinned PyTorch container with Python TensorFold at /tensorfold/src,
this repo at /fork, q4f8.py at /qualification_80851247/lab, and outputs at /out.
"""
from pathlib import Path
import hashlib, json, shutil, sys, os
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, '/source/engine/tools/zig')
sys.path.insert(0, '/source/tools/afd-prep')
import glm_aot as aot
import glm_aot_rules as rules
BASE = Path('/source/engine/zig/tests/cuda/glm/kernels.json')
F8 = Path('/source/engine/zig/tests/cuda/glm/kernels_f8_w1.json')
OUT = Path('/out')
JOBS = int(os.environ.get('BUILD_JOBS', '2'))

def main():
    # Whole-cubin capture SHAs include source-path debug line tables.
    # The release gate compares every non-debug ELF section with the measured
    # sets; do not mutate the checked-in spec or the engine loader.
    spec = json.loads(BASE.read_text())
    for k in spec['kernels']:
        k.pop('cubin_sha256', None)
    prepared = OUT / 'base-no-debug-capture-sha.json'
    OUT.mkdir(parents=True, exist_ok=True)
    prepared.write_text(json.dumps(spec))
    for sm in (120, 121):
        aot.TARGET = ('cuda', sm, 32)
        target = OUT / f'set{sm}'
        if aot.build(prepared, target, [], JOBS):
            raise SystemExit('Base Triton set failed')
        if sm == 120:
            # glm_aot.build checks a package root; the FP8 spec is a module.
            spec = json.loads(F8.read_text())
            with ProcessPoolExecutor(max_workers=JOBS) as pool:
                results = list(pool.map(aot.compile_job, spec['kernels']))
            rows = json.loads((target / 'aot.json').read_text())
            keys = {json.dumps([r['fn'], r['params'], r['consts']], sort_keys=True) for r in rows['kernels']}
            for k, got in zip(spec['kernels'], results):
                if isinstance(got, str):
                    raise SystemExit(got)
                row = aot.aot_row(k, got)
                if row['global_scratch'] or row['profile_scratch']:
                    raise SystemExit('Unsupported scratch')
                key = json.dumps([row['fn'], row['params'], row['consts']], sort_keys=True)
                if key in keys:
                    raise SystemExit('Duplicate FP8 launch key')
                keys.add(key)
                row['specs'] = [rules.spec_key(k)]
                (target / 'cubins' / f'{got.hash}.cubin').write_bytes(got.cubin)
                rows['kernels'].append(row)
            rows['kernels'].sort(key=lambda r: (r['fn'], r['hash']))
            (target / 'aot.json').write_text(json.dumps(rows, indent=1) + '\n')
        files = sorted(p for p in target.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
        (target / 'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(target)}\n' for p in files))
        print(target, 'ready', len(json.loads((target / 'aot.json').read_text())['kernels']), 'kernels')

if __name__ == '__main__':
    main()
