#!/usr/bin/env python3
"""Refuse source drift before building locally, including optimized Python."""
from pathlib import Path
import hashlib,json,os,sys
R=Path(__file__).resolve().parents[1]
p=R/'docs/SOURCE.json'; data=json.loads(p.read_text())
def require(ok,message):
 if not ok: raise SystemExit(message)
require(data['engine_revision']==os.environ.get('ENGINE_REVISION'),'engine pin differs from exported snapshot')
require(data['engine_tree']==os.environ.get('ENGINE_TREE'),'original tree pin differs')
require(hashlib.sha256(p.read_bytes()).hexdigest()==os.environ.get('ENGINE_BUNDLE_SHA256'),'source-manifest pin differs')
for path,digest in data['files'].items():
 q=R/path
 require(q.is_file() and not q.is_symlink(),'missing or symlink source: '+path)
 require(hashlib.sha256(q.read_bytes()).hexdigest()==digest,'source drift: '+path)
actual={str(p.relative_to(R)) for root in ['engine','tools/grammar','tools/afd-prep'] for p in (R/root).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
require(actual==set(data['files']),'source closure differs')
print('SOURCE VERIFIED',len(actual),'files; original engine',data['engine_revision'])
