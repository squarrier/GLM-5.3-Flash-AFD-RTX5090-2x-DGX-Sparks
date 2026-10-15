#!/usr/bin/env python3
"""Offline release checks; fake transports are explicit, never hardware results."""
from pathlib import Path
import ast,hashlib,json,os,re,subprocess,sys,tempfile,py_compile,unittest
R=Path(__file__).resolve().parents[1]
BASE={'PATH':os.environ['PATH'],'HOME':os.environ.get('HOME','/srv/afd'),'LANG':'C.UTF-8'}
TEMP=Path(os.environ.get('STATIC_TEST_TEMP',str(R/'.build/static'))); TEMP.mkdir(parents=True,exist_ok=True)
class ReleaseTests(unittest.TestCase):
 def call(self,*args,env=None,good=True):
  p=subprocess.run(list(args),env={**BASE,**(env or {})},capture_output=True,text=True,timeout=30,cwd=R)
  if good: self.assertEqual(p.returncode,0,p.stderr+p.stdout)
  else: self.assertNotEqual(p.returncode,0,p.stderr+p.stdout)
  return p
 def test_01_shell_syntax(self):
  paths=list(R.rglob('*.sh')); self.assertGreater(len(paths),5)
  for p in paths:
   if '.git' not in p.parts: self.call('bash','-n',str(p))
 def test_02_python_syntax(self):
  for p in R.rglob('*.py'):
   if '.git' in p.parts or '.build' in p.parts: continue
   py_compile.compile(str(p),cfile=str(TEMP/(hashlib.sha256(str(p.relative_to(R)).encode()).hexdigest()+'.pyc')),doraise=True)
 def test_03_topology_and_flags(self):
  p=self.call('bash',str(R/'start.sh'),'attention','print')
  for x in ['TF_AFD_USERS=4','TF_AFD_POOL_TOKENS=720896','TF_AFD_USERS_INTERLEAVE=1','TF_GLM_EXL3_SEQ=1','--context 262144','--drafts auto','TF_AFD_USERS_SHARED=1','TF_AFD_USERS_SHARED_DRAFTS=1','TF_AFD_USERS_BATCHED_DRAFTS=1','TF_AFD_USERS_PING_PONG=1','TF_AFD_GRAMMAR_MASK=1','TF_AFD_USERS_SAMPLING=1']: self.assertIn(x,p.stdout)
  self.assertNotIn('TF_AFD_INFLIGHT=',p.stdout)
 def test_04_expert_halves(self):
  for role,half in [('expert0','0'),('expert1','1')]:
   p=self.call('bash',str(R/'start.sh'),role,'print'); self.assertIn('--half '+half,p.stdout); self.assertIn('/set121',p.stdout); self.assertNotIn('TF_AFD_USERS=',p.stdout)
 def test_05_caller_exports_win(self):
  p=self.call('bash',str(R/'start.sh'),'attention','print',env={'API_PORT':'8123','TF_AFD_USERS_SHARED':'1','TF_AFD_USERS_SHARED_DRAFTS':'1'})
  self.assertIn('--port 8123',p.stdout); self.assertIn('TF_AFD_USERS_SHARED=1',p.stdout); self.assertIn('TF_AFD_USERS_SHARED_DRAFTS=1',p.stdout)
  p=self.call('bash',str(R/'start.sh'),'expert0','print',env={'TF_GLM_EXL3_X1':'0'})
  self.assertIn('TF_GLM_EXL3_X1=0',p.stdout)
  self.assertIn('TF_GLM_EXL3_PGROUP=0',p.stdout); self.assertIn('TF_GLM_EXL3_MMA=0',p.stdout)
 def test_06_fake_native_arg_env(self):
  with tempfile.TemporaryDirectory(dir=TEMP) as d:
   d=Path(d); exe=d/'native'; log=d/'argv.json'
   exe.write_text('#!/usr/bin/env python3\nimport json,os,sys\nfrom pathlib import Path\nPath('+repr(str(log))+').write_text(json.dumps({"args":sys.argv[1:],"env":dict(os.environ)}))\n'); exe.chmod(0o700)
   for role in ['attention','expert0','expert1']:
    self.call('bash',str(R/'start.sh'),role,'check',env={'ATTENTION_BIN':str(exe),'EXPERT_BIN':str(exe),'TF_AFD_GRAPHS':'1'})
    data=json.loads(log.read_text()); self.assertEqual(data['args'][0],'check-flags'); self.assertNotIn('TF_AFD_GRAPHS',data['env'])
    if role=='attention': self.assertEqual(data['env']['TF_AFD_USERS'],'4')
 def test_07_bad_role_and_mode(self):
  self.call('bash',str(R/'start.sh'),'all','print',good=False)
  self.call('bash',str(R/'start.sh'),'attention','up',good=False)
 def test_08_documentation_address_refused(self):
  self.call('bash',str(R/'start.sh'),'attention','run',good=False)
 def test_09_nonloopback_without_key_refused(self):
  p=self.call('bash',str(R/'start.sh'),'attention','run',env={'FABRIC_ADDRESS':'127.0.0.1','API_HOST':'0.0.0.0'},good=False)
  self.assertIn('requires API_KEY_FILE',p.stderr)
 def test_10_keyfile_mode_and_wiring(self):
  with tempfile.TemporaryDirectory(dir=TEMP) as d:
   d=Path(d); key=d/'dummy-key'; key.write_text('CPU-ONLY-NOT-A-REAL-KEY\n'); key.chmod(0o644)
   p=self.call('bash',str(R/'start.sh'),'attention','run',env={'FABRIC_ADDRESS':'127.0.0.1','API_KEY_FILE':str(key)},good=False); self.assertIn('0600',p.stderr)
   p=self.call('bash',str(R/'start.sh'),'attention','print',env={'API_KEY_FILE':str(key)})
   self.assertIn('--api-key-file '+str(key),p.stdout); self.assertNotIn('CPU-ONLY-NOT-A-REAL-KEY',p.stdout)
 def test_11_no_distributed_objects(self):
  forbidden={'.cubin','.fatbin','.so','.safetensors','.gguf','.a'}
  found=[str(p.relative_to(R)) for p in R.rglob('*') if p.is_file() and '.git' not in p.parts and '.build' not in p.parts and (p.suffix in forbidden or p.read_bytes()[:4]==b'\x7fELF')]
  self.assertEqual(found,[])
 def test_12_licence_and_guide(self):
  self.assertEqual(hashlib.sha256((R/'AGENTS.md').read_bytes()).hexdigest(),'81a6dd958181aa1d581cb24d54d30a327842e955bd42b40256e54bfc1abfa8cb')
  self.assertEqual((R/'LICENSE').read_bytes(),(R/'engine/LICENSE').read_bytes())
  self.assertEqual(hashlib.sha256((R/'tools/afd-prep/0005-glm-dense-q4.patch').read_bytes()).hexdigest(),'da13106efa1e2d4ef77e289f60ceba1bcbe1139cabec1595cac1eb6fa73bf91f')
 def test_13_readme_no_false_soak_claim(self):
  text=(R/'README.md').read_text(); self.assertIn('Soak and restart qualification',text); self.assertIn('9/11',text); self.assertIn('23/30',text); self.assertIn('400 W',text); self.assertIn('2,200 MHz',text)
  self.assertLess(text.index('Ash Hart'),text.index('## Quick start'))
 def test_14_download_dry(self):
  p=self.call('bash',str(R/'download.sh'),'print'); self.assertIn('078455ffe6472f9a52fbc1139f58b9db2881b25c',p.stdout); self.assertNotIn('DFlash2',p.stdout)
 def test_17_source_manifest_closure(self):
  pins=dict(re.findall(r'^(ENGINE_(?:REVISION|TREE|BUNDLE_SHA256))=(\S+)$',(R/'.env.example').read_text(),re.M))
  p=self.call(sys.executable,'-B',str(R/'scripts/verify-source.py'),env=pins)
  self.assertIn('SOURCE VERIFIED',p.stdout)
 def test_18_json_syntax(self):
  for p in R.rglob('*.json'):
   if '.git' in p.parts or '.build' in p.parts: continue
   if str(p.relative_to(R))=='engine/zig/tests/server/golden/lifecycle/request-log.json':
    # This upstream .json fixture is deliberately a JSONL request log.
    for line in p.read_text().splitlines():
     if line.strip(): json.loads(line)
   else: json.loads(p.read_text())
 def test_16_integrity_not_assert_only(self):
  env={'PYTHONOPTIMIZE':'1','ENGINE_REVISION':'wrong','ENGINE_TREE':'wrong','ENGINE_BUNDLE_SHA256':'wrong'}
  p=self.call(sys.executable,'-B',str(R/'scripts/verify-source.py'),env=env,good=False)
  self.assertIn('engine pin differs',p.stderr)
 def test_15_real_native_flags(self):
  native=os.environ.get('STATIC_NATIVE_BIN')
  if not native: self.skipTest('native build not supplied; no hardware claimed')
  for role in ['attention','expert0','expert1']:
   self.call('bash',str(R/'start.sh'),role,'check',env={'ATTENTION_BIN':native,'EXPERT_BIN':native})
if __name__=='__main__':
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReleaseTests))
 print('STATIC_RESULT '+json.dumps({'run':result.testsRun,'passed':result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),'failed':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped)}))
 sys.exit(not result.wasSuccessful())
