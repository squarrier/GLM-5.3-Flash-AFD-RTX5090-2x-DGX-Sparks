import hashlib,json,subprocess,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Integrity(unittest.TestCase):
 def test_exact_source_manifest_and_only_measured_observer_overlays(self):
  m=json.loads((ROOT/'manifests/engine-source.json').read_bytes())
  self.assertEqual(len(m['excluded_payloads']),7)
  self.assertEqual(sum(x['disposition']=='measured_observer557_overlay'for x in m['files']),2)
  for x in m['files']:
   with self.subTest(path=x['path']):
    b=(ROOT/x['path']).read_bytes();self.assertEqual(len(b),x['bytes']);self.assertEqual(hashlib.sha256(b).hexdigest(),x['sha256'])
  for x in m['excluded_payloads']:self.assertFalse((ROOT/x['path']).exists())
 def test_canonical_credit_template_and_first_screen_authorship(self):
  self.assertEqual(hashlib.sha256((ROOT/'AGENTS.md').read_bytes()).hexdigest(),'81a6dd958181aa1d581cb24d54d30a327842e955bd42b40256e54bfc1abfa8cb')
  text=(ROOT/'README.md').read_text();above=text.split('## Install and build')[0];credits=text.split('## License and credits')[1]
  for who in ('Ash Hart','Russell Marr','Bizuayeu','Mia','Hugh Madden','T.J. Purtell','taussoe','Turboderp','Z.ai','NVIDIA','Scott Quarrier','Raymond Lucke','Albert Lee','SvangenStudios','Alfonso Fortunato'):
   self.assertIn(who,above);self.assertIn(who,credits)
  self.assertIn('authored',above)
 def test_invalid_session_and_existing_generation_are_refused(self):
  with tempfile.TemporaryDirectory()as td:
   root=Path(td)/'runtime'
   q=['python3',str(ROOT/'scripts/render.py'),'--out',str(root)]
   bad=subprocess.run(q+['--session','0'*32],capture_output=True);self.assertNotEqual(bad.returncode,0);self.assertFalse(root.exists())
   subprocess.run(q,check=True,capture_output=True);before=(root/'attention.json').read_bytes()
   repeated=subprocess.run(q,capture_output=True);self.assertNotEqual(repeated.returncode,0);self.assertEqual((root/'attention.json').read_bytes(),before)
 def test_config_modes_and_all_role_settings_match(self):
  with tempfile.TemporaryDirectory()as td:
   out=Path(td)/'runtime';subprocess.run(['python3',str(ROOT/'scripts/render.py'),'--out',str(out)],check=True,capture_output=True)
   configs=[json.loads((out/(r+'.json')).read_bytes())for r in('attention','expert0','expert1')]
   self.assertEqual({c['session']for c in configs},{configs[0]['session']})
   for c in configs:
    self.assertEqual(c['settings'],configs[0]['settings']);self.assertEqual(c['served_context'],479232);self.assertEqual(c['resident_users'],8)
   for p in out.iterdir():self.assertEqual(p.stat().st_mode&0o777,0o600)
 def test_no_compiled_or_checkpoint_artifacts(self):
  for p in ROOT.rglob('*'):
   if p.is_file()and '.git'not in p.parts:
    self.assertNotIn(p.suffix,('.safetensors','.fatbin','.cubin','.so','.o','.a','.pt'))
if __name__=='__main__':unittest.main(verbosity=2)
