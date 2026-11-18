import json,os,subprocess,tempfile,time,unittest,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Lifecycle(unittest.TestCase):
 def test_real_cpu_child_is_started_owned_and_stopped_once(self):
  with tempfile.TemporaryDirectory()as td:
   tmp=Path(td);install=tmp/'install';b=install/'native/bin/tensorfold-native';b.parent.mkdir(parents=True)
   b.write_text('#!'+sys.executable+'\nimport os,signal,time,json,sys\nfrom pathlib import Path\np=Path(os.environ["TMPDIR"])/"fixture.json"\np.write_text(json.dumps({"argv":sys.argv,"dense":os.environ["TF_GLM_DENSE_BITS"],"tile":os.environ["TF_GLM_ATTENTION_TILE_ROWS"]}))\ndef done(s,f):\n p.with_suffix(".stopped").write_text(str(s));sys.exit(0)\nsignal.signal(signal.SIGTERM,done)\nwhile True:time.sleep(.05)\n');b.chmod(0o700)
   model=tmp/'model';model.mkdir();auth=tmp/'key';auth.write_text('cpu-fixture-only\n');auth.chmod(0o600)
   out=tmp/'runtime';subprocess.run(['python3',str(ROOT/'scripts/render.py'),'--out',str(out),'--install',str(install),'--model',str(model),'--auth',str(auth),'--session','00000000000000000000000000000001'],check=True,capture_output=True)
   self.assertTrue((ROOT/'start.sh').is_file(),'local start entrypoint missing')
   p=subprocess.Popen(['bash',str(ROOT/'start.sh'),str(out/'attention.launch.json'),'--ack-local-resources'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
   try:
    end=time.monotonic()+8
    while time.monotonic()<end and not(out/'fixture.json').exists() and p.poll()is None:time.sleep(.05)
    self.assertTrue((out/'fixture.json').exists(),p.communicate(timeout=1)if p.poll()is not None else 'no actual child')
    duplicate=subprocess.run(['bash',str(ROOT/'start.sh'),str(out/'attention.launch.json'),'--ack-local-resources'],capture_output=True,timeout=3)
    self.assertNotEqual(duplicate.returncode,0)
    stopped=subprocess.run(['bash',str(ROOT/'stop.sh'),str(out/'attention.launch.json')],capture_output=True,timeout=8)
    self.assertEqual(stopped.returncode,0,stopped.stderr)
    raw=p.communicate(timeout=8);self.assertEqual(p.returncode,0,raw)
    self.assertEqual((out/'fixture.stopped').read_text(),'15')
    self.assertFalse((out/'attention.owner.json').exists());self.assertFalse((out/'attention.sock').exists())
    data=json.loads((out/'fixture.json').read_text());self.assertEqual(data['argv'][-2:],['--mtp-confidence','0.50']);self.assertEqual(data['tile'],'16');self.assertEqual(data['dense'],'16')
   finally:
    if p.poll()is None:p.terminate();p.communicate(timeout=8)
if __name__=='__main__':unittest.main(verbosity=2)
