import json,subprocess,tempfile,unittest,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Render(unittest.TestCase):
 def test_selected_native_contract_is_rendered_without_reading_payloads(self):
  with tempfile.TemporaryDirectory() as tmp:
   q=subprocess.run(['python3',str(ROOT/'scripts/render.py'),'--out',tmp,'--session','00000000000000000000000000000001'],capture_output=True,text=True)
   self.assertEqual(q.returncode,0,q.stderr)
   launch=json.loads((Path(tmp)/'attention.launch.json').read_text());config=json.loads((Path(tmp)/'attention.json').read_text())
   self.assertEqual(launch['argv'][-2:],['--mtp-confidence','0.50'])
   self.assertEqual(config['served_context'],479232);self.assertEqual(config['resident_users'],8)
   self.assertEqual(config['settings']['kv_host_gib'],46.69189453125)
   self.assertEqual(launch['env']['TF_GLM_ATTENTION_TILE_ROWS'],'16')
   self.assertEqual(launch['env']['TF_GLM_DENSE_BITS'],'16');self.assertEqual(launch['env']['TF_GLM_HEAD_BITS'],'16')
   self.assertEqual(config['settings']['kv8'],False)
if __name__=='__main__':unittest.main(verbosity=2)
