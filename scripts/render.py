#!/usr/bin/env python3
"""Portable exact-profile renderer. No payload/secret/device access."""
import argparse,json,os,secrets
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--out',type=Path,required=True);p.add_argument('--session',default=None)
 p.add_argument('--fabric-master',default='10.0.0.10');p.add_argument('--verbs-device',default='rdma0')
 p.add_argument('--model',default='/models/GLM-5.3-Flash');p.add_argument('--install',default='/opt/afd')
 p.add_argument('--auth',default='/secrets/api-key');p.add_argument('--device',type=int,default=0)
 a=p.parse_args();session=a.session or secrets.token_hex(16)
 if len(session)!=32 or any(c not in '0123456789abcdef'for c in session) or int(session,16)==0:p.error('session must be nonzero lowercase32hex')
 for v in (a.model,a.install,a.auth):
  if not Path(v).is_absolute():p.error('model/install/auth must be absolute paths')
 if a.device<0:p.error('device must be nonnegative')
 if not a.fabric_master or not a.verbs_device:p.error('fabric and verbs device required')
 recipe=json.loads((ROOT/'recipe.json').read_text());out=a.out.resolve();out.mkdir(mode=0o700,parents=True,exist_ok=True)
 if out.is_symlink() or out.stat().st_uid!=os.getuid() or out.stat().st_mode&0o077:p.error('output must be owner-controlled0700 directory')
 paths=[out/(role+suffix)for role in recipe['roles']for suffix in('.json','.launch.json')]
 if any(x.exists()or x.is_symlink()for x in paths):p.error('refusing an existing generation')
 for role,v in recipe['roles'].items():
  config=dict(recipe['config'],role=role,session=session,master=[a.fabric_master]*2,verbs_device=a.verbs_device,roce_image=a.install+'/assets/tp_mailbox.fatbin')
  cfg=out/(role+'.json');env=dict(v['env'],HOME='/nonexistent',LANG='C',PATH='/usr/bin:/bin',CUDA_VISIBLE_DEVICES=str(a.device),TMPDIR=str(out),XDG_CACHE_HOME=str(out))
  if role=='attention':
   env.update(TF_GLM_HYBRID_CONFIG=str(cfg),TF_LANE_TRACE=str(out/'occupied-lane.trace.jsonl'))
   argv=[a.install+'/native/bin/tensorfold-native']+[x.replace('{model}',a.model).replace('{auth}',a.auth)for x in recipe['attention_tail']]
  else:argv=[a.install+'/bin/tf-glm-expert','--config',str(cfg),'--checkpoint',a.model,'--device',str(a.device)]
  launch=dict(role=role,argv=argv,env=env,config=str(cfg),auth=a.auth if role=='attention'else None,state_dir=str(out),model=a.model)
  for target,data in ((cfg,config),(out/(role+'.launch.json'),launch)):
   fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
   with os.fdopen(fd,'w')as f:json.dump(data,f,indent=2);f.write('\n')
 print('Rendered three roles; no payloads or devices read.')
if __name__=='__main__':main()
