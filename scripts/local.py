#!/usr/bin/env python3
"""Own one local foreground child. No SSH, device probing, restart or forced kill."""
import argparse,fcntl,hashlib,json,os,select,signal,socket,stat,struct,subprocess,sys,time
from pathlib import Path

def trusted(p,private=False):
 s=p.lstat()
 if not stat.S_ISREG(s.st_mode) or s.st_uid!=os.getuid() or s.st_mode&0o022 or (private and s.st_mode&0o077):raise ValueError('untrusted file mode/owner/type')
 return s

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['run','stop']);p.add_argument('launch',type=Path);p.add_argument('--ack-local-resources',action='store_true');a=p.parse_args()
 trusted(a.launch,True);cfg=json.loads(a.launch.read_text());root=Path(cfg['state_dir'])
 s=root.lstat()
 if not stat.S_ISDIR(s.st_mode) or s.st_uid!=os.getuid() or s.st_mode&0o077:raise ValueError('untrusted state directory')
 role=cfg['role']
 if role not in ('attention','expert0','expert1'):raise ValueError('bad role')
 sockpath=root/(role+'.sock');record=root/(role+'.owner.json')
 # Linux abstract Unix address avoids pathname-length limits without an insecure alias.
 address='\0afd-'+hashlib.sha256(str(root.resolve()/role).encode()).hexdigest()[:40]
 if a.action=='stop':
  trusted(record,True)
  with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)as c:
   c.settimeout(35);c.connect(address)
   if struct.unpack('3i',c.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))[1]!=os.getuid():raise ValueError('foreign controller')
   c.sendall(b'STOP\n');reply=c.recv(128)
   if reply!=b'STOPPED\n':raise RuntimeError('exact child closure not confirmed')
   print('Exact local child exited; no force/restart.');return
 if not a.ack_local_resources:p.error('separate local resource/ownership admission must be acknowledged')
 trusted(Path(cfg['config']),True)
 exe=Path(cfg['argv'][0]);trusted(exe)
 if not exe.is_absolute() or not os.access(exe,os.X_OK):raise ValueError('binary must be absolute/executable')
 if not Path(cfg['model']).is_dir():raise ValueError('checkpoint directory required')
 if cfg['auth']:
  k=trusted(Path(cfg['auth']),True)
  if not 1<=k.st_size<=4096:raise ValueError('invalid auth file size')
 lock=os.open(root/(role+'.lock'),os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if record.exists()or record.is_symlink():raise ValueError('unknown previous owner record; refusing')
 requested=False;sent=False;child=None;pidfd=None;listener=None;clients=[]
 def request(*_):
  nonlocal requested;requested=True
 signal.signal(signal.SIGTERM,request);signal.signal(signal.SIGINT,request)
 try:
  listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);listener.bind(address);listener.listen(2);listener.setblocking(False)
  child=subprocess.Popen(cfg['argv'],env=cfg['env'],stdin=subprocess.DEVNULL,close_fds=True)
  # Registration and pidfd creation precede fallible receipt publication.
  try:pidfd=os.pidfd_open(child.pid)
  except ProcessLookupError:
   if child.poll()is None:raise
  fd=os.open(record,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
  with os.fdopen(fd,'w')as f:json.dump(dict(pid=child.pid,role=role,launch_sha256=hashlib.sha256(a.launch.read_bytes()).hexdigest()),f);f.flush();os.fsync(f.fileno())
  while child.poll()is None:
   ready,_,_=select.select([listener],[],[],.05)
   if ready:
    c,_=listener.accept();c.settimeout(1)
    if struct.unpack('3i',c.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))[1]==os.getuid() and c.recv(16)==b'STOP\n':requested=True;clients.append(c)
    else:c.close()
   if requested and not sent and child.poll()is None:
    if pidfd is None:raise RuntimeError('missing child pidfd; ownership retained')
    signal.pidfd_send_signal(pidfd,signal.SIGTERM);sent=True
  for c in clients:
   try:c.sendall(b'STOPPED\n')
   except (BrokenPipeError,ConnectionResetError):pass
  returncode=child.returncode
 finally:
  # Never exit the foreground owner with a live native child, including on publication errors.
  if child is not None:
   if child.poll()is None and not sent:
    if pidfd is not None:signal.pidfd_send_signal(pidfd,signal.SIGTERM);sent=True
    # Popen identity is our direct unreaped child; fallback only if pidfd_open itself failed.
    else:child.terminate();sent=True
   child.wait()
  if pidfd is not None:os.close(pidfd)
  if listener is not None:listener.close()
  for c in clients:c.close()
  if child is not None and record.exists():record.unlink()
  os.close(lock)
 sys.exit(returncode)
if __name__=='__main__':
 try:main()
 except Exception as e:print(type(e).__name__+': '+str(e),file=sys.stderr);sys.exit(1)
