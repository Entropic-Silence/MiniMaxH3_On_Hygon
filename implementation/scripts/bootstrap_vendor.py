"""Install the pinned DAS compiler and attention wheels in the build environment."""
from pathlib import Path
import hashlib,importlib.metadata,json,os,subprocess,sys,urllib.request
root=Path(__file__).resolve().parents[1]
assert sys.prefix!=sys.base_prefix,'Activate the project virtual environment before installing dependencies'
assert sys.platform=='linux' and sys.version_info[:2]==(3,11),'This runtime lock targets Linux CPython 3.11'
import torch
assert torch.__version__.split('+',1)[0]=='2.9.0',torch.__version__
assert torch.cuda.is_available()
arch=getattr(torch.cuda.get_device_properties(0),'gcnArchName','').split(':',1)[0]
assert arch=='gfx936',arch
print('DAS_BASE',importlib.metadata.version('torch'),torch.__version__,arch,flush=True)
wheelhouse=Path(os.environ.get('H3_WHEELHOUSE',str(root/'third_party/wheels')));wheelhouse.mkdir(parents=True,exist_ok=True)
for item in json.loads((root/'configs/vendor-packages.json').read_text()):
 name,version=item['file'].split('-')[:2]
 try:installed=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:installed=None
 if installed==version:
  print('DAS_DEPENDENCY_READY',name,version,flush=True);continue
 target=wheelhouse/item['file']
 def digest(path):
  h=hashlib.sha256()
  with path.open('rb') as f:
   for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
  return h.hexdigest()
 if not target.exists() or digest(target)!=item['sha256']:
  print('DAS_DEPENDENCY_DOWNLOAD',name,version,flush=True)
  with target.open('wb') as out:
   for part in item.get('parts',[{'url':item.get('url')}]):
    h=hashlib.sha256();count=0
    with urllib.request.urlopen(part['url'],timeout=120) as response:
     while b:=response.read(1024*1024):out.write(b);h.update(b);count+=len(b)
    if 'sha256' in part:assert h.hexdigest()==part['sha256'] and count==part['bytes']
    print('DAS_DEPENDENCY_TRANSFER',name,count,flush=True)
 assert digest(target)==item['sha256'],target
 subprocess.run([sys.executable,str(root/'scripts/pip_scoped.py'),'install','--no-deps',str(target)],check=True)
 assert importlib.metadata.version(name)==version
 print('DAS_DEPENDENCY_INSTALLED',name,version,flush=True)
