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
  print('DAS_DEPENDENCY_ASSEMBLE',name,version,flush=True)
  temporary=target.with_suffix(target.suffix+'.assembling')
  with temporary.open('wb') as out:
   for part in item['local_parts']:
    source=root/'third_party/vendor-wheels'/part['file']
    assert source.is_file(),f'Missing bundled dependency: {source}'
    assert source.stat().st_size==part['bytes'] and digest(source)==part['sha256'],source
    with source.open('rb') as response:
     while b:=response.read(1024*1024):out.write(b)
  assert digest(temporary)==item['sha256'],temporary
  # Use copy instead of rename on virtual filesystems that reject rename with EXDEV.
  import shutil
  shutil.copyfile(temporary,target);temporary.unlink()
 assert digest(target)==item['sha256'],target
 subprocess.run([sys.executable,str(root/'scripts/pip_scoped.py'),'install','--no-deps',str(target)],check=True)
 assert importlib.metadata.version(name)==version
 print('DAS_DEPENDENCY_INSTALLED',name,version,flush=True)
