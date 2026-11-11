from pathlib import Path
import json,subprocess,sys,hashlib,os
base=Path(__file__).resolve().parent
if os.name=='nt':base=Path('\\\\?\\'+str(base))
root=base/'implementation'
repos=json.loads((base/'repositories.lock.json').read_text())
for name in ['InfiniCore','InfiniLM','NineToothed','Diffusers']:
 r=repos[name];p=root/r['path']
 manifest=base/'vendored-sources.json'
 if manifest.is_file() and p.is_dir() and not (p/'.git').exists():
  files=json.loads(manifest.read_text())[name]
  for rel,digest in files.items():
   assert hashlib.sha256((p/rel).read_bytes()).hexdigest()==digest, str(p/rel)
  print(name,r['commit'],'verified',len(files),'source files')
  continue
 if not (p/'.git').exists():
  p.parent.mkdir(parents=True,exist_ok=True)
  subprocess.run(['git','-c','core.longpaths=true','clone','--branch',r['branch'],r['url']+'.git',str(p)],check=True)
 if subprocess.check_output(['git','-c','core.longpaths=true','-C',str(p),'status','--porcelain']).strip():
  raise RuntimeError('Modified checkout: '+str(p))
 subprocess.run(['git','-c','core.longpaths=true','-C',str(p),'checkout','--detach',r['commit']],check=True)
 assert subprocess.check_output(['git','-c','core.longpaths=true','-C',str(p),'rev-parse','HEAD'],text=True).strip()==r['commit']
 print(name,r['commit'])
