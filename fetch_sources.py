from pathlib import Path
import json,subprocess,sys
base=Path(__file__).resolve().parent;root=base/'implementation'
repos=json.loads((base/'repositories.lock.json').read_text())
for name in ['InfiniCore','InfiniLM','NineToothed','Diffusers']:
 r=repos[name];p=root/r['path']
 if not (p/'.git').exists():
  p.parent.mkdir(parents=True,exist_ok=True)
  subprocess.run(['git','-c','core.longpaths=true','clone','--branch',r['branch'],r['url']+'.git',str(p)],check=True)
 if subprocess.check_output(['git','-c','core.longpaths=true','-C',str(p),'status','--porcelain']).strip():
  raise RuntimeError('Modified checkout: '+str(p))
 subprocess.run(['git','-c','core.longpaths=true','-C',str(p),'checkout','--detach',r['commit']],check=True)
 assert subprocess.check_output(['git','-c','core.longpaths=true','-C',str(p),'rev-parse','HEAD'],text=True).strip()==r['commit']
 print(name,r['commit'])
