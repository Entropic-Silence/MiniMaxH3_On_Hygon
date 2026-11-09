import argparse,base64,hashlib,json,os,re,shutil,subprocess,tarfile,tempfile,time,urllib.request,zlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('mode',choices=['reference','native']);a=p.parse_args()
r=Path(os.environ['IFAB_AFS_HOME'])/'h3';tag='puppy-'+a.mode+'-v1';d=r/'diagnostics'/tag
assert not d.exists();d.mkdir(parents=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
seal=json.loads((r/'diagnostics/attention-identity-default-seal-v1.json').read_text());before=(r/'env-owned.sh').read_bytes();assert sha(before)==seal['default_env_sha256'] and seal['adopted']
url='https://raw.githubusercontent.com/Entropic-Silence/MiniMaxH3_On_Hygon/a1ecbf8c5a5304717d1fd56d3633ff0e0c31407b/assets/first_frame.png'
b=urllib.request.urlopen(url,timeout=60).read();assert sha(b)=='c053d4ac8eeb7cf2e96c775b6d073a4f51e5bc64af6a8946be664c876c277427'
image=d/'input.png';image.write_bytes(b)
record={'mode':a.mode,'input_sha256':sha(b),'source_sha256':seal['source_sha256'],'prompt':'A dog falls asleep','steps':50,'frames':124,'seed':42,'width':768,'height':768,'gpus':list(range(8)),'environment_before_sha256':sha(before),'performance_scope':'Unprofiled complete generation; generation.pipeline_total excludes startup and model construction.'}
(d/'inputs.json').write_text(json.dumps(record,indent=2)+'\n')
code=1;started=time.time()
try:
 with tempfile.TemporaryDirectory(prefix='puppy-validation-',dir=Path.cwd()) as temp:
  work=Path(temp);raw=zlib.decompress(base64.b64decode('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'));assert sha(raw)=='a8c3fc5061fe0b939d34c64529e41f4b40ab4bfc4869f973f507e3434e4ff784';(work/'run8.sh').write_bytes(raw)
  env=dict(os.environ,TEXT='A dog falls asleep',IMAGE=str(image),FRAMES='124',H3_SMI_INTERVAL='0')
  args=['bash',str(work/'run8.sh'),tag,'--height','768','--width','768']
  if a.mode=='reference':args+=['--legacy','--legacy-memory-fraction','0.4']
  result=subprocess.run(args,cwd=work,env=env,timeout=1600);code=result.returncode
  assert code==0,f'generation failed: {code}'
finally:
 assert (r/'env-owned.sh').read_bytes()==before
 for suffix,name in [('.log','generation.log'),('.mp4','output.mp4')]:
  source=r/'runs'/(tag+suffix)
  if source.exists():shutil.copyfile(source,d/name)
 record.update(exit_code=code,elapsed_seconds=time.time()-started,environment_unchanged=True)
 if (d/'output.mp4').exists():record['output_sha256']=sha((d/'output.mp4').read_bytes())
 if (d/'generation.log').exists():
  log=(d/'generation.log').read_text(errors='replace');values=re.findall(r'generation\.pipeline_total\s+([0-9.]+)\s+s',log)
  if values:record['pipeline_seconds']=float(values[-1])
 (d/'result.json').write_text(json.dumps(record,indent=2)+'\n')
 archive=d.parent/(tag+'-evidence.tar.gz')
 with tarfile.open(archive,'w:gz') as t:
  for f in sorted(d.iterdir()):t.add(f,arcname=tag+'/'+f.name)
 print('PUPPY_VALIDATION_RESULT',json.dumps(record),flush=True);print('PUPPY_ARCHIVE',archive.name,sha(archive.read_bytes()),flush=True)
raise SystemExit(code)
