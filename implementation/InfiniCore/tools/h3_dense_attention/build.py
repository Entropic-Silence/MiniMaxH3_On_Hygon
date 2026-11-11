"""Build the gfx936 dense BF16 attention library from the bundled, pinned sources."""
import argparse,hashlib,json,os,shutil,struct,subprocess,tarfile,tempfile
from pathlib import Path

here=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=here.parents[2]/'InfiniLM/python/infinilm/lib/libh3_dense_attention.so')
args=parser.parse_args();output=args.output.resolve()
manifest=json.loads((here/'source-manifest.json').read_text())
manifest_sha=hashlib.sha256((here/'source-manifest.json').read_bytes()).hexdigest()
builder_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
if output.exists():
    receipt=json.loads(output.with_suffix('.build.json').read_text())
    assert receipt['complete'] and receipt['exit_code']==0
    assert receipt['source_manifest_sha256']==manifest_sha and receipt['builder_sha256']==builder_sha
    assert hashlib.sha256(output.read_bytes()).hexdigest()==receipt['sha256']
    print('Existing attention library matches its source and build receipt:',output)
    raise SystemExit(0)
output.parent.mkdir(parents=True,exist_ok=True)
prepared=here/'prepared'
compiler=shutil.which('hipcc');assert compiler,'Source the DTK environment before building'
import torch
from torch.utils.hipify import hipify_python
torch_root=Path(torch.__file__).resolve().parent;dtk=Path(os.environ.get('ROCM_PATH','/opt/dtk'))
record={'source_manifest_sha256':manifest_sha,'builder_sha256':builder_sha,'gpu_verified':False,'production_modified':False}
with tempfile.TemporaryDirectory(prefix='h3-dense-build-',dir=Path.cwd()) as temp:
    work=Path(temp).resolve();assert work.is_relative_to(Path.cwd().resolve())
    source=work/'input';source.mkdir()
    shutil.copytree(prepared,source,dirs_exist_ok=True)
    for name,digest in manifest['prepared_files'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
    converted=hipify_python.hipify(project_directory=str(source),output_directory=str(work/'hip'),includes=['*'],header_include_dirs=[str(source/'flash'),str(source/'cutlass/include')],is_pytorch_extension=True,show_detailed=False,show_progress=False)
    kernels=[Path(v.hipified_path) for k,v in converted.items() if Path(k).name=='kernel.cu' and v.hipified_path]
    assert len(kernels)==1 and kernels[0].is_file()
    w=work/'hip'
    flags=['-x','hip','-std=c++17','-O3','-shared','-fPIC','--offload-arch=gfx936','-DDCU_ASM','-DBF162FLOAT_USE_ASM','-DHIP_ENABLE_WARP_SYNC_BUILTINS','-DFLASHATTENTION_DISABLE_SOFTCAP','-D__HIP_PLATFORM_AMD__','-DC10_CUDA_NO_CMAKE_CONFIGURE_FILE','-DTORCH_CUDA_CPP_API=TORCH_HIP_CPP_API','-ffast-math','-mno-fma','-fno-finite-math-only','-mcode-object-version=5','--gpu-max-threads-per-block=256','-Rpass-analysis=kernel-resource-usage','-fno-associative-math','-DUSE_ROCM=1','-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP','-mllvm=-enable-num-vgprs-512=true']
    flags+=['-I'+str(p) for p in [w/'flash',w/'cutlass/include',torch_root/'include',torch_root/'include/torch/csrc/api/include',dtk/'include',dtk/'hip/include']]
    flags+=['-mllvm=-disable-machine-sink','-mllvm=-amdgpu-enable-rewrite-partial-reg-uses=false','-mllvm=-amdgpu-early-inline-all=true','-mllvm=-amdgpu-function-calls=false','-mllvm=-unroll-threshold=10000']
    flags+=['-DH3_CACHE_SWIZZLE=0','-DH3_SUM_FIX=1','-DH3_IDENTITY_RESCALE=1','-DH3_VENDOR_REDUCE=0','-DH3_LONG_CONTEXT=0','-DH3_BLOCK_M=128','-DH3_STAGES=3','-DFLASH_ATTENTION_BF16_TYPE=1']
    links=['-L'+str(torch_root/'lib'),'-Wl,-rpath,'+str(torch_root/'lib'),'-ltorch_hip','-lc10_hip','-ltorch_cpu','-lc10','-L'+str(dtk/'lib'),'-lamdhip64']
    command=[compiler,*flags,str(kernels[0]),'-o',str(output),*links]
    log=output.with_suffix('.build.log')
    with log.open('w') as f:result=subprocess.run(command,cwd=w,stdout=f,stderr=subprocess.STDOUT,timeout=600)
    record['command']=command;record['exit_code']=result.returncode
    if result.returncode:raise RuntimeError(log.read_text()[-10000:])
raw=output.read_bytes();magic=b'__CLANG_OFFLOAD_BUNDLE__';at=raw.find(magic)
assert at>=0,'No GPU code emitted'
count=struct.unpack_from('<Q',raw,at+len(magic))[0];cursor=at+len(magic)+8;images=[]
for _ in range(count):
    offset,size,length=struct.unpack_from('<QQQ',raw,cursor);cursor+=24
    target=raw[cursor:cursor+length].decode();cursor+=length;image=raw[at+offset:at+offset+size]
    if 'gfx936' in target and image.startswith(b'\x7fELF') and b'h3_dense_tile_kernel' in image:
        images.append({'target':target,'bytes':size,'sha256':hashlib.sha256(image).hexdigest()})
assert images,'No verified gfx936 kernel image'
symbols=subprocess.check_output(['nm','-D','--defined-only',str(output)],text=True)
assert all(name in symbols for name in ['h3_dense_tile_forward','h3_dense_tile_block_m','h3_dense_tile_stages','h3_dense_tile_round','h3_dense_tile_sum_fix','h3_dense_tile_identity_rescale'])
record.update(complete=True,output=str(output),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),device_images=images)
output.with_suffix('.build.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
