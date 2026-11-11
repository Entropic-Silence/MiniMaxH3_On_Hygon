import functools
import ninetoothed as nt
from ninetoothed import Tensor
from qk_norm_rope_full import apply

def arrange(x,weight,eps,cos,sin,out,rows):
    x=x.tile((1,1,rows,128));x.dtype=x.dtype.squeeze(0).squeeze(0)
    out=out.tile((1,1,rows,128));out.dtype=out.dtype.squeeze(0).squeeze(0)
    weight=weight.tile((128,)).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(x.shape)
    angles=[]
    for angle in (cos,sin):
        a=angle.tile((1,128));a.dtype=a.dtype.squeeze(0)
        angles.append(a.unsqueeze(1).unsqueeze(1).expand((-1,x.shape[1],x.shape[2],-1)))
    return x,weight,eps,*angles,out

def make(output_dir,rows=8,*,caller="hip",archs=None):
    x=lambda:Tensor(shape=(None,8,7,128),dtype=nt.bfloat16)
    angle=lambda:Tensor(shape=(None,48),dtype=nt.bfloat16)
    return nt.make(functools.partial(arrange,rows=rows),apply,
        (x(),Tensor(shape=(128,),dtype=nt.bfloat16),Tensor(0,dtype=nt.float64),angle(),angle(),x()),
        caller=caller,output_dir=output_dir,archs=archs or ['gfx936'],kernel_name=f'qk_direct_pack_rows{rows}',num_warps=1,num_stages=1)
