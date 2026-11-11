"""NineToothed split-reduction RMS plus default FP32-intermediate RoPE."""
import functools
import ninetoothed as nt
from ninetoothed import Tensor
import ninetoothed.language as ntl
from mrope_full import arrange as rope_arrange


def arrange(x,weight,eps,cos,sin,out,rows):
    x,c,s,out = rope_arrange(x,cos,sin,out,rows)
    w = weight.tile((128,)).unsqueeze(0).unsqueeze(0).expand(x.shape)
    return x,w,eps,c,s,out


def apply(x,weight,eps,cos,sin,out):
    a = x
    c = cos
    s = sin
    a = a.to(ntl.float32)
    c = c.to(ntl.float32)
    s = s.to(ntl.float32)
    squared = a*a
    left = ntl.where(x.offsets(-1)<64,squared,0.)
    right = ntl.where(x.offsets(-1)>=64,squared,0.)
    mean = (ntl.sum(left,1)+ntl.sum(right,1))/128
    epsilon = eps
    epsilon = epsilon.to(ntl.float32)
    reciprocal = ntl.rsqrt(mean+epsilon)
    normalized = a*reciprocal[:,None]
    a = (normalized*weight).to(x.dtype).to(ntl.float32)
    index = ntl.arange(0,128)
    rotated = ntl.where(index<48,index+48,ntl.where(index<96,index-48,index))
    indices = ntl.broadcast_to(rotated[None,:],a.shape)
    b = ntl.gather(a,indices,1)
    c = ntl.gather(c,index%48,0)
    s = ntl.gather(s,index%48,0)
    sign = ntl.where(index<48,-1.,1.)
    out = ntl.where(index[None,:]<96,a*c[None,:]+sign[None,:]*b*s[None,:],a)


def make(caller='hip',output_dir=None,archs=None,rows=8,warps=1,
         kernel_name='h3_qk_rms_rope_fp32'):
    a = lambda:Tensor(shape=(None,None,128),dtype=nt.bfloat16)
    angle = lambda:Tensor(shape=(None,48),dtype=nt.bfloat16)
    return nt.make(functools.partial(arrange,rows=rows),apply,
        (a(),Tensor(shape=(128,),dtype=nt.bfloat16),Tensor(0,dtype=nt.float64),angle(),angle(),a()),
        caller=caller,archs=archs,output_dir=output_dir,kernel_name=kernel_name,num_warps=warps,num_stages=1)
