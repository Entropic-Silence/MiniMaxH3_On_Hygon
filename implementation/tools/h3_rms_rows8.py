"""Diagnostic: two independent 64-element sums in a single-warp RMS128."""
import ninetoothed.language as ntl

def arrange(x,weight,eps,out,rows):
    a=x.tile((1,rows,128))
    a.dtype=a.dtype.squeeze(0)
    o=out.tile((1,rows,128))
    o.dtype=o.dtype.squeeze(0)
    w=weight.tile((128,)).unsqueeze(0).unsqueeze(0).expand(a.shape)
    return a,w,eps,o

def apply(x,weight,eps,out):
    value=x
    value=value.to(ntl.float32)
    squared=value*value
    squared=ntl.where(x.offsets(-1)<x.source.shape[-1],squared,0.)
    left=ntl.where(x.offsets(-1)<64,squared,0.)
    right=ntl.where(x.offsets(-1)>=64,squared,0.)
    mean=(ntl.sum(left,1)+ntl.sum(right,1))/x.source.shape[-1]
    epsilon=eps
    epsilon=epsilon.to(ntl.float32)
    reciprocal=ntl.rsqrt(mean+epsilon)
    normalized=value*reciprocal[:,None]
    out=normalized*weight
