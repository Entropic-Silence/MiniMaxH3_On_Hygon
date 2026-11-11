"""NineToothed full-width load/store candidate for H3's 96/128 rotation."""
import ninetoothed.language as ntl


def arrange(x,cos,sin,out,rows):
    x = x.tile((1,rows,128))
    x.dtype = x.dtype.squeeze(0)
    out = out.tile((1,rows,128))
    out.dtype = out.dtype.squeeze(0)
    angles = []
    for angle in (cos,sin):
        a = angle.tile((1,128))
        a.dtype = a.dtype.squeeze(0)
        angles.append(a.unsqueeze(1).expand((-1,x.shape[1],-1)))
    return x,*angles,out


def apply(x,cos,sin,out):
    a = x
    c = cos
    s = sin
    a = a.to(ntl.float32)
    c = c.to(ntl.float32)
    s = s.to(ntl.float32)
    index = ntl.arange(0,128)
    rotated = ntl.where(index<48,index+48,ntl.where(index<96,index-48,index))
    indices = ntl.broadcast_to(rotated[None,:],a.shape)
    b = ntl.gather(a,indices,1)
    c = ntl.gather(c,index%48,0)
    s = ntl.gather(s,index%48,0)
    sign = ntl.where(index<48,-1.,1.)
    out = ntl.where(index[None,:]<96,a*c[None,:]+sign[None,:]*b*s[None,:],a)
