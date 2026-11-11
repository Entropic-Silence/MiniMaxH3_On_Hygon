import functools
import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed import Tensor

def arrange(q,k,v,wq,wk,epsq,epsk,cos,sin,qo,ko,vo,rows):
    tensors=[]
    for x in (q,k,v,qo,ko,vo):
        x=x.tile((rows,1,8,128));x.dtype=x.dtype.squeeze(1)
        tensors.append(x)
    q,k,v,qo,ko,vo=tensors
    weights=[]
    for w in (wq,wk):
        weights.append(w.tile((128,)).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(q.shape))
    angles=[]
    for angle in (cos,sin):
        a=angle.tile((rows,128))
        angles.append(a.unsqueeze(1).unsqueeze(1).expand((-1,q.shape[1],q.shape[2],-1)))
    return q,k,v,*weights,epsq,epsk,*angles,qo,ko,vo

def apply(q, k, v, wq, wk, epsq, epsk, cos, sin, qo, ko, vo):
    q_a = q
    q_c = cos
    q_s = sin
    q_a = q_a.to(ntl.float32)
    q_c = q_c.to(ntl.float32)
    q_s = q_s.to(ntl.float32)
    q_squared = q_a * q_a
    q_left = ntl.where(q.offsets(-1) < 64, q_squared, 0.0)
    q_right = ntl.where(q.offsets(-1) >= 64, q_squared, 0.0)
    q_mean = (ntl.sum(q_left, 2) + ntl.sum(q_right, 2)) / 128
    q_epsilon = epsq
    q_epsilon = q_epsilon.to(ntl.float32)
    q_reciprocal = ntl.rsqrt(q_mean + q_epsilon)
    q_normalized = q_a * q_reciprocal[:, :, None]
    q_a = (q_normalized * wq).to(q.dtype).to(ntl.float32)
    q_index = ntl.arange(0, 128)
    q_rotated = ntl.where(q_index < 48, q_index + 48, ntl.where(q_index < 96, q_index - 48, q_index))
    q_indices = ntl.broadcast_to(q_rotated[None, None, :], q_a.shape)
    q_b = ntl.gather(q_a, q_indices, 2)
    q_c = ntl.gather(q_c, ntl.broadcast_to((q_index % 48)[None, :], q_c.shape), 1)
    q_s = ntl.gather(q_s, ntl.broadcast_to((q_index % 48)[None, :], q_s.shape), 1)
    q_sign = ntl.where(q_index < 48, -1.0, 1.0)
    qo = ntl.where(q_index[None, None, :] < 96, q_a * q_c[:, None, :] + q_sign[None, None, :] * q_b * q_s[:, None, :], q_a)
    k_a = k
    k_c = cos
    k_s = sin
    k_a = k_a.to(ntl.float32)
    k_c = k_c.to(ntl.float32)
    k_s = k_s.to(ntl.float32)
    k_squared = k_a * k_a
    k_left = ntl.where(k.offsets(-1) < 64, k_squared, 0.0)
    k_right = ntl.where(k.offsets(-1) >= 64, k_squared, 0.0)
    k_mean = (ntl.sum(k_left, 2) + ntl.sum(k_right, 2)) / 128
    k_epsilon = epsk
    k_epsilon = k_epsilon.to(ntl.float32)
    k_reciprocal = ntl.rsqrt(k_mean + k_epsilon)
    k_normalized = k_a * k_reciprocal[:, :, None]
    k_a = (k_normalized * wk).to(k.dtype).to(ntl.float32)
    k_index = ntl.arange(0, 128)
    k_rotated = ntl.where(k_index < 48, k_index + 48, ntl.where(k_index < 96, k_index - 48, k_index))
    k_indices = ntl.broadcast_to(k_rotated[None, None, :], k_a.shape)
    k_b = ntl.gather(k_a, k_indices, 2)
    k_c = ntl.gather(k_c, ntl.broadcast_to((k_index % 48)[None, :], k_c.shape), 1)
    k_s = ntl.gather(k_s, ntl.broadcast_to((k_index % 48)[None, :], k_s.shape), 1)
    k_sign = ntl.where(k_index < 48, -1.0, 1.0)
    ko = ntl.where(k_index[None, None, :] < 96, k_a * k_c[:, None, :] + k_sign[None, None, :] * k_b * k_s[:, None, :], k_a)
    vo = v

def make(output_dir,rows=2,warps=1,*,caller="hip",archs=None):
    x=lambda:Tensor(shape=(None,8,7,128),dtype=nt.bfloat16)
    w=lambda:Tensor(shape=(128,),dtype=nt.bfloat16)
    eps=lambda:Tensor(0,dtype=nt.float64)
    angle=lambda:Tensor(shape=(None,48),dtype=nt.bfloat16)
    return nt.make(functools.partial(arrange,rows=rows),apply,
        (x(),x(),x(),w(),w(),eps(),eps(),angle(),angle(),x(),x(),x()),
        caller=caller,output_dir=output_dir,archs=archs or ['gfx936'],kernel_name=f'qkv_direct_token{rows}_w{warps}',num_warps=warps,num_stages=1)
