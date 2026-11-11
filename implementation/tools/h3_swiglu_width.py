"""Same SwiGLU arithmetic with a compile-time inner width for HIP AOT."""
import functools
import ninetoothed
from ninetoothed import Tensor
from h3_pointwise import swiglu_arrange, swiglu_apply, _apply_for_arch

def make_swiglu_width(output_dir, rows=1, cols=1024, num_warps=1, archs=None):
    archs=archs or ["gfx936"]
    dtype=ninetoothed.bfloat16
    return ninetoothed.make(functools.partial(swiglu_arrange,rows=rows,cols=cols),
        _apply_for_arch(swiglu_apply,dtype,archs),
        tuple(Tensor(shape=(None,None,14336),dtype=dtype) for _ in range(3)),
        caller='hip',output_dir=output_dir,archs=archs,
        kernel_name='h3_swiglu_width14336',num_warps=num_warps,num_stages=1)
