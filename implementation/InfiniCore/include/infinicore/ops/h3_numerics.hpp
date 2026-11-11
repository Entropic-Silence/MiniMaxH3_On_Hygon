#pragma once
#include "../graph/graph.hpp"
#include "common/op.hpp"
#include <utility>

namespace infinicore::op {

// Direct packed Q/K normalization and RoPE plus an exact V copy (gfx936/SP8).
INFINICORE_GRAPH_OP_CLASS(H3QKVNormRope, Tensor, const Tensor &, const Tensor &, const Tensor &,
                         const Tensor &, const Tensor &, const Tensor &, const Tensor &, float, float);
void h3_qkv_norm_rope_(Tensor out, const Tensor &q, const Tensor &k, const Tensor &v,
                       const Tensor &wq, const Tensor &wk, const Tensor &cos, const Tensor &sin,
                       float epsq, float epsk);
// H3's torch 2.5 reference rounds each RMSNorm intermediate to BF16.
// Keep this separate from the FP32-accumulating RMSNorm used by Qwen.
INFINICORE_GRAPH_OP_CLASS(H3RMSNorm, Tensor, const Tensor &, const Tensor &, float);
INFINICORE_GRAPH_OP_CLASS(H3SwiGLU, Tensor, const Tensor &, const Tensor &);
INFINICORE_GRAPH_OP_CLASS(H3RMSModulate, Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, float);
INFINICORE_GRAPH_OP_CLASS(H3RoPE, Tensor, const Tensor &, const Tensor &, const Tensor &);
// gfx936 split-reduction RMS followed by default MRoPE arithmetic.
INFINICORE_GRAPH_OP_CLASS(H3QKRMSRoPEFP32, Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, float);
Tensor h3_qk_rms_rope_fp32(const Tensor &x, const Tensor &weight, const Tensor &cos, const Tensor &sin, float eps);

// out = x + gate * y with the product and the sum each rounded to BF16.
INFINICORE_GRAPH_OP_CLASS(H3GatedResidual, Tensor, const Tensor &, const Tensor &, const Tensor &);
INFINICORE_GRAPH_OP_CLASS(H3QKNormRope, Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, float);
INFINICORE_GRAPH_OP_CLASS(H3RoPECache, Tensor, const Tensor &, const Tensor &);
// residual = x + gate * y as above; out = h3_rms_modulate(residual, ...).
INFINICORE_GRAPH_OP_CLASS(H3GatedResidualRMSModulate, Tensor, Tensor, const Tensor &, const Tensor &, const Tensor &,
                          const Tensor &, const Tensor &, const Tensor &, float);
Tensor h3_rms_norm(const Tensor &x, const Tensor &weight, float epsilon);
Tensor h3_swiglu(const Tensor &up, const Tensor &gate);
Tensor h3_rms_modulate(const Tensor &x, const Tensor &weight, const Tensor &scale, const Tensor &shift, float epsilon);
// Angles have already been gathered into token order; each half is repeated.
Tensor h3_rope(const Tensor &x, const Tensor &cos, const Tensor &sin);
Tensor h3_gated_residual(const Tensor &x, const Tensor &gate, const Tensor &y);
// Per-head RMSNorm (weight [128]) then 3-axis RoPE (angles [T, 48]) of
// x [T, P, Hs, 128] (any strides) written into out [T, P, Hs, 128] (any
// strides, e.g. a permuted all-to-all send buffer); k100-extreme.
void h3_qk_norm_rope_(Tensor out, const Tensor &x, const Tensor &weight, const Tensor &cos,
                      const Tensor &sin, float epsilon);
// H3's 3-axis RoPE table [T, 96] BF16 = cat(cos, sin) of position [T, 3]
// (FP64) times 1 / power, power [16] FP32 = theta ** (2k / 32) (float_power).
Tensor h3_rope_cache(const Tensor &position, const Tensor &power);
// Returns {residual, modulated}.
std::pair<Tensor, Tensor> h3_gated_residual_rms_modulate(const Tensor &x, const Tensor &gate, const Tensor &y,
                                                         const Tensor &weight, const Tensor &scale,
                                                         const Tensor &shift, float epsilon);
} // namespace infinicore::op
