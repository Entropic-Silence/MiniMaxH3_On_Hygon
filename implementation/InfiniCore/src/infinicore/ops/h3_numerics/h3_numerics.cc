#include "infinicore/ops/h3_numerics.hpp"
#include "../../utils.hpp"
#include <stdexcept>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(H3QKVNormRope);
H3QKVNormRope::H3QKVNormRope(Tensor out, const Tensor &q, const Tensor &k, const Tensor &v,
                            const Tensor &wq, const Tensor &wk, const Tensor &cos, const Tensor &sin,
                            float epsq, float epsk) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, q, k, v, wq, wk, cos, sin);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(), out, q, k, v, wq, wk, cos, sin, epsq, epsk);
}
void H3QKVNormRope::execute(Tensor out, const Tensor &q, const Tensor &k, const Tensor &v,
                           const Tensor &wq, const Tensor &wk, const Tensor &cos, const Tensor &sin,
                           float epsq, float epsk) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(H3QKVNormRope, out, q, k, v, wq, wk, cos, sin, epsq, epsk);
}
void h3_qkv_norm_rope_(Tensor out, const Tensor &q, const Tensor &k, const Tensor &v,
                       const Tensor &wq, const Tensor &wk, const Tensor &cos, const Tensor &sin,
                       float epsq, float epsk) {
    if (q->ndim()!=4 || q->size(0)==0 || q->size(1)!=8 || q->size(2)!=7 || q->size(3)!=128
        || k->shape()!=q->shape() || v->shape()!=q->shape()
        || out->ndim()!=5 || out->size(0)!=8 || out->size(1)!=q->size(0)
        || out->size(2)!=3 || out->size(3)!=7 || out->size(4)!=128
        || wq->ndim()!=1 || wq->size(0)!=128 || wk->shape()!=wq->shape()
        || cos->ndim()!=2 || cos->size(0)!=q->size(0) || cos->size(1)!=48 || sin->shape()!=cos->shape()
        || out->dtype()!=DataType::BF16 || q->dtype()!=DataType::BF16 || k->dtype()!=DataType::BF16
        || v->dtype()!=DataType::BF16 || wq->dtype()!=DataType::BF16 || wk->dtype()!=DataType::BF16
        || cos->dtype()!=DataType::BF16 || sin->dtype()!=DataType::BF16) {
        throw std::invalid_argument("H3 packed QKV requires BF16 Q/K/V [T,8,7,128], out [8,T,3,7,128], weights [128], angles [T,48]");
    }
    H3QKVNormRope::execute(out, q, k, v, wq, wk, cos, sin, epsq, epsk);
}
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(H3RMSNorm);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(H3SwiGLU);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(H3RMSModulate);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(H3RoPE);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(H3QKRMSRoPEFP32);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(H3GatedResidual);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(H3GatedResidualRMSModulate);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(H3QKNormRope);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(H3RoPECache);

H3RMSNorm::H3RMSNorm(Tensor y, const Tensor &x, const Tensor &w, float eps) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x, w);
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(), y, x, w, eps);
}
void H3RMSNorm::execute(Tensor y, const Tensor &x, const Tensor &w, float eps) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(H3RMSNorm, y, x, w, eps);
}
Tensor h3_rms_norm(const Tensor &x, const Tensor &w, float eps) {
    if (x->ndim() != 3 || w->ndim() != 1 || x->size(2) != w->size(0)
        || x->dtype() != DataType::BF16 || w->dtype() != DataType::BF16
        || (x->size(2) != 128 && x->size(2) != 5376)) {
        throw std::invalid_argument("H3 RMSNorm requires BF16 [B,S,128|5376] and weight [D]");
    }
    auto y = Tensor::empty(x->shape(), x->dtype(), x->device());
    H3RMSNorm::execute(y, x, w, eps);
    return y;
}

H3SwiGLU::H3SwiGLU(Tensor y, const Tensor &up, const Tensor &gate) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, up, gate);
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(), y, up, gate);
}
void H3SwiGLU::execute(Tensor y, const Tensor &up, const Tensor &gate) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(H3SwiGLU, y, up, gate);
}
Tensor h3_swiglu(const Tensor &up, const Tensor &gate) {
    if (up->ndim() != 3 || up->shape() != gate->shape()
        || up->dtype() != DataType::BF16 || gate->dtype() != DataType::BF16) {
        throw std::invalid_argument("H3 SwiGLU requires matching BF16 [B,S,D] inputs");
    }
    auto y = Tensor::empty(up->shape(), up->dtype(), up->device());
    H3SwiGLU::execute(y, up, gate);
    return y;
}

H3RMSModulate::H3RMSModulate(Tensor y, const Tensor &x, const Tensor &w,
                           const Tensor &scale, const Tensor &shift, float eps) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x, w, scale, shift);
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(), y, x, w, scale, shift, eps);
}
void H3RMSModulate::execute(Tensor y, const Tensor &x, const Tensor &w,
                          const Tensor &scale, const Tensor &shift, float eps) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(H3RMSModulate, y, x, w, scale, shift, eps);
}
Tensor h3_rms_modulate(const Tensor &x, const Tensor &w, const Tensor &scale,
                      const Tensor &shift, float eps) {
    if (x->ndim() != 3 || x->size(2) != 5376 || w->ndim() != 1 || w->size(0) != 5376
        || x->shape() != scale->shape() || x->shape() != shift->shape()
        || x->dtype() != DataType::BF16 || w->dtype() != DataType::BF16
        || scale->dtype() != DataType::BF16 || shift->dtype() != DataType::BF16) {
        throw std::invalid_argument("H3 RMSNorm/AdaLN requires BF16 [B,S,5376] and weight [5376]");
    }
    auto y = Tensor::empty(x->shape(), x->dtype(), x->device());
    H3RMSModulate::execute(y, x, w, scale, shift, eps);
    return y;
}
H3RoPE::H3RoPE(Tensor y, const Tensor &x, const Tensor &cos, const Tensor &sin) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x, cos, sin);
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(), y, x, cos, sin);
}
void H3RoPE::execute(Tensor y, const Tensor &x, const Tensor &cos, const Tensor &sin) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(H3RoPE, y, x, cos, sin);
}
Tensor h3_rope(const Tensor &x, const Tensor &cos, const Tensor &sin) {
    if (x->ndim() != 3 || x->size(2) != 128 || cos->ndim() != 2
        || cos->size(0) != x->size(0) || cos->size(1) != 48 || cos->shape() != sin->shape()
        || x->dtype() != DataType::BF16 || cos->dtype() != DataType::BF16 || sin->dtype() != DataType::BF16) {
        throw std::invalid_argument("H3 RoPE requires BF16 [S,H,128] and token-ordered angles [S,48]");
    }
    auto y = Tensor::empty(x->shape(), x->dtype(), x->device());
    H3RoPE::execute(y, x, cos, sin);
    return y;
}

H3QKRMSRoPEFP32::H3QKRMSRoPEFP32(Tensor y, const Tensor &x, const Tensor &w,
                                                   const Tensor &cos, const Tensor &sin, float eps) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x, w, cos, sin);
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(), y, x, w, cos, sin, eps);
}
void H3QKRMSRoPEFP32::execute(Tensor y, const Tensor &x, const Tensor &w,
                             const Tensor &cos, const Tensor &sin, float eps) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(H3QKRMSRoPEFP32, y, x, w, cos, sin, eps);
}
Tensor h3_qk_rms_rope_fp32(const Tensor &x, const Tensor &w, const Tensor &cos,
                          const Tensor &sin, float eps) {
    if (x->ndim() != 3 || x->size(2) != 128 || w->ndim() != 1 || w->size(0) != 128
        || cos->ndim() != 2 || cos->size(0) != x->size(0) || cos->size(1) != 48
        || cos->shape() != sin->shape() || x->dtype() != DataType::BF16
        || w->dtype() != DataType::BF16 || cos->dtype() != DataType::BF16 || sin->dtype() != DataType::BF16) {
        throw std::invalid_argument("H3 fused Q/K RMS/RoPE requires BF16 [S,H,128], weight [128], ordered angles [S,48]");
    }
    auto y = Tensor::empty(x->shape(), x->dtype(), x->device());
    H3QKRMSRoPEFP32::execute(y, x, w, cos, sin, eps);
    return y;
}

namespace {
void check_gated_residual(const Tensor &x, const Tensor &gate, const Tensor &y) {
    if (x->ndim() != 3 || x->size(2) != 5376 || x->shape() != gate->shape() || x->shape() != y->shape()
        || x->dtype() != DataType::BF16 || gate->dtype() != DataType::BF16 || y->dtype() != DataType::BF16) {
        throw std::invalid_argument("H3 gated residual requires matching BF16 [B,S,5376] inputs");
    }
}
} // namespace

H3GatedResidual::H3GatedResidual(Tensor out, const Tensor &x, const Tensor &gate, const Tensor &y) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, x, gate, y);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(), out, x, gate, y);
}
void H3GatedResidual::execute(Tensor out, const Tensor &x, const Tensor &gate, const Tensor &y) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(H3GatedResidual, out, x, gate, y);
}
Tensor h3_gated_residual(const Tensor &x, const Tensor &gate, const Tensor &y) {
    check_gated_residual(x, gate, y);
    auto out = Tensor::empty(x->shape(), x->dtype(), x->device());
    H3GatedResidual::execute(out, x, gate, y);
    return out;
}

H3GatedResidualRMSModulate::H3GatedResidualRMSModulate(Tensor residual, Tensor out, const Tensor &x,
                                                       const Tensor &gate, const Tensor &y, const Tensor &w,
                                                       const Tensor &scale, const Tensor &shift, float eps) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(residual, out, x, gate, y, w, scale, shift);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(), residual, out, x, gate, y, w, scale, shift, eps);
}
void H3GatedResidualRMSModulate::execute(Tensor residual, Tensor out, const Tensor &x, const Tensor &gate,
                                         const Tensor &y, const Tensor &w, const Tensor &scale,
                                         const Tensor &shift, float eps) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(H3GatedResidualRMSModulate, residual, out, x, gate, y, w, scale, shift, eps);
}
std::pair<Tensor, Tensor> h3_gated_residual_rms_modulate(const Tensor &x, const Tensor &gate, const Tensor &y,
                                                         const Tensor &w, const Tensor &scale,
                                                         const Tensor &shift, float eps) {
    check_gated_residual(x, gate, y);
    if (w->ndim() != 1 || w->size(0) != 5376 || x->shape() != scale->shape() || x->shape() != shift->shape()
        || w->dtype() != DataType::BF16 || scale->dtype() != DataType::BF16 || shift->dtype() != DataType::BF16) {
        throw std::invalid_argument("H3 gated residual RMSNorm/AdaLN requires BF16 weight [5376] and [B,S,5376] modulation");
    }
    auto residual = Tensor::empty(x->shape(), x->dtype(), x->device());
    auto out = Tensor::empty(x->shape(), x->dtype(), x->device());
    H3GatedResidualRMSModulate::execute(residual, out, x, gate, y, w, scale, shift, eps);
    return {residual, out};
}

H3QKNormRope::H3QKNormRope(Tensor out, const Tensor &x, const Tensor &w, const Tensor &cos,
                           const Tensor &sin, float eps) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, x, w, cos, sin);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(), out, x, w, cos, sin, eps);
}
void H3QKNormRope::execute(Tensor out, const Tensor &x, const Tensor &w, const Tensor &cos,
                           const Tensor &sin, float eps) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(H3QKNormRope, out, x, w, cos, sin, eps);
}
void h3_qk_norm_rope_(Tensor out, const Tensor &x, const Tensor &w, const Tensor &cos,
                      const Tensor &sin, float eps) {
    if (x->ndim() != 4 || x->size(3) != 128 || out->shape() != x->shape() || w->ndim() != 1
        || w->size(0) != 128 || cos->ndim() != 2 || cos->size(0) != x->size(0) || cos->size(1) != 48
        || cos->shape() != sin->shape() || x->dtype() != DataType::BF16 || out->dtype() != DataType::BF16
        || w->dtype() != DataType::BF16 || cos->dtype() != DataType::BF16 || sin->dtype() != DataType::BF16) {
        throw std::invalid_argument("H3 q/k norm + RoPE requires BF16 [T,P,Hs,128], weight [128], angles [T,48]");
    }
    H3QKNormRope::execute(out, x, w, cos, sin, eps);
}

H3RoPECache::H3RoPECache(Tensor out, const Tensor &position, const Tensor &power) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, position, power);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(), out, position, power);
}
void H3RoPECache::execute(Tensor out, const Tensor &position, const Tensor &power) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(H3RoPECache, out, position, power);
}
Tensor h3_rope_cache(const Tensor &position, const Tensor &power) {
    if (position->ndim() != 2 || position->size(1) != 3 || position->dtype() != DataType::F64
        || power->ndim() != 1 || power->size(0) != 16 || power->dtype() != DataType::F32) {
        throw std::invalid_argument("H3 RoPE cache requires FP64 positions [T,3] and FP32 powers [16]");
    }
    auto out = Tensor::empty({position->size(0), 96}, DataType::BF16, position->device());
    H3RoPECache::execute(out, position, power);
    return out;
}
} // namespace infinicore::op
