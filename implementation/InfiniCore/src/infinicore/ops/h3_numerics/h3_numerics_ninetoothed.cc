#if defined(ENABLE_HYGON_API) && defined(ENABLE_H3_NINETOOTHED)
#include "infinicore/ops/h3_numerics.hpp"
#include "h3_rms_exact_128.h"
#include "h3_rms_exact_5376.h"
#include "h3_swiglu_exact.h"
#include "h3_swiglu_width14336.h"
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <cstring>
#include "h3_rms_modulate_exact.h"
#include "h3_rope_exact.h"
#include "h3_qk_rms_rope_fp32.h"
#include "h3_gated_residual_exact.h"
#include "h3_gated_residual_rms_modulate_exact.h"
#include "h3_qk_norm_rope.h"
#include "qk_direct_pack_rows8.h"
#include "qkv_direct_token2_w1.h"
#include "h3_rope_cache.h"
#include <array>
#include <stdexcept>
#include <string>

namespace infinicore::op::h3_numerics_impl {
template<size_t N> struct View {
    std::array<uint64_t, N> shape;
    std::array<int64_t, N> strides;
    NineToothedTensor tensor;
    explicit View(const graph::GraphTensor &x) {
        for (size_t i = 0; i < N; ++i) {
            shape[i] = x->size(i);
            strides[i] = x->stride(i);
        }
        tensor = {const_cast<void *>(static_cast<const void *>(x->data())), shape.data(), strides.data()};
    }
};
// NineToothed's Python scalar ABI uses double; C++ callers use the same ABI.
struct RMSMeta { graph::GraphTensor y, x, weight; double eps; };
void *rms_plan(Tensor y, const Tensor &x, const Tensor &w, float eps) {
    return new RMSMeta{graph::GraphTensor(y), graph::GraphTensor(x), graph::GraphTensor(w), eps};
}
void rms_run(void *meta) {
    auto *p = static_cast<RMSMeta *>(meta);
    View<3> x(p->x), y(p->y);
    View<1> w(p->weight);
    NineToothedTensor eps{&p->eps, nullptr, nullptr};
    int status = p->x->size(2) == 128
        ? launch_h3_rms_exact_128(context::getStream(), x.tensor, w.tensor, eps, y.tensor)
        : launch_h3_rms_exact_5376(context::getStream(), x.tensor, w.tensor, eps, y.tensor);
    if (status) throw std::runtime_error("H3 NineToothed RMSNorm launch failed: " + std::to_string(status));
}
void rms_cleanup(void **meta) { delete static_cast<RMSMeta *>(*meta); *meta = nullptr; }

struct SwiGLUMeta { graph::GraphTensor y, up, gate; bool fixed_width; };
void *swiglu_plan(Tensor y, const Tensor &up, const Tensor &gate) {
    bool fixed_width = false;
    const char *enabled = std::getenv("H3_BW_SWIGLU_WIDTH");
    if (enabled && std::strcmp(enabled, "1") == 0
        && up->ndim() == 3 && up->size(2) == 14336
        && up->shape() == gate->shape() && up->shape() == y->shape()
        && up->dtype() == DataType::BF16 && gate->dtype() == DataType::BF16
        && y->dtype() == DataType::BF16) {
        hipDeviceProp_t properties{};
        fixed_width = hipGetDeviceProperties(&properties, static_cast<int>(up->device().getIndex())) == hipSuccess
            && std::strncmp(properties.gcnArchName, "gfx936", 6) == 0;
    }
    return new SwiGLUMeta{graph::GraphTensor(y), graph::GraphTensor(up), graph::GraphTensor(gate), fixed_width};
}
void swiglu_run(void *meta) {
    auto *p = static_cast<SwiGLUMeta *>(meta);
    View<3> y(p->y), up(p->up), gate(p->gate);
    const int status = p->fixed_width
        ? launch_h3_swiglu_width14336(context::getStream(), up.tensor, gate.tensor, y.tensor)
        : launch_h3_swiglu_exact(context::getStream(), up.tensor, gate.tensor, y.tensor);
    if (status) throw std::runtime_error("H3 NineToothed SwiGLU launch failed: " + std::to_string(status));
}
void swiglu_cleanup(void **meta) { delete static_cast<SwiGLUMeta *>(*meta); *meta = nullptr; }

struct ModulateMeta { graph::GraphTensor y, x, weight, scale, shift; double eps; };
void *modulate_plan(Tensor y, const Tensor &x, const Tensor &w,
                    const Tensor &scale, const Tensor &shift, float eps) {
    return new ModulateMeta{graph::GraphTensor(y), graph::GraphTensor(x), graph::GraphTensor(w),
                            graph::GraphTensor(scale), graph::GraphTensor(shift), eps};
}
void modulate_run(void *meta) {
    auto *p = static_cast<ModulateMeta *>(meta);
    View<3> y(p->y), x(p->x), scale(p->scale), shift(p->shift);
    View<1> w(p->weight);
    NineToothedTensor eps{&p->eps, nullptr, nullptr};
    const int status = launch_h3_rms_modulate_exact(context::getStream(), x.tensor, w.tensor,
                                                   eps, scale.tensor, shift.tensor, y.tensor);
    if (status) throw std::runtime_error("H3 NineToothed RMSNorm/AdaLN launch failed: " + std::to_string(status));
}
void modulate_cleanup(void **meta) { delete static_cast<ModulateMeta *>(*meta); *meta = nullptr; }

struct RoPEMeta { graph::GraphTensor y, x, cos, sin; };
void *rope_plan(Tensor y, const Tensor &x, const Tensor &cos, const Tensor &sin) {
    return new RoPEMeta{graph::GraphTensor(y), graph::GraphTensor(x), graph::GraphTensor(cos), graph::GraphTensor(sin)};
}
void rope_run(void *meta) {
    auto *p = static_cast<RoPEMeta *>(meta);
    View<3> y(p->y), x(p->x);
    View<2> cos(p->cos), sin(p->sin);
    const int status = launch_h3_rope_exact(context::getStream(), x.tensor, x.tensor, x.tensor, cos.tensor, sin.tensor, y.tensor, y.tensor, y.tensor);
    if (status) throw std::runtime_error("H3 NineToothed RoPE launch failed: " + std::to_string(status));
}
void rope_cleanup(void **meta) { delete static_cast<RoPEMeta *>(*meta); *meta = nullptr; }

struct GatedResidualMeta { graph::GraphTensor out, x, gate, y; };
void *gated_residual_plan(Tensor out, const Tensor &x, const Tensor &gate, const Tensor &y) {
    return new GatedResidualMeta{graph::GraphTensor(out), graph::GraphTensor(x), graph::GraphTensor(gate),
                                 graph::GraphTensor(y)};
}
void gated_residual_run(void *meta) {
    auto *p = static_cast<GatedResidualMeta *>(meta);
    View<3> out(p->out), x(p->x), gate(p->gate), y(p->y);
    const int status = launch_h3_gated_residual_exact(context::getStream(), x.tensor, gate.tensor, y.tensor, out.tensor);
    if (status) throw std::runtime_error("H3 NineToothed gated residual launch failed: " + std::to_string(status));
}
void gated_residual_cleanup(void **meta) { delete static_cast<GatedResidualMeta *>(*meta); *meta = nullptr; }

struct GatedModulateMeta { graph::GraphTensor residual, out, x, gate, y, weight, scale, shift; double eps; };
void *gated_modulate_plan(Tensor residual, Tensor out, const Tensor &x, const Tensor &gate, const Tensor &y,
                          const Tensor &w, const Tensor &scale, const Tensor &shift, float eps) {
    return new GatedModulateMeta{graph::GraphTensor(residual), graph::GraphTensor(out), graph::GraphTensor(x),
                                 graph::GraphTensor(gate), graph::GraphTensor(y), graph::GraphTensor(w),
                                 graph::GraphTensor(scale), graph::GraphTensor(shift), eps};
}
void gated_modulate_run(void *meta) {
    auto *p = static_cast<GatedModulateMeta *>(meta);
    View<3> residual(p->residual), out(p->out), x(p->x), gate(p->gate), y(p->y), scale(p->scale), shift(p->shift);
    View<1> w(p->weight);
    NineToothedTensor eps{&p->eps, nullptr, nullptr};
    const int status = launch_h3_gated_residual_rms_modulate_exact(
        context::getStream(), x.tensor, gate.tensor, y.tensor, w.tensor, eps, scale.tensor, shift.tensor,
        residual.tensor, out.tensor);
    if (status) throw std::runtime_error("H3 NineToothed gated residual RMSNorm/AdaLN launch failed: " + std::to_string(status));
}
void gated_modulate_cleanup(void **meta) { delete static_cast<GatedModulateMeta *>(*meta); *meta = nullptr; }

struct QKNormRopeMeta { graph::GraphTensor out, x, weight, cos, sin; double eps; bool direct_pack; };
void *qk_norm_rope_plan(Tensor out, const Tensor &x, const Tensor &w, const Tensor &cos,
                        const Tensor &sin, float eps) {
    const char *enabled = std::getenv("H3_QK_DIRECT_PACK");
    const bool direct_pack = enabled && std::strcmp(enabled, "1") == 0
        && x->ndim() == 4 && x->size(1) == 8 && x->size(2) == 7 && x->size(3) == 128;
    if (direct_pack) {
        hipDeviceProp_t properties{};
        if (hipGetDeviceProperties(&properties, static_cast<int>(x->device().getIndex())) != hipSuccess
            || std::strncmp(properties.gcnArchName, "gfx936", 6) != 0) {
            throw std::invalid_argument("H3_QK_DIRECT_PACK requires gfx936");
        }
    }
    return new QKNormRopeMeta{graph::GraphTensor(out), graph::GraphTensor(x), graph::GraphTensor(w),
                              graph::GraphTensor(cos), graph::GraphTensor(sin), eps, direct_pack};
}
void qk_norm_rope_run(void *meta) {
    auto *p = static_cast<QKNormRopeMeta *>(meta);
    View<4> out(p->out), x(p->x);
    View<1> w(p->weight);
    View<2> cos(p->cos), sin(p->sin);
    NineToothedTensor eps{&p->eps, nullptr, nullptr};
    const int status = p->direct_pack
        ? launch_qk_direct_pack_rows8(context::getStream(), x.tensor, w.tensor, eps, cos.tensor, sin.tensor, out.tensor)
        : launch_h3_qk_norm_rope(context::getStream(), x.tensor, x.tensor, x.tensor,
                                 w.tensor, w.tensor, w.tensor, eps, cos.tensor, sin.tensor,
                                 out.tensor, out.tensor, out.tensor);
    if (status) throw std::runtime_error("H3 NineToothed q/k norm + RoPE launch failed: " + std::to_string(status));
}
void qk_norm_rope_cleanup(void **meta) { delete static_cast<QKNormRopeMeta *>(*meta); *meta = nullptr; }

struct RoPECacheMeta { graph::GraphTensor cos, sin, position, power; };
void *rope_cache_plan(Tensor out, const Tensor &position, const Tensor &power) {
    return new RoPECacheMeta{graph::GraphTensor(out->narrow({{1, 0, 48}})),
                             graph::GraphTensor(out->narrow({{1, 48, 48}})),
                             graph::GraphTensor(position), graph::GraphTensor(power)};
}
void rope_cache_run(void *meta) {
    auto *p = static_cast<RoPECacheMeta *>(meta);
    View<2> cos(p->cos), sin(p->sin), position(p->position);
    View<1> power(p->power);
    const int status = launch_h3_rope_cache(context::getStream(), position.tensor, power.tensor, cos.tensor, sin.tensor);
    if (status) throw std::runtime_error("H3 NineToothed RoPE cache launch failed: " + std::to_string(status));
}
void rope_cache_cleanup(void **meta) { delete static_cast<RoPECacheMeta *>(*meta); *meta = nullptr; }

struct QKRMSRoPEFP32Meta { graph::GraphTensor y, x, weight, cos, sin; double eps; };
void *qk_fp32_plan(Tensor y, const Tensor &x, const Tensor &weight,
                   const Tensor &cos, const Tensor &sin, float eps) {
    return new QKRMSRoPEFP32Meta{graph::GraphTensor(y), graph::GraphTensor(x), graph::GraphTensor(weight),
                                graph::GraphTensor(cos), graph::GraphTensor(sin), eps};
}
void qk_fp32_run(void *meta) {
    auto *p = static_cast<QKRMSRoPEFP32Meta *>(meta);
    View<3> y(p->y), x(p->x);
    View<1> w(p->weight);
    View<2> cos(p->cos), sin(p->sin);
    NineToothedTensor eps{&p->eps, nullptr, nullptr};
    const int status = launch_h3_qk_rms_rope_fp32(context::getStream(), x.tensor, w.tensor, eps, cos.tensor, sin.tensor, y.tensor);
    if (status) throw std::runtime_error("H3 fused Q/K RMS/RoPE launch failed: " + std::to_string(status));
}
void qk_fp32_cleanup(void **meta) { delete static_cast<QKRMSRoPEFP32Meta *>(*meta); *meta = nullptr; }

struct QKVNormRopeMeta {
    graph::GraphTensor q, k, v, wq, wk, cos, sin, qo, ko, vo;
    double epsq, epsk;
};
void *qkv_norm_rope_plan(Tensor out, const Tensor &q, const Tensor &k, const Tensor &v,
                         const Tensor &wq, const Tensor &wk, const Tensor &cos, const Tensor &sin,
                         float epsq, float epsk) {
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, static_cast<int>(q->device().getIndex())) != hipSuccess
        || std::strncmp(properties.gcnArchName,"gfx936",6)!=0) {
        throw std::invalid_argument("H3 packed QKV kernel requires gfx936");
    }
    auto qo=out->narrow({{2,0,1}})->squeeze(2)->permute({1,0,2,3});
    auto ko=out->narrow({{2,1,1}})->squeeze(2)->permute({1,0,2,3});
    auto vo=out->narrow({{2,2,1}})->squeeze(2)->permute({1,0,2,3});
    return new QKVNormRopeMeta{graph::GraphTensor(q),graph::GraphTensor(k),graph::GraphTensor(v),
        graph::GraphTensor(wq),graph::GraphTensor(wk),graph::GraphTensor(cos),graph::GraphTensor(sin),
        graph::GraphTensor(qo),graph::GraphTensor(ko),graph::GraphTensor(vo),epsq,epsk};
}
void qkv_norm_rope_run(void *meta) {
    auto *p=static_cast<QKVNormRopeMeta *>(meta);
    View<4> q(p->q),k(p->k),v(p->v),qo(p->qo),ko(p->ko),vo(p->vo);
    View<1> wq(p->wq),wk(p->wk);
    View<2> cos(p->cos),sin(p->sin);
    NineToothedTensor epsq{&p->epsq,nullptr,nullptr},epsk{&p->epsk,nullptr,nullptr};
    const int status=launch_qkv_direct_token2_w1(context::getStream(),q.tensor,k.tensor,v.tensor,
        wq.tensor,wk.tensor,epsq,epsk,cos.tensor,sin.tensor,qo.tensor,ko.tensor,vo.tensor);
    if(status)throw std::runtime_error("H3 packed QKV launch failed: "+std::to_string(status));
}
void qkv_norm_rope_cleanup(void **meta) { delete static_cast<QKVNormRopeMeta *>(*meta);*meta=nullptr; }

static bool registered = []() {
    H3QKVNormRope::plan_dispatcher().registerDevice(Device::Type::HYGON, &qkv_norm_rope_plan);
    H3QKVNormRope::run_dispatcher().registerDevice(Device::Type::HYGON, &qkv_norm_rope_run);
    H3QKVNormRope::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &qkv_norm_rope_cleanup);
    H3QKRMSRoPEFP32::plan_dispatcher().registerDevice(Device::Type::HYGON, &qk_fp32_plan);
    H3QKRMSRoPEFP32::run_dispatcher().registerDevice(Device::Type::HYGON, &qk_fp32_run);
    H3QKRMSRoPEFP32::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &qk_fp32_cleanup);
    H3RoPECache::plan_dispatcher().registerDevice(Device::Type::HYGON, &rope_cache_plan);
    H3RoPECache::run_dispatcher().registerDevice(Device::Type::HYGON, &rope_cache_run);
    H3RoPECache::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &rope_cache_cleanup);
    H3QKNormRope::plan_dispatcher().registerDevice(Device::Type::HYGON, &qk_norm_rope_plan);
    H3QKNormRope::run_dispatcher().registerDevice(Device::Type::HYGON, &qk_norm_rope_run);
    H3QKNormRope::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &qk_norm_rope_cleanup);
    H3GatedResidual::plan_dispatcher().registerDevice(Device::Type::HYGON, &gated_residual_plan);
    H3GatedResidual::run_dispatcher().registerDevice(Device::Type::HYGON, &gated_residual_run);
    H3GatedResidual::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &gated_residual_cleanup);
    H3GatedResidualRMSModulate::plan_dispatcher().registerDevice(Device::Type::HYGON, &gated_modulate_plan);
    H3GatedResidualRMSModulate::run_dispatcher().registerDevice(Device::Type::HYGON, &gated_modulate_run);
    H3GatedResidualRMSModulate::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &gated_modulate_cleanup);
    H3RMSNorm::plan_dispatcher().registerDevice(Device::Type::HYGON, &rms_plan);
    H3RMSNorm::run_dispatcher().registerDevice(Device::Type::HYGON, &rms_run);
    H3RMSNorm::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &rms_cleanup);
    H3SwiGLU::plan_dispatcher().registerDevice(Device::Type::HYGON, &swiglu_plan);
    H3SwiGLU::run_dispatcher().registerDevice(Device::Type::HYGON, &swiglu_run);
    H3SwiGLU::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &swiglu_cleanup);
    H3RMSModulate::plan_dispatcher().registerDevice(Device::Type::HYGON, &modulate_plan);
    H3RMSModulate::run_dispatcher().registerDevice(Device::Type::HYGON, &modulate_run);
    H3RMSModulate::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &modulate_cleanup);
    H3RoPE::plan_dispatcher().registerDevice(Device::Type::HYGON, &rope_plan);
    H3RoPE::run_dispatcher().registerDevice(Device::Type::HYGON, &rope_run);
    H3RoPE::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &rope_cleanup);
    return true;
}();
} // namespace infinicore::op::h3_numerics_impl
#endif
