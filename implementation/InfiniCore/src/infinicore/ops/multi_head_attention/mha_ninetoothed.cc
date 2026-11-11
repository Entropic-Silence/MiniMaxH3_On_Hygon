#if defined(ENABLE_HYGON_API) && defined(ENABLE_H3_NINETOOTHED)
#include "infinicore/ops/mha.hpp"
#include "../infiniop_impl.hpp"
#include "h3_full_attention.h"
#include "h3_vision_attention_72.h"
#include <array>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#if defined(ENABLE_FLASH_ATTN)
namespace infinicore::op::mha_impl::flashattn {
void *plan(Tensor, const Tensor &, const Tensor &, const Tensor &,
           std::optional<Tensor>, float, bool);
void run(void *);
void cleanup(void **);
}
#endif

namespace infinicore::op::mha_impl::ninetoothed {
struct PlannedMeta {
    graph::GraphTensor out, q, k, v;
    double scale;
    void *baseline = nullptr;
};

static bool supported(const Tensor &out, const Tensor &q, const Tensor &k,
                      const Tensor &v, const std::optional<Tensor> &alibi, bool causal) {
    const char *vision = std::getenv("H3_QWEN_NINETOOTHED");
    const bool vision_enabled = vision && std::strcmp(vision, "1") == 0;
    // Query and key lengths may differ (sequence-parallel padding of queries):
    // the kernel tiles Q and K/V independently and masks keys by K's length.
    return !causal && !alibi && q->ndim() == 4 && k->ndim() == 4 && k->shape() == v->shape()
        && q->shape() == out->shape() && q->size(0) == k->size(0)
        && q->size(2) == k->size(2) && q->size(3) == k->size(3)
        && q->size(1) > 0 && k->size(1) > 0
        && (q->size(3) == 128 || (vision_enabled && q->size(3) == 72))
        && q->dtype() == DataType::BF16 && k->dtype() == DataType::BF16
        && v->dtype() == DataType::BF16 && out->dtype() == DataType::BF16;
}

void *plan(Tensor out, const Tensor &q, const Tensor &k, const Tensor &v,
           std::optional<Tensor> alibi, float scale, bool causal) {
    // k100-extreme: vendor flash attention (ATen SDPA) unless
    // INFINICORE_HYGON_MHA=ninetoothed.
    const char *mode = std::getenv("INFINICORE_HYGON_MHA");
    const bool baseline = !mode || std::strcmp(mode, "aten") == 0;
    if (mode && std::strcmp(mode, "ninetoothed") != 0 && !baseline) {
        throw std::invalid_argument("INFINICORE_HYGON_MHA must be ninetoothed or aten");
    }
    void *fallback = nullptr;
    if (baseline || !supported(out, q, k, v, alibi, causal)) {
#if defined(ENABLE_FLASH_ATTN)
        // Preserve the existing general MHA for non-H3 shapes, causal/GQA and dtypes.
        fallback = flashattn::plan(out, q, k, v, alibi, scale, causal);
#else
        throw std::invalid_argument("H3 NineToothed MHA requires noncausal BF16 [B,S,H,128] self attention");
#endif
    }
    return new PlannedMeta{graph::GraphTensor(out), graph::GraphTensor(q),
                            graph::GraphTensor(k), graph::GraphTensor(v), scale, fallback};
}

struct View {
    std::array<uint64_t, 4> shape;
    std::array<int64_t, 4> strides;
    NineToothedTensor tensor;
    explicit View(const graph::GraphTensor &x)
        : shape{x->size(0), x->size(2), x->size(1), x->size(3)},
          strides{x->stride(0), x->stride(2), x->stride(1), x->stride(3)},
          tensor{const_cast<void *>(static_cast<const void *>(x->data())), shape.data(), strides.data()} {}
};

void run(void *meta) {
    auto *p = static_cast<PlannedMeta *>(meta);
#if defined(ENABLE_FLASH_ATTN)
    if (p->baseline) { flashattn::run(p->baseline); return; }
#endif
    View q(p->q), k(p->k), v(p->v), out(p->out);
    NineToothedTensor scale{&p->scale, nullptr, nullptr};
    const int status = p->q->size(3) == 72
        ? launch_h3_vision_attention_72(context::getStream(), q.tensor, k.tensor, v.tensor, scale, out.tensor)
        : launch_h3_full_attention(context::getStream(), q.tensor, k.tensor, v.tensor, scale, out.tensor);
    if (status != 0) {
        throw std::runtime_error("NineToothed HIP MHA launch failed: " + std::to_string(status));
    }
}

void cleanup(void **meta) {
    auto *p = static_cast<PlannedMeta *>(*meta);
#if defined(ENABLE_FLASH_ATTN)
    if (p->baseline) flashattn::cleanup(&p->baseline);
#endif
    delete p;
    *meta = nullptr;
}

static bool registered = []() {
    MultiheadAttention::plan_dispatcher().registerDevice(Device::Type::HYGON, &plan);
    MultiheadAttention::run_dispatcher().registerDevice(Device::Type::HYGON, &run);
    MultiheadAttention::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &cleanup);
    return true;
}();
} // namespace infinicore::op::mha_impl::ninetoothed
#endif
