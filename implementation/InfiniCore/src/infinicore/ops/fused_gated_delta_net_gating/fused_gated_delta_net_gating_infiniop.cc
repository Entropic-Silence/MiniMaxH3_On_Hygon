#include "infinicore/ops/fused_gated_delta_net_gating.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::fused_gated_delta_net_gating_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, FusedGatedDeltaNetGating, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, g, beta_output, A_log, a, b, dt_bias;
};

void *plan(Tensor g,
           Tensor beta_output,
           const Tensor &A_log,
           const Tensor &a,
           const Tensor &b,
           const Tensor &dt_bias,
           float beta,
           float threshold) {
    size_t seed = hash_combine(g, beta_output, A_log, a, b, dt_bias, beta, threshold);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, FusedGatedDeltaNetGating, seed,
        g->desc(),
        beta_output->desc(),
        A_log->desc(),
        a->desc(),
        b->desc(),
        dt_bias->desc(),
        beta,
        threshold);

    INFINIOP_WORKSPACE_TENSOR(workspace, FusedGatedDeltaNetGating, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(g),
        graph::GraphTensor(beta_output),
        graph::GraphTensor(A_log),
        graph::GraphTensor(a),
        graph::GraphTensor(b),
        graph::GraphTensor(dt_bias)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopFusedGatedDeltaNetGating(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->g->data(),
        planned->beta_output->data(),
        planned->A_log->data(),
        planned->a->data(),
        planned->b->data(),
        planned->dt_bias->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(FusedGatedDeltaNetGating, &plan, &run, &cleanup);

} // namespace infinicore::op::fused_gated_delta_net_gating_impl::infiniop
