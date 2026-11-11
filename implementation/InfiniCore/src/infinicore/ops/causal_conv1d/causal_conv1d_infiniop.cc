#include "infinicore/ops/causal_conv1d.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::causal_conv1d_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, CausalConv1d, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, out, conv_state, qkv, weight;
    std::optional<graph::GraphTensor> final_conv_state;
    std::optional<graph::GraphTensor> bias;
    std::optional<graph::GraphTensor> cu_seqlens;
    std::optional<graph::GraphTensor> initial_state_indices;
    std::optional<graph::GraphTensor> final_state_indices;
};

void *plan(Tensor out,
           Tensor conv_state,
           std::optional<Tensor> final_conv_state,
           const Tensor &qkv,
           const Tensor &weight,
           std::optional<Tensor> bias,
           std::optional<Tensor> cu_seqlens,
           std::optional<Tensor> initial_state_indices,
           std::optional<Tensor> final_state_indices) {
    size_t seed = hash_combine(out,
                               conv_state,
                               final_conv_state,
                               qkv,
                               weight,
                               bias,
                               cu_seqlens,
                               initial_state_indices,
                               final_state_indices);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, CausalConv1d,
        seed,
        out->desc(),
        conv_state->desc(),
        final_conv_state.has_value() ? final_conv_state.value()->desc() : nullptr,
        qkv->desc(),
        weight->desc(),
        bias.has_value() ? bias.value()->desc() : nullptr,
        cu_seqlens.has_value() ? cu_seqlens.value()->desc() : nullptr,
        initial_state_indices.has_value() ? initial_state_indices.value()->desc() : nullptr,
        final_state_indices.has_value() ? final_state_indices.value()->desc() : nullptr);

    INFINIOP_WORKSPACE_TENSOR(workspace, CausalConv1d, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(out),
        graph::GraphTensor(conv_state),
        graph::GraphTensor(qkv),
        graph::GraphTensor(weight),
        final_conv_state.has_value() ? std::optional<graph::GraphTensor>(graph::GraphTensor(final_conv_state.value())) : std::nullopt,
        bias.has_value() ? std::optional<graph::GraphTensor>(graph::GraphTensor(bias.value())) : std::nullopt,
        cu_seqlens.has_value() ? std::optional<graph::GraphTensor>(graph::GraphTensor(cu_seqlens.value())) : std::nullopt,
        initial_state_indices.has_value() ? std::optional<graph::GraphTensor>(graph::GraphTensor(initial_state_indices.value())) : std::nullopt,
        final_state_indices.has_value() ? std::optional<graph::GraphTensor>(graph::GraphTensor(final_state_indices.value())) : std::nullopt};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopCausalConv1d(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->out->data(),
        planned->conv_state->data(),
        planned->final_conv_state.has_value() ? planned->final_conv_state.value()->data() : nullptr,
        planned->qkv->data(),
        planned->weight->data(),
        planned->bias.has_value() ? planned->bias.value()->data() : nullptr,
        planned->cu_seqlens.has_value() ? planned->cu_seqlens.value()->data() : nullptr,
        planned->initial_state_indices.has_value() ? planned->initial_state_indices.value()->data() : nullptr,
        planned->final_state_indices.has_value() ? planned->final_state_indices.value()->data() : nullptr,
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(CausalConv1d, &plan, &run, &cleanup);

} // namespace infinicore::op::causal_conv1d_impl::infiniop
