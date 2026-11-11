#include "infinicore/ops/hardtanh.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::hardtanh_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, HardTanh, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, output, input;
};

void *plan(Tensor output, Tensor input, float min_val, float max_val) {
    size_t seed = hash_combine(output, input, min_val, max_val);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, HardTanh,
        seed,
        output->desc(), input->desc(), min_val, max_val);

    INFINIOP_WORKSPACE_TENSOR(workspace, HardTanh, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(output),
        graph::GraphTensor(input)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopHardTanh(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->output->data(),
        planned->input->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(HardTanh, &plan, &run, &cleanup);

} // namespace infinicore::op::hardtanh_impl::infiniop
