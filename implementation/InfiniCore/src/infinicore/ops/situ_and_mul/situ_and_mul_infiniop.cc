#include "infinicore/ops/situ_and_mul.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::situ_and_mul_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, SituAndMul, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, output, gate, up;
    float beta;
    float linear_beta;
};

void *plan(Tensor output,
           const Tensor &gate,
           const Tensor &up,
           float beta,
           float linear_beta) {
    size_t seed = hash_combine(output, gate, up);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, SituAndMul,
        seed, output->desc(), gate->desc(), up->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, SituAndMul, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(output),
        graph::GraphTensor(gate),
        graph::GraphTensor(up),
        beta,
        linear_beta};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopSituAndMul(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->output->data(),
        planned->gate->data(),
        planned->up->data(),
        planned->beta,
        planned->linear_beta,
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(SituAndMul, &plan, &run, &cleanup);

} // namespace infinicore::op::situ_and_mul_impl::infiniop
