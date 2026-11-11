#include "infinicore/ops/mrope.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::mrope_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, MRoPE, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace;
    graph::GraphTensor q_out;
    graph::GraphTensor k_out;
    graph::GraphTensor q;
    graph::GraphTensor k;
    graph::GraphTensor cos;
    graph::GraphTensor sin;
    graph::GraphTensor positions;
};

void *plan(Tensor q_out,
           Tensor k_out,
           const Tensor &q,
           const Tensor &k,
           const Tensor &cos,
           const Tensor &sin,
           const Tensor &positions,
           int head_size,
           int rotary_dim,
           int section_t,
           int section_h,
           int section_w,
           bool interleaved) {
    size_t key = hash_combine(q_out,
                              k_out,
                              q,
                              k,
                              cos,
                              sin,
                              positions,
                              head_size,
                              rotary_dim,
                              section_t,
                              section_h,
                              section_w,
                              interleaved);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor,
        descriptor,
        MRoPE,
        key,
        q_out->desc(),
        k_out->desc(),
        q->desc(),
        k->desc(),
        cos->desc(),
        sin->desc(),
        positions->desc(),
        head_size,
        rotary_dim,
        section_t,
        section_h,
        section_w,
        interleaved);

    INFINIOP_WORKSPACE_TENSOR(workspace, MRoPE, descriptor);
    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(q_out),
        graph::GraphTensor(k_out),
        graph::GraphTensor(q),
        graph::GraphTensor(k),
        graph::GraphTensor(cos),
        graph::GraphTensor(sin),
        graph::GraphTensor(positions)};
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(
        infiniopMRoPE(
            p->descriptor->desc,
            p->workspace->data(),
            p->workspace->numel(),
            p->q_out->data(),
            p->k_out->data(),
            p->q->data(),
            p->k->data(),
            p->cos->data(),
            p->sin->data(),
            p->positions->data(),
            context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(MRoPE, &plan, &run, &cleanup);

} // namespace infinicore::op::mrope_impl::infiniop
