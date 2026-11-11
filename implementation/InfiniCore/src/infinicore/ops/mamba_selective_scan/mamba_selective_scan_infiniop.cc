#include "../infiniop_impl.hpp"
#include "infinicore/ops/mamba_selective_scan.hpp"

namespace infinicore::op::mamba_selective_scan_impl::infiniop {
INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, MambaSelectiveScan, 100);
struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, out, x, dt, b, c, a_log, d, gate, dt_bias, state;
};
void *plan(Tensor out, const Tensor &x, const Tensor &dt, const Tensor &b, const Tensor &c, const Tensor &a_log, const Tensor &d, const Tensor &gate, const Tensor &dt_bias, Tensor state) {
    size_t seed = hash_combine(out, x, dt, b, c, a_log, d, gate, dt_bias, state);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(Descriptor, descriptor, MambaSelectiveScan, seed, out->desc(), x->desc(), dt->desc(), b->desc(), c->desc(), a_log->desc(), d->desc(), gate->desc(), dt_bias->desc(), state->desc());
    INFINIOP_WORKSPACE_TENSOR(workspace, MambaSelectiveScan, descriptor);
    return new PlannedMeta{descriptor, graph::GraphTensor(workspace), graph::GraphTensor(out), graph::GraphTensor(x), graph::GraphTensor(dt), graph::GraphTensor(b), graph::GraphTensor(c), graph::GraphTensor(a_log), graph::GraphTensor(d), graph::GraphTensor(gate), graph::GraphTensor(dt_bias), graph::GraphTensor(state)};
}
void run(void *planned_meta) {
    auto p = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopMambaSelectiveScan(p->descriptor->desc, p->workspace->data(), p->workspace->numel(), p->out->data(), p->x->data(), p->dt->data(), p->b->data(), p->c->data(), p->a_log->data(), p->d->data(), p->gate->data(), p->dt_bias->data(), p->state->data(), context::getStream()));
}
void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(MambaSelectiveScan, &plan, &run, &cleanup);
} // namespace infinicore::op::mamba_selective_scan_impl::infiniop
