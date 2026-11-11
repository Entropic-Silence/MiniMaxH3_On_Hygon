#include "infinicore/ops/nsa_paged_attention.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::nsa_paged_attention_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, NsaPagedAttention, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, out, q, k_cmp, v_cmp, k_cache, v_cache, block_tables, cache_lens, gates;
};

void *plan(Tensor out, const Tensor &q, const Tensor &k_cmp, const Tensor &v_cmp, const Tensor &k_cache, const Tensor &v_cache,
           const Tensor &block_tables, const Tensor &cache_lens, const Tensor &gates,
           float scale, int nsa_block_size, int window_size, int select_blocks) {
    size_t seed = hash_combine(out, q, k_cmp, v_cmp, k_cache, v_cache, block_tables, cache_lens, gates, nsa_block_size, window_size, select_blocks);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, NsaPagedAttention,
        seed,
        out->desc(), q->desc(), k_cmp->desc(), v_cmp->desc(), k_cache->desc(), v_cache->desc(),
        block_tables->desc(), cache_lens->desc(), gates->desc(),
        scale, nsa_block_size, window_size, select_blocks);

    INFINIOP_WORKSPACE_TENSOR(workspace, NsaPagedAttention, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(out),
        graph::GraphTensor(q),
        graph::GraphTensor(k_cmp),
        graph::GraphTensor(v_cmp),
        graph::GraphTensor(k_cache),
        graph::GraphTensor(v_cache),
        graph::GraphTensor(block_tables),
        graph::GraphTensor(cache_lens),
        graph::GraphTensor(gates)};
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(
        infiniopNsaPagedAttention(
            p->descriptor->desc,
            p->workspace->data(),
            p->workspace->numel(),
            p->out->data(),
            p->q->data(),
            p->k_cmp->data(),
            p->v_cmp->data(),
            p->k_cache->data(),
            p->v_cache->data(),
            p->block_tables->data(),
            p->cache_lens->data(),
            p->gates->data(),
            context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(NsaPagedAttention, &plan, &run, &cleanup);

} // namespace infinicore::op::nsa_paged_attention_impl::infiniop
