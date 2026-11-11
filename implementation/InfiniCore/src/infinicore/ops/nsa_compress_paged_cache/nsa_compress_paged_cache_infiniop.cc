#include "infinicore/ops/nsa_compress_paged_cache.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::nsa_compress_paged_cache_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, NsaCompressPagedCache, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, k_cmp, v_cmp, k_cache, v_cache, block_tables, cache_lens;
};

void *plan(Tensor k_cmp, Tensor v_cmp, const Tensor &k_cache, const Tensor &v_cache,
           const Tensor &block_tables, const Tensor &cache_lens, int nsa_block_size, bool update_last_only) {
    size_t seed = hash_combine(k_cmp, v_cmp, k_cache, v_cache, block_tables, cache_lens, nsa_block_size, update_last_only);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, NsaCompressPagedCache,
        seed,
        k_cmp->desc(), v_cmp->desc(), k_cache->desc(), v_cache->desc(),
        block_tables->desc(), cache_lens->desc(), nsa_block_size, static_cast<int>(update_last_only));

    INFINIOP_WORKSPACE_TENSOR(workspace, NsaCompressPagedCache, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(k_cmp),
        graph::GraphTensor(v_cmp),
        graph::GraphTensor(k_cache),
        graph::GraphTensor(v_cache),
        graph::GraphTensor(block_tables),
        graph::GraphTensor(cache_lens)};
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(
        infiniopNsaCompressPagedCache(
            p->descriptor->desc,
            p->workspace->data(),
            p->workspace->numel(),
            p->k_cmp->data(),
            p->v_cmp->data(),
            p->k_cache->data(),
            p->v_cache->data(),
            p->block_tables->data(),
            p->cache_lens->data(),
            context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(NsaCompressPagedCache, &plan, &run, &cleanup);

} // namespace infinicore::op::nsa_compress_paged_cache_impl::infiniop
