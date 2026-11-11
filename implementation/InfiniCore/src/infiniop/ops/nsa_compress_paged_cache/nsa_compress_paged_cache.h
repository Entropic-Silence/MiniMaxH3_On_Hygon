#ifndef NSA_COMPRESS_PAGED_CACHE_H
#define NSA_COMPRESS_PAGED_CACHE_H

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                                        \
    namespace op::nsa_compress_paged_cache::NAMESPACE {                              \
    class Descriptor final : public InfiniopDescriptor {                             \
        struct Opaque;                                                               \
        Opaque *_opaque;                                                             \
        NsaCompressPagedCacheInfo _info;                                             \
        size_t _workspace_size;                                                      \
                                                                                     \
        Descriptor(Opaque *opaque, NsaCompressPagedCacheInfo info,                   \
                   size_t workspace_size, infiniDevice_t device_type, int device_id) \
            : InfiniopDescriptor{device_type, device_id}, _opaque(opaque),           \
              _info(info), _workspace_size(workspace_size) {}                        \
                                                                                     \
    public:                                                                          \
        ~Descriptor();                                                               \
        size_t workspaceSize() const { return _workspace_size; }                     \
        static infiniStatus_t create(                                                \
            infiniopHandle_t handle, Descriptor **desc_ptr,                          \
            infiniopTensorDescriptor_t k_cmp_desc,                                   \
            infiniopTensorDescriptor_t v_cmp_desc,                                   \
            infiniopTensorDescriptor_t k_cache_desc,                                 \
            infiniopTensorDescriptor_t v_cache_desc,                                 \
            infiniopTensorDescriptor_t block_tables_desc,                            \
            infiniopTensorDescriptor_t seq_lens_desc, int nsa_block_size,            \
            int update_last_only);                                                   \
        infiniStatus_t calculate(                                                    \
            void *workspace, size_t workspace_size, void *k_cmp, void *v_cmp,        \
            const void *k_cache, const void *v_cache, const void *block_tables,      \
            const void *seq_lens, void *stream) const;                               \
    };                                                                               \
    }

#endif // NSA_COMPRESS_PAGED_CACHE_H
