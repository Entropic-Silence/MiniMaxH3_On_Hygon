#ifndef NSA_PAGED_ATTENTION_H
#define NSA_PAGED_ATTENTION_H

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                         \
    namespace op::nsa_paged_attention::NAMESPACE {                    \
    class Descriptor final : public InfiniopDescriptor {              \
        struct Opaque;                                                \
        Opaque *_opaque;                                              \
        NsaPagedAttentionInfo _info;                                  \
        size_t _workspace_size;                                       \
                                                                      \
        Descriptor(Opaque *opaque, NsaPagedAttentionInfo info,        \
                   size_t workspace_size, infiniDevice_t device_type, \
                   int device_id)                                     \
            : InfiniopDescriptor{device_type, device_id},             \
              _opaque(opaque), _info(info),                           \
              _workspace_size(workspace_size) {}                      \
                                                                      \
    public:                                                           \
        ~Descriptor();                                                \
        size_t workspaceSize() const { return _workspace_size; }      \
                                                                      \
        static infiniStatus_t create(                                 \
            infiniopHandle_t handle, Descriptor **desc_ptr,           \
            infiniopTensorDescriptor_t out_desc,                      \
            infiniopTensorDescriptor_t q_desc,                        \
            infiniopTensorDescriptor_t k_cmp_desc,                    \
            infiniopTensorDescriptor_t v_cmp_desc,                    \
            infiniopTensorDescriptor_t k_cache_desc,                  \
            infiniopTensorDescriptor_t v_cache_desc,                  \
            infiniopTensorDescriptor_t block_tables_desc,             \
            infiniopTensorDescriptor_t seq_lens_desc,                 \
            infiniopTensorDescriptor_t gates_desc, float scale,       \
            int nsa_block_size, int window_size, int select_blocks);  \
                                                                      \
        infiniStatus_t calculate(                                     \
            void *workspace, size_t workspace_size, void *out,        \
            const void *q, const void *k_cmp, const void *v_cmp,      \
            const void *k_cache, const void *v_cache,                 \
            const void *block_tables, const void *seq_lens,           \
            const void *gates, void *stream) const;                   \
    };                                                                \
    }

#endif // NSA_PAGED_ATTENTION_H
