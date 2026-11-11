#ifndef __INFINIOP_CAUSAL_CONV1D_H__
#define __INFINIOP_CAUSAL_CONV1D_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                      \
                                                                   \
    namespace op::causal_conv1d::NAMESPACE {                       \
    class Descriptor final : public InfiniopDescriptor {           \
        struct Opaque;                                             \
        Opaque *_opaque;                                           \
        CausalConv1dInfo _info;                                    \
        size_t _workspace_size;                                    \
                                                                   \
        Descriptor(Opaque *opaque, CausalConv1dInfo info,          \
                   size_t workspace_size,                          \
                   infiniDevice_t device_type, int device_id)      \
            : InfiniopDescriptor{device_type, device_id},          \
              _opaque(opaque), _info(info),                        \
              _workspace_size(workspace_size) {}                   \
                                                                   \
    public:                                                        \
        ~Descriptor();                                             \
        size_t workspaceSize() const { return _workspace_size; }   \
                                                                   \
        static infiniStatus_t create(                              \
            infiniopHandle_t handle, Descriptor **desc_ptr,        \
            infiniopTensorDescriptor_t out_desc,                   \
            infiniopTensorDescriptor_t conv_state_desc,            \
            infiniopTensorDescriptor_t final_conv_state_desc,      \
            infiniopTensorDescriptor_t qkv_desc,                   \
            infiniopTensorDescriptor_t weight_desc,                \
            infiniopTensorDescriptor_t bias_desc,                  \
            infiniopTensorDescriptor_t cu_seqlens_desc,            \
            infiniopTensorDescriptor_t initial_state_indices_desc, \
            infiniopTensorDescriptor_t final_state_indices_desc);  \
                                                                   \
        infiniStatus_t calculate(                                  \
            void *workspace, size_t workspace_size,                \
            void *out, void *conv_state, void *final_conv_state,   \
            const void *qkv, const void *weight, const void *bias, \
            const void *cu_seqlens,                                \
            const void *initial_state_indices,                     \
            const void *final_state_indices,                       \
            void *stream) const;                                   \
    };                                                             \
    }

#endif // __INFINIOP_CAUSAL_CONV1D_H__
