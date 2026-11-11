#ifndef __MROPE_H__
#define __MROPE_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                               \
    namespace op::mrope::NAMESPACE {                                        \
    class Descriptor final : public InfiniopDescriptor {                    \
        struct Opaque;                                                      \
        Opaque *_opaque;                                                    \
        MRoPEInfo _info;                                                    \
        size_t _workspace_size;                                             \
                                                                            \
        Descriptor(MRoPEInfo info, size_t workspace_size,                   \
                   Opaque *opaque, infiniDevice_t device_type,              \
                   int device_id)                                           \
            : InfiniopDescriptor{device_type, device_id},                   \
              _opaque(opaque), _info(info),                                 \
              _workspace_size(workspace_size) {}                            \
                                                                            \
    public:                                                                 \
        ~Descriptor();                                                      \
        size_t workspaceSize() const { return _workspace_size; }            \
                                                                            \
        static infiniStatus_t create(                                       \
            infiniopHandle_t handle, Descriptor **desc_ptr,                 \
            infiniopTensorDescriptor_t q_out_desc,                          \
            infiniopTensorDescriptor_t k_out_desc,                          \
            infiniopTensorDescriptor_t q_desc,                              \
            infiniopTensorDescriptor_t k_desc,                              \
            infiniopTensorDescriptor_t cos_desc,                            \
            infiniopTensorDescriptor_t sin_desc,                            \
            infiniopTensorDescriptor_t positions_desc,                      \
            int head_size, int rotary_dim,                                  \
            int section_t, int section_h, int section_w, bool interleaved); \
                                                                            \
        infiniStatus_t calculate(                                           \
            void *workspace, size_t workspace_size, void *q_out,            \
            void *k_out, const void *q, const void *k,                      \
            const void *cos, const void *sin, const void *positions,        \
            void *stream) const;                                            \
    };                                                                      \
    }

#endif // __MROPE_H__
