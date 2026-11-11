#ifndef MAMBA_SELECTIVE_SCAN_H
#define MAMBA_SELECTIVE_SCAN_H

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                                                 \
    namespace op::mamba_selective_scan::NAMESPACE {                                           \
    class Descriptor final : public InfiniopDescriptor {                                      \
        struct Opaque;                                                                        \
        Opaque *_opaque;                                                                      \
        MambaSelectiveScanInfo _info;                                                         \
        size_t _workspace_size;                                                               \
        Descriptor(Opaque *opaque, MambaSelectiveScanInfo info, size_t workspace_size,        \
                   infiniDevice_t device_type, int device_id)                                 \
            : InfiniopDescriptor{device_type, device_id}, _opaque(opaque),                    \
              _info(info), _workspace_size(workspace_size) {}                                 \
                                                                                              \
    public:                                                                                   \
        ~Descriptor();                                                                        \
        size_t workspaceSize() const { return _workspace_size; }                              \
        static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr,          \
                                     infiniopTensorDescriptor_t out_desc,                     \
                                     infiniopTensorDescriptor_t x_desc,                       \
                                     infiniopTensorDescriptor_t dt_desc,                      \
                                     infiniopTensorDescriptor_t b_desc,                       \
                                     infiniopTensorDescriptor_t c_desc,                       \
                                     infiniopTensorDescriptor_t a_log_desc,                   \
                                     infiniopTensorDescriptor_t d_desc,                       \
                                     infiniopTensorDescriptor_t gate_desc,                    \
                                     infiniopTensorDescriptor_t dt_bias_desc,                 \
                                     infiniopTensorDescriptor_t state_desc);                  \
        infiniStatus_t calculate(void *workspace, size_t workspace_size, void *out,           \
                                 const void *x, const void *dt, const void *b, const void *c, \
                                 const void *a_log, const void *d, const void *gate,          \
                                 const void *dt_bias, void *state, void *stream) const;       \
    };                                                                                        \
    }

#endif
