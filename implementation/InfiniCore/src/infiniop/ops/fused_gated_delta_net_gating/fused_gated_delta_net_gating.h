#ifndef __FUSED_GATED_DELTA_NET_GATING_H__
#define __FUSED_GATED_DELTA_NET_GATING_H__

#include "../../operator.h"
#include "info.h"

#include <utility>

#define DESCRIPTOR(NAMESPACE)                                                                                                           \
    namespace op::fused_gated_delta_net_gating::NAMESPACE {                                                                             \
    class Descriptor final : public InfiniopDescriptor {                                                                                \
        struct Opaque;                                                                                                                  \
        Opaque *_opaque;                                                                                                                \
        FusedGatedDeltaNetGatingInfo _info;                                                                                             \
        size_t _workspace_size;                                                                                                         \
                                                                                                                                        \
        Descriptor(Opaque *opaque, FusedGatedDeltaNetGatingInfo info, size_t workspace_size, infiniDevice_t device_type, int device_id) \
            : InfiniopDescriptor{device_type, device_id}, _opaque(opaque), _info(std::move(info)), _workspace_size(workspace_size) {}   \
                                                                                                                                        \
    public:                                                                                                                             \
        ~Descriptor();                                                                                                                  \
        size_t workspaceSize() const { return _workspace_size; }                                                                        \
                                                                                                                                        \
        static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr,                                                    \
                                     infiniopTensorDescriptor_t g_desc, infiniopTensorDescriptor_t beta_output_desc,                    \
                                     infiniopTensorDescriptor_t A_log_desc, infiniopTensorDescriptor_t a_desc,                          \
                                     infiniopTensorDescriptor_t b_desc, infiniopTensorDescriptor_t dt_bias_desc,                        \
                                     float beta, float threshold);                                                                      \
                                                                                                                                        \
        infiniStatus_t calculate(void *workspace, size_t workspace_size, void *g, void *beta_output,                                    \
                                 const void *A_log, const void *a, const void *b, const void *dt_bias, void *stream) const;             \
    };                                                                                                                                  \
    }

#endif
