#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/situ_and_mul.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_HYGON_API)
#include "nvidia/situ_and_mul_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateSituAndMulDescriptor(
    infiniopHandle_t handle,
    infiniopSituAndMulDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t gate_desc,
    infiniopTensorDescriptor_t up_desc) {

#define CREATE(CASE, NAMESPACE)                                                     \
    case CASE:                                                                      \
        return op::situ_and_mul::NAMESPACE::Descriptor::create(                     \
            handle,                                                                 \
            reinterpret_cast<op::situ_and_mul::NAMESPACE::Descriptor **>(desc_ptr), \
            output_desc,                                                            \
            gate_desc,                                                              \
            up_desc)

    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CREATE(INFINI_DEVICE_HYGON, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetSituAndMulWorkspaceSize(
    infiniopSituAndMulDescriptor_t desc,
    size_t *size) {

#define GET(CASE, NAMESPACE)                                                                        \
    case CASE:                                                                                      \
        *size = reinterpret_cast<op::situ_and_mul::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        GET(INFINI_DEVICE_HYGON, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef GET
}

__INFINI_C infiniStatus_t infiniopSituAndMul(
    infiniopSituAndMulDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *gate,
    const void *up,
    float beta,
    float linear_beta,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                     \
    case CASE:                                                                         \
        return reinterpret_cast<const op::situ_and_mul::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, output, gate, up, beta, linear_beta, stream)

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CALCULATE(INFINI_DEVICE_HYGON, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDestroySituAndMulDescriptor(
    infiniopSituAndMulDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                         \
    case CASE:                                                                          \
        delete reinterpret_cast<const op::situ_and_mul::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DELETE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        DELETE(INFINI_DEVICE_HYGON, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}
