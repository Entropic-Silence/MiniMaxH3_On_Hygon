#ifndef __INFINIOP_SITU_AND_MUL_API_H__
#define __INFINIOP_SITU_AND_MUL_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopSituAndMulDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateSituAndMulDescriptor(
    infiniopHandle_t handle,
    infiniopSituAndMulDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output,
    infiniopTensorDescriptor_t gate,
    infiniopTensorDescriptor_t up);

__INFINI_C __export infiniStatus_t infiniopGetSituAndMulWorkspaceSize(
    infiniopSituAndMulDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopSituAndMul(
    infiniopSituAndMulDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *gate,
    const void *up,
    float beta,
    float linear_beta,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroySituAndMulDescriptor(
    infiniopSituAndMulDescriptor_t desc);

#endif
