#include "situ_and_mul_nvidia.cuh"

#include "../../../elementwise/nvidia/elementwise_nvidia.cuh"
#include "../cuda/kernel.cuh"

namespace op::situ_and_mul::nvidia {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t gate_desc,
    infiniopTensorDescriptor_t up_desc) {

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = output_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
    CHECK_OR_RETURN(gate_desc->dtype() == dtype && up_desc->dtype() == dtype,
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_SAME_SHAPE(output_desc->shape(), gate_desc->shape(), up_desc->shape());

    std::vector<infiniopTensorDescriptor_t> input_descs{gate_desc, up_desc};
    CREATE_ELEMENTWISE_CUDA_DESCRIPTOR(handle, dtype, output_desc, input_descs)
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *gate,
    const void *up,
    float beta,
    float linear_beta,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    if (beta <= 0.0f || linear_beta <= 0.0f) {
        return INFINI_STATUS_BAD_PARAM;
    }

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return _device_info->calculate<256, cuda::SituAndMulOp, half>(
            _info, workspace, output, {gate, up}, stream, beta, linear_beta);
    case INFINI_DTYPE_BF16:
        return _device_info->calculate<256, cuda::SituAndMulOp, cuda_bfloat16>(
            _info, workspace, output, {gate, up}, stream, beta, linear_beta);
    case INFINI_DTYPE_F32:
        return _device_info->calculate<256, cuda::SituAndMulOp, float>(
            _info, workspace, output, {gate, up}, stream, beta, linear_beta);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::situ_and_mul::nvidia
