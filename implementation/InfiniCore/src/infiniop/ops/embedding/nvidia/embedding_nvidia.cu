#include "../../../../utils.h"
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../tensor.h"
#include "../cuda/embedding_kernel.cuh"
#include "embedding_nvidia.cuh"
#include <cstdint>
#include <cuda_runtime.h>

template <typename T, typename IndexType>
INFINIOP_CUDA_KERNEL embeddingKernel(
    T *__restrict__ output,
    const IndexType *__restrict__ indices,
    const T *__restrict__ weight,
    size_t num_indices,
    size_t embedding_dim,
    size_t vocab_size) {
    const size_t idx = blockIdx.x;
    if (idx >= num_indices) {
        return;
    }

    __shared__ IndexType index_val;
    if (threadIdx.x == 0) {
        index_val = indices[idx];
    }
    __syncthreads();

    if (index_val < 0 || static_cast<size_t>(index_val) >= vocab_size) {
        return;
    }

    const T *src = weight + static_cast<size_t>(index_val) * embedding_dim;
    T *dst = output + idx * embedding_dim;

    constexpr size_t VECTOR_BYTES = sizeof(uint4);
    constexpr size_t ELEMENTS_PER_VECTOR = VECTOR_BYTES / sizeof(T);
    const bool vectorized = reinterpret_cast<uintptr_t>(src) % VECTOR_BYTES == 0
                         && reinterpret_cast<uintptr_t>(dst) % VECTOR_BYTES == 0
                         && embedding_dim % ELEMENTS_PER_VECTOR == 0;

    if (vectorized) {
        const auto *src_vec = reinterpret_cast<const uint4 *>(src);
        auto *dst_vec = reinterpret_cast<uint4 *>(dst);
        const size_t vector_count = embedding_dim / ELEMENTS_PER_VECTOR;
        for (size_t i = threadIdx.x; i < vector_count; i += blockDim.x) {
            dst_vec[i] = src_vec[i];
        }
    } else {
        for (size_t i = threadIdx.x; i < embedding_dim; i += blockDim.x) {
            dst[i] = src[i];
        }
    }
}

namespace op::embedding::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc) {

    auto input_shape = input_desc->shape();
    auto weight_shape = weight_desc->shape();

    // Validate shapes
    CHECK_OR_RETURN(weight_shape.size() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(output_desc->shape().size() == input_shape.size() + 1, INFINI_STATUS_BAD_TENSOR_SHAPE);

    // Check output shape matches input shape + embedding_dim
    auto output_shape = output_desc->shape();
    size_t embedding_dim = weight_shape[1];
    CHECK_OR_RETURN(output_shape.back() == embedding_dim, INFINI_STATUS_BAD_TENSOR_SHAPE);

    for (size_t i = 0; i < input_shape.size(); ++i) {
        CHECK_OR_RETURN(output_shape[i] == input_shape[i], INFINI_STATUS_BAD_TENSOR_SHAPE);
    }

    // Validate dtypes
    auto input_dtype = input_desc->dtype();
    auto weight_dtype = weight_desc->dtype();
    CHECK_OR_RETURN(input_dtype == INFINI_DTYPE_I32 || input_dtype == INFINI_DTYPE_I64,
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(weight_dtype == INFINI_DTYPE_F32 || weight_dtype == INFINI_DTYPE_F16 || weight_dtype == INFINI_DTYPE_BF16, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(output_desc->dtype() == weight_dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);

    // Calculate number of indices (supporting batch dimension)
    size_t num_indices = 1;
    for (auto dim : input_shape) {
        num_indices *= dim;
    }

    size_t vocab_size = weight_shape[0];

    *desc_ptr = new Descriptor(
        num_indices,
        embedding_dim,
        vocab_size,
        input_dtype,
        weight_dtype,
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *output,
    const void *input,
    const void *weight,
    void *stream) const {

    if (_num_indices == 0) {
        return INFINI_STATUS_SUCCESS;
    }

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    // One block cooperatively copies one embedding row. The previous mapping used
    // one thread per row and serialized large language-model embeddings.
    size_t block_size = 256;
    if (_embedding_dim <= 64) {
        block_size = 32;
    } else if (_embedding_dim <= 256) {
        block_size = 64;
    } else if (_embedding_dim <= 1024) {
        block_size = 128;
    }

    size_t grid_size = _num_indices;

    // Launch kernel based on dtypes
    if (_input_dtype == INFINI_DTYPE_I32) {
        const int32_t *indices_ptr = reinterpret_cast<const int32_t *>(input);

        if (_weight_dtype == INFINI_DTYPE_F32) {
            embeddingKernel<float, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                reinterpret_cast<float *>(output),
                indices_ptr,
                reinterpret_cast<const float *>(weight),
                _num_indices,
                _embedding_dim,
                _vocab_size);
        } else if (_weight_dtype == INFINI_DTYPE_F16) {
            embeddingKernel<half, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                reinterpret_cast<half *>(output),
                indices_ptr,
                reinterpret_cast<const half *>(weight),
                _num_indices,
                _embedding_dim,
                _vocab_size);
        } else if (_weight_dtype == INFINI_DTYPE_BF16) {
            embeddingKernel<cuda_bfloat16, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                reinterpret_cast<cuda_bfloat16 *>(output),
                indices_ptr,
                reinterpret_cast<const cuda_bfloat16 *>(weight),
                _num_indices,
                _embedding_dim,
                _vocab_size);
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else if (_input_dtype == INFINI_DTYPE_I64) {
        const int64_t *indices_ptr = reinterpret_cast<const int64_t *>(input);

        if (_weight_dtype == INFINI_DTYPE_F32) {
            embeddingKernel<float, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                reinterpret_cast<float *>(output),
                indices_ptr,
                reinterpret_cast<const float *>(weight),
                _num_indices,
                _embedding_dim,
                _vocab_size);
        } else if (_weight_dtype == INFINI_DTYPE_F16) {
            embeddingKernel<half, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                reinterpret_cast<half *>(output),
                indices_ptr,
                reinterpret_cast<const half *>(weight),
                _num_indices,
                _embedding_dim,
                _vocab_size);
        } else if (_weight_dtype == INFINI_DTYPE_BF16) {
            embeddingKernel<cuda_bfloat16, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                reinterpret_cast<cuda_bfloat16 *>(output),
                indices_ptr,
                reinterpret_cast<const cuda_bfloat16 *>(weight),
                _num_indices,
                _embedding_dim,
                _vocab_size);
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::embedding::nvidia
