#ifndef __SELECT_LAST_TOKEN_HIDDEN_CUDA_KERNEL_CUH__
#define __SELECT_LAST_TOKEN_HIDDEN_CUDA_KERNEL_CUH__

#include <cstddef>
#include <cstdint>

namespace op::select_last_token_hidden::cuda {

template <typename CopyT>
__device__ __forceinline__ void selectLastTokenHiddenBlock(
    CopyT *__restrict__ output,
    const CopyT *__restrict__ hidden_states,
    const int32_t *__restrict__ input_offsets,
    size_t row_width,
    size_t total_tokens) {
    const size_t request = blockIdx.x;
    const int32_t row = input_offsets[request + 1] - 1;
    if (row < 0 || static_cast<size_t>(row) >= total_tokens) {
        return;
    }

    const CopyT *src = hidden_states + static_cast<size_t>(row) * row_width;
    CopyT *dst = output + request * row_width;
    for (size_t column = threadIdx.x; column < row_width; column += blockDim.x) {
        dst[column] = src[column];
    }
}

} // namespace op::select_last_token_hidden::cuda

#endif
