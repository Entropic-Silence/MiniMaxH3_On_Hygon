// infiniop/ops/causal_conv1d/info.h

#ifndef __CAUSAL_CONV1D_INFO_H__
#define __CAUSAL_CONV1D_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

#include <vector>

namespace op {
namespace causal_conv1d {

class CausalConv1dInfo {
    CausalConv1dInfo() = default;

public:
    infiniDtype_t data_dtype;
    infiniDtype_t cu_seqlens_dtype;
    infiniDtype_t initial_state_indices_dtype;
    infiniDtype_t final_state_indices_dtype;

    bool has_bias;
    bool has_cu_seqlens;
    bool has_initial_state_indices;
    bool has_final_state_indices;
    bool indexed_state_pool;

    size_t B;
    size_t T;
    size_t C;
    size_t state_len;
    size_t kernel_size;
    size_t request_count;
    size_t total_tokens;
    size_t pool_size;

    std::vector<ptrdiff_t> out_strides;
    std::vector<ptrdiff_t> conv_state_strides;
    std::vector<ptrdiff_t> final_conv_state_strides;
    std::vector<ptrdiff_t> qkv_strides;
    std::vector<ptrdiff_t> weight_strides;
    std::vector<ptrdiff_t> bias_strides;

    static utils::Result<CausalConv1dInfo>
    create(infiniopTensorDescriptor_t out_desc,
           infiniopTensorDescriptor_t conv_state_desc,
           infiniopTensorDescriptor_t final_conv_state_desc,
           infiniopTensorDescriptor_t qkv_desc,
           infiniopTensorDescriptor_t weight_desc,
           infiniopTensorDescriptor_t bias_desc,
           infiniopTensorDescriptor_t cu_seqlens_desc,
           infiniopTensorDescriptor_t initial_state_indices_desc,
           infiniopTensorDescriptor_t final_state_indices_desc) {

        if (out_desc == nullptr || conv_state_desc == nullptr || qkv_desc == nullptr || weight_desc == nullptr) {
            return INFINI_STATUS_NULL_POINTER;
        }

        auto data_dtype = qkv_desc->dtype();
        CHECK_DTYPE(data_dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
        if (out_desc->dtype() != data_dtype || conv_state_desc->dtype() != data_dtype || weight_desc->dtype() != data_dtype || (final_conv_state_desc != nullptr && final_conv_state_desc->dtype() != data_dtype) || (bias_desc != nullptr && bias_desc->dtype() != data_dtype)) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        bool has_cu = cu_seqlens_desc != nullptr;
        bool has_initial_indices = initial_state_indices_desc != nullptr;
        bool has_final_indices = final_state_indices_desc != nullptr;
        bool indexed_pool = has_initial_indices || has_final_indices;

        if (has_final_indices && final_conv_state_desc != nullptr) {
            return INFINI_STATUS_BAD_PARAM;
        }
        if (!has_final_indices && final_conv_state_desc == nullptr) {
            return INFINI_STATUS_NULL_POINTER;
        }

        if (out_desc->ndim() != 3 || qkv_desc->ndim() != 3 || conv_state_desc->ndim() != 3 || weight_desc->ndim() != 3 || (!has_final_indices && final_conv_state_desc->ndim() != 3)) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (bias_desc != nullptr && bias_desc->ndim() != 1) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        auto qkv_shape = qkv_desc->shape();
        auto out_shape = out_desc->shape();
        auto state_shape = conv_state_desc->shape();
        auto weight_shape = weight_desc->shape();

        size_t B = qkv_shape[0];
        size_t T = qkv_shape[1];
        size_t C = qkv_shape[2];
        size_t total_tokens = B * T;
        size_t request_count = B;

        if (has_cu) {
            if (cu_seqlens_desc->ndim() != 1 || cu_seqlens_desc->shape()[0] < 2) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
            CHECK_DTYPE(cu_seqlens_desc->dtype(), INFINI_DTYPE_I32, INFINI_DTYPE_I64);
            if (B != 1 || out_shape[0] != 1) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
            request_count = cu_seqlens_desc->shape()[0] - 1;
            total_tokens = T;
        }

        if (out_shape[0] != qkv_shape[0] || out_shape[1] != qkv_shape[1] || out_shape[2] != C) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (qkv_desc->strides()[2] != 1 || out_desc->strides()[2] != 1) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }

        if (weight_shape[0] != C || weight_shape[1] != 1 || weight_shape[2] < 1) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        size_t kernel_size = weight_shape[2];
        size_t state_len = kernel_size - 1;
        if (kernel_size != 4) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (state_shape[1] != C || state_shape[2] != state_len) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (bias_desc != nullptr && bias_desc->shape()[0] != C) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        size_t pool_size = state_shape[0];
        if (!indexed_pool && state_shape[0] != request_count) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (!has_final_indices) {
            auto final_shape = final_conv_state_desc->shape();
            if (final_shape[0] != request_count || final_shape[1] != C || final_shape[2] != state_len) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }

        infiniDtype_t initial_indices_dtype = INFINI_DTYPE_INVALID;
        infiniDtype_t final_indices_dtype = INFINI_DTYPE_INVALID;
        if (has_initial_indices) {
            if (initial_state_indices_desc->ndim() != 1 || initial_state_indices_desc->shape()[0] != request_count) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
            initial_indices_dtype = initial_state_indices_desc->dtype();
            CHECK_DTYPE(initial_indices_dtype, INFINI_DTYPE_I32, INFINI_DTYPE_I64);
        }
        if (has_final_indices) {
            if (final_state_indices_desc->ndim() != 1 || final_state_indices_desc->shape()[0] != request_count) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
            final_indices_dtype = final_state_indices_desc->dtype();
            CHECK_DTYPE(final_indices_dtype, INFINI_DTYPE_I32, INFINI_DTYPE_I64);
        }

        CausalConv1dInfo info;
        info.data_dtype = data_dtype;
        info.cu_seqlens_dtype = has_cu ? cu_seqlens_desc->dtype() : INFINI_DTYPE_INVALID;
        info.initial_state_indices_dtype = initial_indices_dtype;
        info.final_state_indices_dtype = final_indices_dtype;
        info.has_bias = bias_desc != nullptr;
        info.has_cu_seqlens = has_cu;
        info.has_initial_state_indices = has_initial_indices;
        info.has_final_state_indices = has_final_indices;
        info.indexed_state_pool = indexed_pool;
        info.B = B;
        info.T = T;
        info.C = C;
        info.state_len = state_len;
        info.kernel_size = kernel_size;
        info.request_count = request_count;
        info.total_tokens = total_tokens;
        info.pool_size = pool_size;
        info.out_strides = out_desc->strides();
        info.conv_state_strides = conv_state_desc->strides();
        if (final_conv_state_desc != nullptr) {
            info.final_conv_state_strides = final_conv_state_desc->strides();
        }
        info.qkv_strides = qkv_desc->strides();
        info.weight_strides = weight_desc->strides();
        if (bias_desc != nullptr) {
            info.bias_strides = bias_desc->strides();
        }
        return utils::Result<CausalConv1dInfo>(info);
    }
};

} // namespace causal_conv1d
} // namespace op

#endif // __CAUSAL_CONV1D_INFO_H__
