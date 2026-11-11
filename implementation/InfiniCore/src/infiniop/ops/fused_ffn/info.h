#ifndef __FUSED_FFN_INFO_H__
#define __FUSED_FFN_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::fused_ffn {

class FusedFFNInfo {
    FusedFFNInfo() = default;

public:
    infiniDtype_t dtype;
    infiniDtype_t wtype; // norm weight dtype
    infiniDtype_t mtype; // matrix weight dtype (gate_up, down)
    float epsilon;
    std::vector<size_t> shape;
    ptrdiff_t in_stride;
    ptrdiff_t out_stride;
    ptrdiff_t residual_stride;
    size_t hidden_dim;
    size_t intermediate_dim;
    bool has_residual;

    size_t ntok() const { return shape[0]; }
    size_t d() const { return hidden_dim; }
    size_t di() const { return intermediate_dim; }

    static utils::Result<FusedFFNInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t in_desc,
        infiniopTensorDescriptor_t residual_desc,
        infiniopTensorDescriptor_t norm_weight_desc,
        infiniopTensorDescriptor_t gate_up_weight_desc,
        infiniopTensorDescriptor_t down_weight_desc,
        float epsilon) {

        auto dtype = out_desc->dtype();
        auto wtype = norm_weight_desc->dtype();
        auto mtype = gate_up_weight_desc->dtype();

        // Check that input and output have the same dtype
        if (in_desc->dtype() != dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        // Check matrix weight dtypes (gate_up and down must match each other)
        if (down_weight_desc->dtype() != mtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        // Validate norm weight dtype: for half-precision activations, can be same or FP32
        if (dtype == INFINI_DTYPE_F16 || dtype == INFINI_DTYPE_BF16) {
            if (wtype != dtype && wtype != INFINI_DTYPE_F32) {
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }
        } else if (dtype == INFINI_DTYPE_F32) {
            if (wtype != INFINI_DTYPE_F32) {
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        // Validate matrix weight dtype: for half-precision activations, can be same or FP32
        if (dtype == INFINI_DTYPE_F16 || dtype == INFINI_DTYPE_BF16) {
            if (mtype != dtype && mtype != INFINI_DTYPE_F32) {
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }
        } else if (dtype == INFINI_DTYPE_F32) {
            if (mtype != INFINI_DTYPE_F32) {
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        // Check tensor dimensions
        const size_t out_ndim = out_desc->ndim();
        const size_t in_ndim = in_desc->ndim();

        // Must be 2D tensors [ntok, hidden_dim]
        if (out_ndim != 2 || in_ndim != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        size_t ntok = out_desc->dim(0);
        size_t hidden_dim = out_desc->dim(1);

        // Check input shape matches output
        if (in_desc->dim(0) != ntok || in_desc->dim(1) != hidden_dim) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // Check norm_weight is 1D [hidden_dim]
        if (norm_weight_desc->ndim() != 1 || norm_weight_desc->dim(0) != hidden_dim) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // Check gate_up_weight is 2D with shape either [2*intermediate_dim, hidden_dim] or [hidden_dim, 2*intermediate_dim]
        if (gate_up_weight_desc->ndim() != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        size_t gate_up_dim0 = gate_up_weight_desc->dim(0);
        size_t gate_up_dim1 = gate_up_weight_desc->dim(1);
        size_t intermediate_dim;
        if (gate_up_dim1 == hidden_dim && gate_up_dim0 % 2 == 0) {
            // Layout A: [2*intermediate_dim, hidden_dim]
            intermediate_dim = gate_up_dim0 / 2;
        } else if (gate_up_dim0 == hidden_dim && gate_up_dim1 % 2 == 0) {
            // Layout B: [hidden_dim, 2*intermediate_dim]
            intermediate_dim = gate_up_dim1 / 2;
        } else {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // Check down_weight is 2D with shape either [hidden_dim, intermediate_dim] or [intermediate_dim, hidden_dim]
        if (down_weight_desc->ndim() != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        {
            size_t dw_dim0 = down_weight_desc->dim(0);
            size_t dw_dim1 = down_weight_desc->dim(1);
            if (!((dw_dim0 == hidden_dim && dw_dim1 == intermediate_dim) || (dw_dim0 == intermediate_dim && dw_dim1 == hidden_dim))) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }

        // Check contiguity of the last dimension for activation tensors and norm weights
        if (out_desc->stride(out_ndim - 1) != 1 || in_desc->stride(in_ndim - 1) != 1 || norm_weight_desc->stride(0) != 1) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }
        // For matrix weights, at least one stride dimension must be 1 (contiguous along one axis)
        if (gate_up_weight_desc->stride(0) != 1 && gate_up_weight_desc->stride(1) != 1) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }
        if (down_weight_desc->stride(0) != 1 && down_weight_desc->stride(1) != 1) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }

        // Check residual if provided
        bool has_residual = residual_desc != nullptr;
        ptrdiff_t residual_stride = 0;
        if (has_residual) {
            if (residual_desc->ndim() != 2) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
            if (residual_desc->dim(0) != ntok || residual_desc->dim(1) != hidden_dim) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
            if (residual_desc->dtype() != dtype) {
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }
            if (residual_desc->stride(1) != 1) {
                return INFINI_STATUS_BAD_TENSOR_STRIDES;
            }
            residual_stride = residual_desc->stride(0);
        }

        FusedFFNInfo info;
        info.dtype = dtype;
        info.wtype = wtype;
        info.mtype = mtype;
        info.epsilon = epsilon;
        info.shape = out_desc->shape();
        info.in_stride = in_desc->stride(0);
        info.out_stride = out_desc->stride(0);
        info.residual_stride = residual_stride;
        info.hidden_dim = hidden_dim;
        info.intermediate_dim = intermediate_dim;
        info.has_residual = has_residual;

        return utils::Result<FusedFFNInfo>(info);
    }
};

} // namespace op::fused_ffn

#endif // __FUSED_FFN_INFO_H__
