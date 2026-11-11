#ifndef __MAMBA_SELECTIVE_SCAN_INFO_H__
#define __MAMBA_SELECTIVE_SCAN_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::mamba_selective_scan {

class MambaSelectiveScanInfo {
    MambaSelectiveScanInfo() = default;

public:
    infiniDtype_t dtype;
    size_t batch;
    size_t seq_len;
    size_t intermediate;
    size_t state_size;

    static utils::Result<MambaSelectiveScanInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t dt_desc,
        infiniopTensorDescriptor_t b_desc,
        infiniopTensorDescriptor_t c_desc,
        infiniopTensorDescriptor_t a_log_desc,
        infiniopTensorDescriptor_t d_desc,
        infiniopTensorDescriptor_t gate_desc,
        infiniopTensorDescriptor_t dt_bias_desc,
        infiniopTensorDescriptor_t state_desc) {
        auto dtype = x_desc->dtype();
        if (dtype != INFINI_DTYPE_F16 && dtype != INFINI_DTYPE_BF16 && dtype != INFINI_DTYPE_F32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (out_desc->dtype() != dtype || dt_desc->dtype() != dtype || b_desc->dtype() != dtype || c_desc->dtype() != dtype || gate_desc->dtype() != dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (a_log_desc->dtype() != dtype || d_desc->dtype() != dtype || dt_bias_desc->dtype() != dtype || state_desc->dtype() != INFINI_DTYPE_F32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (out_desc->ndim() != 3 || x_desc->ndim() != 3 || dt_desc->ndim() != 3 || b_desc->ndim() != 3 || c_desc->ndim() != 3 || a_log_desc->ndim() != 2 || d_desc->ndim() != 1 || gate_desc->ndim() != 3 || dt_bias_desc->ndim() != 1 || state_desc->ndim() != 3) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        auto xs = x_desc->shape();
        if (out_desc->shape() != xs || dt_desc->shape() != xs || gate_desc->shape() != xs) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        size_t batch = xs[0], seq_len = xs[1], intermediate = xs[2];
        auto bs = b_desc->shape();
        if (c_desc->shape() != bs || bs[0] != batch || bs[1] != seq_len) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        size_t state_size = bs[2];
        if (a_log_desc->shape()[0] != intermediate || a_log_desc->shape()[1] != state_size) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (d_desc->shape()[0] != intermediate || dt_bias_desc->shape()[0] != intermediate) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        auto ss = state_desc->shape();
        if (ss[0] < batch || ss[1] != intermediate || ss[2] != state_size) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (out_desc->strides()[2] != 1 || x_desc->strides()[2] != 1 || dt_desc->strides()[2] != 1 || b_desc->strides()[2] != 1 || c_desc->strides()[2] != 1 || a_log_desc->strides()[1] != 1 || state_desc->strides()[2] != 1) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }
        return utils::Result<MambaSelectiveScanInfo>(MambaSelectiveScanInfo{dtype, batch, seq_len, intermediate, state_size});
    }
};

} // namespace op::mamba_selective_scan
#endif
