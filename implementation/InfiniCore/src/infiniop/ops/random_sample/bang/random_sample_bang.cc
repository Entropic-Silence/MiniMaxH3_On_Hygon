#include "random_sample_bang.h"

#include "../../../devices/bang/common_bang.h"

#include <algorithm>
#include <cstdint>

namespace op::random_sample::bang {

static size_t alignUp(size_t value, size_t alignment = 128) {
    return (value + alignment - 1) / alignment * alignment;
}

struct Descriptor::Opaque {
    std::shared_ptr<device::bang::Handle::Internal> internal;
    cnnlTensorDescriptor_t values_desc = nullptr;
    cnnlTensorDescriptor_t softmax_desc = nullptr;
    cnnlTensorDescriptor_t indices_desc = nullptr;
    cnnlTensorDescriptor_t scalar_value_desc = nullptr;
    cnnlTensorDescriptor_t scalar_index_desc = nullptr;
    cnnlTensorDescriptor_t result_desc = nullptr;
    cnnlTensorDescriptor_t mask_desc = nullptr;
    cnnlTensorDescriptor_t scalar_mask_desc = nullptr;
    size_t op_workspace_size = 0;
    size_t sorted_values_offset = 0;
    size_t probabilities_offset = 0;
    size_t sorted_indices_offset = 0;
    size_t threshold_pk_offset = 0;
    size_t threshold_pp_offset = 0;
    size_t mask_pk_offset = 0;
    size_t mask_pp_offset = 0;
    size_t selected_value_offset = 0;
    size_t selected_position_offset = 0;
    size_t selected_index_offset = 0;

    ~Opaque() {
        if (values_desc) {
            cnnlDestroyTensorDescriptor(values_desc);
        }
        if (softmax_desc) {
            cnnlDestroyTensorDescriptor(softmax_desc);
        }
        if (indices_desc) {
            cnnlDestroyTensorDescriptor(indices_desc);
        }
        if (scalar_value_desc) {
            cnnlDestroyTensorDescriptor(scalar_value_desc);
        }
        if (scalar_index_desc) {
            cnnlDestroyTensorDescriptor(scalar_index_desc);
        }
        if (result_desc) {
            cnnlDestroyTensorDescriptor(result_desc);
        }
        if (mask_desc) {
            cnnlDestroyTensorDescriptor(mask_desc);
        }
        if (scalar_mask_desc) {
            cnnlDestroyTensorDescriptor(scalar_mask_desc);
        }
    }
};

Descriptor::~Descriptor() { delete _opaque; }

static infiniStatus_t setTensor1D(
    cnnlTensorDescriptor_t desc, cnnlDataType_t dtype, int64_t length) {
    CHECK_BANG(cnnlSetTensorDescriptor_v2(
        desc, CNNL_LAYOUT_ARRAY, dtype, 1, &length));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t result_desc,
    infiniopTensorDescriptor_t probs_desc) {

    auto result = RandomSampleInfo::create(result_desc, probs_desc);
    CHECK_RESULT(result);
    auto info = result.take();
    CHECK_OR_RETURN(info.dt_i == INFINI_DTYPE_I32
                        || info.dt_i == INFINI_DTYPE_I64,
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(info.dt_p == INFINI_DTYPE_F16
                        || info.dt_p == INFINI_DTYPE_BF16
                        || info.dt_p == INFINI_DTYPE_F32,
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(info.n > 0 && info.n <= static_cast<size_t>(INT32_MAX),
                    INFINI_STATUS_BAD_TENSOR_SHAPE);

    auto handle = reinterpret_cast<device::bang::Handle *>(handle_);
    auto opaque = new Opaque{handle->internal()};
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->values_desc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->softmax_desc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->indices_desc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->scalar_value_desc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->scalar_index_desc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->result_desc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->mask_desc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->scalar_mask_desc));

    const auto value_dtype = device::bang::getCnnlDtype(info.dt_p);
    const int64_t n = static_cast<int64_t>(info.n);
    CHECK_STATUS(setTensor1D(opaque->values_desc, value_dtype, n));
    CHECK_STATUS(setTensor1D(opaque->indices_desc, CNNL_DTYPE_INT32, n));
    CHECK_STATUS(setTensor1D(opaque->scalar_value_desc, value_dtype, 1));
    CHECK_STATUS(setTensor1D(opaque->scalar_index_desc, CNNL_DTYPE_INT32, 1));
    CHECK_STATUS(setTensor1D(
        opaque->result_desc, device::bang::getCnnlDtype(info.dt_i), 1));
    CHECK_STATUS(setTensor1D(opaque->mask_desc, value_dtype, n));
    CHECK_STATUS(setTensor1D(opaque->scalar_mask_desc, value_dtype, 1));
    int64_t softmax_dims[3] = {1, n, 1};
    CHECK_BANG(cnnlSetTensorDescriptor_v2(
        opaque->softmax_desc, CNNL_LAYOUT_ARRAY, value_dtype,
        3, softmax_dims));

    CHECK_STATUS(opaque->internal->useCnnl(
        nullptr,
        [&](cnnlHandle_t cnnl_handle) {
            size_t size = 0;
            CHECK_BANG(cnnlGetTopKTensorWorkspaceSize(
                cnnl_handle, opaque->values_desc, static_cast<int>(info.n), 0, true,
                opaque->values_desc, opaque->indices_desc, &size));
            opaque->op_workspace_size = std::max(opaque->op_workspace_size, size);
            CHECK_BANG(cnnlGetTopKTensorWorkspaceSize(
                cnnl_handle, opaque->values_desc, 1, 0, true,
                opaque->scalar_value_desc, opaque->scalar_index_desc, &size));
            opaque->op_workspace_size = std::max(opaque->op_workspace_size, size);
            CHECK_BANG(cnnlGetTopKTensorWorkspaceSize(
                cnnl_handle, opaque->mask_desc, 1, 0, true,
                opaque->scalar_mask_desc, opaque->scalar_index_desc, &size));
            opaque->op_workspace_size = std::max(opaque->op_workspace_size, size);
            CHECK_BANG(cnnlGetCumsumWorkspaceSize(
                cnnl_handle, opaque->values_desc, 0, &size));
            opaque->op_workspace_size = std::max(opaque->op_workspace_size, size);
            CHECK_BANG(cnnlGetLogicOpWorkspaceSize(
                cnnl_handle, opaque->values_desc, opaque->scalar_value_desc,
                opaque->mask_desc, &size));
            opaque->op_workspace_size = std::max(opaque->op_workspace_size, size);
            CHECK_BANG(cnnlGetLogicOpWorkspaceSize(
                cnnl_handle, opaque->mask_desc, opaque->mask_desc,
                opaque->mask_desc, &size));
            opaque->op_workspace_size = std::max(opaque->op_workspace_size, size);
            CHECK_BANG(cnnlGetGatherWorkspaceSize(
                cnnl_handle, opaque->indices_desc, opaque->scalar_index_desc,
                opaque->scalar_index_desc, 0, &size));
            opaque->op_workspace_size = std::max(opaque->op_workspace_size, size);
            CHECK_BANG(cnnlGetCopyWorkspaceSize(
                cnnl_handle, opaque->scalar_index_desc,
                opaque->scalar_index_desc, &size));
            opaque->op_workspace_size = std::max(opaque->op_workspace_size, size);
            return INFINI_STATUS_SUCCESS;
        }));

    size_t workspace_size = alignUp(opaque->op_workspace_size);
    const size_t values_bytes = info.n * infiniSizeOf(info.dt_p);
    const size_t indices_bytes = info.n * sizeof(int32_t);
    opaque->sorted_values_offset = workspace_size;
    workspace_size += alignUp(values_bytes);
    opaque->probabilities_offset = workspace_size;
    workspace_size += alignUp(values_bytes);
    opaque->sorted_indices_offset = workspace_size;
    workspace_size += alignUp(indices_bytes);
    opaque->threshold_pk_offset = workspace_size;
    workspace_size += alignUp(infiniSizeOf(info.dt_p));
    opaque->threshold_pp_offset = workspace_size;
    workspace_size += alignUp(infiniSizeOf(info.dt_p));
    opaque->mask_pk_offset = workspace_size;
    workspace_size += alignUp(values_bytes);
    opaque->mask_pp_offset = workspace_size;
    workspace_size += alignUp(values_bytes);
    opaque->selected_value_offset = workspace_size;
    workspace_size += alignUp(infiniSizeOf(info.dt_p));
    opaque->selected_position_offset = workspace_size;
    workspace_size += alignUp(sizeof(int32_t));
    opaque->selected_index_offset = workspace_size;
    workspace_size += alignUp(sizeof(int32_t));

    *desc_ptr = new Descriptor(
        std::move(info), workspace_size, opaque,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

size_t Descriptor::minWorkspaceSize() const { return _min_workspace_size; }

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *result,
    const void *probs,
    float random_val,
    float topp,
    int topk,
    float temperature,
    void *stream) const {

    if (workspace_size < _min_workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    CHECK_OR_RETURN(topk > 0, INFINI_STATUS_BAD_PARAM);

    auto base = reinterpret_cast<char *>(workspace);
    void *sorted_values = base + _opaque->sorted_values_offset;
    void *probabilities = base + _opaque->probabilities_offset;
    void *sorted_indices = base + _opaque->sorted_indices_offset;
    void *threshold_pk = base + _opaque->threshold_pk_offset;
    void *threshold_pp = base + _opaque->threshold_pp_offset;
    void *mask_pk = base + _opaque->mask_pk_offset;
    void *mask_pp = base + _opaque->mask_pp_offset;
    void *selected_value = base + _opaque->selected_value_offset;
    void *selected_position = base + _opaque->selected_position_offset;
    void *selected_index = base + _opaque->selected_index_offset;
    const size_t value_size = infiniSizeOf(_info.dt_p);
    auto queue = reinterpret_cast<cnrtQueue_t>(stream);

    CHECK_STATUS(_opaque->internal->useCnnl(
        queue,
        [&](cnnlHandle_t cnnl_handle) {
            auto write_result = [&]() -> infiniStatus_t {
                if (_info.dt_i == INFINI_DTYPE_I32) {
                    CHECK_BANG(cnnlCopy_v2(
                        cnnl_handle,
                        _opaque->scalar_index_desc, selected_index,
                        _opaque->result_desc, result,
                        workspace, _opaque->op_workspace_size));
                } else {
                    CHECK_BANG(cnnlCastDataType(
                        cnnl_handle,
                        _opaque->scalar_index_desc, selected_index,
                        CNNL_CAST_INT32_TO_INT64,
                        _opaque->result_desc, result));
                }
                return INFINI_STATUS_SUCCESS;
            };

            if (random_val == 0.0f || topp == 0.0f
                || topk == 1 || temperature == 0.0f) {
                CHECK_BANG(cnnlTopKTensor_v3(
                    cnnl_handle, _opaque->values_desc, probs,
                    1, 0, true, true, true,
                    workspace, _opaque->op_workspace_size,
                    _opaque->scalar_value_desc, selected_value,
                    _opaque->scalar_index_desc, selected_index));
                return write_result();
            }

            CHECK_BANG(cnnlTopKTensor_v3(
                cnnl_handle, _opaque->values_desc, probs,
                static_cast<int>(_info.n), 0, true, true, true,
                workspace, _opaque->op_workspace_size,
                _opaque->values_desc, sorted_values,
                _opaque->indices_desc, sorted_indices));

            const float alpha_temperature = 1.0f / temperature;
            const float zero = 0.0f;
            CHECK_BANG(cnnlTransform_v2(
                cnnl_handle, CNNL_POINTER_MODE_HOST,
                &alpha_temperature, _opaque->values_desc, sorted_values,
                &zero, _opaque->values_desc, sorted_values));
            CHECK_BANG(cnnlSoftmaxForward(
                cnnl_handle, CNNL_SOFTMAX_ACCURATE,
                CNNL_SOFTMAX_MODE_MEDIUM_DIMENSION,
                nullptr, _opaque->softmax_desc, sorted_values,
                nullptr, _opaque->softmax_desc, probabilities));
            CHECK_BANG(cnnlCumsum_v2(
                cnnl_handle, _opaque->values_desc, probabilities,
                0, false, false, CNNL_NOT_PROPAGATE_NAN,
                _opaque->values_desc, probabilities,
                workspace, _opaque->op_workspace_size));

            const size_t k = std::min(static_cast<size_t>(topk), _info.n);
            const void *pk = reinterpret_cast<const char *>(probabilities)
                           + (k - 1) * value_size;
            CHECK_BANG(cnnlTransform_v2(
                cnnl_handle, CNNL_POINTER_MODE_HOST,
                &random_val, _opaque->scalar_value_desc, pk,
                &zero, _opaque->scalar_value_desc, threshold_pk));
            const float pp = random_val * topp;
            CHECK_BANG(cnnlTransform_v2(
                cnnl_handle, CNNL_POINTER_MODE_HOST,
                &zero, _opaque->scalar_value_desc, threshold_pk,
                &pp, _opaque->scalar_value_desc, threshold_pp));

            CHECK_BANG(cnnlLogicOp(
                cnnl_handle, CNNL_LOGIC_OP_GE,
                _opaque->values_desc, probabilities,
                _opaque->scalar_value_desc, threshold_pk,
                workspace, _opaque->op_workspace_size,
                _opaque->mask_desc, mask_pk));
            CHECK_BANG(cnnlLogicOp(
                cnnl_handle, CNNL_LOGIC_OP_GE,
                _opaque->values_desc, probabilities,
                _opaque->scalar_value_desc, threshold_pp,
                workspace, _opaque->op_workspace_size,
                _opaque->mask_desc, mask_pp));
            CHECK_BANG(cnnlLogicOp(
                cnnl_handle, CNNL_LOGIC_OP_OR,
                _opaque->mask_desc, mask_pk,
                _opaque->mask_desc, mask_pp,
                workspace, _opaque->op_workspace_size,
                _opaque->mask_desc, mask_pk));

            CHECK_BANG(cnnlTopKTensor_v3(
                cnnl_handle, _opaque->mask_desc, mask_pk,
                1, 0, true, true, true,
                workspace, _opaque->op_workspace_size,
                _opaque->scalar_mask_desc, selected_value,
                _opaque->scalar_index_desc, selected_position));
            CHECK_BANG(cnnlGather_v2(
                cnnl_handle, 0,
                _opaque->indices_desc, sorted_indices,
                _opaque->scalar_index_desc, selected_position,
                workspace, _opaque->op_workspace_size,
                _opaque->scalar_index_desc, selected_index));
            return write_result();
        }));
    cnrtQueueSync(queue);
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::random_sample::bang
