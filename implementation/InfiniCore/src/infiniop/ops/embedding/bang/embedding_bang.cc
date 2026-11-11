#include "embedding_bang.h"

#include "../../../devices/bang/common_bang.h"

namespace op::embedding::bang {

struct Descriptor::Opaque {
    std::shared_ptr<device::bang::Handle::Internal> internal;
    cnnlTensorDescriptor_t filter_desc = nullptr;
    cnnlTensorDescriptor_t indices_desc = nullptr;
    cnnlTensorDescriptor_t output_desc = nullptr;

    ~Opaque() {
        if (filter_desc) {
            cnnlDestroyTensorDescriptor(filter_desc);
        }
        if (indices_desc) {
            cnnlDestroyTensorDescriptor(indices_desc);
        }
        if (output_desc) {
            cnnlDestroyTensorDescriptor(output_desc);
        }
    }
};

Descriptor::~Descriptor() { delete _opaque; }

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc) {

    const auto input_shape = input_desc->shape();
    const auto weight_shape = weight_desc->shape();
    CHECK_OR_RETURN(weight_shape.size() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(output_desc->ndim() == input_shape.size() + 1,
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(output_desc->dim(output_desc->ndim() - 1) == weight_shape[1],
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    for (size_t i = 0; i < input_shape.size(); ++i) {
        CHECK_OR_RETURN(output_desc->dim(i) == input_shape[i],
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
    }

    const auto input_dtype = input_desc->dtype();
    const auto weight_dtype = weight_desc->dtype();
    CHECK_OR_RETURN(input_dtype == INFINI_DTYPE_I32 || input_dtype == INFINI_DTYPE_I64,
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_DTYPE(weight_dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
    CHECK_OR_RETURN(output_desc->dtype() == weight_dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(input_desc->isContiguous() && weight_desc->isContiguous()
                        && output_desc->isContiguous(),
                    INFINI_STATUS_BAD_TENSOR_STRIDES);

    auto handle = reinterpret_cast<device::bang::Handle *>(handle_);
    auto opaque = new Opaque{handle->internal()};
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->filter_desc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->indices_desc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->output_desc));
    CHECK_STATUS(device::bang::setCnnlTensor(opaque->filter_desc, weight_desc));
    CHECK_STATUS(device::bang::setCnnlTensor(opaque->indices_desc, input_desc));
    CHECK_STATUS(device::bang::setCnnlTensor(opaque->output_desc, output_desc));

    *desc_ptr = new Descriptor(
        input_desc->numel(), weight_shape[1], weight_shape[0],
        input_dtype, weight_dtype, opaque, handle->device, handle->device_id);
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
    auto queue = reinterpret_cast<cnrtQueue_t>(stream);
    CHECK_STATUS(_opaque->internal->useCnnl(
        queue,
        [&](cnnlHandle_t cnnl_handle) {
            CHECK_BANG(cnnlEmbeddingForward_v2(
                cnnl_handle, _opaque->filter_desc, weight,
                _opaque->indices_desc, input, -1, nullptr, nullptr,
                _opaque->output_desc, output));
            return INFINI_STATUS_SUCCESS;
        }));
    cnrtQueueSync(queue);
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::embedding::bang
