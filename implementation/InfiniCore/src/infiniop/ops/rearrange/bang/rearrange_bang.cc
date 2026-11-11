#include "rearrange_bang.h"

#include "../../../devices/bang/common_bang.h"

namespace op::rearrange::bang {

struct Descriptor::Opaque {
    std::shared_ptr<device::bang::Handle::Internal> internal;
    cnnlTensorDescriptor_t x_desc = nullptr;
    cnnlTensorDescriptor_t y_desc = nullptr;
    size_t workspace_size = 0;
    void *workspace = nullptr;

    ~Opaque() {
        if (x_desc) {
            cnnlDestroyTensorDescriptor(x_desc);
        }
        if (y_desc) {
            cnnlDestroyTensorDescriptor(y_desc);
        }
        if (workspace) {
            cnrtFree(workspace);
        }
    }
};

Descriptor::~Descriptor() { delete _opaque; }

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc) {

    CHECK_OR_RETURN(x_desc->dtype() == y_desc->dtype(), INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(x_desc->ndim() == y_desc->ndim(), INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_SAME_SHAPE(x_desc->shape(), y_desc->shape());
    CHECK_OR_RETURN(!y_desc->hasBroadcastDim(), INFINI_STATUS_BAD_TENSOR_STRIDES);

    auto meta_result = utils::RearrangeMeta::create(
        y_desc->shape().data(), y_desc->strides().data(), x_desc->strides().data(),
        y_desc->ndim(), infiniSizeOf(y_desc->dtype()));
    CHECK_RESULT(meta_result);
    auto meta = meta_result.take();

    auto handle = reinterpret_cast<device::bang::Handle *>(handle_);
    auto opaque = new Opaque{handle->internal()};
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->x_desc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->y_desc));
    CHECK_STATUS(device::bang::setCnnlTensorEx(opaque->x_desc, x_desc));
    CHECK_STATUS(device::bang::setCnnlTensorEx(opaque->y_desc, y_desc));
    CHECK_STATUS(opaque->internal->useCnnl(
        nullptr,
        [&](cnnlHandle_t cnnl_handle) {
            CHECK_BANG(cnnlGetCopyWorkspaceSize(
                cnnl_handle, opaque->x_desc, opaque->y_desc,
                &opaque->workspace_size));
            return INFINI_STATUS_SUCCESS;
        }));

    if (opaque->workspace_size > 0) {
        CHECK_INTERNAL(cnrtMalloc(&opaque->workspace, opaque->workspace_size), cnrtSuccess);
    }

    *desc_ptr = new Descriptor(std::move(meta), opaque, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *y,
    const void *x,
    void *stream) const {

    auto queue = reinterpret_cast<cnrtQueue_t>(stream);
    auto status = _opaque->internal->useCnnl(
        queue,
        [&](cnnlHandle_t cnnl_handle) {
            CHECK_BANG(cnnlCopy_v2(
                cnnl_handle, _opaque->x_desc, x, _opaque->y_desc, y,
                _opaque->workspace, _opaque->workspace_size));
            return INFINI_STATUS_SUCCESS;
        });
    cnrtQueueSync(queue);
    CHECK_STATUS(status);
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::rearrange::bang
