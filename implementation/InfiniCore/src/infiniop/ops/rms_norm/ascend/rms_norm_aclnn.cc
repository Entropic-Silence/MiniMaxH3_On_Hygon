#include "rms_norm_aclnn.h"
#include "../../../devices/ascend/common_ascend.h"
#include <aclnnop/aclnn_rms_norm.h>

extern "C" infiniStatus_t rms_norm_cast_w_launch(
    void *dst, const void *src,
    infiniDtype_t src_dtype, infiniDtype_t dst_dtype,
    size_t count, void *stream);

namespace op::rms_norm::ascend {

struct Descriptor::Opaque {
    aclnnTensorDescriptor_t y;
    aclnnTensorDescriptor_t x;
    aclnnTensorDescriptor_t w;
    aclnnTensorDescriptor_t rstd;
    size_t workspaceSize;
    aclOpExecutor *executor;
    bool full_tensor;
    bool needs_cast_w;
    size_t cast_w_offset;
    size_t w_padded_offset;
    size_t w_padded_size;
    size_t x_work_offset;
    size_t y_work_offset;
    size_t logical_rows;

    Opaque(aclnnTensorDescriptor_t y_, aclnnTensorDescriptor_t x_,
           aclnnTensorDescriptor_t w_, aclnnTensorDescriptor_t rstd_,
           size_t ws, aclOpExecutor *exec,
           bool full, bool cast_w, size_t cast_off, size_t pad_off, size_t pad_sz)
        : y(y_), x(x_), w(w_), rstd(rstd_), workspaceSize(ws), executor(exec),
          full_tensor(full), needs_cast_w(cast_w), cast_w_offset(cast_off),
          w_padded_offset(pad_off), w_padded_size(pad_sz),
          x_work_offset(0), y_work_offset(0), logical_rows(0) {}

    ~Opaque() {
        delete y;
        delete x;
        delete w;
        delete rstd;
        aclDestroyAclOpExecutor(executor);
    }
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    float epsilon) {

    auto result = RMSNormInfo::create(y_desc, x_desc, w_desc, epsilon);
    CHECK_RESULT(result);
    auto info = result.take();

    auto handle_ascend = reinterpret_cast<device::ascend::Handle *>(handle);

    const bool full_tensor = [&]() {
        ptrdiff_t expected_stride = 1;
        for (size_t i = info.ndim(); i-- > 0;) {
            if (info.x_strides[i] != expected_stride
                || info.y_strides[i] != expected_stride) {
                return false;
            }
            expected_stride *= static_cast<ptrdiff_t>(info.shape[i]);
        }
        return true;
    }();

    std::vector<int64_t> tensor_shape;
    tensor_shape.reserve(info.ndim());
    for (size_t dim : info.shape) {
        tensor_shape.push_back(static_cast<int64_t>(dim));
    }
    std::vector<int64_t> contiguous_strides(tensor_shape.size(), 1);
    for (ptrdiff_t i = static_cast<ptrdiff_t>(tensor_shape.size()) - 2; i >= 0; --i) {
        contiguous_strides[i] = contiguous_strides[i + 1] * tensor_shape[i + 1];
    }

    aclnnTensorDescriptor_t y;
    aclnnTensorDescriptor_t x;
    if (full_tensor) {
        y = new aclnnTensorDescriptor(y_desc);
        x = new aclnnTensorDescriptor(x_desc);
    } else {
        y = new aclnnTensorDescriptor(toAclDataType(info.atype), tensor_shape, contiguous_strides);
        x = new aclnnTensorDescriptor(toAclDataType(info.atype), tensor_shape, contiguous_strides);
    }

    // 仅在跨半精度组合时需要将 w cast 到 atype
    // (F16 atype + BF16 w, 或 BF16 atype + F16 w)
    bool needs_cast_w = (info.atype != info.wtype && info.wtype != INFINI_DTYPE_F32);
    aclnnTensorDescriptor_t w = nullptr;
    if (needs_cast_w) {
        // 规避 constructor #2 的 ndim 内存 corruption 问题
        // 先用 constructor #1 从 w_desc 正确构造，再替换 tensor 为正确的 dtype
        w = new aclnnTensorDescriptor(w_desc);
        if (w->tensor) {
            aclDestroyTensor(w->tensor);
        }
        w->dataType = toAclDataType(INFINI_DTYPE_F32);
        w->tensor = aclCreateTensor(w->shape.data(), w->ndim, w->dataType,
                                    w->strides.data(), w->offset, w->format,
                                    w->storageShape.data(), w->storageNdim, nullptr);
    } else {
        w = new aclnnTensorDescriptor(w_desc);
    }

    std::vector<int64_t> rstd_shape;
    rstd_shape.reserve(info.ndim() - 1);
    for (size_t i = 0; i + 1 < info.ndim(); ++i) {
        rstd_shape.push_back(static_cast<int64_t>(info.shape[i]));
    }
    std::vector<int64_t> rstd_strides(rstd_shape.size(), 1);
    for (ptrdiff_t i = static_cast<ptrdiff_t>(rstd_shape.size()) - 2; i >= 0; --i) {
        rstd_strides[i] = rstd_strides[i + 1] * rstd_shape[i + 1];
    }
    aclnnTensorDescriptor_t rstd = new aclnnTensorDescriptor(toAclDataType(INFINI_DTYPE_F32), rstd_shape, rstd_strides);

    size_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    CHECK_ACL(aclnnRmsNormGetWorkspaceSize(
        x->tensor,
        w->tensor,
        static_cast<double>(epsilon),
        y->tensor,
        rstd->tensor,
        &workspace_size,
        &executor));

    aclSetAclOpExecutorRepeatable(executor);

    size_t rstd_size = rstd->numel() * aclDataTypeSize(rstd->dataType);
    size_t cast_w_dst_size = needs_cast_w ? info.dim() * sizeof(float) : 0;
    size_t w_padded_size = 0;
    if (needs_cast_w) {
        size_t w_raw_bytes = info.dim() * infiniSizeOf(info.wtype);
        w_padded_size = ((w_raw_bytes + 31) / 32) * 32;
    }
    auto align_32 = [](size_t offset) {
        return (offset + 31) & ~static_cast<size_t>(31);
    };
    size_t cast_w_offset = align_32(workspace_size + rstd_size);
    size_t w_padded_offset = align_32(cast_w_offset + cast_w_dst_size);
    size_t x_work_offset = align_32(w_padded_offset + w_padded_size);

    size_t logical_rows = 1;
    for (size_t i = 0; i + 1 < info.ndim(); ++i) {
        logical_rows *= info.shape[i];
    }
    size_t tensor_bytes = full_tensor
                            ? 0
                            : logical_rows * info.dim() * infiniSizeOf(info.atype);
    size_t y_work_offset = align_32(x_work_offset + tensor_bytes);
    size_t all_workspace_size = y_work_offset + tensor_bytes;

    auto *opaque = new Opaque{
        y, x, w, rstd, workspace_size, executor, full_tensor, needs_cast_w,
        cast_w_offset, w_padded_offset, w_padded_size};
    opaque->x_work_offset = x_work_offset;
    opaque->y_work_offset = y_work_offset;
    opaque->logical_rows = logical_rows;
    *desc_ptr = new Descriptor(
        opaque,
        std::move(info),
        all_workspace_size,
        handle_ascend->device,
        handle_ascend->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *y, const void *x, const void *w,
    void *stream) const {

    if (workspace_size < workspaceSize()) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    void *rstd_ptr = static_cast<uint8_t *>(workspace) + _opaque->workspaceSize;
    void *w_ptr = nullptr;
    if (_opaque->needs_cast_w) {
        void *cast_w_ptr = static_cast<uint8_t *>(workspace) + _opaque->cast_w_offset;
        void *w_padded_src = static_cast<uint8_t *>(workspace) + _opaque->w_padded_offset;
        size_t w_bytes = _info.dim() * infiniSizeOf(_info.wtype);
        CHECK_ACL(aclrtMemcpyAsync(
            w_padded_src, _opaque->w_padded_size, const_cast<void *>(w), w_bytes,
            ACL_MEMCPY_DEVICE_TO_DEVICE, static_cast<aclrtStream>(stream)));
        CHECK_STATUS(rms_norm_cast_w_launch(
            cast_w_ptr, w_padded_src, _info.wtype, INFINI_DTYPE_F32,
            _info.dim(), stream));
        w_ptr = cast_w_ptr;
    } else {
        w_ptr = const_cast<void *>(w);
    }

    void *x_ptr = const_cast<void *>(x);
    void *y_ptr = y;
    auto *workspace_bytes = static_cast<uint8_t *>(workspace);
    const size_t unit = infiniSizeOf(_info.atype);
    const size_t row_bytes = _info.dim() * unit;
    if (!_opaque->full_tensor) {
        auto *x_work = workspace_bytes + _opaque->x_work_offset;
        auto *y_work = workspace_bytes + _opaque->y_work_offset;
        const size_t ndim = _info.ndim();
        for (size_t row = 0; row < _opaque->logical_rows; ++row) {
            size_t remaining = row;
            ptrdiff_t x_offset = 0;
            for (size_t axis = ndim - 1; axis-- > 0;) {
                const size_t coordinate = remaining % _info.shape[axis];
                remaining /= _info.shape[axis];
                x_offset += coordinate * _info.x_strides[axis];
            }
            CHECK_ACL(aclrtMemcpyAsync(
                x_work + row * row_bytes, row_bytes,
                const_cast<char *>(static_cast<const char *>(x)) + x_offset * unit,
                row_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE,
                static_cast<aclrtStream>(stream)));
        }
        x_ptr = x_work;
        y_ptr = y_work;
    }

    CHECK_ACL(AclSetTensorAddr(_opaque->executor, 1, _opaque->w->tensor, w_ptr));
    CHECK_ACL(AclSetTensorAddr(_opaque->executor, 3, _opaque->rstd->tensor, rstd_ptr));
    CHECK_ACL(AclSetTensorAddr(_opaque->executor, 0, _opaque->x->tensor, x_ptr));
    CHECK_ACL(AclSetTensorAddr(_opaque->executor, 2, _opaque->y->tensor, y_ptr));
    CHECK_ACL(aclnnRmsNorm(
        workspace, _opaque->workspaceSize, _opaque->executor, stream));

    if (!_opaque->full_tensor) {
        auto *y_work = workspace_bytes + _opaque->y_work_offset;
        const size_t ndim = _info.ndim();
        for (size_t row = 0; row < _opaque->logical_rows; ++row) {
            size_t remaining = row;
            ptrdiff_t y_offset = 0;
            for (size_t axis = ndim - 1; axis-- > 0;) {
                const size_t coordinate = remaining % _info.shape[axis];
                remaining /= _info.shape[axis];
                y_offset += coordinate * _info.y_strides[axis];
            }
            CHECK_ACL(aclrtMemcpyAsync(
                static_cast<char *>(y) + y_offset * unit, row_bytes,
                y_work + row * row_bytes, row_bytes,
                ACL_MEMCPY_DEVICE_TO_DEVICE, static_cast<aclrtStream>(stream)));
        }
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::rms_norm::ascend
