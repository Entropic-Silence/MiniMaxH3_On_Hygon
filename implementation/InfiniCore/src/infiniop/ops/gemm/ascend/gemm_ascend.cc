#include "gemm_ascend.h"
#include "../../../devices/ascend/common_ascend.h"
#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/level2/aclnn_gemm.h>

#include <cstring>
#include <unordered_map>

// Custom hash function for alpha beta pair<float, float>
struct FloatPairHash {
    size_t operator()(const std::pair<float, float> &p) const {
        uint64_t combined;
        std::memcpy(reinterpret_cast<char *>(&combined), &p.first, sizeof(float));
        std::memcpy(reinterpret_cast<char *>(&combined) + sizeof(float), &p.second, sizeof(float));

        return std::hash<uint64_t>()(combined);
    }
};

struct FloatPairEqual {
    bool operator()(const std::pair<float, float> &a, const std::pair<float, float> &b) const {
        return a.first == b.first && a.second == b.second;
    }
};

namespace op::gemm::ascend {

struct Descriptor::Opaque {
    aclnnTensorDescriptor_t c, a, b;
    // cubeMathType
    // see doc:
    // https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha002/apiref/appdevgapi/context/aclnnBatchMatMul.md
    int8_t mt;
    // whether B is handed to aclnnGemm as a contiguous [out, in] tensor with
    // transB=1 (see Descriptor::create); 0 = original behaviour (transB = 0).
    int8_t transB;
    // alpha&beta hashmap
    std::unordered_map<std::pair<float, float>, aclOpExecutor *, FloatPairHash, FloatPairEqual> lookup;

    ~Opaque() {
        delete c;
        delete a;
        delete b;
        for (auto &item : lookup) {
            aclDestroyAclOpExecutor(item.second);
        }
        lookup.clear();
    }
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {
    auto handle = reinterpret_cast<device::ascend::Handle *>(handle_);
    auto dtype = c_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);

    auto result = MatmulInfo::create(c_desc, a_desc, b_desc, MatrixLayout::ROW_MAJOR);
    CHECK_RESULT(result);
    auto info = result.take();

    auto c = new aclnnTensorDescriptor(toAclDataType(c_desc->dtype()),
                                       {static_cast<int64_t>(info.m), static_cast<int64_t>(info.n)},
                                       {info.c_matrix.row_stride, info.c_matrix.col_stride});
    auto a = new aclnnTensorDescriptor(toAclDataType(a_desc->dtype()),
                                       {static_cast<int64_t>(info.a_matrix.rows), static_cast<int64_t>(info.a_matrix.cols)},
                                       {info.a_matrix.row_stride, info.a_matrix.col_stride});
    // The common Linear case stores the weight physically as [out, in] row-major
    // and views it here as a logical [in, out] column-major matrix (row_stride == 1).
    // Passing that column-major view to aclnnGemm with transB = 0 makes it emit a
    // physical Transpose kernel on every launch -- this dominated runtime (~60%).
    // Instead, describe the SAME memory as a contiguous [out, in] tensor and set
    // transB = 1, so the cube core handles the transpose via addressing (no kernel).
    // Mathematically identical; only affects Ascend. For any other B layout, keep
    // the original behaviour (transB = 0).
    bool trans_b = false;
    aclnnTensorDescriptor *b;
    if (info.b_matrix.row_stride == 1 && info.b_matrix.col_stride > 1) {
        b = new aclnnTensorDescriptor(toAclDataType(b_desc->dtype()),
                                      {static_cast<int64_t>(info.b_matrix.cols), static_cast<int64_t>(info.b_matrix.rows)},
                                      {info.b_matrix.col_stride, info.b_matrix.row_stride});
        trans_b = true;
    } else {
        b = new aclnnTensorDescriptor(toAclDataType(b_desc->dtype()),
                                      {static_cast<int64_t>(info.b_matrix.rows), static_cast<int64_t>(info.b_matrix.cols)},
                                      {info.b_matrix.row_stride, info.b_matrix.col_stride});
    }

    auto tc = c->tensor,
         ta = a->tensor,
         tb = b->tensor;

    std::unordered_map<std::pair<float, float>, aclOpExecutor *, FloatPairHash, FloatPairEqual> lookup;
    aclOpExecutor *executor = nullptr;
    size_t workspace_size = 0;
    int8_t mt = 1;
    CHECK_ACL(aclnnGemmGetWorkspaceSize(ta, tb, tc, 1., 0., 0, trans_b ? 1 : 0, tc, mt, &workspace_size, &executor));
    CHECK_ACL(aclSetAclOpExecutorRepeatable(executor));
    lookup[std::make_pair(1.0f, 0.0f)] = executor;
    CHECK_ACL(aclnnGemmGetWorkspaceSize(ta, tb, tc, 1., 1., 0, trans_b ? 1 : 0, tc, mt, &workspace_size, &executor));
    CHECK_ACL(aclSetAclOpExecutorRepeatable(executor));
    lookup[std::make_pair(1.0f, 1.0f)] = executor;

    *desc_ptr = new Descriptor(
        dtype, info, workspace_size,
        new Opaque{
            c,
            a,
            b,
            mt,
            static_cast<int8_t>(trans_b ? 1 : 0),
            std::move(lookup)},
        handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspaceSize_,
    void *c,
    float beta,
    const void *a,
    const void *b,
    float alpha,
    void *stream) const {

    auto tc = _opaque->c->tensor,
         ta = _opaque->a->tensor,
         tb = _opaque->b->tensor;

    size_t workspace_size = _workspace_size;
    aclOpExecutor *executor;
    auto key = std::make_pair(alpha, beta);
    if (_opaque->lookup.find(key) != _opaque->lookup.end()) {
        executor = _opaque->lookup[key];
    } else {
        CHECK_ACL(aclnnGemmGetWorkspaceSize(
            ta, tb, tc, alpha, beta, 0, _opaque->transB, tc, _opaque->mt,
            &workspace_size, &executor));
        CHECK_ACL(aclSetAclOpExecutorRepeatable(executor));
        _opaque->lookup[key] = executor;
    }

    if (workspaceSize_ < workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    auto unit = infiniSizeOf(_dtype);
    for (size_t i = 0; i < _info.batch; ++i) {
        AclSetTensorAddr(executor, 0, ta, ((char *)a) + i * _info.a_matrix.stride * unit);
        AclSetTensorAddr(executor, 1, tb, ((char *)b) + i * _info.b_matrix.stride * unit);
        AclSetTensorAddr(executor, 2, tc, ((char *)c) + i * _info.c_matrix.stride * unit);
        AclSetTensorAddr(executor, 3, tc, ((char *)c) + i * _info.c_matrix.stride * unit);
        CHECK_ACL(aclnnGemm(workspace, workspace_size, executor, stream));
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::gemm::ascend
