#include "select_last_token_hidden_ascend.h"
#include "../../../devices/ascend/common_ascend.h"
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_index_select.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>

namespace op::select_last_token_hidden::ascend {

struct Descriptor::Opaque {
    aclnnTensorDescriptor_t output;
    aclnnTensorDescriptor_t hidden_states;
    aclnnTensorDescriptor_t offsets;
    aclnnTensorDescriptor_t indices;
    aclnnScalarDescriptor_t minus_one;
    aclnnScalarDescriptor_t alpha;
    void *minus_one_value;
    void *alpha_value;
    void *indices_data;
    void *workspace;
    uint64_t adds_workspace_size;
    uint64_t index_select_workspace_size;
    aclOpExecutor *adds_executor;
    aclOpExecutor *index_select_executor;

    ~Opaque() {
        delete output;
        delete hidden_states;
        delete offsets;
        delete indices;
        delete minus_one;
        delete alpha;
        std::free(minus_one_value);
        std::free(alpha_value);
        if (indices_data != nullptr) {
            aclrtFree(indices_data);
        }
        if (workspace != nullptr) {
            aclrtFree(workspace);
        }
        aclDestroyAclOpExecutor(adds_executor);
        aclDestroyAclOpExecutor(index_select_executor);
    }
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t hidden_states_desc,
    infiniopTensorDescriptor_t input_offsets_desc) {
    const auto output_shape = output_desc->shape();
    const auto hidden_shape = hidden_states_desc->shape();
    const auto offsets_shape = input_offsets_desc->shape();

    CHECK_OR_RETURN(output_shape.size() == 3 && hidden_shape.size() == 3 && offsets_shape.size() == 1,
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(offsets_shape[0] >= 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
    const size_t num_requests = offsets_shape[0] - 1;
    CHECK_OR_RETURN(output_shape[0] == 1 && output_shape[1] == num_requests
                        && output_shape[2] == hidden_shape[2],
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(output_desc->isContiguous() && hidden_states_desc->isContiguous()
                        && input_offsets_desc->isContiguous(),
                    INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(input_offsets_desc->dtype() == INFINI_DTYPE_I32,
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    const auto hidden_dtype = hidden_states_desc->dtype();
    CHECK_OR_RETURN(hidden_dtype == INFINI_DTYPE_F16 || hidden_dtype == INFINI_DTYPE_BF16
                        || hidden_dtype == INFINI_DTYPE_F32,
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(output_desc->dtype() == hidden_dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);

    const size_t total_tokens = hidden_shape[0] * hidden_shape[1];
    const size_t hidden_size = hidden_shape[2];
    CHECK_OR_RETURN(total_tokens > 0 && hidden_size > 0, INFINI_STATUS_BAD_TENSOR_SHAPE);

    const auto hidden_acl_dtype = toAclDataType(hidden_dtype);
    auto output = new aclnnTensorDescriptor(
        hidden_acl_dtype,
        {static_cast<int64_t>(num_requests), static_cast<int64_t>(hidden_size)},
        {static_cast<int64_t>(hidden_size), 1});
    auto hidden_states = new aclnnTensorDescriptor(
        hidden_acl_dtype,
        {static_cast<int64_t>(total_tokens), static_cast<int64_t>(hidden_size)},
        {static_cast<int64_t>(hidden_size), 1});
    auto offsets = new aclnnTensorDescriptor(
        ACL_INT32,
        {static_cast<int64_t>(num_requests)},
        {1});
    auto indices = new aclnnTensorDescriptor(
        ACL_INT32,
        {static_cast<int64_t>(num_requests)},
        {1});

    auto minus_one_value = std::malloc(sizeof(int32_t));
    auto alpha_value = std::malloc(sizeof(int32_t));
    CHECK_OR_RETURN(minus_one_value != nullptr && alpha_value != nullptr,
                    INFINI_STATUS_INSUFFICIENT_WORKSPACE);
    *static_cast<int32_t *>(minus_one_value) = -1;
    *static_cast<int32_t *>(alpha_value) = 1;
    auto minus_one = new aclnnScalarDescriptor(
        ACL_INT32, minus_one_value, sizeof(int32_t));
    auto alpha = new aclnnScalarDescriptor(
        ACL_INT32, alpha_value, sizeof(int32_t));

    uint64_t adds_workspace_size = 0;
    aclOpExecutor *adds_executor = nullptr;
    CHECK_ACL(aclnnAddsGetWorkspaceSize(
        offsets->tensor,
        minus_one->scalar,
        alpha->scalar,
        indices->tensor,
        &adds_workspace_size,
        &adds_executor));
    aclSetAclOpExecutorRepeatable(adds_executor);

    uint64_t index_select_workspace_size = 0;
    aclOpExecutor *index_select_executor = nullptr;
    CHECK_ACL(aclnnIndexSelectGetWorkspaceSize(
        hidden_states->tensor,
        0,
        indices->tensor,
        output->tensor,
        &index_select_workspace_size,
        &index_select_executor));
    aclSetAclOpExecutorRepeatable(index_select_executor);

    void *indices_data = nullptr;
    CHECK_ACL(aclrtMalloc(
        &indices_data,
        num_requests * sizeof(int32_t),
        ACL_MEM_MALLOC_HUGE_FIRST));

    void *workspace = nullptr;
    const uint64_t workspace_size = std::max(
        adds_workspace_size,
        index_select_workspace_size);
    if (workspace_size != 0) {
        CHECK_ACL(aclrtMalloc(
            &workspace,
            workspace_size,
            ACL_MEM_MALLOC_HUGE_FIRST));
    }

    auto opaque = new Opaque{
        output,
        hidden_states,
        offsets,
        indices,
        minus_one,
        alpha,
        minus_one_value,
        alpha_value,
        indices_data,
        workspace,
        adds_workspace_size,
        index_select_workspace_size,
        adds_executor,
        index_select_executor};

    auto handle_ascend = reinterpret_cast<device::ascend::Handle *>(handle);
    *desc_ptr = new Descriptor(
        num_requests,
        total_tokens,
        hidden_size * infiniSizeOf(hidden_dtype),
        opaque,
        handle_ascend->device,
        handle_ascend->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *output,
    const void *hidden_states,
    const void *input_offsets,
    void *stream) const {
    auto offsets_data = const_cast<int32_t *>(
        static_cast<const int32_t *>(input_offsets) + 1);

    AclSetTensorAddr(
        _opaque->adds_executor,
        0,
        _opaque->offsets->tensor,
        offsets_data);
    AclSetTensorAddr(
        _opaque->adds_executor,
        1,
        _opaque->indices->tensor,
        _opaque->indices_data);
    CHECK_ACL(aclnnAdds(
        _opaque->workspace,
        _opaque->adds_workspace_size,
        _opaque->adds_executor,
        stream));

    AclSetTensorAddr(
        _opaque->index_select_executor,
        0,
        _opaque->hidden_states->tensor,
        const_cast<void *>(hidden_states));
    AclSetTensorAddr(
        _opaque->index_select_executor,
        1,
        _opaque->indices->tensor,
        _opaque->indices_data);
    AclSetTensorAddr(
        _opaque->index_select_executor,
        2,
        _opaque->output->tensor,
        output);
    CHECK_ACL(aclnnIndexSelect(
        _opaque->workspace,
        _opaque->index_select_workspace_size,
        _opaque->index_select_executor,
        stream));
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::select_last_token_hidden::ascend
