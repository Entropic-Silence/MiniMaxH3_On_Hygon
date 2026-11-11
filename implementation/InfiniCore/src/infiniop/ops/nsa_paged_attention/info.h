#ifndef __NSA_PAGED_ATTENTION_INFO_H__
#define __NSA_PAGED_ATTENTION_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::nsa_paged_attention {

class NsaPagedAttentionInfo {
    NsaPagedAttentionInfo() = default;

public:
    infiniDtype_t dtype;
    infiniDtype_t gates_dtype;
    infiniDtype_t index_dtype;
    float scale;
    int nsa_block_size;
    int window_size;
    int select_blocks;

    size_t num_seqs;
    size_t num_heads;
    size_t num_kv_heads;
    size_t head_size;
    size_t page_block_size;
    size_t subblocks_per_page;
    size_t max_num_blocks_per_seq;

    ptrdiff_t q_stride;
    ptrdiff_t q_head_stride;
    ptrdiff_t k_cmp_block_stride;
    ptrdiff_t k_cmp_head_stride;
    ptrdiff_t v_cmp_block_stride;
    ptrdiff_t v_cmp_head_stride;
    ptrdiff_t k_batch_stride;
    ptrdiff_t k_head_stride;
    ptrdiff_t k_row_stride;
    ptrdiff_t v_batch_stride;
    ptrdiff_t v_head_stride;
    ptrdiff_t v_row_stride;
    ptrdiff_t o_stride;
    ptrdiff_t o_head_stride;
    ptrdiff_t block_table_batch_stride;
    ptrdiff_t cache_lens_stride;
    ptrdiff_t gates_seq_stride;
    ptrdiff_t gates_branch_stride;
    ptrdiff_t gates_head_stride;

    static utils::Result<NsaPagedAttentionInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t q_desc,
        infiniopTensorDescriptor_t k_cmp_desc,
        infiniopTensorDescriptor_t v_cmp_desc,
        infiniopTensorDescriptor_t k_cache_desc,
        infiniopTensorDescriptor_t v_cache_desc,
        infiniopTensorDescriptor_t block_tables_desc,
        infiniopTensorDescriptor_t cache_lens_desc,
        infiniopTensorDescriptor_t gates_desc,
        float scale,
        int nsa_block_size,
        int window_size,
        int select_blocks) {
        auto dtype = q_desc->dtype();
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16);
        if (out_desc->dtype() != dtype || k_cache_desc->dtype() != dtype || v_cache_desc->dtype() != dtype
            || k_cmp_desc->dtype() != dtype || v_cmp_desc->dtype() != dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (gates_desc->dtype() != dtype && gates_desc->dtype() != INFINI_DTYPE_F32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (q_desc->ndim() != 3 || out_desc->ndim() != 3 || k_cmp_desc->ndim() != 3 || v_cmp_desc->ndim() != 3
            || k_cache_desc->ndim() != 4 || v_cache_desc->ndim() != 4 || block_tables_desc->ndim() != 2
            || cache_lens_desc->ndim() != 1 || gates_desc->ndim() != 3) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        CHECK_OR_RETURN(q_desc->stride(2) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(out_desc->stride(2) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(k_cmp_desc->stride(2) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(v_cmp_desc->stride(2) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(k_cache_desc->stride(3) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(v_cache_desc->stride(3) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(block_tables_desc->stride(1) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(cache_lens_desc->stride(0) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);

        const auto index_dtype = block_tables_desc->dtype();
        if (index_dtype != cache_lens_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (index_dtype != INFINI_DTYPE_I64 && index_dtype != INFINI_DTYPE_I32 && index_dtype != INFINI_DTYPE_U32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        auto q_shape = q_desc->shape();
        auto k_shape = k_cache_desc->shape();
        const size_t num_seqs = q_shape[0];
        const size_t num_heads = q_shape[1];
        const size_t head_size = q_shape[2];
        const size_t num_kv_heads = k_shape[1];
        const size_t page_block_size = k_shape[2];
        if (head_size != 128 || page_block_size == 0 || nsa_block_size <= 0 || window_size < 0 || select_blocks <= 0) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (num_heads % num_kv_heads != 0 || page_block_size % static_cast<size_t>(nsa_block_size) != 0) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (k_shape[3] != head_size || v_cache_desc->shape() != k_shape) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        const size_t subblocks_per_page = page_block_size / static_cast<size_t>(nsa_block_size);
        if (k_cmp_desc->shape()[0] != k_shape[0] * subblocks_per_page || k_cmp_desc->shape()[1] != num_kv_heads
            || k_cmp_desc->shape()[2] != head_size || v_cmp_desc->shape() != k_cmp_desc->shape()) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (out_desc->shape()[0] != num_seqs || out_desc->shape()[1] != num_heads || out_desc->shape()[2] != head_size) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (cache_lens_desc->shape()[0] != num_seqs) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (gates_desc->shape()[0] != num_seqs || gates_desc->shape()[1] != 3 || gates_desc->shape()[2] != num_heads) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        return utils::Result<NsaPagedAttentionInfo>(NsaPagedAttentionInfo{
            dtype,
            gates_desc->dtype(),
            index_dtype,
            scale,
            nsa_block_size,
            window_size,
            select_blocks,
            num_seqs,
            num_heads,
            num_kv_heads,
            head_size,
            page_block_size,
            subblocks_per_page,
            block_tables_desc->shape()[1],
            q_desc->stride(0),
            q_desc->stride(1),
            k_cmp_desc->stride(0),
            k_cmp_desc->stride(1),
            v_cmp_desc->stride(0),
            v_cmp_desc->stride(1),
            k_cache_desc->stride(0),
            k_cache_desc->stride(1),
            k_cache_desc->stride(2),
            v_cache_desc->stride(0),
            v_cache_desc->stride(1),
            v_cache_desc->stride(2),
            out_desc->stride(0),
            out_desc->stride(1),
            block_tables_desc->stride(0),
            cache_lens_desc->stride(0),
            gates_desc->stride(0),
            gates_desc->stride(1),
            gates_desc->stride(2),
        });
    }
};

} // namespace op::nsa_paged_attention

#endif // __NSA_PAGED_ATTENTION_INFO_H__
