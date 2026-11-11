#ifndef __NSA_COMPRESS_PAGED_CACHE_INFO_H__
#define __NSA_COMPRESS_PAGED_CACHE_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::nsa_compress_paged_cache {

class NsaCompressPagedCacheInfo {
    NsaCompressPagedCacheInfo() = default;

public:
    infiniDtype_t dtype;
    infiniDtype_t index_dtype;
    int nsa_block_size;
    int update_last_only;
    size_t num_seqs;
    size_t num_kv_heads;
    size_t head_size;
    size_t page_block_size;
    size_t subblocks_per_page;
    size_t max_num_blocks_per_seq;

    ptrdiff_t k_cmp_block_stride;
    ptrdiff_t k_cmp_head_stride;
    ptrdiff_t v_cmp_block_stride;
    ptrdiff_t v_cmp_head_stride;
    ptrdiff_t k_cache_batch_stride;
    ptrdiff_t k_cache_head_stride;
    ptrdiff_t k_cache_row_stride;
    ptrdiff_t v_cache_batch_stride;
    ptrdiff_t v_cache_head_stride;
    ptrdiff_t v_cache_row_stride;
    ptrdiff_t block_table_batch_stride;
    ptrdiff_t cache_lens_stride;

    static utils::Result<NsaCompressPagedCacheInfo> create(
        infiniopTensorDescriptor_t k_cmp_desc,
        infiniopTensorDescriptor_t v_cmp_desc,
        infiniopTensorDescriptor_t k_cache_desc,
        infiniopTensorDescriptor_t v_cache_desc,
        infiniopTensorDescriptor_t block_tables_desc,
        infiniopTensorDescriptor_t cache_lens_desc,
        int nsa_block_size,
        int update_last_only) {
        auto dtype = k_cache_desc->dtype();
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16);
        if (v_cache_desc->dtype() != dtype || k_cmp_desc->dtype() != dtype || v_cmp_desc->dtype() != dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (k_cmp_desc->ndim() != 3 || v_cmp_desc->ndim() != 3 || k_cache_desc->ndim() != 4 || v_cache_desc->ndim() != 4 || block_tables_desc->ndim() != 2 || cache_lens_desc->ndim() != 1) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
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
        auto k_shape = k_cache_desc->shape();
        const size_t num_blocks = k_shape[0];
        const size_t num_kv_heads = k_shape[1];
        const size_t page_block_size = k_shape[2];
        const size_t head_size = k_shape[3];
        if (head_size != 128 || nsa_block_size <= 0 || page_block_size % static_cast<size_t>(nsa_block_size) != 0) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (v_cache_desc->shape() != k_shape) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        const size_t subblocks_per_page = page_block_size / static_cast<size_t>(nsa_block_size);
        if (k_cmp_desc->shape()[0] != num_blocks * subblocks_per_page || k_cmp_desc->shape()[1] != num_kv_heads || k_cmp_desc->shape()[2] != head_size) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (v_cmp_desc->shape() != k_cmp_desc->shape()) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (cache_lens_desc->shape()[0] != block_tables_desc->shape()[0]) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        return utils::Result<NsaCompressPagedCacheInfo>(NsaCompressPagedCacheInfo{
            dtype,
            index_dtype,
            nsa_block_size,
            update_last_only,
            block_tables_desc->shape()[0],
            num_kv_heads,
            head_size,
            page_block_size,
            subblocks_per_page,
            block_tables_desc->shape()[1],
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
            block_tables_desc->stride(0),
            cache_lens_desc->stride(0),
        });
    }
};

} // namespace op::nsa_compress_paged_cache

#endif // __NSA_COMPRESS_PAGED_CACHE_INFO_H__
