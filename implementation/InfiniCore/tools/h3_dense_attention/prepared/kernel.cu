#ifndef H3_LONG_CONTEXT
#define H3_LONG_CONTEXT 0
#endif
// Experimental instantiation of the official dense BF16 prefetch kernel.
// Only the query block size and prefetch depth vary; K blocks remain 64.
#include <hip/hip_runtime.h>
#include <cmath>
#include "flash.h"
#include "flash_fwd_kernel.h"
#ifndef H3_BLOCK_M
#define H3_BLOCK_M 128
#endif
#ifndef H3_STAGES
#define H3_STAGES 3
#endif

using H3Traits=Flash_fwd_kernel_16x64_prefetch_traits<128,H3_BLOCK_M,64,4,cutlass::bfloat16_t,H3_STAGES,false,H3_LONG_CONTEXT>;
__global__ __launch_bounds__(256) void h3_dense_tile_kernel(const Flash_fwd_params params) {
    flash::compute_attn_16x64_prefetch<H3Traits,false,false,false,false,false,true,false,false>(params);
}

extern "C" int h3_dense_tile_forward(
    void *q,void *k,void *v,void *out,void *lse,int lq,int lk,int heads,
    int64_t qrow,int64_t qhead,int64_t krow,int64_t khead,int64_t vrow,int64_t vhead,
    float scale,void *stream) {
    if(!q||!k||!v||!out||!lse||lq<1||lk<1||heads<1) return -1;
    Flash_fwd_params p{};
    p.q_ptr=q;p.k_ptr=k;p.v_ptr=v;p.o_ptr=out;p.softmax_lse_ptr=lse;
    p.b=1;p.h=heads;p.h_k=heads;p.h_h_k_ratio=1;
    p.q_row_stride=qrow;p.q_head_stride=qhead;p.q_batch_stride=lq*qrow;
    p.k_row_stride=krow;p.k_head_stride=khead;p.k_batch_stride=lk*krow;
    p.v_row_stride=vrow;p.v_head_stride=vhead;p.v_batch_stride=lk*vrow;
    p.o_row_stride=heads*128;p.o_head_stride=128;p.o_batch_stride=int64_t(lq)*heads*128;
    p.seqlen_q=lq;p.seqlen_k=lk;p.total_q=lq;
    p.seqlen_q_rounded=((lq+127)/128)*128;p.seqlen_k_rounded=((lk+127)/128)*128;
    p.d=p.d_rounded=p.d_value=p.d_value_rounded=128;
    p.scale_softmax=scale;p.scale_softmax_log2=scale*M_LOG2E;
    p.p_dropout=1.0f;p.p_dropout_in_uint8_t=255;p.rp_dropout=1.0f;p.scale_softmax_rp_dropout=scale;
    p.window_size_left=-1;p.window_size_right=-1;p.is_bf16=true;
    p.is_seqlens_k_cumulative=true;p.num_splits=1;
    dim3 grid((lq+H3_BLOCK_M-1)/H3_BLOCK_M,heads,1);
    h3_dense_tile_kernel<<<grid,256,H3Traits::kSmemSize,reinterpret_cast<hipStream_t>(stream)>>>(p);
    return static_cast<int>(hipGetLastError());
}
extern "C" int h3_dense_tile_block_m(){return H3_BLOCK_M;}
extern "C" int h3_dense_tile_stages(){return H3_STAGES;}

extern "C" int h3_dense_tile_round(){return FLASH_ATTENTION_BF16_TYPE;}

extern "C" int h3_dense_tile_long_context(){return H3_LONG_CONTEXT;}

extern "C" int h3_dense_tile_reduce(){return H3_VENDOR_REDUCE;}

extern "C" int h3_dense_tile_sum_fix(){return H3_SUM_FIX;}

extern "C" int h3_dense_tile_identity_rescale(){return H3_IDENTITY_RESCALE;}

extern "C" int h3_dense_tile_cache_swizzle(){return H3_CACHE_SWIZZLE;}
