/***************************************************************************************************
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/semaphore.h"

#include "cute/tensor.hpp"

#define PADDING_A 96
#define PADDING_B 32
#define PADDING_BLOCK_SIZE 256
namespace cutlass::gemm::kernel {


template <typename T>
__global__
#ifdef __CUDACC__
// Enclosing this in __CUDACC__ suppresses MSVC warnings.
__launch_bounds__(PADDING_BLOCK_SIZE)
#endif // __CUDACC__
void device_padding_kernel(const void *input, void* ouput, int lda, int row, int col, int ldb, T *dumy = nullptr)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // rowIdx
  int y = blockIdx.y * blockDim.y + threadIdx.y;  // colIdx
  
  float4 * tempIn  = (float4*)input;
	float4 * tempOut = (float4*)ouput;
  //float4 对应的数据T类型 一次copy的数据长度
  static constexpr int scale = sizeof(float4) / sizeof(T);
  {
    if(y < col && x < row / scale){
      tempOut[y * ldb / scale + x] = tempIn[y * lda / scale + x];
    }
  }
}

template <class IntT>
CUTLASS_DEVICE
cute::Stride<IntT, cute::Int<1>, int64_t>
make_cute_packed_stride(cute::Stride<IntT, cute::Int<1>, int64_t> s, cute::Shape<int,int,int> shape_MKL) {
  static_assert(std::is_integral_v<IntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");
  auto s_copy = s;
  cute::get<0>(s_copy) = static_cast<IntT>(cute::get<1>(shape_MKL));
  int batch_count =  cute::get<2>(shape_MKL);
  if (batch_count > 1) {
    cute::get<2>(s_copy) = static_cast<IntT>(cute::get<0>(shape_MKL) * cute::get<1>(shape_MKL));
  }
  else {
    cute::get<2>(s_copy) = static_cast<IntT>(0);
  }
  return s_copy;
}


template <class IntT>
CUTLASS_DEVICE
cute::Stride<cute::Int<1>, IntT, int64_t>
make_cute_packed_stride(cute::Stride<cute::Int<1>, IntT, int64_t> s, cute::Shape<int,int,int> shape_MKL) {
  static_assert(std::is_integral_v<IntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");
  auto s_copy = s;
  cute::get<1>(s_copy) = static_cast<IntT>(cute::get<0>(shape_MKL));
  int batch_count =  cute::get<2>(shape_MKL);
  if (batch_count > 1) {
    cute::get<2>(s_copy) = static_cast<IntT>(cute::get<0>(shape_MKL) * cute::get<1>(shape_MKL));
  }
  else {
    cute::get<2>(s_copy) = static_cast<IntT>(0);
  }
  return s_copy;
}
///////////////////////////////////////////////////////////////////////////////

template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileScheduler_
>
class GemmUniversal<
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileScheduler_,
  cute::enable_if_t<cute::is_base_of_v<KernelMultistage, typename CollectiveMainloop_::DispatchPolicy::Schedule>>>
{
public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;

  static_assert(rank(ProblemShape{}) == 3 or rank(ProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  static_assert(cute::is_void_v<TileScheduler_> or cute::is_same_v<TileScheduler_, PersistentScheduler>,
    "SM70 kernel does not support specializing the tile scheduler.");
  using TileSchedulerTag = TileScheduler_;
  using TileScheduler = typename detail::TileSchedulerSelector<
    TileScheduler_, ArchTag, TileShape,
    cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::Scheduler;
  using TileSchedulerArguments = typename TileScheduler::Arguments;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  static_assert(cute::is_same_v<ElementAccumulator, typename CollectiveEpilogue::ElementAccumulator>,
    "Mainloop and epilogue do not agree on accumulator value type.");

  // acc-smem require
  using MNK = typename TiledMma::TiledShape_MNK;
  //vector方式使用tiled_mma的大小创建共享内存进行向量化中转
  static constexpr int SmemEpilogue =  static_cast<int>(cute::max(sizeof(typename CollectiveEpilogue::SharedStorage),  get<0>(MNK{}) * get<1>(MNK{}) * sizeof(ElementAccumulator)));

  static constexpr int SharedStorageSize = static_cast<int>(cute::max(
      sizeof(typename CollectiveMainloop::SharedStorage),
      SmemEpilogue));

  // // MSVC requires the cast to fix a warning-as-error.
  // static constexpr int SharedStorageSize = static_cast<int>(cute::max(
  //     sizeof(typename CollectiveMainloop::SharedStorage),
  //     sizeof(typename CollectiveEpilogue::SharedStorage)));

  static constexpr uint32_t MaxThreadsPerBlock = cute::size(TiledMma{});
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  // Device side arguments
  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
  };

  // Kernel entry point API
  struct Params {
    GemmUniversalMode mode;
    ProblemShape problem_shape;
    MainloopParams mainloop;
    EpilogueParams epilogue;
    void * workspace;
    int * semaphore = nullptr;     
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    (void) workspace;
    auto * semaphore = workspace;
    size_t bytes = 0;
    if (args.mode == GemmUniversalMode::kGemm && get<3>(args.problem_shape) > 1) {
      auto grid_tiled_m = cute::size(cute::ceil_div(cute::shape<0>(args.problem_shape), cute::shape<0>(TileShape{})));
      auto grid_tiled_n = cute::size(cute::ceil_div(cute::shape<1>(args.problem_shape), cute::shape<1>(TileShape{})));
      bytes = sizeof(int) * grid_tiled_m * grid_tiled_n;
    } else if (args.mode == GemmUniversalMode::kGemmSplitKParallel) {
      bytes = sizeof(ElementC) * size_t(get<0>(args.problem_shape)) *
              size_t(get<1>(args.problem_shape)) *
              size_t(get<3>(args.problem_shape));
    }
    auto * gmem_workspace = (void *)(reinterpret_cast<uint8_t*>(workspace) + bytes);
    return {
      args.mode,
      args.problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, workspace),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, workspace),
      gmem_workspace,
      reinterpret_cast<int*>(semaphore)
    };
  }

  static bool
  can_implement(Arguments const& args) {
    return args.mode == GemmUniversalMode::kGemm or
          (args.mode == GemmUniversalMode::kBatched && rank(ProblemShape{}) == 4) or
          args.mode == GemmUniversalMode::kGemmSplitKParallel;
  }

  // GFX928下TCC冲突时(连续方向为sizeof(half) * 4096)，使用Padding提升访存性能
  static size_t
  get_workspace_size(Arguments const& args) {
    size_t workspace_size = 0;
    auto M = cute::shape<0>(args.problem_shape);
    auto N = cute::shape<1>(args.problem_shape);
    auto K = cute::shape<2>(args.problem_shape);
    //TN类型
    if(is_same_v<StrideA, typename TagToStrideA<layout::RowMajor>::type> && is_same_v<StrideB, typename TagToStrideB<layout::ColumnMajor>::type>){
        //a矩阵
        if(K % 2048 == 0){
          workspace_size +=  (PADDING_A + K) * M * sizeof(ElementA);
          workspace_size +=  (PADDING_B + K) * N * sizeof(ElementB);
        }
    }
    //NN类型
    if(is_same_v<StrideA, typename TagToStrideA<layout::ColumnMajor>::type> && is_same_v<StrideB, typename TagToStrideB<layout::ColumnMajor>::type>){
        if(M % 2048 == 0){
          workspace_size +=  (PADDING_A + M) * K * sizeof(ElementA);
        }
        if(K % 2048 == 0){
          workspace_size +=  (PADDING_B + K) * N * sizeof(ElementB);
        }
    }
    //NT类型
    if(is_same_v<StrideA, typename TagToStrideA<layout::ColumnMajor>::type> && is_same_v<StrideB, typename TagToStrideB<layout::RowMajor>::type>){
        if(M % 2048 == 0){
          workspace_size +=  (PADDING_A + M) * K * sizeof(ElementA);
        }
        if(N % 2048 == 0){
          workspace_size +=  (PADDING_B + N) * K * sizeof(ElementB);
        }
    }
    //TT类型
    if(is_same_v<StrideA, typename TagToStrideA<layout::RowMajor>::type> && is_same_v<StrideB, typename TagToStrideB<layout::RowMajor>::type>){
        if(K % 2048 == 0){
          workspace_size +=  (PADDING_A + K) * M * sizeof(ElementA);
        }
        if(N % 2048 == 0){
          workspace_size +=  (PADDING_B + N) * K * sizeof(ElementB);
        }
    }
    return workspace_size;
  }

  static
  cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, hipStream_t stream = nullptr) {
    //
    //可能存在L的情况需要关注
    //可能存在TCP冲突L1 TCC冲突L2  TCP冲突是128KB分为4个bank？
    //padding的数据需要进一步确定并非TCC冲突完全消失会好 要权衡TCC以及TCP冲突
    #if 1
    auto M = cute::shape<0>(args.problem_shape);
    auto N = cute::shape<1>(args.problem_shape);
    auto K = cute::shape<2>(args.problem_shape);
    int ptr_B_offset = 0;
    // TCC padding对于一个threadblock处理(256线程 * 128bit)的数据
    constexpr int padding_stride_A = PADDING_BLOCK_SIZE * (sizeof(float4) / sizeof(ElementA));
    constexpr int padding_stride_B = PADDING_BLOCK_SIZE * (sizeof(float4) / sizeof(ElementA));
    //TN类型
    if(is_same_v<StrideA, typename TagToStrideA<layout::RowMajor>::type> && is_same_v<StrideB, typename TagToStrideB<layout::ColumnMajor>::type>){
        //a矩阵
        if(K % 2048 == 0){
          dim3 grid_A((K + padding_stride_A - 1) / padding_stride_A, M, 1);
          dim3 block_A(PADDING_BLOCK_SIZE, 1, 1);
          device_padding_kernel<ElementA><<<grid_A,block_A>>>((const void*)(args.mainloop.ptr_A),
                                        workspace,
                                        (int)K,
                                        (int)K,
                                        (int)M,
                                        (int)(PADDING_A + K));
          ptr_B_offset += (PADDING_A + K) * M * sizeof(ElementA);
          dim3 grid_B((K + padding_stride_B - 1) / padding_stride_B, N, 1);
          dim3 block_B(PADDING_BLOCK_SIZE, 1, 1);
          device_padding_kernel<ElementB><<<grid_B,block_B>>>((const void*)(args.mainloop.ptr_B),
                                        (void *)(reinterpret_cast<uint8_t*>(workspace) + ptr_B_offset),
                                        (int)K,
                                        (int)K,
                                        (int)N,
                                        (int)(PADDING_B + K));

        }
    }
    //NN类型
    if(is_same_v<StrideA, typename TagToStrideA<layout::ColumnMajor>::type> && is_same_v<StrideB, typename TagToStrideB<layout::ColumnMajor>::type>){
        if(M % 2048 == 0){
          ptr_B_offset +=  (PADDING_A + M) * K * sizeof(ElementA);
          dim3 grid_A((M + padding_stride_A - 1) / padding_stride_A, K, 1);
          dim3 block_A(PADDING_BLOCK_SIZE, 1, 1);
          device_padding_kernel<ElementA><<<grid_A,block_A>>>((const void*)(args.mainloop.ptr_A),
                                        workspace,
                                        (int)M,
                                        (int)M,
                                        (int)K,
                                        (int)(PADDING_A + M));
        }
        if(K % 2048 == 0){
          dim3 grid_B((K + padding_stride_B - 1) / padding_stride_B, N, 1);
          dim3 block_B(PADDING_BLOCK_SIZE, 1, 1);
          device_padding_kernel<ElementB><<<grid_B,block_B>>>((const void*)(args.mainloop.ptr_B),
                                        (void *)(reinterpret_cast<uint8_t*>(workspace) + ptr_B_offset),
                                        (int)K,
                                        (int)K,
                                        (int)N,
                                        (int)(PADDING_B + K));
        }
    }
    //NT类型
    if(is_same_v<StrideA, typename TagToStrideA<layout::ColumnMajor>::type> && is_same_v<StrideB, typename TagToStrideB<layout::RowMajor>::type>){
        if(M % 2048 == 0){
          ptr_B_offset +=  (PADDING_A + M) * K * sizeof(ElementA);
          dim3 grid_A((M + padding_stride_A - 1) / padding_stride_A, K, 1);
          dim3 block_A(PADDING_BLOCK_SIZE, 1, 1);
          device_padding_kernel<ElementA><<<grid_A,block_A>>>((const void*)(args.mainloop.ptr_A),
                                        workspace,
                                        (int)M,
                                        (int)M,
                                        (int)K,
                                        (int)(PADDING_A + M));
        }
        if(N % 2048 == 0){
          dim3 grid_B((N + padding_stride_B - 1) / padding_stride_B, K, 1);
          dim3 block_B(PADDING_BLOCK_SIZE, 1, 1);
          device_padding_kernel<ElementB><<<grid_B,block_B>>>((const void*)(args.mainloop.ptr_B),
                                        (void *)(reinterpret_cast<uint8_t*>(workspace) + ptr_B_offset),
                                        (int)N,
                                        (int)N,
                                        (int)K,
                                        (int)(PADDING_B + N));
        }
    }
    //TT类型
    if(is_same_v<StrideA, typename TagToStrideA<layout::RowMajor>::type> && is_same_v<StrideB, typename TagToStrideB<layout::RowMajor>::type>){
        if(K % 2048 == 0){
          ptr_B_offset +=  (PADDING_A + K) * M * sizeof(ElementA);
          dim3 grid_A((K + padding_stride_A - 1) / padding_stride_A, M, 1);
          dim3 block_A(PADDING_BLOCK_SIZE, 1, 1);
          device_padding_kernel<ElementA><<<grid_A,block_A>>>((const void*)(args.mainloop.ptr_A),
                                        workspace,
                                        (int)K,
                                        (int)K,
                                        (int)M,
                                        (int)(PADDING_A + K));
        }
        if(N % 2048 == 0){
          dim3 grid_B((N + padding_stride_B - 1) / padding_stride_B, K, 1);
          dim3 block_B(PADDING_BLOCK_SIZE, 1, 1);
          device_padding_kernel<ElementB><<<grid_B,block_B>>>((const void*)(args.mainloop.ptr_B),
                                        (void *)(reinterpret_cast<uint8_t*>(workspace) + ptr_B_offset),
                                        (int)N,
                                        (int)N,
                                        (int)K,
                                        (int)(PADDING_B + N));
        }
    }
    #endif
    return Status::kSuccess;
  }

  static dim3
  get_grid_shape(Params const& params) {
    int slice_k = 1;
    if constexpr (rank(ProblemShape{}) == 4) {
      if ((params.mode == GemmUniversalMode::kGemm && get<3>(params.problem_shape) > 1)) {
        slice_k = get<3>(params.problem_shape);
        auto blk_shape = TileShape{};
        auto grid_tiled_shape_k = (get<2>(params.problem_shape) + size<2>(blk_shape) - 1) / size<2>(blk_shape); // k方向block数量
        auto k_subproblem = (grid_tiled_shape_k + slice_k - 1) / slice_k;                                       // 前面若干子问题所占的block数量相同，最后若干个子问题可能小于此block大小或为0
        auto gemm_k_size = k_subproblem * size<2>(blk_shape);                                                   // 子问题在k方向所占元素个数
        slice_k = (get<2>(params.problem_shape) + gemm_k_size - 1) / gemm_k_size;                      // 去除了block数量为0的slice_k，计算的是真正需要的slice个数                                                           // 将L设置为真实需要的slice_k数量
      } else if (params.mode == GemmUniversalMode::kGemmSplitKParallel) {
        slice_k = get<3>(params.problem_shape);
        auto blk_shape = TileShape{};
        auto gemm_k_size = ceil_div(get<2>(params.problem_shape), slice_k);
        slice_k = ceil_div(get<2>(params.problem_shape), gemm_k_size);
      }
    }
    return dim3(
      cute::size(cute::ceil_div(cute::shape<0>(params.problem_shape), cute::shape<0>(TileShape{}))),
      cute::size(cute::ceil_div(cute::shape<1>(params.problem_shape), cute::shape<1>(TileShape{}))),
      slice_k
    );
  }

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }
  CUTLASS_DEVICE
  auto get_mA_mkl_tensor(Params const& params){
    decltype(params.problem_shape) problem_shape_MNKL;
    if constexpr (cute::rank(decltype(params.problem_shape){}) == 3) {
      problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    } else {
      problem_shape_MNKL = params.problem_shape;
    }
    auto M = get<0>(problem_shape_MNKL);
    auto N = get<1>(problem_shape_MNKL);
    auto K = get<2>(problem_shape_MNKL);
    auto L = 1;         // 由于L维度存的是splitK, 而splitk与batch不能共存，因此batch_count只能为1
    if(is_same_v<StrideA, typename TagToStrideA<layout::RowMajor>::type> && K % 2048 == 0)
    {
      auto da_copy = make_cute_packed_stride(StrideA{},make_shape(M,  K+PADDING_A,  L));
      return make_tensor(make_gmem_ptr((const ElementA *)params.workspace), make_shape(M,K,L), da_copy);
    }
    if(is_same_v<StrideA, typename TagToStrideA<layout::ColumnMajor>::type> && M % 2048 == 0)
    {
 
      auto da_copy = make_cute_packed_stride(StrideA{},make_shape(M + PADDING_A,  K,  L));
      return make_tensor(make_gmem_ptr((const ElementA *)params.workspace), make_shape(M,K,L), da_copy);
    }

    return make_tensor(make_gmem_ptr(params.mainloop.ptr_A), make_shape(M,K,L), params.mainloop.dA);
  }

  CUTLASS_DEVICE
  auto get_mB_nkl_tensor(Params const& params){
    decltype(params.problem_shape) problem_shape_MNKL;
    if constexpr (cute::rank(decltype(params.problem_shape){}) == 3) {
      problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    } else {
      problem_shape_MNKL = params.problem_shape;
    }
    auto M = get<0>(problem_shape_MNKL);
    auto N = get<1>(problem_shape_MNKL);
    auto K = get<2>(problem_shape_MNKL);
    auto L = 1;         // 由于L维度存的是splitK, 而splitk与batch不能共存，因此batch_count只能为1
    int offset_padding = 0;
    if(is_same_v<StrideA, typename TagToStrideA<layout::RowMajor>::type> && K % 2048 == 0)
    {
      offset_padding +=  (PADDING_A + K) * M * sizeof(ElementA);
    }
    if(is_same_v<StrideA, typename TagToStrideA<layout::ColumnMajor>::type> && M % 2048 == 0)
    {
      offset_padding +=  (PADDING_A + M) * K * sizeof(ElementA);
    }

    if(is_same_v<StrideB, typename TagToStrideB<layout::ColumnMajor>::type> && K % 2048 == 0)
    {
      auto db_copy = make_cute_packed_stride(StrideB{},make_shape(N,  K+PADDING_B,  L));
      return make_tensor(make_gmem_ptr((const ElementB *)((reinterpret_cast<uint8_t*>(params.workspace) + offset_padding))), make_shape(N,K,L), db_copy);
    }

    if(is_same_v<StrideB, typename TagToStrideB<layout::RowMajor>::type> && N % 2048 == 0)
    {
      auto db_copy = make_cute_packed_stride(StrideB{},make_shape(N + PADDING_B,  K,  L));
      return make_tensor(make_gmem_ptr((const ElementB *)((reinterpret_cast<uint8_t*>(params.workspace) + offset_padding))), make_shape(N,K,L), db_copy);
    }

    return make_tensor(make_gmem_ptr(params.mainloop.ptr_B), make_shape(N,K,L), params.mainloop.dB); 
  }
  CUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf) {
    using namespace cute;
    using X = Underscore;

    // Preconditions
    CUTE_STATIC_ASSERT(is_static<TileShape>::value);

    // Separate out problem shape for convenience
    // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    // auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    decltype(params.problem_shape) problem_shape_MNKL;
    if constexpr (cute::rank(decltype(params.problem_shape){}) == 3) {
      problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    } else {
      problem_shape_MNKL = params.problem_shape;
    }

    auto M = get<0>(problem_shape_MNKL);
    auto N = get<1>(problem_shape_MNKL);
    auto K = get<2>(problem_shape_MNKL);
    auto L = get<3>(problem_shape_MNKL);

    // Preconditions
    static_assert(rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    // Get the appropriate blocks for this thread block -- potential for thread block locality
    int thread_idx = int(threadIdx.x);
    auto blk_shape = TileShape{};

#ifndef DCU_ASM                                                                // (BLK_M,BLK_N,BLK_K)
    auto [m_coord, n_coord, l_coord] = blockIdx;
#else
    auto m_coord = blockIdx.x;
    auto n_coord = blockIdx.y;
    auto l_coord = blockIdx.z;
#endif
    auto blk_coord_mnkl = make_coord(m_coord, n_coord, _, l_coord);                                        // (m,n,k,l)

#if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__) ||defined(__gfx92a__))  && defined(DCU_ASM)
    Tensor mA_mkl = get_mA_mkl_tensor(params);
    Tensor mB_nkl = get_mB_nkl_tensor(params);
#else
    Tensor mA_mkl = make_tensor(make_gmem_ptr(params.mainloop.ptr_A), make_shape(M,K,Int<1>{}), params.mainloop.dA); //(m,k,1)  splitk与batch不能共存，因此在splitk中batch_count必须为1
    Tensor mB_nkl = make_tensor(make_gmem_ptr(params.mainloop.ptr_B), make_shape(N,K,Int<1>{}), params.mainloop.dB); //(n,k,1)  而splitk实现是复用batch_count(L维度)来传参slice_k，因此这里L写为1
#endif

    auto slice_k = L;                                                                        // slice_k存储在problem_shape的L中
    auto grid_tiled_shape_k = 0;
    auto gemm_k_size = 0;
    auto k_offset = 0;
    if (params.mode == GemmUniversalMode::kGemm) {
      grid_tiled_shape_k = (K + size<2>(blk_shape) - 1) / size<2>(blk_shape);             // k方向block数量
      auto k_subproblem = (grid_tiled_shape_k + slice_k - 1) / slice_k;                   // 前slice_k-1个子问题k方向所占的block数量               
      gemm_k_size = k_subproblem * size<2>(blk_shape);                                    // 子问题在k方向所占元素个数
      k_offset = l_coord * gemm_k_size;                                                   // 根据blockIdx.z索引，每个子问题的起始偏移
    } else if (params.mode == GemmUniversalMode::kGemmSplitKParallel) {
      gemm_k_size = ceil_div(K, slice_k);
      k_offset = l_coord * gemm_k_size;
      grid_tiled_shape_k = ceil_div(gemm_k_size, size<2>(blk_shape));
      if (K > k_offset && K <= gemm_k_size * (l_coord + 1)) {
        grid_tiled_shape_k = ceil_div(K - k_offset, size<2>(blk_shape));
      } else if (K < gemm_k_size * (l_coord + 1)) {
        return;
      }
    }
    
    auto split_k_shape = make_shape(M, N, gemm_k_size);                                       // 将整个问题规模根据slice_k切分
    
    // Slice to get the tiles this thread block is responsible for
    Tensor mA_mk_split = local_tile(mA_mkl, split_k_shape, make_coord(_,_,l_coord), Step<_1, X,_1>{});                    // (m,gemm_k_size)
    Tensor gA_layout = local_tile(mA_mk_split, blk_shape, make_coord(m_coord,n_coord,_), Step<_1, X,_1>{});              // (BLK_M,BLK_K,k_subproblem) 
    Tensor gA = take<0,3>(gA_layout);
    Tensor mB_nk_split = local_tile(mB_nkl, split_k_shape, make_coord(_,_,l_coord), Step<X,_1,_1>{});                     // (n,gemm_k_size)
    Tensor gB_layout = local_tile(mB_nk_split, blk_shape, make_coord(m_coord,n_coord,_), Step<X,_1,_1>{});               // (BLK_N,BLK_K,k_subproblem)
    Tensor gB = take<0,3>(gB_layout);

    // Compute tile residues for predication
    auto m_max_coord = M - size<0>(gA) * get<0>(blk_coord_mnkl);                             // M - BLK_M * m_coord
    auto n_max_coord = N - size<0>(gB) * get<1>(blk_coord_mnkl);                             // N - BLK_N * n_coord
                                 
    int k_residue = 0;
    if (params.mode == GemmUniversalMode::kGemm) {
      if (l_coord == gridDim.z - 1) {
        k_residue = K - grid_tiled_shape_k * size<1>(gA);                                       
      }
    } else if (params.mode == GemmUniversalMode::kGemmSplitKParallel) {
      k_residue = gemm_k_size - grid_tiled_shape_k * size<1>(gA);
      if (K > k_offset && K <= gemm_k_size * (l_coord + 1)) {
        k_residue = K - k_offset - grid_tiled_shape_k * size<1>(gA);
      }
    }

    auto residue_mnk = make_tuple(m_max_coord, n_max_coord, k_residue);

    // Allocate the tiled_mma and the accumulators for the (M,N) blk_shape
    TiledMma tiled_mma;
    Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape)); // (MMA,MMA_M,MMA_N)
    clear(accumulators);

    K = min(K, (l_coord + 1) * gemm_k_size);                      // K取problem_size_k和slice_k个子问题k方向长度总和的较小值

    int k_tile_count = (K - k_offset + size<2>(blk_shape) - 1) / size<2>(blk_shape);        // 每个子问题的迭代次数
    auto k_tile_iter  = cute::make_coord_iterator(k_tile_count); 

    auto grid_tiled_shape_m = (M + size<0>(blk_shape) - 1) / size<0>(blk_shape); 
    int block_idx = get<0>(blk_coord_mnkl) + get<1>(blk_coord_mnkl) * grid_tiled_shape_m;

    // Perform the collective scoped MMA
    CollectiveMainloop collective_mma;
    collective_mma(
      accumulators,
      gA,
      gB,
      accumulators,
      k_tile_iter, k_tile_count,
      residue_mnk,
      thread_idx,
      smem_buf
    );

    EpilogueParams epilogue_params = params.epilogue;
    // 在串行splitk中，需要将每一段子问题的计算结果D依次暂存到C中
    if (params.mode == GemmUniversalMode::kGemm && slice_k > 1 && l_coord) {
      epilogue_params.ptr_C = epilogue_params.ptr_D;
      epilogue_params.dC = epilogue_params.dD;
    } else if (params.mode == GemmUniversalMode::kGemmSplitKParallel) {
      // 在并行splitk中，需要将每一段子问题的计算结果存储到相对应的workspace中
      size_t splitk_slice_stride = size_t(M) * size_t(N) * sizeof(ElementC);
      auto * parallel_workspace = reinterpret_cast<uint8_t*>(params.semaphore);
      epilogue_params.ptr_D = reinterpret_cast<ElementD*>(
          parallel_workspace + splitk_slice_stride * l_coord);
      epilogue_params.thread.beta = 0;  // parallel中alpha*A*B+beta*C的操作是在reductionKernel中实现的
      epilogue_params.thread.alpha = 1;
    }

    // Epilogue and write to gD
    CollectiveEpilogue epilogue{epilogue_params};

    // Construct the semaphore.
    Semaphore semaphore(params.semaphore + block_idx, thread_idx);

    // If performing a reduction via split-K, fetch the initial synchronization
    if (params.mode == GemmUniversalMode::kGemm && slice_k > 1) {
      // Fetch the synchronization lock initially but do not block.
      semaphore.fetch();
      // Indicate which position in a serial reduction the output operator is currently updating
      epilogue.set_k_partition(l_coord, slice_k);
    }

    // Wait on the semaphore - this latency may have been covered by iterator construction
    if (params.mode == GemmUniversalMode::kGemm && slice_k > 1) {    
      semaphore.wait(l_coord);
    }
    
    epilogue(
      problem_shape_MNKL,
      blk_shape,
      blk_coord_mnkl,
      accumulators,
      tiled_mma,
      residue_mnk,
      thread_idx,
      smem_buf
    );

    // Release the semaphore
    if (params.mode == GemmUniversalMode::kGemm && slice_k > 1) {
      int lock = 0;
      if (slice_k == l_coord + 1) {
        // The final threadblock resets the semaphore for subsequent grids.
        lock = 0;
      } else {
        // Otherwise, the semaphore is incremented
        lock = l_coord + 1;
      }
      semaphore.release(lock);
    }

  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel
