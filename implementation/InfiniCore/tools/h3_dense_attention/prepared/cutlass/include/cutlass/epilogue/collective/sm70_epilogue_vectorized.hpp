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
/*! \file
  \brief Functor performing elementwise operations used by epilogues.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace collective {


template <class StrideType, class ElementType, class BLK_MN, class BLK_K>
CUTE_HOST_DEVICE constexpr
auto
tiled_smem_acc_layout()
{
  auto BLK_MN0 = size<0>(BLK_MN{});
  auto BLK_K0  = size<0>(BLK_K{});

  static_assert(BLK_MN0 % 16 == 0, "BLK_MN0 must be a multiple of 8.");
  static_assert(BLK_K0 % 16 == 0,  "BLK_K0 must be a multiple of 8.");
  //细化swizzle参数
  if constexpr (cutlass::gemm::detail::is_mn_major<StrideType>()) {
    using smem_layout = decltype(make_layout(make_shape(BLK_MN0, BLK_K0),
                                             make_stride(Int<1>{}, BLK_MN0)));

    return composition(Swizzle<2, 3, 3>{}, smem_layout{});
  } else if constexpr (cutlass::gemm::detail::is_k_major<StrideType>()) {
    using smem_layout = decltype(make_layout(make_shape(BLK_MN0, BLK_K0),
                                             make_stride(BLK_K0, Int<1>{})));
    return composition(Swizzle<2, 3, 3>{}, smem_layout{});
  }
}

//需要考虑swizzle

template <int ThreadCount, class Element, int Alignment, class StrideType, class TileM, class TileN>
CUTE_HOST_DEVICE constexpr
auto
make_acc_tiled_smem_copy()
{

  using AlignmentType = cute::uint_byte_t<static_cast<int>(sizeof(Element)) * Alignment>;
  constexpr int TileSizeM  = cute::size(TileM{});
  constexpr int TileSizeN   = cute::size(TileN{});

  // Maximize the number of threads along the gmem major mode to promote coalesced reads
  // While making sure our thread layout tiles the threadblock tile evenly

  if constexpr (cutlass::gemm::detail::is_k_major<StrideType>()) {
    // K major thread layout for K major gmem
    constexpr int threads_major = TileSizeN   / Alignment;
    constexpr int threads_minor = ThreadCount / threads_major;
    static_assert(threads_major > 0);
    static_assert(ThreadCount % threads_major == 0);
    static_assert(threads_minor == 0 || (TileSizeM % threads_minor == 0));

    return make_tiled_copy(
      Copy_Atom<UniversalCopy<AlignmentType>, Element>{},
      Layout<Shape <Int<threads_minor>,Int<threads_major>>,
             Stride<Int<threads_major>,                _1>>{},
      Layout<Shape<Int<TileSizeM / threads_minor>,Int<Alignment>>>{});
  }
  else if constexpr (cutlass::gemm::detail::is_mn_major<StrideType>()) {
    // MN major thread layout for MN major gmem
    constexpr int threads_major = TileSizeM  / Alignment;
    constexpr int threads_minor = ThreadCount / threads_major;
    static_assert(threads_major > 0);
    static_assert(ThreadCount % threads_major == 0);
    static_assert(threads_minor == 0 || (TileSizeN % threads_minor == 0));
    return make_tiled_copy(
        Copy_Atom<UniversalCopy<AlignmentType>, Element>{},
        Layout<Shape <Int<threads_major>,Int<threads_minor>>,
              Stride<                _1,Int<threads_major>>>{},
          Layout<Shape<Int<Alignment>,Int<1>>>{});
        // Layout<Shape<Int<Alignment>,Int<TileSizeN / threads_minor>>>{});
  }
  else {
    static_assert(cute::is_void_v<Element>, "Unsupported gmem layout for automatic gmem tiled copy builder.");
  }
}
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies an element wise operation to all elements within the fragment
/// and writes it out to destination storage.
///
/// Ways to generalize this:
/// - CTA tile shape
/// - vectorization requirements (GMEM)
/// - vectoriz(able) transform() 
///
template <
  class StrideC_,
  class StrideD_,
  class ThreadEpilogueOp_,
  class SmemLayout_,
  class CopyAtomR2S_,
  class TiledCopyS2R_,
  class CopyAtomR2G_
>
class Epilogue {
public:
  //
  // Type Aliases
  //
  // derived types of output thread level operator
  using ThreadEpilogueOp = ThreadEpilogueOp_;
  using ElementAccumulator = typename ThreadEpilogueOp::ElementAccumulator;
  using ElementCompute = typename ThreadEpilogueOp::ElementCompute;
  using ElementScalar = ElementCompute;
  using ElementOutput = typename ThreadEpilogueOp::ElementOutput;
  using ElementC = typename ThreadEpilogueOp::ElementC;
  using StrideC = StrideC_;
  using ElementD = typename ThreadEpilogueOp::ElementD;
  using StrideD = StrideD_;

  using SmemLayout   = SmemLayout_;
  using CopyAtomR2S  = CopyAtomR2S_;
  using TiledCopyS2R = TiledCopyS2R_;
  using CopyAtomR2G  = CopyAtomR2G_;

  using GmemTiledCopyC = void;
  using GmemTiledCopyD = void;
  
  static const int kOutputAlignment = ThreadEpilogueOp::kCount;
  using AlignmentType = typename cute::uint_bit<sizeof_bits<ElementOutput>::value * kOutputAlignment>::type;

  static_assert(rank(StrideC{}) == 3, "StrideCD must be rank-3: [M, N, L]");
  static_assert(rank(StrideD{}) == 3, "StrideCD must be rank-3: [M, N, L]");

  struct SharedStorage {
      // cute::array_aligned<ElementAccumulator, cute::cosize_v<SmemLayout>> smem_epilogue;
  };

  // Host side epilogue arguments
  struct Arguments {
    typename ThreadEpilogueOp::Params thread{};
    ElementC const* ptr_C = nullptr;
    StrideC dC{};
    ElementD* ptr_D = nullptr;
    StrideD dD{};
  };

  // Device side epilogue params
  using Params = Arguments;

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      [[maybe_unused]] ProblemShape const& _,
      Arguments const& args,
      [[maybe_unused]] void* workspace) {
    return args;
  }

  template<class ProblemShape>
  CUTLASS_HOST_DEVICE static bool
  can_implement(
      [[maybe_unused]] ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  Epilogue(Params const& params_)
      : params(params_), epilogue_op(params_.thread) { }

  CUTLASS_DEVICE
  bool
  is_source_needed() {
    return epilogue_op.is_source_needed();
  }

  CUTLASS_HOST_DEVICE
  auto set_k_partition(int k_partition, int k_partition_count)
  {
    return epilogue_op.set_k_partition(k_partition,k_partition_count);
  }

  template<
    class ProblemShapeMNKL,
    class BlockShapeMNK,
    class BlockCoordMNKL,
    class FrgEngine, class FrgLayout,
    class TiledMma,
    class ResidueMNK
  >
  CUTLASS_DEVICE void
  operator()(
      ProblemShapeMNKL problem_shape_mnkl,
      BlockShapeMNK blk_shape_MNK,
      BlockCoordMNKL blk_coord_mnkl,
      cute::Tensor<FrgEngine,FrgLayout> const& accumulators,                   // (MMA,MMA_M,MMA_N)
      TiledMma tiled_mma,
      ResidueMNK residue_mnk,
      int thread_idx,
      char* smem_buf)
  {

    using namespace cute;
    using X = Underscore;

    static_assert(rank(ProblemShapeMNKL{}) == 4, "ProblemShapeMNKL must be rank 4");
    static_assert(is_static<BlockShapeMNK>::value, "ThreadBlock tile shape must be static");
    static_assert(rank(BlockShapeMNK{}) == 3, "BlockShapeMNK must be rank 3");
    static_assert(rank(BlockCoordMNKL{}) == 4, "BlockCoordMNKL must be rank 3");

    // synchronizing function for smem reads/writes
#if CUDA_BARRIER_ENABLED
    auto synchronize = [] () { cutlass::arch::NamedBarrier::sync(typename TiledCopyS2R::TiledNumThr{}, 0); };
#else
    auto synchronize = [] () { __syncthreads(); };
#endif

    // Separate out problem shape for convenience
    auto M = get<0>(problem_shape_mnkl);
    auto N = get<1>(problem_shape_mnkl);
    auto L = get<3>(problem_shape_mnkl);

    // Represent the full output tensor
    Tensor mC_mnl = make_tensor(make_gmem_ptr(params.ptr_C), make_shape(M,N,L), params.dC);      //             (m,n,l)
    Tensor mD_mnl = make_tensor(make_gmem_ptr(params.ptr_D), make_shape(M,N,L), params.dD);      //             (m,n,l)
    Tensor gC_mnl = local_tile(mC_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});      // (BLK_M,BLK_N,m,n,l)
    Tensor gD_mnl = local_tile(mD_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});      // (BLK_M,BLK_N,m,n,l)

    // Slice to get the tile this CTA is responsible for
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord_mnkl;
    Tensor gC = gC_mnl(_,_,m_coord,n_coord,l_coord);                                                   // (BLK_M,BLK_N)
    Tensor gD = gD_mnl(_,_,m_coord,n_coord,l_coord);                                                   // (BLK_M,BLK_N)
    // // Construct a tensor in SMEM that we can partition for rearranging data

    using MNK = typename TiledMma::TiledShape_MNK;
    // 128x32
    // using SmemLayoutC = decltype(make_layout(make_shape(get<0>(MNK{}), get<1>(MNK{})),
    //                                         make_stride(Int<1>{}, get<0>(MNK{}))));
    using SmemLayoutC = decltype(tiled_smem_acc_layout<StrideC, 
                                                        ElementAccumulator,
                                                        decltype(cute::get<0>(MNK{})),
                                                        decltype(cute::get<1>(MNK{}))>());

    using SmemArray =  cute::array_aligned<ElementAccumulator, cute::cosize_v<SmemLayoutC>>;

    SmemArray& smem_epilogue = *reinterpret_cast<SmemArray* >(smem_buf);
    Tensor sC = make_tensor(make_smem_ptr(smem_epilogue.data()), SmemLayoutC{});  

    // SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    // Tensor sC = make_tensor(make_smem_ptr(storage.smem_epilogue.data()), SmemLayout{});              // (SMEM_M,SMEM_N)
    // Partition sC to match the accumulator partitioning
    auto tiled_r2s = make_tiled_copy_C(CopyAtomR2S{}, tiled_mma);
    auto tC     = tiled_r2s.get_thread_slice(thread_idx);
    Tensor tCaC = tC.retile_S(accumulators);                                          // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor tCsC = tC.partition_D(sC);                                                 // ((Atom,AtomNum),PIPE_M,PIPE_N)


    // Tile gD and gC by the shape of SmemLayout first
    auto tile  = make_shape(size<0>(sC), size<1>(sC));
    Tensor gCt = local_tile(gC, tile, _);                                              // (SMEM_M,SMEM_N,TILE_M,TILE_N)
    Tensor gDt = local_tile(gD, tile, _);                                              // (SMEM_M,SMEM_N,TILE_M,TILE_N)

    // Partition sC, gC, and gD for the output
    auto tiled_s2r = make_acc_tiled_smem_copy<cute::size(TiledMma{}),ElementAccumulator,
                                              32 / sizeof_bits<ElementAccumulator>::value,
                                              StrideC,
                                              decltype(cute::get<0>(SmemLayoutC{})),
                                              decltype(cute::get<1>(SmemLayoutC{}))>();
    //  auto tiled_s2r  = TiledCopyS2R{};                                                      
    auto tD     = tiled_s2r.get_thread_slice(thread_idx);
    Tensor tDsC = tD.partition_S(sC);                                   //               ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tDgC = tD.partition_D(gCt);                                  // ((Atom,AtomNum),ATOM_M,ATOM_N,TILE_M,TILE_N)
    Tensor tDgD = tD.partition_D(gDt);                                  // ((Atom,AtomNum),ATOM_M,ATOM_N,TILE_M,TILE_N)

    //tCaC ---> tCsC ---->tDrC ---> ep --->copy

    // Allocate intermediate registers on the dst tensors
    Tensor tDrC = make_tensor<ElementAccumulator>(take<0,3>(shape(tDgC)));            // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tDrD = make_tensor<ElementOutput>(shape(tDrC));                            // ((Atom,AtomNum),ATOM_M,ATOM_N)

    // // Repeat the D-partitioning for coordinates and predication
    Tensor cD   = make_identity_tensor(make_shape(size<0>(gD),size<1>(gD)));          // (BLK_M,BLK_N) -> (blk_m,blk_n)
    Tensor cDt  = local_tile(cD, tile, _);                              //                (SMEM_M,SMEM_N,TILE_M,TILE_N)
    Tensor tDcD = tD.partition_D(cDt);                                  // ((Atom,AtomNum),ATOM_M,ATOM_N,TILE_M,TILE_N)

    CUTE_STATIC_ASSERT(size<1>(tCaC) % size<3>(tDgC) == 0);  // TILE_M divides MMA_M
    CUTE_STATIC_ASSERT(size<2>(tCaC) % size<4>(tDgC) == 0);  // TILE_N divides MMA_N

#if 0
    if(thread0()){
      print(cD.layout());print("\n");
      print(cDt.layout());print("\n");
    }
    if (thread_idx == 0 && m_coord == 0 && n_coord == 0) {
      print("aC   : "); print(accumulators.layout()); print("\n");
      print("gC   : "); print(gC.layout()); print("\n");
      print("gD   : "); print(gD.layout()); print("\n");
      print("sC   : "); print(sC.layout()); print("\n");
      print("\n");
      print("tCsC : "); print(tCsC.layout()); print("\n");
      print("tCaC : "); print(tCaC.layout()); print("\n");
      print("\n");
      print("gDt  : "); print(gDt.layout()); print("\n");
      print("tDsC : "); print(tDsC.layout()); print("\n");
      print("tDrC : "); print(tDrC.layout()); print("\n");
      print("\n");
      print("tDrD : "); print(tDrD.layout()); print("\n");
      print("tDgC : "); print(tDgC.layout()); print("\n");
      print("tDgD : "); print(tDgD.layout()); print("\n");
      print("\n");
    }
#endif
    // For each tiling needed for SmemLayout to cover shape(gD)

#if 1
    //128x128 ---> 32x32 的迭代次数 block--->tiled
    //先分析一个block 其他block情况类似
    // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
    {
      CUTLASS_PRAGMA_UNROLL
      for (int step_m = 0; step_m < size<2>(cDt); ++step_m) // 128 / 32
      {
        CUTLASS_PRAGMA_UNROLL
        for (int step_n = 0; step_n < size<3>(cDt); ++step_n)// 128 / 32
        {
          // Step 1. Copy to SMEM
          CUTLASS_PRAGMA_UNROLL
          for (int pipe_m = 0; pipe_m < size<1>(tCsC); ++pipe_m) {

            CUTLASS_PRAGMA_UNROLL
            for (int pipe_n = 0; pipe_n < size<2>(tCsC); ++pipe_n) {
                int mma_m = step_m * size<1>(tCsC) + pipe_m;
                int mma_n = step_n * size<2>(tCsC) + pipe_n;

              copy(tiled_r2s, tCaC(_,mma_m,mma_n), tCsC(_,pipe_m,pipe_n));

            }
          }
          // Step 2. Wait for SMEM writes to complete
          synchronize();
   
          // Step 3. Copy from SMEM into a fragment
          copy(tiled_s2r, tDsC, tDrC);
          // Step 4. Wait for SMEM reads to complete
          synchronize();
          Tensor tDgDmn = tDgD(_,_,_,step_m,step_n);
          Tensor tDcDmn = tDcD(_,_,_,step_m,step_n);

          if (epilogue_op.is_source_needed()) {
            // source is needed
            Tensor tDgCmn = tDgC(_,_,_,step_m,step_n);
            CUTLASS_PRAGMA_UNROLL
            for (int m = 0; m < size<1>(tDgDmn); ++m) 
            {
              CUTLASS_PRAGMA_UNROLL
              for (int n = 0; n < size<2>(tDgDmn); ++n) 
              {
                // Predication
                if (get<0>(tDcDmn(0,m,n)) < get<0>(residue_mnk) &&
                    get<1>(tDcDmn(0,m,n)) < get<1>(residue_mnk)) 
                {
                  // Step 5. Elementwise operation with conversion
                  CUTLASS_PRAGMA_UNROLL
                  for (int i = 0; i < size<0>(tDrC); ++i) {
                    tDrD(i,m,n) = epilogue_op(tDrC(i,m,n), tDgCmn(i,m,n));
                  }
                  // Step 6. Copy to GMEM
                  copy(CopyAtomR2G{}, tDrD(_,m,n), tDgDmn(_,m,n));
                }
              }
            }
          }
          else {
            // source is not needed, avoid load and lift compute
            
            // Step 5. Elementwise operation with conversion

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(tDrC); ++i) {
              tDrD(i) = epilogue_op(tDrC(i));
            }
            CUTLASS_PRAGMA_UNROLL
            for (int m = 0; m < size<1>(tDgDmn); ++m) 
            {
              CUTLASS_PRAGMA_UNROLL
              for (int n = 0; n < size<2>(tDgDmn); ++n) 
              {
                // Predication
                if (get<0>(tDcDmn(0,m,n)) < get<0>(residue_mnk) &&
                    get<1>(tDcDmn(0,m,n)) < get<1>(residue_mnk)) 
                {
                  
                  // Step 6. Copy to GMEM
                  copy(CopyAtomR2G{},tDrD(_,m,n), tDgDmn(_,m,n));
                 
                }
              }
            }
          }
        }
      }
    }
  
#endif 
  }

private:
  Params params;
  ThreadEpilogueOp epilogue_op;
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
