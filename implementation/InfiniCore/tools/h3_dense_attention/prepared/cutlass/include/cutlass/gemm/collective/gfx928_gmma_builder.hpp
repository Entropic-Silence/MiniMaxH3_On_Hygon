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

#include "cutlass/arch/mma.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/sm70_mma_twostage.hpp"

#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_atom.hpp"
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
//config A和B的copy范式
namespace detail {


using TiledMma_16x16x16 = TiledMMA<MMA_Atom<GFX928_16x16x16_F32F16F16F32_NT>,
                                  Layout<Shape<_2,_2,_1>>,
                                  Layout<Shape<_1,_1,_1>>>; 
                                  
using TiledMma_32x32x16 = TiledMMA<MMA_Atom<GFX928_32x32x16_F32F16F16F32_NT_ALT>,
                                  Layout<Shape<_2,_2,_1>>, 
                                  Layout<Shape<_1,_1,_1>>>; 

//to do根据shape进行选择tile mma的大小设置
template <
  class ElementA,
  class ElementB,
  class ElementC,
  class TileShape_MNK,
  auto... Args                       
>
CUTE_HOST_DEVICE constexpr
auto
mmac_op_selector() {
  static_assert(is_static<TileShape_MNK>::value, "TileShape_MNK must be static.");
  static_assert(rank(TileShape_MNK{}) == 3, "TileShape_MNK must be rank 3.");
  //最小的mnk中m应该为32
  static_assert(size<0>(TileShape_MNK{}) % 32 == 0, "Tile_M must be a multiple of 32.");
  auto Tile_N = size<1>(TileShape_MNK{});

  // FP16 accumulator
  if constexpr (is_same_v<ElementC, half_t>) {
    static_assert(is_same_v<ElementA, half_t>, "Element types for AB must be half if ElementC is half.");
    static_assert(is_same_v<ElementB, half_t>, "Element types for AB must be half if ElementC is half.");
    static_assert(size<2>(TileShape_MNK{}) % 16 == 0, "Tile_K must be a multiple of 16.");
  }

  // FP32 accumulator
  else if constexpr (is_same_v<ElementC, float>) {
    // FP16 inputs
    if constexpr (is_same_v<ElementA, half_t>) {
      static_assert(is_same_v<ElementA, ElementB>, "ElementA and ElementB must be the same type for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 16 == 0, "Tile_K must be a multiple of 16.");
      if constexpr (Tile_N % 32 == 0) {
        return GFX928_32x32x16_F32F16F16F32_NT_ALT{};
      } else if constexpr (Tile_N % 16 == 0) {
        return GFX928_16x16x16_F32F16F16F32_NT{};
      }
    }

    // BF16 inputs
    else if constexpr (is_same_v<ElementA, bfloat16_t>) {
      static_assert(is_same_v<ElementA, ElementB>, "ElementA and ElementB must be the same type for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 16 == 0, "Tile_K must be a multiple of 16.");
      if constexpr (Tile_N % 32 == 0) {
        return GFX928_32x32x16_F32BF16BF16F32_NT_ALT{};
      } else if constexpr (Tile_N % 16 == 0) {
        return GFX928_16x16x16_F32BF16BF16F32_NT{};
      }
    }

    // TF32 inputs
    else if constexpr (is_same_v<ElementA, tfloat32_t>) {
      static_assert(is_same_v<ElementA, ElementB>, "ElementA and ElementB must be the same type for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 8 == 0, "Tile_K must be a multiple of 8.");
      if constexpr (Tile_N % 16 == 0) {
        return GFX928_16x16x8_F32TF32TF32F32_NT{};
      }
    }
    else if constexpr (is_same_v<ElementA, float>) {
      static_assert(is_same_v<ElementA, ElementB>, "ElementA and ElementB must be the same type for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 8 == 0, "Tile_K must be a multiple of 8.");
      if constexpr (Tile_N % 16 == 0) {
        return GFX928_16x16x8_F32F32F32F32_NT{};
      }
    } else {
      static_assert(sizeof(ElementA) == 0, "No eligible GMMA operator for request configuration.");
    }
  }
  // int32_t accumulators
  else if constexpr (is_same_v<ElementC, int32_t>) {
    static_assert(is_same_v<ElementA, ElementB>, "ElementA and ElementB must be the same type for this config.");
    static_assert(size<2>(TileShape_MNK{}) % 32 == 0, "Tile_K must be a multiple of 32.");
    if constexpr (is_same_v<ElementA, int8_t>) {
      if constexpr (Tile_N % 32 == 0) {
        return GFX928_32x32x32_I32I8I8I32_NT{};
      } else if constexpr (Tile_N % 16 == 0) {
        return GFX928_16x16x32_I32I8I8I32_NT{};
      } else {
        static_assert(sizeof(ElementA) == 0, "No eligible GMMA operator for request configuration.");
      }
    } else if constexpr (is_same_v<ElementA, uint8_t>) {
      if constexpr (Tile_N % 32 == 0) {
        return GFX928_32x32x32_I32U8U8I32_NT{};
      } else if constexpr (Tile_N % 16 == 0) {
        return GFX928_16x16x32_I32U8U8I32_NT{};
      } else {
        static_assert(sizeof(ElementA) == 0, "No eligible GMMA operator for request configuration.");
      }
    }
  }
  // Unknown accumulator type
  else {
    static_assert(sizeof(ElementC) == 0, "Unknown ElementC accumulator type.");
  }
}
//
// F16
//

// Generates the most efficient possible TiledCopy with cp  atom given a set of parameters.
template<int ThreadCount, class Element, int Alignment, class StrideType, class TileMN, class TileK>
constexpr auto
make_cp_gmem_tiled_copy() {

  constexpr int TileSizeMN  = cute::size(TileMN{});
  constexpr int TileSizeK   = cute::size(TileK{});

  // Maximize the number of threads along the gmem major mode to promote coalesced reads
  // While making sure our thread layout tiles the threadblock tile evenly

  if constexpr (cutlass::gemm::detail::is_k_major<StrideType>()) {

    constexpr int MaxPerThreads = TileSizeMN * TileSizeK / ThreadCount;
    constexpr int Alignment_ = 1;//cute::min(MaxPerThreads,Alignment);
    using AlignmentType = cute::uint_byte_t<static_cast<int>(sizeof(Element)) * Alignment_>;

    // K major thread layout for K major gmem
    constexpr int threads_major = TileSizeK / Alignment_;
    constexpr int threads_minor = ThreadCount / threads_major;
    static_assert(threads_major > 0);
    static_assert(ThreadCount % threads_major == 0);
    static_assert(threads_minor == 0 || (TileSizeMN % threads_minor == 0));
    return make_tiled_copy(
          Copy_Atom<UniversalCopy<AlignmentType>, Element>{},
          Layout<Shape <Int<threads_minor>,Int<threads_major>>,
                Stride<Int<threads_major>,                _1>>{},
          Layout<Shape<_1,Int<Alignment_>>>{});
  }
  else if constexpr (cutlass::gemm::detail::is_mn_major<StrideType>()) {
    // MN major thread layout for MN major gmem
    // 进行通用推算
    static_assert(TileSizeMN * TileSizeK / ThreadCount > 0);
    constexpr int MaxPerThreads = TileSizeMN * TileSizeK / ThreadCount;
    constexpr int Alignment_ = cute::min(MaxPerThreads,Alignment);
    using AlignmentType = cute::uint_byte_t<static_cast<int>(sizeof(Element)) * Alignment_>;
    // 约束条件：
    // threads_major * threads_minor = ThreadCount
    // 尽可能进行合并访存 Alignment_
    // threads_major * Alignment_ <= TileSizeMN
    // threads_minor * 1 <= TileSizeK
    // TileSizeMN % threads_major * Alignment_ == 0
    constexpr int threads_minor = TileSizeK;
    constexpr int threads_major = ThreadCount / threads_minor;
    static_assert(ThreadCount % threads_minor == 0);
    static_assert(TileSizeMN % (threads_major * Alignment_) == 0);
    return make_tiled_copy(
            Copy_Atom<UniversalCopy<AlignmentType>, Element>{},
            Layout<Shape <Int<threads_major>,Int<threads_minor>>,
                  Stride<                _1,Int<threads_major>>>{},
            Layout<Shape<Int<Alignment_>,_1>>{});
  }
  else {
    static_assert(cute::is_void_v<Element>, "Unsupported gmem layout for automatic gmem tiled copy builder.");
  }
}

template <class StrideType, class ElementType, class BLK_MN, class BLK_K>
CUTE_HOST_DEVICE constexpr
auto
tiled_smem_selector()
{
  constexpr auto BLK_MN0 = size<0>(BLK_MN{});
  constexpr auto BLK_K0  = size<0>(BLK_K{});

  static_assert(BLK_MN0 % 32 == 0, "BLK_MN0 must be a multiple of 32.");
  if constexpr (is_same_v<ElementType, float> || is_same_v<ElementType, tfloat32_t>) {
    if constexpr (BLK_MN0 % 32 == 0) {
      return composition(Swizzle<3, 2, 3>{}, Layout<Shape<_32, _8>, Stride<_1, _32>>{});
    }
  }
  //为了兼容 ds read 指令 共享内存默认是m/n major 需要进一步根据数据类型进行拆分
  else if constexpr (is_same_v<ElementType, half_t> || is_same_v<ElementType, bfloat16_t>) {
    if constexpr (BLK_MN0 % 64 == 0) {
      return composition(Swizzle<3, 3, 3>{}, Layout<Shape<_64, _16>, Stride<_1, _64>>{});
    } else if constexpr (BLK_MN0 % 32 == 0) {
      return composition(Swizzle<2, 3, 3>{}, Layout<Shape<_32, _16>, Stride<_1, _32>>{});
    }
  }
  // int8_t && uint8_ta
  else if constexpr (is_same_v<ElementType, int8_t> || is_same_v<ElementType, uint8_t>) {
    if constexpr (BLK_MN0 % 64 == 0) {
      return composition(Swizzle<3, 4, 3>{}, Layout<Shape<_64, _16>, Stride<_1, _64>>{});
    } else if constexpr (BLK_MN0 % 32 == 0) {
      return composition(Swizzle<2, 4, 3>{}, Layout<Shape<_32, _16>, Stride<_1, _64>>{});
    }
  } else {
    static_assert(sizeof(ElementType) == 0, "Unsupported type to inference SmemLayout.");
  }
}

template <class ElementType>
CUTE_HOST_DEVICE constexpr
auto
ds_read_selector() {
  if constexpr (is_same_v<ElementType, half_t> || is_same_v<ElementType, bfloat16_t>) {
    return GFX928_DS_READ_DS_M32x16_B16_ALT{};
  } else if constexpr (is_same_v<ElementType, float> || is_same_v<ElementType, tfloat32_t>) {
    return UniversalCopy<float>{};
  } else if constexpr (is_same_v<ElementType, int8_t> || is_same_v<ElementType, uint8_t>) {
    return GFX928_DS_READ_DS_M32x32_B8{};
  } else {
    static_assert(sizeof(ElementType) == 0, "Unsupported ds_read_matrix data type.");
  }
}
}

/////////////////////////////////////////////////////////////////////////////////////////////////

// MainloopSm70TwoStage
template <
  class ElementA,
  class GmemLayoutA,
  int AlignmentA,
  class ElementB,
  class GmemLayoutB,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class WarpShape_MNK,
  class InstructionShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm75,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutA,
    AlignmentA,
    ElementB,
    GmemLayoutB,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    WarpShape_MNK,
    InstructionShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
      cute::is_same_v<KernelScheduleType, KernelMultistage>>
> {
  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);

  using TileShape = TileShape_MNK;

  using DispatchPolicy = MainloopSm70TwoStage;

  using AtomLayoutMNK = Layout<Shape<decltype(cute::get<0>(WarpShape_MNK{})),decltype(cute::get<1>(WarpShape_MNK{})),decltype(cute::get<2>(WarpShape_MNK{}))>>;

  using TiledMma = decltype(cute::make_tiled_mma(detail::mmac_op_selector<
      ElementA, ElementB, ElementAccumulator, TileShape_MNK>(), AtomLayoutMNK{}));

  static constexpr uint32_t blockSize = cute::size(TiledMma{});


  // A
  using GmemTiledCopyA = decltype(detail::make_cp_gmem_tiled_copy<
      blockSize, ElementA, AlignmentA, TagToStrideA_t<GmemLayoutA>,decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
  
  // B
  using GmemTiledCopyB = decltype(detail::make_cp_gmem_tiled_copy<
      blockSize, ElementB, AlignmentB, TagToStrideB_t<GmemLayoutB>,decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

  using SmemLayoutAtomA = decltype(detail::tiled_smem_selector<
      TagToStrideA_t<GmemLayoutA>, ElementA, decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

  using SmemLayoutAtomB = decltype(detail::tiled_smem_selector<
      TagToStrideB_t<GmemLayoutB>, ElementB, decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

  // GFX928_DS_READ_DS_M32x16_B16_ALT only support M/N major
  using SmemCopyAtomA = Copy_Atom<decltype(detail::ds_read_selector<ElementA>()), ElementA>;//decltype(detail::ds_read_selector<ElementA>);//

  using SmemCopyAtomB = Copy_Atom<decltype(detail::ds_read_selector<ElementB>()), ElementB>;//decltype(detail::ds_read_selector<ElementB>);//
  
  // Mainloop
  using CollectiveOp = collective::CollectiveMma<
      MainloopSm70TwoStage, TileShape,
      ElementA,
      TagToStrideA_t<GmemLayoutA>,
      ElementB,
      TagToStrideB_t<GmemLayoutB>,
      TiledMma,
      GmemTiledCopyA, 
      SmemLayoutAtomA, 
      SmemCopyAtomA, 
      cute::identity,  // A
      GmemTiledCopyB, 
      SmemLayoutAtomB, 
      SmemCopyAtomB, 
      cute::identity   // B
    >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// 目前默认走一个 multistage 后续直接走sm70的two stage pipline 
// GMMA auto kernel schedule
template <
  class ElementA,
  class GmemLayoutA,
  int AlignmentA,
  class ElementB,
  class GmemLayoutB,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class WarpShape_MNK,
  class InstructionShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm75,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutA,
    AlignmentA,
    ElementB,
    GmemLayoutB,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    WarpShape_MNK,
    InstructionShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<cute::is_same_v<KernelScheduleType, KernelScheduleAuto>>
> {
  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);

  using CollectiveOp = typename CollectiveBuilder<
      arch::Sm75,
      arch::OpClassTensorOp,
      ElementA,
      GmemLayoutA,
      AlignmentA,
      ElementB,
      GmemLayoutB,
      AlignmentB,
      ElementAccumulator,
      TileShape_MNK,
      WarpShape_MNK,
      InstructionShape_MNK,
      ClusterShape_MNK,
      StageCountType,
      KernelMultistage
    >::CollectiveOp;
};
/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
