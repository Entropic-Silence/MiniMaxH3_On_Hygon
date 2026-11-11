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

#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/detail/layout.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_generic.h"
#include "cutlass/epilogue/thread/linear_combination_bias_elementwise.h"


#if defined(__CUDACC_RTC__)
#include <cuda/std/type_traits>
#else
#include <type_traits>
#endif

///////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::collective {

///////////////////////////////////////////////////////////////////////////////

namespace detail {




} // namespace detail

///////////////////////////////////////////////////////////////////////////////
// No-smem builder
template <
  class TileShape_MNK,
  class ClusterShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC_,
  class GmemLayoutTagC_,
  int AlignmentC,
  class ElementD,
  class GmemLayoutTagD,
  int AlignmentD,
  FloatRoundStyle RoundStyle
>
struct CollectiveBuilder<
    arch::Sm70,
    arch::OpClassTensorOp,
    TileShape_MNK,
    ClusterShape_MNK,
    EpilogueTileType,
    ElementAccumulator,
    ElementCompute,
    ElementC_,
    GmemLayoutTagC_,
    AlignmentC,
    ElementD,
    GmemLayoutTagD,
    AlignmentD,
    NoSmemWarpSpecialized,
    fusion::LinearCombination<ElementD,ElementCompute,ElementCompute,RoundStyle>,
    void> {

    // Passing void C disables source load
    using ElementC = cute::conditional_t<cute::is_void_v<ElementC_>,
        ElementD, ElementC_>; // prevents cute breakages
    using GmemLayoutTagC = cute::conditional_t<cute::is_void_v<ElementC_>,
        GmemLayoutTagD, GmemLayoutTagC_>;
    static constexpr thread::ScaleType::Kind ScaleType = cute::is_void_v<ElementC_> ?
        thread::ScaleType::OnlyAlphaScaling : thread::ScaleType::Default;

    static constexpr int FragmentSize = 1;
    using ThreadOp = thread::LinearCombination<
      ElementD, FragmentSize, ElementAccumulator, ElementCompute,
      ScaleType, RoundStyle, ElementC>;

    using CollectiveOp = cutlass::epilogue::collective::DefaultEpilogue<
                              cutlass::detail::TagToStrideC_t<GmemLayoutTagC>,
                              cutlass::detail::TagToStrideC_t<GmemLayoutTagD>,
                              ThreadOp,
                              cutlass::gemm::EpilogueDefault>;
};
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class TileShape_MNK,
  class ClusterShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC_,
  class GmemLayoutTagC_,
  int AlignmentC,
  class ElementD,
  class GmemLayoutTagD,
  int AlignmentD,
  FloatRoundStyle RoundStyle
>
struct CollectiveBuilder<
    arch::Sm70,
    arch::OpClassTensorOp,
    TileShape_MNK,
    ClusterShape_MNK,
    EpilogueTileType,
    ElementAccumulator,
    ElementCompute,
    ElementC_,
    GmemLayoutTagC_,
    AlignmentC,
    ElementD,
    GmemLayoutTagD,
    AlignmentD,
    VectorSpecialized,
    fusion::LinearCombination<ElementD,ElementCompute,ElementCompute,RoundStyle>,
    void> {
  // Passing void C disables source load
  using ElementC = cute::conditional_t<cute::is_void_v<ElementC_>,
      ElementD, ElementC_>; // prevents cute breakages
  using GmemLayoutTagC = cute::conditional_t<cute::is_void_v<ElementC_>,
      GmemLayoutTagD, GmemLayoutTagC_>;
  static constexpr thread::ScaleType::Kind ScaleType = cute::is_void_v<ElementC_> ?
      thread::ScaleType::OnlyAlphaScaling : thread::ScaleType::Default;

  static constexpr int FragmentSize = 1;
  using ThreadOp = thread::LinearCombination<
        ElementD, FragmentSize, ElementAccumulator, ElementCompute,
        ScaleType, RoundStyle, ElementC>;

  //当前使用void作为标识，因为该接口无法接入mma参数 实际调用的时候会使用gemm collective builder的mma
  //因此可以在构建pipline的时候进行创建SmemLayoutC以及TiledCopyS2R
  //这里使用void传参 pipline内部进行判断该参数是否是void
  
  using SmemLayoutC = void;

  using TiledCopyS2R = void;

  using CollectiveOp = cutlass::epilogue::collective::Epilogue<
                                cutlass::detail::TagToStrideC_t<GmemLayoutTagC>,
                                cutlass::detail::TagToStrideC_t<GmemLayoutTagD>,
                                ThreadOp,
                                SmemLayoutC,   
                                Copy_Atom<DefaultCopy,ElementAccumulator>,                                        // R2S with tiled_mma layout
                                TiledCopyS2R,                     // Value layout
                                Copy_Atom<DefaultCopy,ElementD>                                        // R2G with S2R_dst layout
                                >;
};
// Auto builder
template <
  class TileShape_MNK,
  class ClusterShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC,
  class GmemLayoutTagC,
  int AlignmentC,
  class ElementD,
  class GmemLayoutTagD,
  int AlignmentD,
  class FusionOperation
>
struct CollectiveBuilder<
    arch::Sm75,
    arch::OpClassTensorOp,
    TileShape_MNK,
    ClusterShape_MNK,
    EpilogueTileType,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    GmemLayoutTagC,
    AlignmentC,
    ElementD,
    GmemLayoutTagD,
    AlignmentD,
    EpilogueScheduleAuto,
    FusionOperation,
    void> {
private:
  // Pick No-Smem epilogue as the Auto Epilogue Schedule (Auto schedules do not guarantee best performance) 
  // since TMA epilogues are not compatible with non-TMA non-WS mainloops
  using EpilogueSchedule = VectorSpecialized;//NoSmemWarpSpecialized;
  using _CollectiveBuilder = CollectiveBuilder<
    arch::Sm70,
    arch::OpClassTensorOp,
    TileShape_MNK,
    ClusterShape_MNK,
    EpilogueTileType,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    GmemLayoutTagC,
    AlignmentC,
    ElementD,
    GmemLayoutTagD,
    AlignmentD,
    EpilogueSchedule,
    FusionOperation
  >;

public:
  using CollectiveOp = typename _CollectiveBuilder::CollectiveOp;
};
///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::collective
