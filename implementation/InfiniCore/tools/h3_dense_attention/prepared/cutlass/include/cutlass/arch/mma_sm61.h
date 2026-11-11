/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Matrix multiply
*/

#pragma once

#include "cutlass/layout/matrix.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace arch {

/////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef DCU_ASM
#if defined(__cplusplus)
    extern "C" {
#endif
__device__
__attribute__((const))
unsigned int __ockl_sdot4(char4, char4, unsigned int, bool);
#if defined(__cplusplus)
    } // extern "C"
#endif
#endif

/// Matrix multiply-add operation
template <typename LayoutA, typename LayoutB, typename LayoutC>
struct Mma<
  gemm::GemmShape<1,1,4>,
  1,
  int8_t,
  LayoutA,
  int8_t,
  LayoutB,
  int,
  LayoutC,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<1, 1, 4>;
  using Operator = OpMultiplyAdd;
  using ElementC = int;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<int, 1> &d,
    Array<int8_t, 4> const &a,
    Array<int8_t, 4> const &b,
    Array<int, 1> const &c
  ) {

#if 1//(defined(__HIP_DEVICE_COMPILE__) && (__HIP_DEVICE_COMPILE__ >= 610))

    unsigned const &A = reinterpret_cast<unsigned const &>(a);
    unsigned const &B = reinterpret_cast<unsigned const &>(b);

    asm volatile("v_dot4_i32_i8 %0, %1, %2, %3;"
                 : "=v"(d[0])
                 : "v"(A), "v"(B), "v"(c[0]));

#else

    d[0] = c[0];

    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < 4; ++k) {
      d[0] += a[k] * b[k];
    }

#endif
  }
};
#ifdef MIX_FP16_DOT2
/////////////////////////////////////////////////////////////////////////////////////////////
/// Matrix multiply-add operation
template <typename LayoutA, typename LayoutB, typename LayoutC>
struct Mma<
  gemm::GemmShape<1,1,2>,
  1,
  half_t,
  LayoutA,
  half_t,
  LayoutB,
  float,
  LayoutC,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<1, 1, 2>;
  using Operator = OpMultiplyAdd;
  using ElementC = float;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<float, 1> &d,
    Array<half_t, 2> const &a,
    Array<half_t, 2> const &b,
    Array<float, 1> const &c
  ) {
    __half2 const & A = reinterpret_cast<__half2 const &>(a);
    __half2 const & B = reinterpret_cast<__half2 const &>(b);
    // printf("inner dot2================\n");
    //d[0] = amd_mixed_dot(A, B, c[0], false);  // 饱和处理情况暂时不清楚
    asm volatile("v_dot2_f32_f16 %0,%1,%2,%3\n\t"
                 :"=v"(d[0])
                 :"v"(A),"v"(B),"v"(c[0]));
  }
};
#endif
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <typename LayoutC>
struct Mma<
  gemm::GemmShape<1, 1, 2>,
  1,
  int16_t,
  layout::RowMajor,
  int16_t,
  layout::ColumnMajor,
  int,
  LayoutC,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<1, 1, 2>;
  using Operator = OpMultiplyAdd;
  using ElementC = int;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<int, 1> &d,
    Array<int16_t, 2> const &a,
    Array<int16_t, 2> const &b,
    Array<int, 1> const &c
  ) {

#if (defined(__HIP_DEVICE_COMPILE__) && (__HIP_DEVICE_COMPILE__ >= 610))

    unsigned const &A = reinterpret_cast<unsigned const &>(a);
    unsigned const &B = reinterpret_cast<unsigned const &>(b);

    asm volatile("dp2a.s32.s32 %0, %1, %2, %3;"
                 : "=r"(d[0])
                 : "r"(A), "r"(B), "r"(c[0]));
#else
    d[0] = c[0];

    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < 2; ++k) {
      d[0] += a[k] * b[k];
    }
#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}
}
