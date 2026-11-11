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

#include <hip/hip_fp16.h>

#include "cutlass/arch/mma.h"

#include "cutlass/layout/matrix.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace arch {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <typename LayoutA, typename LayoutB, typename LayoutC>
struct Mma<
  gemm::GemmShape<2,1,1>,
  1,
  half_t,
  LayoutA,
  half_t,
  LayoutB,
  half_t,
  LayoutC,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<2, 1, 1>;
  using Operator = OpMultiplyAdd;
  using ElementC = half_t;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<half_t, 2> &d,
    Array<half_t, 2> const &a,
    Array<half_t, 1> const &b,
    Array<half_t, 2> const &c
  ) {

#if  1 //(defined(__HIP_DEVICE_COMPILE__) && (__HIP_DEVICE_COMPILE__ >= 600))
    // printf("111111111111111111111111111111111\n");
    __half2 const & A = reinterpret_cast<__half2 const &>(a);
    __half2 B = __half2half2(reinterpret_cast<__half const &>(b));
    __half2 const & C = reinterpret_cast<__half2 const &>(c);

    __half2 D = __hfma2(A, B, C);
      // asm volatile(
      //   "v_pk_fma_fp16 %0,%1,%2,%3\n\t"
      //   :"+v"(D),"v"(A),"v"(B),"v"(C)
      // );

    d = reinterpret_cast<Array<half_t, 2> &>(D);
 
#else
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      d[i] = a[i] * b[0] + c[i];
    }
#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <typename LayoutA, typename LayoutB>
struct Mma<
  gemm::GemmShape<1,2,1>,
  1,
  half_t,
  LayoutA,
  half_t,
  LayoutB,
  half_t,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<1, 2, 1>;
  using Operator = OpMultiplyAdd;
  using ElementC = half_t;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<half_t, 2> &d,
    Array<half_t, 1> const &a,
    Array<half_t, 2> const &b,
    Array<half_t, 2> const &c
  ) {

#if   1 //(defined(__HIP_DEVICE_COMPILE__) && (__HIP_DEVICE_COMPILE__ >= 600))
        // printf("22222222222222222222222222222222\n");
    __half2 const & A = __half2half2(reinterpret_cast<__half const &>(a));
    __half2 B = reinterpret_cast<__half2 const &>(b);
    __half2 const & C = reinterpret_cast<__half2 const &>(c);
  
    __half2 D = __hfma2(A, B, C);
    // printf("A:%f %f B:%f %f C:%f %f D:%f %f\n",__half2float(A.x),__half2float(A.y),__half2float(B.x),__half2float(B.y),__half2float(C.x),__half2float(C.y),__half2float(D.x),__half2float(D.y));
    d = reinterpret_cast<Array<half_t, 2> &>(D);

#else
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      d[i] = a[0] * b[i] + c[i];
    }
#endif
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////
/// Matrix multiply-add operation
#ifdef MIX_FP16XFP16_SM60

template <typename LayoutA, typename LayoutB>
struct Mma<
  gemm::GemmShape<1,2,1>,
  1,
  half_t,
  LayoutA,
  half_t,
  LayoutB,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<1, 2, 1>;
  using Operator = OpMultiplyAdd;
  using ElementC = float;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<float, 2> &d,
    Array<half_t, 1> const &a,
    Array<half_t, 2> const &b,
    Array<float, 2> const &c
  ) {

#if  1//(defined(__HIP_DEVICE_COMPILE__) && (__HIP_DEVICE_COMPILE__ >= 600))
    __half2 A =  __half2half2(reinterpret_cast<__half const &>(a));
    __half2 B = reinterpret_cast<__half2 const &>(b);
    float C1 = reinterpret_cast<float const &>(c[0]);
    float C2 = reinterpret_cast<float const &>(c[1]);

    asm volatile(
        "v_fma_mix_f32 %0,%2,%3,%4 op_sel:[0,0] op_sel_hi:[1,1,0]\n\t"
        "v_fma_mix_f32 %1,%2,%3,%5 op_sel:[1,1] op_sel_hi:[1,1,0]\n\t"
        :"=v"(d[0]),"=v"(d[1]),"+v"(A),"+v"(B),"+v"(C1),"+v"(C2)
      );
#else
    // printf("22222222222222222222222222222222\n");
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      d[i] = a[0] * b[i] + c[i];
    }
#endif
  }
};
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <>
struct Mma <
  gemm::GemmShape<2, 2, 1>,
  1,
  half_t,
  layout::ColumnMajor,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<2, 2, 1>;
  using Operator = OpMultiplyAdd;
  using ElementC = half_t;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<half_t, 4> &d,
    Array<half_t, 2> const &a,
    Array<half_t, 2> const &b,
    Array<half_t, 4> const &c
  ) {

#if   1 //(defined(__HIP_DEVICE_COMPILE__) && (__HIP_DEVICE_COMPILE__ >= 600))
      //  printf("333333333333333333333333333333\n");
    __half2 const & A = reinterpret_cast<__half2 const &>(a);
    __half2 Blo = __low2half2(reinterpret_cast<__half2 const &>(b));
    __half2 Bhi = __high2half2(reinterpret_cast<__half2 const &>(b));

    __half2 const *C = reinterpret_cast<__half2 const *>(&c);

    __half2 Dlo = __hfma2(A, Blo, C[0]);
    __half2 Dhi = __hfma2(A, Bhi, C[1]);

    Array<half_t, 2> * D = reinterpret_cast<Array<half_t, 2> *>(&d);

    D[0] = reinterpret_cast<Array<half_t, 2> const &>(Dlo);
    D[1] = reinterpret_cast<Array<half_t, 2> const &>(Dhi);

#else
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < 2; ++j) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < 2; ++i) {
        d[i + 2 * j] = a[i] * b[j] + c[i + 2 * j];
      }
    }
#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <>
struct Mma<
  gemm::GemmShape<2, 2, 1>,
  1,
  half_t,
  layout::ColumnMajor,
  half_t,
  layout::RowMajor,
  half_t,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<2, 2, 1>;
  using Operator = OpMultiplyAdd;
  using ElementC = half_t;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<half_t, 4> &d,
    Array<half_t, 2> const &a,
    Array<half_t, 2> const &b,
    Array<half_t, 4> const &c
  ) {

#if   1 //(defined(__HIP_DEVICE_COMPILE__) && (__HIP_DEVICE_COMPILE__ >= 600))

    __half2 Alo = __low2half2(reinterpret_cast<__half2 const &>(a));
    __half2 Ahi = __high2half2(reinterpret_cast<__half2 const &>(a));
    __half2 const & B = reinterpret_cast<__half2 const &>(b);

    __half2 const *C = reinterpret_cast<__half2 const *>(&c);

    __half2 Dlo = __hfma2(Alo, B, C[0]);
    __half2 Dhi = __hfma2(Ahi, B, C[0]);

    Array<half_t, 2> * D = reinterpret_cast<Array<half_t, 2> *>(&d);

    D[0] = reinterpret_cast<Array<half_t, 2> &>(Dlo);
    D[1] = reinterpret_cast<Array<half_t, 2> &>(Dhi);
#else
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < 2; ++j) {
        d[i * 2 + j] = a[i] * b[j] + c[i * 2 + j];
      }
    }
#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
}
}

