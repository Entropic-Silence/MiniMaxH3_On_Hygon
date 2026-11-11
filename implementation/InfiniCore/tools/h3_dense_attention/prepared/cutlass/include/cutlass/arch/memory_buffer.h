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
    \brief Architecture-specific operators on memory added for AMDGPU
*/

#pragma once

#include "cutlass/array.h"
#include "cute/arch/util.hpp"

namespace cutlass {
namespace arch {

typedef uint32_t uint32x4_t __attribute__((ext_vector_type(4)));
typedef int32_t int32x4_t __attribute__((ext_vector_type(4)));
typedef int32_t int32x2_t __attribute__((ext_vector_type(2)));

struct alignas(16) BufferPointer {
    unsigned long long ptr : 64;
    uint num_records : 32;
    uint sel : 12;
    uint num_format : 3;
    uint data_format : 4;
    bool vm_enable : 1;
    bool vm_mode : 1;
    uint idx_stride : 2;
    bool add_tid_enable : 1;
    uint _ : 3;
    bool non_volatile : 1;
    uint __ : 2;
    uint type : 2;
    public:
    CUTLASS_HOST_DEVICE BufferPointer(unsigned long long _ptr, uint size = 0xffffffff) :
        ptr((unsigned long long)_ptr), 
        num_records(size * 4), 
        sel(0), 
        num_format(0), 
        data_format(4), 
        vm_enable(false), 
        vm_mode(false), 
        idx_stride(0), 
        add_tid_enable(false), 
        _(0), 
        non_volatile(false), 
        __(0), 
        type(0)
    {}
};

struct alignas(16) BufferAccessor {
  union {
    BufferPointer buffer_pointer;
    uint32x4_t buffer_pointer_data;
  };

  mutable uint vofft = 0;
  mutable uint sofft = 0;

  CUTLASS_HOST_DEVICE void add_to_vofft(uint32_t offt){
    vofft += offt;
  }

  CUTLASS_HOST_DEVICE void add_to_sofft(uint32_t offt){
    sofft += offt;
  }

  CUTLASS_HOST_DEVICE BufferAccessor(void* ptr, uint size = 0xffffffff) : buffer_pointer((unsigned long long)ptr, size) {}
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType, int Bytes>
struct buffer_load_op {}; // 这个

template <typename AccessType>
CUTLASS_DEVICE void buffer_load(AccessType &D, BufferAccessor &ptr, uint32_t voffset, uint32_t soffset) {
#ifdef DCU_ASM
  buffer_load_op<AccessType, int(sizeof(AccessType))>(D, ptr, voffset, soffset);
#else
  cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
    D, ptr.buffer_pointer.ptr + voffset + soffset, true);
#endif
}

#ifdef DCU_ASM

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType>
struct buffer_load_op<AccessType, 16> {
  CUTLASS_DEVICE
  buffer_load_op(AccessType &D, BufferAccessor &ptr, uint32_t voffset, uint32_t soffset) {
    int32x4_t &data = reinterpret_cast<int32x4_t &>(D);
    data = intrinsic_buffer_load(ptr.buffer_pointer_data, ptr.vofft + voffset, ptr.sofft + soffset, 0);
  }

  __device__ int32x4_t static intrinsic_buffer_load(uint32x4_t srsrc, uint voffset, uint soffset, uint glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4i32");

};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType>
struct buffer_load_op<AccessType, 8> {
  CUTLASS_DEVICE
  buffer_load_op(AccessType &D, BufferAccessor &ptr, uint32_t voffset, uint32_t soffset) {
    int32x2_t &data = reinterpret_cast<int32x2_t &>(D);
    data = intrinsic_buffer_load(ptr.buffer_pointer_data, ptr.vofft + voffset, ptr.sofft + soffset, 0);
  }

  __device__ int32x2_t static intrinsic_buffer_load(uint32x4_t srsrc, uint voffset, uint soffset, uint glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2i32");

};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType>
struct buffer_load_op<AccessType, 4> {
  CUTLASS_DEVICE
  buffer_load_op(AccessType &D, BufferAccessor &ptr, uint32_t voffset, uint32_t soffset) {
    
    int &data = reinterpret_cast<int &>(D);
    data = intrinsic_buffer_load(ptr.buffer_pointer_data, ptr.vofft + voffset, ptr.sofft + soffset, 0);

    // D = reinterpret_cast<AccessType>(intrinsic_buffer_load<typename AccessType::Element>(ptr.buffer_pointer_data, ptr.vofft + voffset, ptr.sofft + soffset, 0));
    // D = intrinsic_buffer_load<typename AccessType::Element>(ptr.buffer_pointer_data, ptr.vofft + voffset, ptr.sofft + soffset, 0);
    
    // float tmp = intrinsic_buffer_load(ptr.buffer_pointer_data, ptr.vofft + voffset, ptr.sofft + soffset, 0);
    // D[0] = tmp;
  }

  __device__ int static intrinsic_buffer_load(uint32x4_t srsrc, uint voffset, uint soffset, uint glc_slc) __asm("llvm.amdgcn.raw.buffer.load.i32");
  // __device__ float static intrinsic_buffer_load(uint32x4_t srsrc, uint voffset, uint soffset, uint glc_slc) __asm("llvm.amdgcn.raw.buffer.load.f32");

  // template <typename T>
  // __device__ AccessType intrinsic_buffer_load(uint32x4_t srsrc, uint voffset, uint soffset, uint glc_slc);

  // template <>
  // __device__ AccessType intrinsic_buffer_load<float>(uint32x4_t srsrc, uint voffset, uint soffset, uint glc_slc) __asm("llvm.amdgcn.raw.buffer.load.f32");

  // template <>
  // CUTLASS_DEVICE AccessType intrinsic_buffer_load<half_t>(uint32x4_t srsrc, uint voffset, uint soffset, uint glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2f16");

  // template <>
  // CUTLASS_DEVICE AccessType intrinsic_buffer_load<int8_t>(uint32x4_t srsrc, uint voffset, uint soffset, uint glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4i8");
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType>
struct buffer_load_op<AccessType, 2> {
  CUTLASS_DEVICE
  buffer_load_op(AccessType &D, BufferAccessor &ptr, uint32_t voffset, uint32_t soffset) {
    int16_t &data = reinterpret_cast<int16_t &>(D);
    data = intrinsic_buffer_load(ptr.buffer_pointer_data, ptr.vofft + voffset, ptr.sofft + soffset, 0);
  }

  __device__ int16_t static intrinsic_buffer_load(uint32x4_t srsrc, uint voffset, uint soffset, uint glc_slc) __asm("llvm.amdgcn.raw.buffer.load.i16");
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType>
struct buffer_load_op<AccessType, 1> {
  CUTLASS_DEVICE
  buffer_load_op(AccessType &D, BufferAccessor &ptr, uint32_t voffset, uint32_t soffset) {
    int8_t &data = reinterpret_cast<int8_t &>(D);
    data = intrinsic_buffer_load(ptr.buffer_pointer_data, ptr.vofft + voffset, ptr.sofft + soffset, 0);
  }

  __device__ int8_t static intrinsic_buffer_load(uint32x4_t srsrc, uint voffset, uint soffset, uint glc_slc) __asm("llvm.amdgcn.raw.buffer.load.i8");
};
#endif

} // namespace arch
} // namespace cutlass
