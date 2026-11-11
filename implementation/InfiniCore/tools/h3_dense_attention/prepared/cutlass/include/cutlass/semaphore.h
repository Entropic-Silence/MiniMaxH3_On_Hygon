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
    \brief Implementation of a CTA-wide semaphore for inter-CTA synchronization.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/array.h"

#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/gemm/gemm.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// CTA-wide semaphore for inter-CTA synchronization.
class Semaphore { 
public:

  int *lock;
  bool wait_thread;
  int state;

public:

  /// Implements a semaphore to wait for a flag to reach a given value
  CUTLASS_HOST_DEVICE
  Semaphore(int *lock_, int thread_id): 
    lock(lock_), 
    wait_thread(thread_id < 0 || thread_id == 0),
    state(-1) {

  }

  /// Permit fetching the synchronization mechanism early
  CUTLASS_DEVICE
  void fetch() {
    if (wait_thread) {
#ifdef DCU_ASM
      state = *lock;
      // asm volatile("v_mov_b32 %0, %1\n\t":
      // "+v"(state),"+v"(*lock));
#else
      #if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__ >= 700
      asm volatile ("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));  
      #else
      asm volatile ("ld.global.cg.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));  
      #endif
#endif
    }
  }

  /// Gets the internal state
  CUTLASS_DEVICE
  int get_state() const {
    return state;
  }

  /// Waits until the semaphore is equal to the given value
  CUTLASS_DEVICE
  void wait(int status = 0) {
    /*
    *一直等待线程0的state == status 所有线程才会继续向下执行,这里status就是类似于blockIdx.z的值
    * 非0线程的state 和 status 一直不相等，非零线程的state默认值是-1 status的值为0，1，2...
    * 因此对于非0线程来说__syncthreads_and的predict为1，如果线程0的state和status相等，那么返回值为0 如果不等则一直处于fetch状态
    * 刚开始运行时候state默认值为0 如果status为0 wait解锁 release进行更新lock值,约定了每一层blockIdx.z的执行顺序
    * __syncthreads_and函数是快内线程都执行到同一个位置
    * 
    */
    while( __syncthreads_and(state != status) ) {
      fetch();
      __threadfence();
    }

    __syncthreads();
  }

  /// Updates the lock with the given result
  CUTLASS_DEVICE
  void release(int status = 0) {
    __syncthreads();

    if (wait_thread) {
#ifdef DCU_ASM
      *lock = status;
#else
      #if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__ >= 700
      asm volatile ("st.global.release.gpu.b32 [%0], %1;\n" : : "l"(lock), "r"(status));
      #else
      asm volatile ("st.global.cg.b32 [%0], %1;\n" : : "l"(lock), "r"(status));
      #endif
#endif
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
