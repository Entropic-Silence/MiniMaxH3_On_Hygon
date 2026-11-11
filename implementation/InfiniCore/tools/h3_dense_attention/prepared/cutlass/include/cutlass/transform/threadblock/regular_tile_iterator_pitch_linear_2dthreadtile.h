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
    \brief Templates implementing loading of tiles from pitch-linear rank=2 tensors. 

    This iterator uses masks to guard out-of-bounds accesses and visits the last "residue" tile
    first, with the objective of minimizing predicate mask updates during steady-state operation.

    A precomputed "Params" object minimizes the amount of state that must be stored in registers,
    and integer addition is used to advance the pointer through memory.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"

#include "regular_tile_iterator.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace transform {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////
template <
  typename Shape,
  typename Element,
  typename Layout,
  int AdvanceRank,
  typename ThreadMap,
  int Alignment = sizeof_bits<Element>::value * ThreadMap::kElementsPerAccess / 8,
  bool Transpose = false
>
class RegularTileIterator2dThreadTile;

//该实例针对half数据类型 重映射 保持与2x2的threadMapThreadAccessShape对应的共享内存一致
/// Regular tile iterator specialized for pitch-linear + 2d thread-tiled threadmapping
//Transpose为false不需要进行转换
template <
  typename Shape_,
  int AdvanceRank,
  typename ThreadMap_,
  int Alignment
>
class RegularTileIterator2dThreadTile<Shape_, half_t, layout::PitchLinear, AdvanceRank, ThreadMap_, Alignment, false> {
public:

  using Shape = Shape_;
  using Element = half_t;
  using Layout = layout::PitchLinear;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using StrideIndex = typename Layout::Stride::Index;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Fragment = Array<Element, ThreadMap::Iterations::kCount * ThreadMap::ThreadAccessShape::kCount>;

  static_assert(kAdvanceRank == 0 || kAdvanceRank == 1, 
    "Advance rank may only be along the contiguous or strided dimensions.");

private:

  //
  // Types
  //
  
  using AccessType = AlignedArray<Element, ThreadMap::ThreadAccessShape::kCount, kAlignment>;

  //
  // Data members
  //

  /// Pointer to memory
  uint8_t *pointer_;

  /// Stride quantity
  StrideIndex stride_;

  /// Amount to increment pointer along strided dimension
  LongIndex increment_strided_;

  /// Amount to advance pointer between tiles
  LongIndex increment_advance_;

public:

  CUTLASS_DEVICE
  RegularTileIterator2dThreadTile(): pointer_(nullptr), increment_strided_(0), increment_advance_(0) { }

  CUTLASS_DEVICE
  RegularTileIterator2dThreadTile(
    TensorRef const &ref, 
    int thread_idx,
    int interleave
  ){ 
    
    TensorCoord t = ThreadMap::initial_offset(thread_idx);
    long int offset = t[0] * interleave + t[1] * ref.stride()[0]/interleave;
    pointer_ = reinterpret_cast<uint8_t *>(ref.data() + offset);

    stride_ = ref.stride()[0] / interleave;
    increment_strided_ = (ref.stride()[0] * sizeof_bits<Element>::value / 8) * ThreadMap::Delta::kStrided / interleave;

    increment_advance_ = 
      (kAdvanceRank == 0 ? 
        Shape::kContiguous * sizeof_bits<Element>::value / 8 : 
        Shape::kStrided * (ref.stride()[0] * sizeof_bits<Element>::value / 8) / interleave);
  }

  /// Loads a fragment
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {

    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);
    uint8_t const *byte_pointer = pointer_ + pointer_offset * sizeof_bits<Element>::value / 8;

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {

      AccessType const *access_ptr = reinterpret_cast<AccessType const *>(byte_pointer);

      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {

          int idx = c + s * ThreadMap::Iterations::kContiguous;
          frag_ptr[idx] = access_ptr[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided];
        
        }

      if (s + 1 < ThreadMap::Iterations::kStrided) {
        byte_pointer += increment_strided_;
      }
    }
  }

  /// Loads a fragment
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag, TensorCoord const & tile_offset) {
    load_with_pointer_offset(
      frag, 
      tile_offset.contiguous() * Shape::kContiguous / ThreadMap::kElementsPerAccess + 
        tile_offset.strided() * Shape::kStrided * stride_
    );
  }

  /// Loads a fragment
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Stores a fragment
  CUTLASS_HOST_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {

    AccessType const *frag_ptr = reinterpret_cast<AccessType const*>(&frag);
    uint8_t *byte_pointer = pointer_ + pointer_offset * sizeof_bits<Element>::value / 8;
    //ThreadAccessMap 

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      AccessType *access_ptr = reinterpret_cast<AccessType *>(byte_pointer);
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {

          int idx = c + s * ThreadMap::Iterations::kContiguous;
          //access_ptr[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided] = frag_ptr[idx];
          //dwordx4 8x1
          //不变量是否需要外提
          if(ThreadMap::ThreadAccessShape::kContiguous == 8 && ThreadMap::ThreadAccessShape::kStrided == 1)
          {
              // 理论上偏移地址应该是 access_ptr + 4 * (c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided)
              // 但是 access_ptr 的type改变
              AccessType *access_ptr_v0 = reinterpret_cast<AccessType *>(byte_pointer +  4 * (c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided));
              access_ptr_v0[0][0] = frag[8 * idx + 0];
              access_ptr_v0[0][1] = frag[8 * idx + 1];
              AccessType *access_ptr_v1 = reinterpret_cast<AccessType *>(byte_pointer + stride_ * 4 + 4 * (c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided));
              access_ptr_v1[0][0] = frag[8 * idx + 2];
              access_ptr_v1[0][1] = frag[8 * idx + 3];
              AccessType *access_ptr_v2 = reinterpret_cast<AccessType *>(byte_pointer + stride_ * 8 + 4 * (c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided));
              access_ptr_v2[0][0] = frag[8 * idx + 4];
              access_ptr_v2[0][1] = frag[8 * idx + 5];
              AccessType *access_ptr_v3 = reinterpret_cast<AccessType *>(byte_pointer + stride_ * 12 + 4 * (c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided));
              access_ptr_v3[0][0] = frag[8 * idx + 6];
              access_ptr_v3[0][1] = frag[8 * idx + 7];    
          }
          //dwordx2 4x1
          else if(ThreadMap::ThreadAccessShape::kContiguous == 4 && ThreadMap::ThreadAccessShape::kStrided == 1)
          {
                AccessType *access_ptr_v0 = reinterpret_cast<AccessType *>(byte_pointer +  4 * (c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided));
                access_ptr_v0[0][0] = frag[4 * idx + 0];
                access_ptr_v0[0][1] = frag[4 * idx + 1];
                AccessType *access_ptr_v1 = reinterpret_cast<AccessType *>(byte_pointer + stride_ * 4 +  4 * (c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided));
                access_ptr_v1[0][0] = frag[4 * idx + 2];
                access_ptr_v1[0][1] = frag[4 * idx + 3];
          }
          else
          {
              access_ptr[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided] = frag_ptr[idx];
          }
          
      }

      if (s + 1 < ThreadMap::Iterations::kStrided) {
        
        byte_pointer += increment_strided_;
      }
    }
  }

  /// Stores a fragment
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag, TensorCoord const & tile_offset) {
    store_with_pointer_offset(
      frag,
      tile_offset.contiguous() * Shape::kContiguous + tile_offset.strided() * Shape::kStrided * stride_
    );
  }

  /// Stores a fragment
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }

  /// Advances the pointer
  CUTLASS_HOST_DEVICE
  RegularTileIterator2dThreadTile &operator++() {
    pointer_ += increment_advance_;
    return *this;
  }

  /// Advances the pointer
  CUTLASS_HOST_DEVICE
  RegularTileIterator2dThreadTile &operator--() {
    pointer_ -= increment_advance_;
    return *this;
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    pointer_ += pointer_offset;
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    int offset = sizeof_bits<Element>::value *
        (coord.contiguous() * Shape::kContiguous + coord.strided() * Shape::kStrided * stride_) / 8;
    add_pointer_offset(offset);
  }

};

//该实例针对half数据类型 重映射 保持与2x2的threadMapThreadAccessShape对应的共享内存一致
/// Regular tile iterator specialized for pitch-linear + 2d thread-tiled threadmapping
template <
  typename Shape_,
    int AdvanceRank,
  typename ThreadMap_,
  int Alignment
>
class RegularTileIterator2dThreadTile<Shape_, half_t, layout::PitchLinear, AdvanceRank, ThreadMap_, Alignment, true> {
public:

  using Shape = Shape_;
  using Element = half_t;
  using Layout = layout::PitchLinear;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using StrideIndex = typename Layout::Stride::Index;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Fragment = Array<Element, ThreadMap::Iterations::kCount * ThreadMap::ThreadAccessShape::kCount>;

  static_assert(kAdvanceRank == 0 || kAdvanceRank == 1, 
    "Advance rank may only be along the contiguous or strided dimensions.");

private:

  //
  // Types
  //
  
  using AccessType = AlignedArray<Element, ThreadMap::ThreadAccessShape::kCount, kAlignment>;

  //
  // Data members
  //

  /// Pointer to memory
  uint8_t *pointer_;

  /// Stride quantity
  StrideIndex stride_;

  /// Amount to increment pointer along strided dimension
  LongIndex increment_strided_;

  /// Amount to advance pointer between tiles
  LongIndex increment_advance_;

public:

  CUTLASS_DEVICE
  RegularTileIterator2dThreadTile(): pointer_(nullptr), increment_strided_(0), increment_advance_(0) { }

  CUTLASS_DEVICE
  RegularTileIterator2dThreadTile(
    TensorRef const &ref, 
    int thread_idx,
    int interleave
  ){ 
    
    TensorCoord t = ThreadMap::initial_offset(thread_idx);
    long int offset = t[0] * interleave + t[1] * ref.stride()[0]/interleave;
    pointer_ = reinterpret_cast<uint8_t *>(ref.data() + offset);

    stride_ = ref.stride()[0] / interleave;
    increment_strided_ = (ref.stride()[0] * sizeof_bits<Element>::value / 8) * ThreadMap::Delta::kStrided / interleave;

    increment_advance_ = 
      (kAdvanceRank == 0 ? 
        Shape::kContiguous * sizeof_bits<Element>::value / 8 : 
        Shape::kStrided * (ref.stride()[0] * sizeof_bits<Element>::value / 8) / interleave);
  }

  /// Loads a fragment
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {

    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);
    uint8_t const *byte_pointer = pointer_ + pointer_offset * sizeof_bits<Element>::value / 8;

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {

      AccessType const *access_ptr = reinterpret_cast<AccessType const *>(byte_pointer);

      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {

          int idx = c + s * ThreadMap::Iterations::kContiguous;
           frag_ptr[idx] = access_ptr[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided];
        }

      if (s + 1 < ThreadMap::Iterations::kStrided) {
        byte_pointer += increment_strided_;
      }
    }
  }

  /// Loads a fragment
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag, TensorCoord const & tile_offset) {
    load_with_pointer_offset(
      frag, 
      tile_offset.contiguous() * Shape::kContiguous / ThreadMap::kElementsPerAccess + 
        tile_offset.strided() * Shape::kStrided * stride_
    );
  }

  /// Loads a fragment
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Stores a fragment
  CUTLASS_HOST_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {

    AccessType const *frag_ptr = reinterpret_cast<AccessType const*>(&frag);
    uint8_t *byte_pointer = pointer_ + pointer_offset * sizeof_bits<Element>::value / 8;

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {

      AccessType *access_ptr = reinterpret_cast<AccessType *>(byte_pointer);

      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
          int idx = c + s * ThreadMap::Iterations::kContiguous;
          //dwordx4 8x2
          if(ThreadMap::ThreadAccessShape::kContiguous == 8 && ThreadMap::ThreadAccessShape::kStrided == 2)
          {
              access_ptr[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][0] = frag[16 * idx + 0];
              access_ptr[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][1] = frag[16 * idx + 8];
              access_ptr[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][2] = frag[16 * idx + 1];
              access_ptr[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][3] = frag[16 * idx + 9];
              access_ptr[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][4] = frag[16 * idx + 2];
              access_ptr[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][5] = frag[16 * idx + 10];
              access_ptr[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][6] = frag[16 * idx + 3];
              access_ptr[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][7] = frag[16 * idx + 11];
              access_ptr[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][8] = frag[16 * idx + 4];
              access_ptr[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][9] = frag[16 * idx + 12];
              access_ptr[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][10] = frag[16 * idx + 5];
              access_ptr[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][11] = frag[16 * idx + 13];
              access_ptr[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][12] = frag[16 * idx + 6];
              access_ptr[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][13] = frag[16 * idx + 14];
              access_ptr[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][14] = frag[16 * idx + 7];
              access_ptr[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][15] = frag[16 * idx + 15];
            
          }
          //dwordx2
          else if(ThreadMap::ThreadAccessShape::kContiguous == 4 && ThreadMap::ThreadAccessShape::kStrided == 2)
          {
              access_ptr[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][0] = frag[8 * idx + 0];
              access_ptr[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][1] = frag[8 * idx + 4];
              access_ptr[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][2] = frag[8 * idx + 1];
              access_ptr[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][3] = frag[8 * idx + 5];
              access_ptr[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][4] = frag[8 * idx + 2];
              access_ptr[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][5] = frag[8 * idx + 6];
              access_ptr[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][6] = frag[8 * idx + 3];
              access_ptr[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][7] = frag[8 * idx + 7];
      
          }
          else if(ThreadMap::ThreadAccessShape::kContiguous == 4 && ThreadMap::ThreadAccessShape::kStrided == 4)
          {
              access_ptr[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][0] = frag[16 * idx + 0];
              access_ptr[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][1] = frag[16 * idx + 4];
              access_ptr[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][2] = frag[16 * idx + 1];
              access_ptr[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][3] = frag[16 * idx + 5];
              AccessType *access_ptr_v1 = reinterpret_cast<AccessType *>(byte_pointer + 8);
              access_ptr_v1[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][0] = frag[16 * idx + 2];
              access_ptr_v1[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][1] = frag[16 * idx + 6];
              access_ptr_v1[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][2] = frag[16 * idx + 3];
              access_ptr_v1[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][3] = frag[16 * idx + 7];
              AccessType *access_ptr_v2 = reinterpret_cast<AccessType *>(byte_pointer + stride_ * 4);
              access_ptr_v2[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][0] = frag[16 * idx + 8];
              access_ptr_v2[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][1] = frag[16 * idx + 12];
              access_ptr_v2[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][2] = frag[16 * idx + 9];
              access_ptr_v2[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][3] = frag[16 * idx + 13];
              AccessType *access_ptr_v3 = reinterpret_cast<AccessType *>(byte_pointer + stride_ * 4 + 8);
              access_ptr_v3[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][0] = frag[16 * idx + 10];
              access_ptr_v3[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][1] = frag[16 * idx + 14];
              access_ptr_v3[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][2] = frag[16 * idx + 11];
              access_ptr_v3[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided][3] = frag[16 * idx + 15];
          }
          else
          {
              access_ptr[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided] = frag_ptr[idx];
          }
      }

      if (s + 1 < ThreadMap::Iterations::kStrided) {
        
        byte_pointer += increment_strided_;
      }
    }
  }

  /// Stores a fragment
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag, TensorCoord const & tile_offset) {
    store_with_pointer_offset(
      frag,
      tile_offset.contiguous() * Shape::kContiguous + tile_offset.strided() * Shape::kStrided * stride_
    );
  }

  /// Stores a fragment
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }

  /// Advances the pointer
  CUTLASS_HOST_DEVICE
  RegularTileIterator2dThreadTile &operator++() {
    pointer_ += increment_advance_;
    return *this;
  }

  /// Advances the pointer
  CUTLASS_HOST_DEVICE
  RegularTileIterator2dThreadTile &operator--() {
    pointer_ -= increment_advance_;
    return *this;
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    pointer_ += pointer_offset;
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    int offset = sizeof_bits<Element>::value *
        (coord.contiguous() * Shape::kContiguous + coord.strided() * Shape::kStrided * stride_) / 8;
    add_pointer_offset(offset);
  }

};


/// Regular tile iterator specialized for pitch-linear + 2d thread-tiled threadmapping
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  int Alignment,
  bool Transpose
>
class RegularTileIterator2dThreadTile<Shape_, Element_, layout::PitchLinear, AdvanceRank, ThreadMap_, Alignment, Transpose> {
public:

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::PitchLinear;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using StrideIndex = typename Layout::Stride::Index;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Fragment = Array<Element, ThreadMap::Iterations::kCount * ThreadMap::ThreadAccessShape::kCount>;

  static_assert(kAdvanceRank == 0 || kAdvanceRank == 1, 
    "Advance rank may only be along the contiguous or strided dimensions.");

private:

  //
  // Types
  //
  
  using AccessType = AlignedArray<Element, ThreadMap::ThreadAccessShape::kCount, kAlignment>;

  //
  // Data members
  //

  /// Pointer to memory
  uint8_t *pointer_;

  /// Stride quantity
  StrideIndex stride_;

  /// Amount to increment pointer along strided dimension
  LongIndex increment_strided_;

  /// Amount to advance pointer between tiles
  LongIndex increment_advance_;

public:

  CUTLASS_DEVICE
  RegularTileIterator2dThreadTile(): pointer_(nullptr), increment_strided_(0), increment_advance_(0) { }

  CUTLASS_DEVICE
  RegularTileIterator2dThreadTile(
    TensorRef const &ref, 
    int thread_idx,
    int interleave
  ){ 
    
    TensorCoord t = ThreadMap::initial_offset(thread_idx);
    long int offset = t[0] * interleave + t[1] * ref.stride()[0]/interleave;
    pointer_ = reinterpret_cast<uint8_t *>(ref.data() + offset);

    stride_ = ref.stride()[0] / interleave;
    increment_strided_ = (ref.stride()[0] * sizeof_bits<Element>::value / 8) * ThreadMap::Delta::kStrided / interleave;

    increment_advance_ = 
      (kAdvanceRank == 0 ? 
        Shape::kContiguous * sizeof_bits<Element>::value / 8 : 
        Shape::kStrided * (ref.stride()[0] * sizeof_bits<Element>::value / 8) / interleave);

  }

  /// Loads a fragment
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {

    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);
    uint8_t const *byte_pointer = pointer_ + pointer_offset * sizeof_bits<Element>::value / 8;

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {

      AccessType const *access_ptr = reinterpret_cast<AccessType const *>(byte_pointer);

      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {

int idx = c + s * ThreadMap::Iterations::kContiguous;
          frag_ptr[idx] = access_ptr[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided];
        
        }

      if (s + 1 < ThreadMap::Iterations::kStrided) {
        byte_pointer += increment_strided_;
      }
    }
  }

  /// Loads a fragment
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag, TensorCoord const & tile_offset) {
    load_with_pointer_offset(
      frag, 
      tile_offset.contiguous() * Shape::kContiguous / ThreadMap::kElementsPerAccess + 
        tile_offset.strided() * Shape::kStrided * stride_
    );
  }

  /// Loads a fragment
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Stores a fragment
  CUTLASS_HOST_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {

    AccessType const *frag_ptr = reinterpret_cast<AccessType const*>(&frag);
    uint8_t *byte_pointer = pointer_ + pointer_offset * sizeof_bits<Element>::value / 8;
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {

      AccessType *access_ptr = reinterpret_cast<AccessType *>(byte_pointer);

      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
          int idx = c + s * ThreadMap::Iterations::kContiguous;
          access_ptr[c * ThreadMap::Delta::kContiguous / ThreadMap::ThreadAccessShape::kStrided] = frag_ptr[idx];
      }

      if (s + 1 < ThreadMap::Iterations::kStrided) {
        byte_pointer += increment_strided_;
      }
    }
  }

  /// Stores a fragment
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag, TensorCoord const & tile_offset) {
    store_with_pointer_offset(
      frag,
      tile_offset.contiguous() * Shape::kContiguous + tile_offset.strided() * Shape::kStrided * stride_
    );
  }

  /// Stores a fragment
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }

  /// Advances the pointer
  CUTLASS_HOST_DEVICE
  RegularTileIterator2dThreadTile &operator++() {
    pointer_ += increment_advance_;
    return *this;
  }

  /// Advances the pointer
  CUTLASS_HOST_DEVICE
  RegularTileIterator2dThreadTile &operator--() {
    pointer_ -= increment_advance_;
    return *this;
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    pointer_ += pointer_offset;
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    int offset = sizeof_bits<Element>::value *
        (coord.contiguous() * Shape::kContiguous + coord.strided() * Shape::kStrided * stride_) / 8;
    add_pointer_offset(offset);
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Regular tile iterator specialized for interleaved layout + 2d thread-tiled threadmapping
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  int Alignment
>
class RegularTileIterator2dThreadTile<Shape_, Element_, layout::RowMajorInterleaved<4>, AdvanceRank, ThreadMap_, Alignment> {
public:

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::RowMajorInterleaved<4>;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Fragment = Array<Element, ThreadMap::Iterations::kCount * ThreadMap::ThreadAccessShape::kCount>;

  using Underlying = RegularTileIterator2dThreadTile<
    layout::PitchLinearShape<Shape::kColumn, Shape::kRow>,
    Element,
    layout::PitchLinear,
    (kAdvanceRank == 0 ? 1 : 0),
    ThreadMap,
    kAlignment
  >;

  static_assert(kAdvanceRank == 0 || kAdvanceRank == 1, 
    "Advance rank may only be along the row or column dimensions.");

private:

  Underlying iterator_;

public:

  CUTLASS_DEVICE
  RegularTileIterator2dThreadTile() { }

  CUTLASS_DEVICE
  RegularTileIterator2dThreadTile(
    TensorRef const &ref, 
    int thread_idx
  ):
    iterator_({ref.data(), ref.stride()}, thread_idx, 4) {

  }

  /// Loads a fragment
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag, TensorCoord const & tile_offset) {
    iterator_.load_with_pointer_offset(frag, {tile_offset.column(), tile_offset.row()});
  }

  /// Loads a fragment
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) {
    iterator_.load_with_pointer_offset(frag, 0);
  }

  /// Stores a fragment
  CUTLASS_HOST_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Stores a fragment
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag, TensorCoord const & tile_offset) {
    iterator_.store_with_pointer_offset(frag, {tile_offset.column(), tile_offset.row()});
  }

  /// Stores a fragment
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) {
    iterator_.store_with_pointer_offset(frag, 0);
  }

  /// Advances the pointer
  CUTLASS_HOST_DEVICE
  RegularTileIterator2dThreadTile &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances the pointer
  CUTLASS_HOST_DEVICE
  RegularTileIterator2dThreadTile &operator--() {
    --iterator_;
    return *this;
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    iterator_.add_tile_offset({coord.column(), coord.row()});
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Regular tile iterator specialized for interleaved layout + 2d thread-tiled threadmapping
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  int Alignment
>
class RegularTileIterator2dThreadTile<Shape_, Element_, layout::ColumnMajorInterleaved<4>, AdvanceRank, ThreadMap_, Alignment> {
public:

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::ColumnMajorInterleaved<4>;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Fragment = Array<Element, ThreadMap::Iterations::kCount * ThreadMap::ThreadAccessShape::kCount>;
  using PitchLinearThreadMap = PitchLinearStripminedThreadMap< layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, 
                                  ThreadMap::kThreads, ThreadMap::ThreadAccessShape::kCount >;
                        

  using Underlying = RegularTileIterator2dThreadTile<
    layout::PitchLinearShape<Shape::kRow, Shape::kColumn>,
    Element,
    layout::PitchLinear,
    (kAdvanceRank == 0 ? 0 : 1),
    ThreadMap,
    kAlignment
  >;

  static_assert(kAdvanceRank == 0 || kAdvanceRank == 1, 
    "Advance rank may only be along the row or column dimensions.");

private:

  Underlying iterator_;

public:

  CUTLASS_DEVICE
  RegularTileIterator2dThreadTile() { }

  CUTLASS_DEVICE
  RegularTileIterator2dThreadTile(
    TensorRef const &ref, 
    int thread_idx
  ):
    iterator_({ref.data(), ref.stride()}, thread_idx, 4) {

  }

  /// Loads a fragment
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag, TensorCoord const & tile_offset) {
    iterator_.load_with_pointer_offset(frag, {tile_offset.row(), tile_offset.column()});
  }

  /// Loads a fragment
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) {
    iterator_.load_with_pointer_offset(frag, 0);
  }

  /// Stores a fragment
  CUTLASS_HOST_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Stores a fragment
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag, TensorCoord const & tile_offset) {
    iterator_.store_with_pointer_offset(frag, {tile_offset.row(), tile_offset.column()});
  }

  /// Stores a fragment
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) {
    iterator_.store_with_pointer_offset(frag, 0);
  }

  /// Advances the pointer
  CUTLASS_HOST_DEVICE
  RegularTileIterator2dThreadTile &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances the pointer
  CUTLASS_HOST_DEVICE
  RegularTileIterator2dThreadTile &operator--() {
    --iterator_;
    return *this;
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    iterator_.add_tile_offset({coord.row(), coord.column()});
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef MIX_FP16_DOT2
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Regular tile iterator specialized for interleaved layout + 2d thread-tiled threadmapping
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  int Alignment,
  bool Transpose
>
class RegularTileIterator2dThreadTile<Shape_, Element_, layout::RowMajorInterleaved<2>, AdvanceRank, ThreadMap_, Alignment, Transpose> {
public:

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::RowMajorInterleaved<2>;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  static int const kAlignment = Alignment;
static bool const kTranspose = Transpose;
  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Fragment = Array<Element, ThreadMap::Iterations::kCount * ThreadMap::ThreadAccessShape::kCount>;

  using Underlying = RegularTileIterator2dThreadTile<
    layout::PitchLinearShape<Shape::kColumn, Shape::kRow>,
    Element,
    layout::PitchLinear,
    (kAdvanceRank == 0 ? 1 : 0),
    ThreadMap,
    kAlignment,
    kTranspose
  >;

  static_assert(kAdvanceRank == 0 || kAdvanceRank == 1, 
    "Advance rank may only be along the row or column dimensions.");

private:

  Underlying iterator_;

public:

  CUTLASS_DEVICE
  RegularTileIterator2dThreadTile() { }

  CUTLASS_DEVICE
  RegularTileIterator2dThreadTile(
    TensorRef const &ref, 
    int thread_idx
  ):
    iterator_({ref.data(), ref.stride()}, thread_idx, 2) {

  }

  /// Loads a fragment
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag, TensorCoord const & tile_offset) {
    iterator_.load_with_pointer_offset(frag, {tile_offset.column(), tile_offset.row()});
  }

  /// Loads a fragment
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) {
    iterator_.load_with_pointer_offset(frag, 0);
  }

  /// Stores a fragment
  CUTLASS_HOST_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Stores a fragment
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag, TensorCoord const & tile_offset) {
    iterator_.store_with_pointer_offset(frag, {tile_offset.column(), tile_offset.row()});
  }

  /// Stores a fragment
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) {
    iterator_.store_with_pointer_offset(frag, 0);
  }

  /// Advances the pointer
  CUTLASS_HOST_DEVICE
  RegularTileIterator2dThreadTile &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances the pointer
  CUTLASS_HOST_DEVICE
  RegularTileIterator2dThreadTile &operator--() {
    --iterator_;
    return *this;
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    iterator_.add_tile_offset({coord.column(), coord.row()});
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Regular tile iterator specialized for interleaved layout + 2d thread-tiled threadmapping
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  int Alignment,
  bool Transpose
>
class RegularTileIterator2dThreadTile<Shape_, Element_, layout::ColumnMajorInterleaved<2>, AdvanceRank, ThreadMap_, Alignment,Transpose> {
public:

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::ColumnMajorInterleaved<2>;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  static int const kAlignment = Alignment;
static bool const kTranspose = Transpose;
  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Fragment = Array<Element, ThreadMap::Iterations::kCount * ThreadMap::ThreadAccessShape::kCount>;

  using PitchLinearThreadMap = PitchLinearStripminedThreadMap< layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, 
                                  ThreadMap::kThreads, ThreadMap::ThreadAccessShape::kCount >;
                        

  using Underlying = RegularTileIterator2dThreadTile<
    layout::PitchLinearShape<Shape::kRow, Shape::kColumn>,
    Element,
    layout::PitchLinear,
    (kAdvanceRank == 0 ? 0 : 1),
    ThreadMap,
    kAlignment,
    kTranspose
  >;

  static_assert(kAdvanceRank == 0 || kAdvanceRank == 1, 
    "Advance rank may only be along the row or column dimensions.");

private:

  Underlying iterator_;

public:

  CUTLASS_DEVICE
  RegularTileIterator2dThreadTile() { }

  CUTLASS_DEVICE
  RegularTileIterator2dThreadTile(
    TensorRef const &ref, 
    int thread_idx
  ):
    iterator_({ref.data(), ref.stride()}, thread_idx, 2) {

  }

  /// Loads a fragment
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag, TensorCoord const & tile_offset) {
    iterator_.load_with_pointer_offset(frag, {tile_offset.row(), tile_offset.column()});
  }

  /// Loads a fragment
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) {
    iterator_.load_with_pointer_offset(frag, 0);
  }

  /// Stores a fragment
  CUTLASS_HOST_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Stores a fragment
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag, TensorCoord const & tile_offset) {
    iterator_.store_with_pointer_offset(frag, {tile_offset.row(), tile_offset.column()});
  }

  /// Stores a fragment
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) {
    iterator_.store_with_pointer_offset(frag, 0);
  }

  /// Advances the pointer
  CUTLASS_HOST_DEVICE
  RegularTileIterator2dThreadTile &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances the pointer
  CUTLASS_HOST_DEVICE
  RegularTileIterator2dThreadTile &operator--() {
    --iterator_;
    return *this;
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    iterator_.add_tile_offset({coord.row(), coord.column()});
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////


#endif
} // namespace threadblock
} // namespace transform
} // namespace cutlass

