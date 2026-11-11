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
    \brief Templates calculating the address and predicates to the load of tiles
   from pitch-linear rank=2 tensors.

    This iterator uses masks to guard out-of-bounds accesses and visits the last
   "residue" tile first, with the objective of minimizing predicate mask updates
   during steady-state operation.

    A precomputed "Params" object minimizes the amount of state that must be
   stored in registers, and integer addition is used to advance the pointer
   through memory.
*/

#pragma once

#include "cutlass/arch/memory_buffer.h"
#include "cutlass/array.h"
#include "cutlass/coord.h"
#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/predicate_vector.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/tensor_view.h"
#include "cutlass/transform/threadblock/predicated_tile_access_iterator_params.h"

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace transform {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// PredicatedTileAccessIterator2dThreadTile
///
template <typename Shape, typename Element, typename Layout, int AdvanceRank,
          typename ThreadMap, typename AccessType, int OffsetNoGuard = 0,
          bool BufferAccess = false, bool EnStaggerK = false,
          bool DirectInc = false>
class PredicatedTileAccessIterator2dThreadTile;

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileAccessIterator2dThreadTile for pitch-linear data.
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, typename AccessType_, int OffsetNoGuard,
          bool BufferAccess, bool EnStaggerK_, bool DirectInc>
class PredicatedTileAccessIterator2dThreadTile<
    Shape_, Element_, layout::PitchLinear, AdvanceRank, ThreadMap_, AccessType_,
    OffsetNoGuard, BufferAccess, EnStaggerK_, DirectInc> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::PitchLinear;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using StrideIndex = typename Layout::Stride::Index;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  static bool constexpr EnStaggerK = EnStaggerK_;
  // static bool constexpr EnStaggerK = true;

  static int const kPredicatesPerByte = 4;
  static int const kPredicatesPerWord = 4 * kPredicatesPerByte;

  /// Number of 32b words containing predicates
  // static int const kPredicateByteCount = (ThreadMap::Iterations::kCount * ThreadMap::ThreadAccessShape::kStrided + kPredicatesPerByte - 1) / kPredicatesPerByte;
  /// Number of 32b words containing predicates
  static int const kPredicateByteCount = (ThreadMap::Iterations::kCount * ThreadMap::ThreadAccessShape::kStrided * (ThreadMap::ThreadAccessShape::kContiguous / ThreadMap::kElementsPerAccess)
                                           + kPredicatesPerByte - 1) / kPredicatesPerByte;

  static int const kPredicateWordCount = (kPredicateByteCount + 3) / 4;

  static unsigned const kPredicateMask = (1u << kPredicatesPerByte) - 1u;

  // static_assert(kPredicateWordCount <= 4, "Too many predicates.");

  static int const kAccessesPerVector = ThreadMap::kElementsPerAccess / AccessType::kElements;
  static_assert(kAccessesPerVector >= 1);
  static_assert(OffsetNoGuard == 0 || ThreadMap::Iterations::kContiguous == 1,
                "OffsetNoGuard == 1 requires ThreadMap::Iterations::kContiguous == 1.");
  static_assert(OffsetNoGuard == 0 || kAccessesPerVector == 1,
                "OffsetNoGuard=1 requires kAccessesPerVector == 1.");
  static_assert(EnStaggerK == false || BufferAccess == true, "EnStaggerK requires BufferAccess");
  static_assert(DirectInc == false || BufferAccess == true, "DirectInc requires BufferAccess");

  /// Predicate vector stores mask to guard accesses
  using Mask = Array<uint32_t, kPredicateWordCount>;

  /// Uses a non-template class
  struct Params : PredicatedTileAccessIteratorParams {

   public:
    friend PredicatedTileAccessIterator2dThreadTile;

    using Base = PredicatedTileAccessIteratorParams;

    // Default ctor
    CUTLASS_HOST_DEVICE
    Params() { }

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout) : 
      Base(layout.stride(0),
            MakePredicatedTileAccessIteratorDesc<Shape, Element, Layout, kAdvanceRank, ThreadMap>()()
        ) { }

    CUTLASS_HOST_DEVICE
    Params(Layout const &layout, Index stagger_k_log2, Index stagger_k_stride_log2) : 
      Base(layout.stride(0),
           MakePredicatedTileAccessIteratorDesc<Shape, Element, Layout, kAdvanceRank, ThreadMap>()(),
           stagger_k_log2, stagger_k_stride_log2
        ) { }

    CUTLASS_HOST_DEVICE
    Params(Base const &base) : 
      Base(base) { }
  };


 private:
  /// Internal pointer type permits fast address arithmetic
  using BytePointer = char *;

 private:
  //
  // Data members
  //

  /// Parameters object with precomputed internal state
  Params const &params_;

  /// Internal pointer to first access of tile
  BytePointer pointer_;
cutlass::arch::BufferAccessor buffer_accessor;

  /// Guard predicates
  uint32_t predicates_[kPredicateWordCount];

  /// Guard offsets
  int guard_offsets[ThreadMap::Iterations::kCount * ThreadMap::ThreadAccessShape::kStrided];

  /// Size of tensor
  TensorCoord extent_;

  /// Initial offset for each thread
  TensorCoord thread_offset_;

  /// Index of residue tile
  int residue_tile_idx_;

  /// Used for out-of-order visitation
  bool is_residue_tile_;

  /// Iteration in the contiguous dimension
  int iteration_contiguous_;

  /// Iteration in the strided dimension
  int iteration_strided_;

  /// Tracks iterations within the thread loop
  int iteration_thread_;

  int iteration_threadAccessShape_contiguous_ = 0;
  /// The id of tile
  int tile_idx = 0;

  /// Number of tiles (excluding the residue tile)
  int num_of_tiles;

  /// Maximum tile address offset
  uint offset_max = UINT32_MAX;

  // ///
  // int tile_offset = 0;


 private:
  /// Computes predicates based on internally tracked per-thread offset.
  CUTLASS_HOST_DEVICE
  void compute_predicates_(
      /// optionally, simplify predicate calculation during 'steady state' phase
      bool is_steady_state = false) {

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPredicateWordCount; ++i) {
      predicates_[i] = 0u;
    }

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        CUTLASS_PRAGMA_UNROLL
        for (int ts = 0; ts < ThreadMap::ThreadAccessShape::kStrided; ts++) {

          TensorCoord iteration_coord(c * ThreadMap::Delta::kContiguous,
                                      ts + s * ThreadMap::Delta::kStrided);

          TensorCoord coord = thread_offset_ + iteration_coord;

          bool guard;

          int access_idx = ts + c *  ThreadMap::ThreadAccessShape::kStrided + s * ThreadMap::Iterations::kContiguous *  ThreadMap::ThreadAccessShape::kStrided;
          guard_offsets[access_idx] = 0;

          if (is_steady_state) {
            if (kAdvanceRank == 0) {
              guard = (coord.strided() < extent_.strided());
              if constexpr (OffsetNoGuard == 3) {
                guard_offsets[access_idx] -= (coord.strided() >= extent_.strided()) ? 
                  (params_.stride_ * (coord.strided() - extent_.strided() + 1) * sizeof_bits<Element>::value / 8) : 0;
              }
            } else {
              guard = (coord.contiguous() < extent_.contiguous());
              if constexpr (OffsetNoGuard == 3) {
                guard_offsets[access_idx] -= (coord.contiguous() >= extent_.contiguous()) ? 
                  ((coord.contiguous() - extent_.contiguous() + ThreadMap::kElementsPerAccess) * sizeof_bits<Element>::value / 8) : 0;
              }
            }
          } else {
            guard = (coord.strided() < extent_.strided() &&
                     coord.contiguous() < extent_.contiguous());
          }
        }
      }
    }

    // 2. 以 ThreadMap::kElementsPerAccess 为最小粒度进行访存（对应于 alignment in units）
    // 处理的是 is_steady_state == 0 时的情况
    // 例如对于 half 是 1xhalf 为粒度，这样可以确保尾部边界的访存合法
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        CUTLASS_PRAGMA_UNROLL
        for (int ts = 0; ts < ThreadMap::ThreadAccessShape::kStrided; ts++) {
          CUTLASS_PRAGMA_UNROLL
          for (int tc = 0; tc < (ThreadMap::ThreadAccessShape::kContiguous / ThreadMap::kElementsPerAccess); tc++) {

            TensorCoord iteration_coord(c * ThreadMap::Delta::kContiguous + tc * ThreadMap::kElementsPerAccess,
                                        ts + s * ThreadMap::Delta::kStrided);

            TensorCoord coord = thread_offset_ + iteration_coord;

            bool guard;

            // Different from before
            int access_idx = tc + (ThreadMap::ThreadAccessShape::kContiguous / ThreadMap::kElementsPerAccess) * 
                             (ts + c *  ThreadMap::ThreadAccessShape::kStrided + s * ThreadMap::Iterations::kContiguous *  ThreadMap::ThreadAccessShape::kStrided);

            if (is_steady_state) {
              if (kAdvanceRank == 0) {
                guard = (coord.strided() < extent_.strided());
              } else {
                guard = (coord.contiguous() < extent_.contiguous());
              }
            } else {
              guard = (coord.strided() < extent_.strided() &&
                       coord.contiguous() < extent_.contiguous());
            }

            int pred_idx = access_idx;
            int word_idx = pred_idx / kPredicatesPerWord;
            int residual = pred_idx % kPredicatesPerWord;
            int byte_idx = residual / kPredicatesPerByte;
            int bit_idx = residual % kPredicatesPerByte;

            predicates_[word_idx] |= (unsigned(guard) << (byte_idx * 8 + bit_idx));

          }
        }
      }
    }


  }

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator2dThreadTile(
      /// Precomputed parameters object
      Params const &params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      /// ID of each participating thread
      int thread_id,
      /// Initial offset of threadblock
      TensorCoord const &threadblock_offset)
      : params_(params),
        pointer_(reinterpret_cast<BytePointer>(
            const_cast<NonConstPointer>(pointer))),
        buffer_accessor(reinterpret_cast<BytePointer>(
            const_cast<NonConstPointer>(pointer))),
        extent_(extent),
        is_residue_tile_(true) {
          

    TensorCoord residue_offset;
    if (kAdvanceRank) {
      residue_tile_idx_ =
          (extent_[kAdvanceRank] - threadblock_offset[kAdvanceRank] - 1) /
          Shape::kStrided;
      num_of_tiles = residue_tile_idx_;
      residue_offset = make_Coord(0, residue_tile_idx_ * Shape::kStrided);
    } else {
      residue_tile_idx_ =
          (extent_[kAdvanceRank] - threadblock_offset[kAdvanceRank] - 1) /
          Shape::kContiguous;
      num_of_tiles = residue_tile_idx_;
      residue_offset = make_Coord(residue_tile_idx_ * Shape::kContiguous, 0);
    }

    // Per-thread offset in logical coordinates of tensor
    thread_offset_ = threadblock_offset + residue_offset +
                     ThreadMap::initial_offset(thread_id);

    if(OffsetNoGuard == 2){
      // OffsetNoGuard == 1 temporarily requires kAdvanceRank == 1,
      // so residue_offset.contiguous() == 0
      if (thread_offset_.contiguous() >= extent_.contiguous()) {
        thread_offset_.contiguous() = extent_.contiguous() - ThreadMap::kElementsPerAccess;
      }
    }

    // update internal pointers
    Layout layout(params_.stride_);
    add_pointer_offset(layout(thread_offset_));
    offset_max = uint(sizeof(Element)) * layout(residue_offset);

    compute_predicates_(false);

    set_iteration_index(0);
  }

  /// Construct a PredicatedTileAccessIterator2dThreadTile with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator2dThreadTile(
      /// Precomputed parameters object
      Params const &params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      ///< ID of each participating thread
      int thread_id)
      : PredicatedTileAccessIterator2dThreadTile(params, pointer, extent, thread_id,
                                     make_Coord(0, 0)) {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index, int tc = 0) {

    int residual = index % (ThreadMap::Iterations::kContiguous * ThreadMap::ThreadAccessShape::kStrided);
    iteration_strided_ = index / (ThreadMap::Iterations::kContiguous * ThreadMap::ThreadAccessShape::kStrided);
    
    iteration_contiguous_ = residual / ThreadMap::ThreadAccessShape::kStrided;
    iteration_thread_ = residual % ThreadMap::ThreadAccessShape::kStrided;

    iteration_threadAccessShape_contiguous_ = tc;
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    pointer_ += int(sizeof(Element)) * pointer_offset;
    if constexpr (BufferAccess) {
      buffer_accessor.add_to_vofft(int(sizeof(Element)) * pointer_offset);
    }
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_DEVICE
  void add_tile_offset(
      TensorCoord const &tile_offset) {
    if (is_residue_tile_) {
      TensorCoord residue_offset;
      if (kAdvanceRank) {
        residue_offset = TensorCoord(0, residue_tile_idx_ * Shape::kStrided);
      } else {
        residue_offset = TensorCoord(residue_tile_idx_ * Shape::kContiguous, 0);
      }

      thread_offset_ -= residue_offset;

      Layout layout(params_.stride_);
      add_pointer_offset(-layout(residue_offset));

      compute_predicates_(true);

      if (kAdvanceRank) {
        pointer_ += params_.inc_advance_ * (tile_offset.strided() - 1);
        pointer_ += Shape::kContiguous * tile_offset.contiguous();
        if constexpr (BufferAccess) {
          buffer_accessor.add_to_sofft(params_.inc_advance_ * (tile_offset.strided() - 1));
          buffer_accessor.add_to_sofft(Shape::kContiguous * tile_offset.contiguous());
        }
      } else {
        pointer_ += params_.inc_advance_ * (tile_offset.contiguous() - 1);
        pointer_ += Shape::kStrided * tile_offset.strided();
        if constexpr (BufferAccess) {
          buffer_accessor.add_to_sofft(params_.inc_advance_ * (tile_offset.contiguous() - 1));
          buffer_accessor.add_to_sofft(Shape::kStrided * tile_offset.strided());
        }
      }

      tile_idx = 0;

      // TODO may be change to thread_id passed from outside is better
      int flat_block_id_xy = gemm::threadblock::RematerializeBlockIdxY() * gemm::threadblock::RematerializeBlockDimY() +
                             gemm::threadblock::RematerializeBlockIdxX();

      // stagger_k count how many threadBlocks are in each stragger group.
      // step_tile count the number of tiles between adjacent threadBlocks in each stragger group.
      // stagger_k_stride count step_tile in bytes
      // stagger_k_stride_log2 = log2(stagger_k_stride)
      int step_tile = 0;
      if constexpr (EnStaggerK) {
        if (kAdvanceRank) {
          step_tile += (1 << params_.stagger_k_stride_log2) / (Shape::kStrided * (sizeof_bits<Element>::value / 8));
        } else {
          step_tile += (1 << params_.stagger_k_stride_log2) / (Shape::kContiguous * (sizeof_bits<Element>::value / 8));
        }

        tile_idx = step_tile * (flat_block_id_xy & ((1 << params_.stagger_k_log2) - 1));
        // buffer_accessor.add_to_sofft(step_tile * (flat_block_id_xy & ((1 << params_.stagger_k_log2) - 1)) * params_.inc_advance_);

        // Method 1. More abusive.
        tile_idx = tile_idx % num_of_tiles;
        tile_idx = __builtin_amdgcn_readfirstlane(tile_idx);

        // Method 2. A little faster, but need checking in kernel's can_implement method.
        // [WARNING!] Otherwise, when tile_idx >= 2 * num_of_tiles, it will cause access errors.
        // tile_idx = (tile_idx >= num_of_tiles) ? (tile_idx - num_of_tiles) : tile_idx;

        // Method 1. More abusive.
        // buffer_accessor.sofft = buffer_accessor.sofft % offset_max;
        // buffer_accessor.sofft = __builtin_amdgcn_readfirstlane(buffer_accessor.sofft);

        // Method 2. A little faster, but need checking in kernel's can_implement method.
        // buffer_accessor.sofft = (buffer_accessor.sofft >= offset_max) ? (buffer_accessor.sofft - offset_max) : buffer_accessor.sofft;
      }
    } else {
      if (kAdvanceRank) {
        pointer_ += params_.inc_advance_ * tile_offset.strided();
        pointer_ += Shape::kContiguous * tile_offset.contiguous();
        if constexpr (BufferAccess) {
          if constexpr (!EnStaggerK) {
            buffer_accessor.add_to_sofft(params_.inc_advance_ * tile_offset.strided());
          }
          buffer_accessor.add_to_sofft(Shape::kContiguous * tile_offset.contiguous());
        }
      } else {
        pointer_ += params_.inc_advance_ * tile_offset.contiguous();
        pointer_ += Shape::kStrided * tile_offset.strided();
        if constexpr (BufferAccess) {
          if constexpr (!EnStaggerK) {
            buffer_accessor.add_to_sofft(params_.inc_advance_ * tile_offset.contiguous());
          }
          buffer_accessor.add_to_sofft(Shape::kStrided * tile_offset.strided());
        }
      }

      if constexpr (EnStaggerK) {

        tile_idx++;
        tile_idx = (tile_idx >= num_of_tiles) ? (tile_idx - num_of_tiles) : tile_idx;

        // buffer_accessor.sofft = (buffer_accessor.sofft >= offset_max) ? (buffer_accessor.sofft - offset_max) : buffer_accessor.sofft;

      }
    }
    is_residue_tile_ = false;
  }

  CUTLASS_HOST_DEVICE
  AccessType *get() const {

    AccessType *ret_val;

    if constexpr (OffsetNoGuard == 3) {
      ret_val = reinterpret_cast<AccessType *>(
          pointer_ +
          (iteration_thread_ * params_.stride_ + iteration_contiguous_ * ThreadMap::Delta::kContiguous + iteration_threadAccessShape_contiguous_) * int(sizeof(Element)) +
          offset());
    } else {
      ret_val = reinterpret_cast<AccessType *>(
          pointer_ +
          (iteration_thread_ * params_.stride_ + iteration_contiguous_ * ThreadMap::Delta::kContiguous + iteration_threadAccessShape_contiguous_) * int(sizeof(Element)));
    }
    return ret_val;
  }

  CUTLASS_HOST_DEVICE
  cutlass::arch::BufferAccessor get_buffer_accessor() const {

    static_assert(BufferAccess == true,
                  "Function clear_buffer_sofft need to enable BufferAccess "
                  "property of the class PredicatedTileAccessIterator2dThreadTile.");

    cutlass::arch::BufferAccessor ret_val = buffer_accessor;
    ret_val.add_to_vofft((iteration_thread_ * params_.stride_  + iteration_contiguous_ * ThreadMap::Delta::kContiguous) * int(sizeof(Element)));
    if constexpr (OffsetNoGuard == 3) {
      ret_val.add_to_vofft(offset());
    }
    if constexpr (EnStaggerK) {
      ret_val.add_to_sofft(tile_idx * params_.inc_advance_);
    }
    if constexpr (DirectInc) {
      // It can be confirm that (in function operator++)
      // ThreadMap::Iterations::kStrided * inc_strided_ + inc_next_ - inc_advance_ == 0;
      // What's more, func load* will always call func set_iteration_index before call this func.
      // So we can use the following method instead of add_to_sofft in func ++,
      // which can reduce the num of sgprs(sofft) to 1 while keeping the vgprs(vofft) no change during buffer_load
      ret_val.add_to_vofft(iteration_strided_ * params_.inc_strided_);
    }
    // TODO 未来改成 buffer_load 的常数索引
    ret_val.add_to_vofft(iteration_threadAccessShape_contiguous_ * int(sizeof(Element)));
    
    return ret_val;
  }

  /// Increment and return an instance to self.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator2dThreadTile &operator++() {

    iteration_thread_++;

    if (iteration_thread_ < ThreadMap::ThreadAccessShape::kStrided)
      return *this;

    iteration_thread_ = 0;

    ++iteration_contiguous_;

    if (iteration_contiguous_ < ThreadMap::Iterations::kContiguous)
      return *this;

    // Enter here only if (iteration_contiguous_ ==
    // ThreadMap::Iteration::kContiguous)
    iteration_contiguous_ = 0;
    ++iteration_strided_;

    if (iteration_strided_ < ThreadMap::Iterations::kStrided) {
      pointer_ += params_.inc_strided_;
      if constexpr (BufferAccess) {
        buffer_accessor.add_to_sofft(params_.inc_strided_);
      }
      return *this;
    }

    // Enter here only if (iteration_stride_ == ThreadMap::Iteration::kStrided)
    // which means we enter the next tile.
    iteration_strided_ = 0;

    // advance to next tile
    pointer_ += params_.inc_next_;
    if constexpr (BufferAccess) {
      buffer_accessor.add_to_sofft(params_.inc_next_);
    }

    // now return to start tile - if the iterator is subsequently advanced, this
    // subtraction as well as the subsequent integer addition are both elided by
    // the compiler.
    pointer_ -= params_.inc_advance_;
    if constexpr (BufferAccess) {
      buffer_accessor.add_to_sofft(-params_.inc_advance_);
    }

    return *this;
  }

  /// Increment and return an instance to self.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator2dThreadTile operator++(int) {
    PredicatedTileAccessIterator2dThreadTile self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPredicateWordCount; ++i) {
      predicates_[i] = enable ? 0u : predicates_[i];
    }

  }

  /// Clears the sgpr offset of the buffer_accessor
  CUTLASS_HOST_DEVICE
  void clear_buffer_sofft(bool enable = true) {
    static_assert(BufferAccess == true,
                  "Function clear_buffer_sofft need to enable BufferAccess "
                  "property of the class PredicatedTileAccessIterator2dThreadTile.");
    if constexpr (!EnStaggerK) {
      buffer_accessor.sofft -= enable ? buffer_accessor.sofft : 0;
      // Only required by some older compilers.
      // buffer_accessor.sofft = __builtin_amdgcn_readfirstlane(buffer_accessor.sofft);
    }
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPredicateWordCount; ++i) {
      predicates_[i] = 0xffffffff;
    }
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) { 
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPredicateWordCount; ++i) {
      predicates_[i] = mask[i];
    }

  }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) {
     CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPredicateWordCount; ++i) {
      mask[i] = predicates_[i];
    }
  }

  /// Returns whether access is valid or not
  CUTLASS_HOST_DEVICE
  bool valid() {

    int pred_idx = 
      iteration_threadAccessShape_contiguous_ + (ThreadMap::ThreadAccessShape::kContiguous / ThreadMap::kElementsPerAccess) * (
      iteration_thread_ + 
      iteration_contiguous_ * ThreadMap::ThreadAccessShape::kStrided + 
      iteration_strided_ * ThreadMap::Iterations::kContiguous * ThreadMap::ThreadAccessShape::kStrided);

    int word_idx = pred_idx / kPredicatesPerWord;
    int residual = pred_idx % kPredicatesPerWord;
    int byte_idx = residual / kPredicatesPerByte;
    int bit_idx = residual % kPredicatesPerByte;
    
    bool pred = (predicates_[word_idx] & (1u << (byte_idx * 8 + bit_idx))) != 0;
    
    return pred;
  }

  /// Returns the offset to replace invalid access with valid access
  CUTLASS_HOST_DEVICE
  int offset() const {

    int pred_idx = 
      iteration_thread_ + 
      iteration_contiguous_ * ThreadMap::ThreadAccessShape::kStrided + 
      iteration_strided_ * ThreadMap::Iterations::kContiguous * ThreadMap::ThreadAccessShape::kStrided;

    return guard_offsets[pred_idx];
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileAccessIterator2dThreadTile for pitch-linear data.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, typename AccessType_, int OffsetNoGuard, bool BufferAccess>
class PredicatedTileAccessIterator2dThreadTile<Shape_, Element_, layout::ColumnMajor,
                                   AdvanceRank, ThreadMap_, AccessType_, OffsetNoGuard, BufferAccess> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::ColumnMajor;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  using UnderlyingIterator = PredicatedTileAccessIterator2dThreadTile<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, Element,
      layout::PitchLinear, (kAdvanceRank == 0 ? 0 : 1), ThreadMap,
      AccessType, OffsetNoGuard, BufferAccess>;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   private:
    friend PredicatedTileAccessIterator2dThreadTile;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

   public:

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Params() { }

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout)
        : params_(layout::PitchLinear(layout.stride(0))){}

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(typename UnderlyingIterator::Params::Base const &base) 
        : params_(base) {}
  };

 private:
  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator2dThreadTile(
      ///< Precomputed parameters object
      Params const &params,
      ///< Pointer to start of tensor
      Pointer pointer,
      ///< Extent of tensor
      TensorCoord extent,
      ///< ID of each participating thread
      int thread_id,
      ///< Initial offset of threadblock
      TensorCoord const &threadblock_offset)
      : iterator_(params.params_, pointer,
                  layout::PitchLinearCoord(extent.row(), extent.column()),
                  thread_id,
                  layout::PitchLinearCoord(threadblock_offset.row(),
                                           threadblock_offset.column())) {}

  /// Construct a PredicatedTileAccessIterator2dThreadTile with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator2dThreadTile(
      Params const &params,  ///< Precomputed parameters object
      Pointer pointer,       ///< Pointer to start of tensor
      TensorCoord extent,    ///< Extent of tensor
      int thread_id          ///< ID of each participating thread
      )
      : PredicatedTileAccessIterator2dThreadTile(params, pointer, extent, thread_id,
                                     make_Coord(0, 0)) {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) { iterator_.set_iteration_index(index); }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_HOST_DEVICE
  void add_tile_offset(TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.row(), tile_offset.column()});
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType *get() const {
    return reinterpret_cast<AccessType *>(iterator_.get());
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator2dThreadTile &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator2dThreadTile operator++(int) {
    PredicatedTileAccessIterator2dThreadTile self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) { iterator_.clear_mask(enable); }

  /// Clears the sgpr offset of the buffer_accessor
  CUTLASS_HOST_DEVICE
  void clear_buffer_sofft(bool enable = true) { iterator_.clear_buffer_sofft(enable); }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() { iterator_.enable_mask(); }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) { iterator_.set_mask(mask); }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) { iterator_.get_mask(mask); }

  /// Returns whether access is valid or not
  CUTLASS_HOST_DEVICE
  bool valid() {
    return iterator_.valid();
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileAccessIterator2dThreadTile for pitch-linear data.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, typename AccessType_, int OffsetNoGuard, bool BufferAccess>
class PredicatedTileAccessIterator2dThreadTile<Shape_, Element_, layout::RowMajor,
                                   AdvanceRank, ThreadMap_, AccessType_, OffsetNoGuard, BufferAccess> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::RowMajor;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  using UnderlyingIterator = PredicatedTileAccessIterator2dThreadTile<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, Element,
      layout::PitchLinear, (kAdvanceRank == 0 ? 1 : 0), ThreadMap,
      AccessType, OffsetNoGuard, BufferAccess>;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   private:
    friend PredicatedTileAccessIterator2dThreadTile;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

   public:

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Params() { }

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout)
        : params_(layout::PitchLinear(layout.stride(0))){}

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(typename UnderlyingIterator::Params::Base const &base) 
        : params_(base) {}
  };

 private:
  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator2dThreadTile(
      ///< Precomputed parameters object
      Params const &params,
      ///< Pointer to start of tensor
      Pointer pointer,
      ///< Extent of tensor
      TensorCoord extent,
      ///< ID of each participating thread
      int thread_id,
      ///< Initial offset of threadblock
      TensorCoord const &threadblock_offset)
      : iterator_(params.params_, pointer,
                  layout::PitchLinearCoord(extent.column(), extent.row()),
                  thread_id,
                  layout::PitchLinearCoord(threadblock_offset.column(),
                                           threadblock_offset.row())) {}

  /// Construct a PredicatedTileAccessIterator2dThreadTile with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator2dThreadTile(
      Params const &params,  ///< Precomputed parameters object
      Pointer pointer,       ///< Pointer to start of tensor
      TensorCoord extent,    ///< Extent of tensor
      int thread_id          ///< ID of each participating thread
      )
      : PredicatedTileAccessIterator2dThreadTile(params, pointer, extent, thread_id,
                                     make_Coord(0, 0)) {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) { iterator_.set_iteration_index(index); }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_HOST_DEVICE
  void add_tile_offset(TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType *get() const {
    return reinterpret_cast<AccessType *>(iterator_.get());
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator2dThreadTile &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator2dThreadTile operator++(int) {
    PredicatedTileAccessIterator2dThreadTile self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) { iterator_.clear_mask(enable); }

  /// Clears the sgpr offset of the buffer_accessor
  CUTLASS_HOST_DEVICE
  void clear_buffer_sofft(bool enable = true) { iterator_.clear_buffer_sofft(enable); }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() { iterator_.enable_mask(); }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) { iterator_.set_mask(mask); }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) { iterator_.get_mask(mask); }

  /// Returns whether access is valid or not
  CUTLASS_HOST_DEVICE
  bool valid() {
    return iterator_.valid();
  }
};

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace transform
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
