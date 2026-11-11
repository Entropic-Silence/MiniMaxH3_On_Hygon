#ifndef __INFINIOP_ELEMENTWISE_CUDA_H__
#define __INFINIOP_ELEMENTWISE_CUDA_H__

#include "../../../utils.h"
#include "../../devices/nvidia/nvidia_common.cuh"
#include "../../devices/nvidia/nvidia_kernel_common.cuh"
#include "elementwise_nvidia_api.cuh"

#include <type_traits>

namespace op::elementwise::nvidia {

template <size_t N>
struct InputPointerArray {
    const void *values[N];
};

template <size_t N>
struct InlineElementwiseMeta {
    size_t ndim;
    size_t output_shape[INLINE_META_MAX_NDIM];
    ptrdiff_t output_strides[INLINE_META_MAX_NDIM];
    size_t input_shapes[N][INLINE_META_MAX_NDIM];
    ptrdiff_t input_strides[N][INLINE_META_MAX_NDIM];
};

template <size_t N>
struct InlineElementwiseOffsets {
    ptrdiff_t output;
    ptrdiff_t inputs[N];
};

template <size_t N>
InlineElementwiseMeta<N> makeInlineElementwiseMeta(
    const op::elementwise::ElementwiseInfo &info) {
    InlineElementwiseMeta<N> meta{};
    meta.ndim = info.getNdim();
    for (size_t dim = 0; dim < meta.ndim; ++dim) {
        meta.output_shape[dim] = info.getOutputShape()[dim];
        meta.output_strides[dim] = info.getOutputStrides()[dim];
        for (size_t input = 0; input < N; ++input) {
            meta.input_shapes[input][dim] = info.getInputShape(input)[dim];
            meta.input_strides[input][dim] = info.getInputStrides(input)[dim];
        }
    }
    return meta;
}

template <size_t N>
__device__ __forceinline__ InlineElementwiseOffsets<N> getInlineElementwiseOffsets(
    size_t idx,
    const InlineElementwiseMeta<N> &meta) {
    InlineElementwiseOffsets<N> offsets{};
    for (size_t dim = meta.ndim; dim-- > 0;) {
        const size_t coordinate = idx % meta.output_shape[dim];
        idx /= meta.output_shape[dim];
        offsets.output += static_cast<ptrdiff_t>(coordinate) * meta.output_strides[dim];
#pragma unroll
        for (size_t input = 0; input < N; ++input) {
            const size_t input_coordinate = meta.input_shapes[input][dim] == 1
                                              ? 0
                                              : coordinate;
            offsets.inputs[input] += static_cast<ptrdiff_t>(input_coordinate)
                                   * meta.input_strides[input][dim];
        }
    }
    return offsets;
}

/**
 * @brief Casts an untyped device pointer to a typed pointer of type T.
 *
 * @tparam T   Desired pointer type.
 *
 * @param ptr  Untyped pointer.
 * @return     Pointer of type const T*.
 */
template <typename T>
__device__ __forceinline__ const T *typedInputPtr(const void *ptr) {
    return reinterpret_cast<const T *>(ptr);
}

/**
 * @brief Computes the output index in memory, accounting for strides if non-contiguous.
 *
 * @param idx            Linear index.
 * @param is_contiguous  Whether the output tensor is contiguous.
 * @param ndim           Number of dimensions.
 * @param shape          Shape of the output tensor.
 * @param strides        Strides of the output tensor.
 * @return               Memory offset index.
 */
__device__ __forceinline__ size_t getOutputIndex(size_t idx, bool is_contiguous, size_t ndim,
                                                 const size_t *shape, const ptrdiff_t *strides) {
    return is_contiguous ? idx : device::nvidia::indexToOffset(idx, ndim, shape, strides);
}

/**
 * @brief Computes input element offset for broadcasting and strided access.
 *
 * Used to map a linear output index to the corresponding index in an input tensor,
 * considering contiguity and broadcasting.
 */
struct InputIndexer {
    size_t idx;
    size_t ndim;
    const bool *input_contiguous;
    const bool *input_broadcasted;
    const size_t *input_shapes;
    const ptrdiff_t *input_strides;
    const ptrdiff_t *output_strides;

    /**
     * @brief Computes the memory offset for a given input tensor at current index.
     *
     * @param input_id  ID of the input tensor.
     * @return          Offset into the input tensor.
     */
    __device__ __forceinline__ size_t operator()(size_t input_id) const {
        return input_contiguous[input_id]
                 ? idx
                 : device::nvidia::indexToOffset(idx, ndim, input_shapes + input_id * ndim, input_strides + input_id * ndim);
    }
};

/**
 * @brief Invokes a callable with compile-time index constants.
 *
 * Used to unpack index sequence for variadic template processing of inputs.
 *
 * @tparam F    Callable type.
 * @tparam Is   Compile-time index sequence.
 *
 * @param f     Callable to invoke with index constants.
 */
template <typename F, size_t... Is>
__device__ __forceinline__ void unpackInputsAndApply(F &&f, std::index_sequence<Is...>) {
    f(std::integral_constant<size_t, Is>{}...);
}

/**
 * @brief CUDA kernel for performing elementwise operations on tensors where all inputs share the same data type.
 *
 * @tparam N        Number of input tensors.
 * @tparam Op       Operator type implementing operator()(Tdata...).
 * @tparam Tdata    Common data type for inputs and output.
 * @tparam Args     Additional arguments to pass to the operator.
 *
 * @param output_size         Total number of output elements.
 * @param ndim                Number of dimensions in tensors.
 * @param output_contiguous   Whether the output tensor is contiguous in memory.
 * @param input_contiguous    Array indicating if each input tensor is contiguous.
 * @param input_broadcasted   Array indicating if each input tensor is broadcasted.
 * @param output_shape        Shape of the output tensor.
 * @param input_shapes        Shapes of the input tensors.
 * @param output_strides      Strides for the output tensor.
 * @param input_strides       Strides for each input tensor.
 * @param output              Output buffer.
 * @param inputs              Array of input pointers, all of type Tdata.
 * @param offset              Linear offset to support partitioned execution.
 * @param args                Additional arguments passed to the operator.
 */
template <size_t N, typename Op, typename Tdata, typename... Args>
INFINIOP_CUDA_KERNEL elementwiseKernel(
    size_t output_size,
    size_t ndim,
    bool output_contiguous,
    const bool *__restrict__ input_contiguous,
    const bool *__restrict__ input_broadcasted,
    const size_t *__restrict__ output_shape,
    const size_t *__restrict__ input_shapes,
    const ptrdiff_t *__restrict__ output_strides,
    const ptrdiff_t *__restrict__ input_strides,
    Tdata *output,
    InputPointerArray<N> inputs,
    size_t offset,
    Args... args) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

    if (idx < output_size) {
        size_t out_idx = getOutputIndex(idx, output_contiguous, ndim, output_shape, output_strides);
        InputIndexer indexer{idx, ndim, input_contiguous, input_broadcasted, input_shapes, input_strides, output_strides};

        unpackInputsAndApply(
            [&](auto... Is) {
#if defined(ENABLE_HYGON_API)
                output[out_idx] = Op{}(typedInputPtr<Tdata>(inputs.values[Is.value])[indexer(Is.value)]..., args...);
#else
                output[out_idx] = Op{}(typedInputPtr<Tdata>(inputs.values[Is.value])[indexer(Is.value)]..., std::forward<Args>(args)...);
#endif
            },
            std::make_index_sequence<N>{});
    }
}

template <size_t N, typename Op, typename Tdata, typename... Args>
INFINIOP_CUDA_KERNEL contiguousElementwiseKernel(
    size_t output_size,
    Tdata *output,
    InputPointerArray<N> inputs,
    size_t offset,
    Args... args) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx < output_size) {
        unpackInputsAndApply(
            [&](auto... Is) {
#if defined(ENABLE_HYGON_API)
                output[idx] = Op{}(typedInputPtr<Tdata>(inputs.values[Is.value])[idx]..., args...);
#else
                output[idx] = Op{}(typedInputPtr<Tdata>(inputs.values[Is.value])[idx]..., std::forward<Args>(args)...);
#endif
            },
            std::make_index_sequence<N>{});
    }
}

template <size_t N, typename Op, typename Tdata, typename... Args>
INFINIOP_CUDA_KERNEL inlineMetaElementwiseKernel(
    size_t output_size,
    Tdata *output,
    InputPointerArray<N> inputs,
    InlineElementwiseMeta<N> meta,
    size_t offset,
    Args... args) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx < output_size) {
        const auto offsets = getInlineElementwiseOffsets(idx, meta);
        unpackInputsAndApply(
            [&](auto... Is) {
#if defined(ENABLE_HYGON_API)
                output[offsets.output] = Op{}(
                    typedInputPtr<Tdata>(inputs.values[Is.value])[offsets.inputs[Is.value]]...,
                    args...);
#else
                output[offsets.output] = Op{}(
                    typedInputPtr<Tdata>(inputs.values[Is.value])[offsets.inputs[Is.value]]...,
                    std::forward<Args>(args)...);
#endif
            },
            std::make_index_sequence<N>{});
    }
}

/**
 * @brief CUDA kernel for performing an elementwise operation on tensors with support
 *        for broadcasting and mixed data types.
 *
 * @tparam Op     Operator type implementing a templated operator() for (Tout, Tin...).
 * @tparam Tout   Output data type.
 * @tparam Tin    Variadic input data types.
 *
 * @param output_size         Total number of output elements.
 * @param ndim                Number of dimensions in the tensors.
 * @param output_contiguous   Whether the output tensor is contiguous.
 * @param input_contiguous    Array indicating whether each input is contiguous.
 * @param input_broadcasted   Array indicating whether each input is broadcasted.
 * @param output_shape        Shape of the output tensor.
 * @param input_shapes        Shapes of the input tensors.
 * @param output_strides      Strides of the output tensor.
 * @param input_strides       Strides of the input tensors.
 * @param output              Pointer to the output buffer.
 * @param inputs              Array of untyped input pointers.
 * @param offset              Linear offset into the output for partitioned execution.
 */
template <typename Op, typename Tout, typename... Tin>
INFINIOP_CUDA_KERNEL elementwiseKernel(
    size_t output_size,
    size_t ndim,
    bool output_contiguous,
    const bool *__restrict__ input_contiguous,
    const bool *__restrict__ input_broadcasted,
    const size_t *__restrict__ output_shape,
    const size_t *__restrict__ input_shapes,
    const ptrdiff_t *__restrict__ output_strides,
    const ptrdiff_t *__restrict__ input_strides,
    Tout *output,
    InputPointerArray<sizeof...(Tin)> inputs,
    size_t offset) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

    if (idx < output_size) {
        size_t out_idx = getOutputIndex(idx, output_contiguous, ndim, output_shape, output_strides);
        InputIndexer indexer{idx, ndim, input_contiguous, input_broadcasted, input_shapes, input_strides, output_strides};

        unpackInputsAndApply(
            [&](auto... Is) {
                output[out_idx] = Op{}.template operator()<Tout, Tin...>(
                    (typedInputPtr<Tin>(inputs.values[Is.value])[indexer(Is.value)])...);
            },
            std::index_sequence_for<Tin...>{});
    }
}

template <typename Op, typename Tout, typename... Tin>
INFINIOP_CUDA_KERNEL contiguousElementwiseKernel(
    size_t output_size,
    Tout *output,
    InputPointerArray<sizeof...(Tin)> inputs,
    size_t offset) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx < output_size) {
        unpackInputsAndApply(
            [&](auto... Is) {
                output[idx] = Op{}.template operator()<Tout, Tin...>(
                    (typedInputPtr<Tin>(inputs.values[Is.value])[idx])...);
            },
            std::index_sequence_for<Tin...>{});
    }
}

template <typename Op, typename Tout, typename... Tin>
INFINIOP_CUDA_KERNEL inlineMetaElementwiseKernel(
    size_t output_size,
    Tout *output,
    InputPointerArray<sizeof...(Tin)> inputs,
    InlineElementwiseMeta<sizeof...(Tin)> meta,
    size_t offset) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx < output_size) {
        const auto offsets = getInlineElementwiseOffsets(idx, meta);
        unpackInputsAndApply(
            [&](auto... Is) {
                output[offsets.output] = Op{}.template operator()<Tout, Tin...>(
                    (typedInputPtr<Tin>(inputs.values[Is.value])[offsets.inputs[Is.value]])...);
            },
            std::index_sequence_for<Tin...>{});
    }
}

struct DeviceImpl::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
    void *device_meta = nullptr;
    const bool *input_contiguous = nullptr;
    const bool *input_broadcasted = nullptr;
    const size_t *output_shape = nullptr;
    const ptrdiff_t *output_strides = nullptr;
    const size_t *input_shapes = nullptr;
    const ptrdiff_t *input_strides = nullptr;
    infiniStatus_t init_status = INFINI_STATUS_SUCCESS;

    Opaque(const std::shared_ptr<device::nvidia::Handle::Internal> &internal_,
           const op::elementwise::ElementwiseInfo &info)
        : internal(internal_), init_status(initialize(info)) {}

    ~Opaque() {
        if (device_meta != nullptr) {
            cudaFree(device_meta);
        }
    }

    infiniStatus_t initialize(const op::elementwise::ElementwiseInfo &info) {
        if (info.canUseContiguousFastPath() || info.canUseInlineMetaFastPath()) {
            return INFINI_STATUS_SUCCESS;
        }

        const auto meta_size = info.getMetaMemSize();
        if (meta_size == 0) {
            return INFINI_STATUS_SUCCESS;
        }

        CHECK_CUDA(cudaMalloc(&device_meta, meta_size));
        CHECK_CUDA(cudaMemcpy(device_meta,
                              info.getMetaStart(),
                              meta_size,
                              cudaMemcpyHostToDevice));

        const auto ndim = info.getNdim();
        const auto input_size = info.getInputSize();
        output_shape = reinterpret_cast<const size_t *>(device_meta);
        output_strides = reinterpret_cast<const ptrdiff_t *>(output_shape + ndim);
        input_shapes = reinterpret_cast<const size_t *>(output_strides + ndim);
        input_strides = reinterpret_cast<const ptrdiff_t *>(input_shapes + input_size * ndim);
        input_contiguous = reinterpret_cast<const bool *>(input_strides + input_size * ndim);
        input_broadcasted = input_contiguous + input_size;

        return INFINI_STATUS_SUCCESS;
    }

    /**
     * @brief Executes an elementwise operation where all inputs and the output share the same data type.
     *
     * @tparam BLOCK_SIZE    CUDA block size used for kernel launch.
     * @tparam N             Number of input tensors.
     * @tparam Op            Functor representing the elementwise operation.
     * @tparam Tdata         Data type of both input and output tensors.
     * @tparam Args          Optional additional arguments passed to the operation.
     *
     * @param info           Metadata about the operation including shape, size, and dimensionality.
     * @param workspace      Temporary workspace used for storing metadata on device.
     * @param output         Pointer to the output buffer.
     * @param inputs         Vector of pointers to input buffers.
     * @param stream         CUDA stream for asynchronous execution.
     * @param args           Additional arguments forwarded to the operation.
     * @return infiniStatus_t Returns success or failure status.
     */
    template <uint32_t BLOCK_SIZE, size_t N, typename Op, typename Tdata, typename... Args>
    infiniStatus_t calculateImpl(const op::elementwise::ElementwiseInfo &info,
                                 void *workspace,
                                 void *output,
                                 const std::vector<const void *> &inputs,
                                 cudaStream_t stream,
                                 Args &&...args) {
        if (info.canUseContiguousFastPath()) {
            return launchContiguousElementwiseKernel<
                BLOCK_SIZE, N, Op, Tdata, std::decay_t<Args>...>(
                info,
                reinterpret_cast<Tdata *>(output),
                inputs,
                stream,
                std::decay_t<Args>(args)...);
        }
        if (info.canUseInlineMetaFastPath()) {
            return launchInlineMetaElementwiseKernel<
                BLOCK_SIZE, N, Op, Tdata, std::decay_t<Args>...>(
                info,
                reinterpret_cast<Tdata *>(output),
                inputs,
                stream,
                std::decay_t<Args>(args)...);
        }
        return launchElementwiseKernel<BLOCK_SIZE, N>(
            info, workspace,
            reinterpret_cast<Tdata *>(output), inputs,
            elementwiseKernel<N, Op, Tdata, std::decay_t<Args>...>,
            stream,
            std::decay_t<Args>(args)...);
    }

    /**
     * @brief Executes an elementwise operation with mixed input and output data types.
     *
     * @tparam BLOCK_SIZE    CUDA block size used for kernel launch.
     * @tparam N             Number of input tensors.
     * @tparam Op            Functor representing the elementwise operation.
     * @tparam Tout          Data type of the output tensor.
     * @tparam Tin...        Data types of the input tensors.
     * @tparam Args          Optional additional arguments passed to the operation.(UNUSED)
     *
     * @param info           Metadata about the operation including shape, size, and dimensionality.
     * @param workspace      Temporary workspace used for storing metadata on device.
     * @param output         Pointer to the output buffer.
     * @param inputs         Vector of pointers to input buffers.
     * @param stream         CUDA stream for asynchronous execution.
     * @param args           Additional arguments forwarded to the operation.
     * @return infiniStatus_t Returns success or failure status.
     */
    template <uint32_t BLOCK_SIZE, size_t N, typename Op, typename Tout, typename... Tin, typename... Args,
              std::enable_if_t<(sizeof...(Tin) == Op::num_inputs), int> = 0>
    infiniStatus_t calculateImpl(const op::elementwise::ElementwiseInfo &info,
                                 void *workspace,
                                 void *output,
                                 const std::vector<const void *> &inputs,
                                 cudaStream_t stream,
                                 Args &&...args) {
        if (info.canUseContiguousFastPath()) {
            return launchContiguousMixedElementwiseKernel<BLOCK_SIZE, N, Op, Tout, Tin...>(
                info,
                reinterpret_cast<Tout *>(output),
                inputs,
                stream);
        }
        if (info.canUseInlineMetaFastPath()) {
            return launchInlineMetaMixedElementwiseKernel<BLOCK_SIZE, N, Op, Tout, Tin...>(
                info,
                reinterpret_cast<Tout *>(output),
                inputs,
                stream);
        }
        return launchElementwiseKernel<BLOCK_SIZE, N>(
            info, workspace,
            reinterpret_cast<Tout *>(output), inputs,
            elementwiseKernel<Op, Tout, Tin...>,
            stream);
    }

private:
    template <uint32_t BLOCK_SIZE, size_t N, typename Op, typename Tdata, typename... Args>
    infiniStatus_t launchContiguousElementwiseKernel(
        const op::elementwise::ElementwiseInfo &info,
        Tdata *output,
        const std::vector<const void *> &inputs,
        cudaStream_t stream,
        Args &&...args) {

        auto output_size = info.getOutputSize();
        if (output_size == 0) {
            return INFINI_STATUS_SUCCESS;
        }

        InputPointerArray<N> input_ptrs{};
        std::copy_n(inputs.begin(), N, input_ptrs.values);
        dim3 block_dims(std::min(BLOCK_SIZE, static_cast<uint32_t>(internal->maxThreadsPerBlock())));
        dim3 grid_dims(std::min(uint32_t(CEIL_DIV(output_size, block_dims.x)), static_cast<uint32_t>(internal->gridSizeX())));
        size_t step = grid_dims.x * block_dims.x;

        for (size_t i = 0; i < output_size; i += step) {
            contiguousElementwiseKernel<N, Op, Tdata, Args...><<<grid_dims, block_dims, 0, stream>>>(
                output_size, output, input_ptrs, i, std::forward<Args>(args)...);
        }
        return INFINI_STATUS_SUCCESS;
    }

    template <uint32_t BLOCK_SIZE, size_t N, typename Op, typename Tout, typename... Tin>
    infiniStatus_t launchContiguousMixedElementwiseKernel(
        const op::elementwise::ElementwiseInfo &info,
        Tout *output,
        const std::vector<const void *> &inputs,
        cudaStream_t stream) {

        auto output_size = info.getOutputSize();
        if (output_size == 0) {
            return INFINI_STATUS_SUCCESS;
        }

        InputPointerArray<N> input_ptrs{};
        std::copy_n(inputs.begin(), N, input_ptrs.values);
        dim3 block_dims(std::min(BLOCK_SIZE, static_cast<uint32_t>(internal->maxThreadsPerBlock())));
        dim3 grid_dims(std::min(uint32_t(CEIL_DIV(output_size, block_dims.x)), static_cast<uint32_t>(internal->gridSizeX())));
        size_t step = grid_dims.x * block_dims.x;

        for (size_t i = 0; i < output_size; i += step) {
            contiguousElementwiseKernel<Op, Tout, Tin...><<<grid_dims, block_dims, 0, stream>>>(
                output_size, output, input_ptrs, i);
        }
        return INFINI_STATUS_SUCCESS;
    }

    template <uint32_t BLOCK_SIZE, size_t N, typename Op, typename Tdata, typename... Args>
    infiniStatus_t launchInlineMetaElementwiseKernel(
        const op::elementwise::ElementwiseInfo &info,
        Tdata *output,
        const std::vector<const void *> &inputs,
        cudaStream_t stream,
        Args &&...args) {

        auto output_size = info.getOutputSize();
        if (output_size == 0) {
            return INFINI_STATUS_SUCCESS;
        }

        InputPointerArray<N> input_ptrs{};
        std::copy_n(inputs.begin(), N, input_ptrs.values);
        const auto meta = makeInlineElementwiseMeta<N>(info);
        dim3 block_dims(std::min(BLOCK_SIZE, static_cast<uint32_t>(internal->maxThreadsPerBlock())));
        dim3 grid_dims(std::min(uint32_t(CEIL_DIV(output_size, block_dims.x)), static_cast<uint32_t>(internal->gridSizeX())));
        size_t step = grid_dims.x * block_dims.x;

        for (size_t i = 0; i < output_size; i += step) {
            inlineMetaElementwiseKernel<N, Op, Tdata, Args...><<<grid_dims, block_dims, 0, stream>>>(
                output_size, output, input_ptrs, meta, i, std::forward<Args>(args)...);
        }
        return INFINI_STATUS_SUCCESS;
    }

    template <uint32_t BLOCK_SIZE, size_t N, typename Op, typename Tout, typename... Tin>
    infiniStatus_t launchInlineMetaMixedElementwiseKernel(
        const op::elementwise::ElementwiseInfo &info,
        Tout *output,
        const std::vector<const void *> &inputs,
        cudaStream_t stream) {

        auto output_size = info.getOutputSize();
        if (output_size == 0) {
            return INFINI_STATUS_SUCCESS;
        }

        InputPointerArray<N> input_ptrs{};
        std::copy_n(inputs.begin(), N, input_ptrs.values);
        const auto meta = makeInlineElementwiseMeta<N>(info);
        dim3 block_dims(std::min(BLOCK_SIZE, static_cast<uint32_t>(internal->maxThreadsPerBlock())));
        dim3 grid_dims(std::min(uint32_t(CEIL_DIV(output_size, block_dims.x)), static_cast<uint32_t>(internal->gridSizeX())));
        size_t step = grid_dims.x * block_dims.x;

        for (size_t i = 0; i < output_size; i += step) {
            inlineMetaElementwiseKernel<Op, Tout, Tin...><<<grid_dims, block_dims, 0, stream>>>(
                output_size, output, input_ptrs, meta, i);
        }
        return INFINI_STATUS_SUCCESS;
    }

    /**
     * @brief Launches the elementwise kernel for the specified operation.
     *
     * @tparam BLOCK_SIZE   Number of threads per block.
     * @tparam N            Number of input tensors.
     * @tparam KernelFunc   Type of the kernel function pointer.
     * @tparam Tout         Output data type.
     * @tparam Args         Additional arguments to be forwarded to the kernel.
     *
     * @param info          Metadata about the elementwise operation (shapes, strides, etc.).
     * @param workspace     CUDA memory used for storing metadata.
     * @param output        Pointer to output buffer on device.
     * @param inputs        Vector of device pointers to input tensors.
     * @param kernel_func   Kernel function to launch.
     * @param stream        CUDA stream for asynchronous execution.
     * @param args          Additional arguments passed to the kernel.
     * @return infiniStatus_t  Status code indicating success or failure.
     */
    template <uint32_t BLOCK_SIZE, size_t N, typename KernelFunc, typename Tout, typename... Args>
    infiniStatus_t launchElementwiseKernel(
        const op::elementwise::ElementwiseInfo &info,
        void *workspace,
        Tout *output,
        const std::vector<const void *> &inputs,
        KernelFunc kernel_func,
        cudaStream_t stream,
        Args &&...args) {

        auto output_size = info.getOutputSize();
        if (output_size == 0) {
            return INFINI_STATUS_SUCCESS;
        }

        (void)workspace;
        InputPointerArray<N> input_ptrs{};
        std::copy_n(inputs.begin(), N, input_ptrs.values);

        dim3 blockDims(std::min(BLOCK_SIZE, static_cast<uint32_t>(internal->maxThreadsPerBlock())));
        dim3 gridDims(std::min(uint32_t(CEIL_DIV(output_size, blockDims.x)), static_cast<uint32_t>(internal->gridSizeX())));
        size_t step = gridDims.x * blockDims.x;

        for (size_t i = 0; i < output_size; i += step) {
            kernel_func<<<gridDims, blockDims, 0, stream>>>(
                output_size, info.getNdim(), info.isOutputContiguous(),
                input_contiguous, input_broadcasted,
                output_shape, input_shapes,
                output_strides, input_strides,
                output, input_ptrs,
                i, std::forward<Args>(args)...);
        }

        return INFINI_STATUS_SUCCESS;
    }
};

template <typename... Args>
utils::Result<DeviceImpl *> DeviceImpl::create(Args &&...args) {
    auto opaque = std::make_shared<Opaque>(std::forward<Args>(args)...);
    if (opaque->init_status != INFINI_STATUS_SUCCESS) {
        return opaque->init_status;
    }
    return utils::Result<DeviceImpl *>(new DeviceImpl(opaque));
}

/* Invoke elementwise operation for different input types */
template <unsigned int BLOCK_SIZE, typename Op, typename Tout, typename... Tin, typename... Args,
          std::enable_if_t<(sizeof...(Tin) == Op::num_inputs), int>>
infiniStatus_t DeviceImpl::calculate(const op::elementwise::ElementwiseInfo &info,
                                     void *workspace,
                                     void *output,
                                     const std::vector<const void *> &inputs,
                                     void *stream,
                                     Args &&...args) {
    constexpr size_t N = Op::num_inputs;
    static_assert(sizeof...(Tin) == N, "Input type count mismatch");
    return _opaque->calculateImpl<BLOCK_SIZE, N, Op, Tout, Tin...>(
        info, workspace, output, inputs,
        reinterpret_cast<cudaStream_t>(stream),
        std::forward<Args>(args)...);
}

/* Invoke elementwise operation when all inputs have the same dtype */
template <unsigned int BLOCK_SIZE, typename Op, typename Tdata, typename... Args>
infiniStatus_t DeviceImpl::calculate(const op::elementwise::ElementwiseInfo &info,
                                     void *workspace,
                                     void *output,
                                     const std::vector<const void *> &inputs,
                                     void *stream,
                                     Args &&...args) {
    constexpr size_t N = Op::num_inputs;
    return _opaque->calculateImpl<BLOCK_SIZE, N, Op, Tdata>(
        info, workspace, output, inputs,
        reinterpret_cast<cudaStream_t>(stream),
        std::forward<Args>(args)...);
}

} // namespace op::elementwise::nvidia

#endif // __INFINIOP_ELEMENTWISE_CUDA_H__
