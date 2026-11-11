#ifndef __INFINIOP_ELEMENTWISE_BANG_H__
#define __INFINIOP_ELEMENTWISE_BANG_H__

#include "../../../utils.h"
#include "../../devices/bang/common_bang.h"
#include "elementwise_bang_api.h"

namespace op::elementwise::bang {

/**
 * @brief Opaque implementation structure for BANG device operations.
 *
 * Contains device-specific resources and implementation methods.
 */
struct DeviceImpl::Opaque {
    std::shared_ptr<device::bang::Handle::Internal> internal;
    void *device_meta = nullptr;
    const bool *input_contiguous = nullptr;
    const bool *input_broadcasted = nullptr;
    const size_t *output_shape = nullptr;
    const ptrdiff_t *output_strides = nullptr;
    const size_t *input_shapes = nullptr;
    const ptrdiff_t *input_strides = nullptr;
    infiniStatus_t init_status = INFINI_STATUS_SUCCESS;

    /**
     * @brief Constructs an Opaque instance with device handle internals.
     *
     * @param internal_ Shared pointer to BANG device handle internals.
     * @param info Elementwise metadata to persist on the device.
     */
    Opaque(const std::shared_ptr<device::bang::Handle::Internal> &internal_,
           const op::elementwise::ElementwiseInfo &info)
        : internal(internal_), init_status(initialize(info)) {}

    ~Opaque() {
        if (device_meta != nullptr) {
            cnrtFree(device_meta);
        }
    }

    infiniStatus_t initialize(const op::elementwise::ElementwiseInfo &info) {
        const auto meta_size = info.getMetaMemSize();
        if (meta_size == 0) {
            return INFINI_STATUS_SUCCESS;
        }

        CHECK_INTERNAL(cnrtMalloc(&device_meta, meta_size), cnrtSuccess);
        CHECK_INTERNAL(cnrtMemcpy(device_meta,
                                  const_cast<int8_t *>(info.getMetaStart()),
                                  meta_size,
                                  cnrtMemcpyHostToDev),
                       cnrtSuccess);

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
     * @brief Implements elementwise calculation for BANG device.
     *
     * @tparam N        Number of input tensors.
     * @tparam Op       Operator functor type.
     * @tparam Tdata    Data type for inputs and output.
     * @tparam Args     Additional arguments for the operator.
     *
     * @param info      Elementwise operation metadata (shapes, strides, etc.).
     * @param workspace Device workspace memory.
     * @param output    Output tensor buffer.
     * @param inputs    Vector of input tensor pointers.
     * @param queue     BANG queue for asynchronous execution.
     * @param args      Additional arguments for the operator.
     * @return infiniStatus_t Status indicating success or failure.
     */
    template <size_t N, typename Op, typename Tdata, typename... Args>
    infiniStatus_t calculateImpl(const op::elementwise::ElementwiseInfo &info,
                                 void *workspace,
                                 void *output,
                                 const std::vector<const void *> &inputs,
                                 cnrtQueue_t queue,
                                 Args &&...args) {
        auto output_size = info.getOutputSize();
        if (output_size == 0) {
            return INFINI_STATUS_SUCCESS;
        }

        CHECK_OR_RETURN(inputs.size() == N, INFINI_STATUS_BAD_PARAM);
        (void)workspace;

        return Op::template launch<Tdata>(
            output_size,
            info.getNdim(),
            info.isOutputContiguous(),
            reinterpret_cast<const void *>(input_contiguous),
            reinterpret_cast<const void *>(input_broadcasted),
            reinterpret_cast<const void *>(output_shape),
            reinterpret_cast<const void *>(input_shapes),
            reinterpret_cast<const void *>(output_strides),
            reinterpret_cast<const void *>(input_strides),
            output,
            inputs.data(),
            queue,
            internal,
            args...);
    }
};

/**
 * @brief Creates a DeviceImpl instance for BANG device.
 *
 * @tparam Args Argument types for Opaque construction.
 * @param args Arguments forwarded to Opaque constructor.
 * @return utils::Result<DeviceImpl*> Result containing new DeviceImpl instance.
 */
template <typename... Args>
utils::Result<DeviceImpl *> DeviceImpl::create(Args &&...args) {
    auto opaque = std::make_shared<Opaque>(std::forward<Args>(args)...);
    if (opaque->init_status != INFINI_STATUS_SUCCESS) {
        return opaque->init_status;
    }
    return utils::Result<DeviceImpl *>(new DeviceImpl(opaque));
}

/**
 * @brief Calculates elementwise operation for BANG device.
 *
 * @tparam Op       Operator functor type.
 * @tparam Tdata    Data type for inputs and output.
 * @tparam Args     Additional arguments for the operator.
 *
 * @param info      Elementwise operation metadata.
 * @param workspace Device workspace memory.
 * @param output    Output tensor buffer.
 * @param inputs    Vector of input tensor pointers.
 * @param queue     BANG queue (as void*).
 * @param args      Additional arguments for the operator.
 * @return infiniStatus_t Status indicating success or failure.
 */
template <typename Op, typename Tdata, typename... Args>
infiniStatus_t DeviceImpl::calculate(const op::elementwise::ElementwiseInfo &info,
                                     void *workspace,
                                     void *output,
                                     const std::vector<const void *> &inputs,
                                     void *queue,
                                     Args &&...args) {
    constexpr size_t N = Op::num_inputs;
    return _opaque->calculateImpl<N, Op, Tdata>(
        info, workspace, output, inputs,
        reinterpret_cast<cnrtQueue_t>(queue),
        std::forward<Args>(args)...);
}
} // namespace op::elementwise::bang

/**
 * @brief Macro for declaring BANG kernel interface.
 *
 * @param OpName Name of the elementwise operation.
 */
#define LAUNCH_ELEMENTWISE_KERNEL(OpName)                                \
    template <typename Tdata, typename... Args>                          \
    void launch##OpName##Kernel(                                         \
        size_t output_size,                                              \
        size_t ndim,                                                     \
        bool output_contiguous,                                          \
        const void *input_contiguous,                                    \
        const void *input_broadcasted,                                   \
        const void *output_shape,                                        \
        const void *input_shapes,                                        \
        const void *output_strides,                                      \
        const void *input_strides,                                       \
        void *output,                                                    \
        const void *const *inputs,                                       \
        cnrtQueue_t queue,                                               \
        const std::shared_ptr<device::bang::Handle::Internal> &internal, \
        Args... args);

#endif // __INFINIOP_ELEMENTWISE_BANG_H__
