#ifndef __ZEROS_BANG_API_H__
#define __ZEROS_BANG_API_H__

#include "../../../elementwise/bang/elementwise_bang.h"

namespace op::zeros::bang {

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    op::elementwise::ElementwiseInfo _info;
    std::unique_ptr<op::elementwise::bang::DeviceImpl> _device_info;
    std::shared_ptr<device::bang::Handle::Internal> _internal;
    cnnlTensorDescriptor_t _value_desc;
    cnnlTensorDescriptor_t _output_desc;
    size_t _workspace_size;

    Descriptor(
        infiniDtype_t dtype,
        op::elementwise::ElementwiseInfo info,
        op::elementwise::bang::DeviceImpl *device_info,
        std::shared_ptr<device::bang::Handle::Internal> internal,
        cnnlTensorDescriptor_t value_desc,
        cnnlTensorDescriptor_t output_desc,
        size_t workspace_size,
        infiniDevice_t device_type,
        int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _dtype(dtype),
          _info(std::move(info)),
          _device_info(device_info),
          _internal(std::move(internal)),
          _value_desc(value_desc),
          _output_desc(output_desc),
          _workspace_size(workspace_size) {}

public:
    ~Descriptor();

    size_t workspaceSize() const { return _workspace_size; }

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t output_desc,
        std::vector<infiniopTensorDescriptor_t> input_descs);

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *output,
        std::vector<const void *> inputs,
        void *stream) const;
};

} // namespace op::zeros::bang

#endif // __ZEROS_BANG_API_H__
