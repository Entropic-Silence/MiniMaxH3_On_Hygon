#ifndef __SITU_AND_MUL_H__
#define __SITU_AND_MUL_H__

#include "../../elementwise/elementwise.h"

#define SITU_AND_MUL_DESCRIPTOR(NAMESPACE)                                    \
                                                                              \
    namespace op::situ_and_mul::NAMESPACE {                                   \
    class Descriptor final : public InfiniopDescriptor {                      \
        infiniDtype_t _dtype;                                                 \
        op::elementwise::ElementwiseInfo _info;                               \
        std::unique_ptr<op::elementwise::NAMESPACE::DeviceImpl> _device_info; \
        size_t _workspace_size;                                               \
                                                                              \
        Descriptor(                                                           \
            infiniDtype_t dtype,                                              \
            op::elementwise::ElementwiseInfo info,                            \
            op::elementwise::NAMESPACE::DeviceImpl *device_info,              \
            size_t workspace_size,                                            \
            infiniDevice_t device_type,                                       \
            int device_id)                                                    \
            : InfiniopDescriptor{device_type, device_id},                     \
              _dtype(dtype),                                                  \
              _info(std::move(info)),                                         \
              _device_info(std::move(device_info)),                           \
              _workspace_size(workspace_size) {}                              \
                                                                              \
    public:                                                                   \
        ~Descriptor();                                                        \
                                                                              \
        size_t workspaceSize() const { return _workspace_size; }              \
                                                                              \
        static infiniStatus_t create(                                         \
            infiniopHandle_t handle,                                          \
            Descriptor **desc_ptr,                                            \
            infiniopTensorDescriptor_t output_desc,                           \
            infiniopTensorDescriptor_t gate_desc,                             \
            infiniopTensorDescriptor_t up_desc);                              \
                                                                              \
        infiniStatus_t calculate(                                             \
            void *workspace,                                                  \
            size_t workspace_size,                                            \
            void *output,                                                     \
            const void *gate,                                                 \
            const void *up,                                                   \
            float beta,                                                       \
            float linear_beta,                                                \
            void *stream) const;                                              \
    };                                                                        \
    }

#endif
