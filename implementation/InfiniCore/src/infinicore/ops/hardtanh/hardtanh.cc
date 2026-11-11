#include "infinicore/ops/hardtanh.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(HardTanh);

HardTanh::HardTanh(Tensor output, Tensor input, float min_val, float max_val) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(), output, input, min_val, max_val);
}

void HardTanh::execute(Tensor output, Tensor input, float min_val, float max_val) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(HardTanh, output, input, min_val, max_val);
}

Tensor hardtanh(Tensor input, float min_val, float max_val) {
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    hardtanh_(output, input, min_val, max_val);
    return output;
}

void hardtanh_(Tensor output, Tensor input, float min_val, float max_val) {
    HardTanh::execute(output, input, min_val, max_val);
}

} // namespace infinicore::op
