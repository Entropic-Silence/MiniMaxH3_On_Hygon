#include "infinicore/ops/situ_and_mul.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(SituAndMul);

SituAndMul::SituAndMul(Tensor output,
                       const Tensor &gate,
                       const Tensor &up,
                       float beta,
                       float linear_beta) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, gate, up);
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(), output, gate, up, beta, linear_beta);
}

void SituAndMul::execute(Tensor output,
                         const Tensor &gate,
                         const Tensor &up,
                         float beta,
                         float linear_beta) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(SituAndMul, output, gate, up, beta, linear_beta);
}

Tensor situ_and_mul(const Tensor &gate,
                    const Tensor &up,
                    float beta,
                    float linear_beta) {
    auto output = Tensor::empty(gate->shape(), gate->dtype(), gate->device());
    situ_and_mul_(output, gate, up, beta, linear_beta);
    return output;
}

void situ_and_mul_(Tensor output,
                   const Tensor &gate,
                   const Tensor &up,
                   float beta,
                   float linear_beta) {
    if (gate->shape() != up->shape() || gate->shape() != output->shape()) {
        throw std::runtime_error("situ_and_mul expects output, gate, and up to have the same shape");
    }
    if (gate->dtype() != up->dtype() || gate->dtype() != output->dtype()) {
        throw std::runtime_error("situ_and_mul expects output, gate, and up to have the same dtype");
    }
    if (beta <= 0.0f || linear_beta <= 0.0f) {
        throw std::runtime_error("situ_and_mul expects positive beta values");
    }
    SituAndMul::execute(output, gate, up, beta, linear_beta);
}

} // namespace infinicore::op
