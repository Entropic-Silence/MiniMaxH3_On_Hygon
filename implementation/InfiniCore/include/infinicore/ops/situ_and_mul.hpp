#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(SituAndMul, Tensor, const Tensor &, const Tensor &, float, float);

Tensor situ_and_mul(const Tensor &gate,
                    const Tensor &up,
                    float beta = 4.0f,
                    float linear_beta = 25.0f);

void situ_and_mul_(Tensor output,
                   const Tensor &gate,
                   const Tensor &up,
                   float beta = 4.0f,
                   float linear_beta = 25.0f);

} // namespace infinicore::op
