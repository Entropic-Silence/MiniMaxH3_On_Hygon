#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(HardTanh, Tensor, Tensor, float, float);

Tensor hardtanh(Tensor input, float min_val = -1.0f, float max_val = 1.0f);
void hardtanh_(Tensor output, Tensor input, float min_val = -1.0f, float max_val = 1.0f);

} // namespace infinicore::op
