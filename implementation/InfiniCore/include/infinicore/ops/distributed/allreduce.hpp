#pragma once

#include "../../device.hpp"
#include "../../graph/graph.hpp"
#include "../common/op.hpp"

#include <infiniccl.h>
#include <infinirt.h>

namespace infinicore::op::distributed {
class AllReduce : public graph::GraphOperator {
public:
    AllReduce(Tensor output, const Tensor &input, infinicclReduceOp_t op, infinicclComm_t communicator);
    ~AllReduce();
    void run() const override;
    static void execute(Tensor output, const Tensor &input, infinicclReduceOp_t op, infinicclComm_t communicator);

private:
    void *planned_meta_;
};

Tensor allreduce(const Tensor &input, infinicclReduceOp_t op, infinicclComm_t communicator);
void allreduce_(Tensor output, const Tensor &input, infinicclReduceOp_t op, infinicclComm_t communicator);

// Same, issued on `stream` instead of the current runtime stream so it can
// overlap computation; the caller orders producers and consumers with events
// and keeps both tensors alive until the consumer has waited.
void allreduce_on_stream_(Tensor output, const Tensor &input, infinicclReduceOp_t op,
                          infinicclComm_t communicator, infinirtStream_t stream);

} // namespace infinicore::op::distributed
