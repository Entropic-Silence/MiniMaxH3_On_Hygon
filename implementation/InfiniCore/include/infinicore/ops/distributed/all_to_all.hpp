#pragma once

#include "../../device.hpp"
#include "../../tensor.hpp"

#include <infiniccl.h>
#include <infinirt.h>

namespace infinicore::op::distributed {

// Equal-split all-to-all along dim 0. `input` and `output` are contiguous with
// the same shape and dtype, and dim 0 is a multiple of `world_size`: the p-th
// block of `input` is sent to rank p, and the q-th block of `output` receives
// rank q's block for this rank. Built from grouped point-to-point transfers;
// the block for this rank is copied locally.
void all_to_all_(Tensor output, const Tensor &input, int rank, int world_size,
                 infinicclComm_t communicator);

// Same, issued on `stream` instead of the current runtime stream so it can
// overlap computation; the caller orders producers and consumers with events
// and keeps both tensors alive until the consumer has waited.
void all_to_all_on_stream_(Tensor output, const Tensor &input, int rank, int world_size,
                           infinicclComm_t communicator, infinirtStream_t stream);

// One grouped point-to-point step on `stream`: `send` goes to rank
// `send_peer` while `recv` is filled from rank `recv_peer` (both contiguous,
// same dtype). A rank sequence where every rank sends to (rank + k) and
// receives from (rank - k) in step k pairs up without deadlock; used to return
// attention outputs destination by destination as soon as each is computed.
void send_recv_on_stream_(const Tensor &send, int send_peer, Tensor recv, int recv_peer,
                          infinicclComm_t communicator, infinirtStream_t stream);

} // namespace infinicore::op::distributed
