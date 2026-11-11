#include "infinicore/ops/distributed/all_to_all.hpp"
#include "../../utils.hpp"
#include "infinicore/ops/distributed/collective_sync.hpp"

#include "infinicore/context/context.hpp"

namespace infinicore::op::distributed {
namespace {

void issue_all_to_all(Tensor output, const Tensor &input, int rank, int world_size,
                      infinicclComm_t communicator, infinirtStream_t stream) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    INFINICORE_ASSERT(output->is_contiguous() && input->is_contiguous());
    INFINICORE_ASSERT(output->dtype() == input->dtype());
    INFINICORE_ASSERT(output->numel() == input->numel());
    INFINICORE_ASSERT(world_size > 0 && rank >= 0 && rank < world_size);
    INFINICORE_ASSERT(input->numel() % static_cast<Size>(world_size) == 0);

    const Size count = input->numel() / static_cast<Size>(world_size);
    const Size bytes = count * dsize(input->dtype());
    const auto dtype = static_cast<infiniDtype_t>(static_cast<int>(input->dtype()));
    const auto *send = static_cast<const std::byte *>(static_cast<const void *>(input->data()));
    auto *recv = static_cast<std::byte *>(static_cast<void *>(output->data()));

    if (count > 0) {
        INFINICORE_CHECK_ERROR(infinirtMemcpyAsync(recv + rank * bytes, send + rank * bytes, bytes,
                                                   INFINIRT_MEMCPY_D2D, stream));
    }
    if (world_size > 1 && count > 0) {
        INFINICORE_CHECK_ERROR(infinicclGroupStart(communicator));
        for (int peer = 0; peer < world_size; ++peer) {
            if (peer == rank) {
                continue;
            }
            INFINICORE_CHECK_ERROR(infinicclSend(send + peer * bytes, count, dtype, peer,
                                                 communicator, stream));
            INFINICORE_CHECK_ERROR(infinicclRecv(recv + peer * bytes, count, dtype, peer,
                                                 communicator, stream));
        }
        INFINICORE_CHECK_ERROR(infinicclGroupEnd(communicator));
    }
}

} // namespace

void all_to_all_(Tensor output, const Tensor &input, int rank, int world_size,
                 infinicclComm_t communicator) {
    release_launch_guard();
    issue_all_to_all(output, input, rank, world_size, communicator,
                     infinicore::context::getStream());
    finish_collective();
}

void all_to_all_on_stream_(Tensor output, const Tensor &input, int rank, int world_size,
                           infinicclComm_t communicator, infinirtStream_t stream) {
    release_launch_guard();
    issue_all_to_all(output, input, rank, world_size, communicator, stream);
    if (synchronous_collectives()) {
        INFINICORE_CHECK_ERROR(infinirtStreamSynchronize(stream));
    }
}

void send_recv_on_stream_(const Tensor &send, int send_peer, Tensor recv, int recv_peer,
                          infinicclComm_t communicator, infinirtStream_t stream) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(recv, send);
    INFINICORE_ASSERT(recv->is_contiguous() && send->is_contiguous());
    INFINICORE_ASSERT(recv->dtype() == send->dtype());
    const auto dtype = static_cast<infiniDtype_t>(static_cast<int>(send->dtype()));
    release_launch_guard();
    INFINICORE_CHECK_ERROR(infinicclGroupStart(communicator));
    if (send->numel() > 0) {
        INFINICORE_CHECK_ERROR(infinicclSend(const_cast<std::byte *>(static_cast<const std::byte *>(send->data())),
                                             send->numel(), dtype, send_peer, communicator, stream));
    }
    if (recv->numel() > 0) {
        INFINICORE_CHECK_ERROR(infinicclRecv(recv->data(), recv->numel(), dtype, recv_peer,
                                             communicator, stream));
    }
    INFINICORE_CHECK_ERROR(infinicclGroupEnd(communicator));
    if (synchronous_collectives()) {
        INFINICORE_CHECK_ERROR(infinirtStreamSynchronize(stream));
    }
}

} // namespace infinicore::op::distributed
