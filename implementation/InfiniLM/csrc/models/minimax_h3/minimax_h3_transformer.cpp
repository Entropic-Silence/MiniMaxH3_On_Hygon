#include "minimax_h3_transformer.hpp"

#include "../../global_state/global_state.hpp"
#include "../models_registry.hpp"

#include <infinicore/ops/add.hpp>
#include <infinicore/ops/cast.hpp>
#include <infinicore/ops/distributed/all_to_all.hpp>
#include <infinicore/ops/distributed/allgather.hpp>
#include <infinicore/ops/distributed/allreduce.hpp>
#include <infinicore/ops/distributed/collective_sync.hpp>
#include <infinicore/ops/embedding.hpp>
#include <infinicore/ops/index_copy.hpp>
#include <infinicore/ops/h3_numerics.hpp>
#include <infinicore/ops/mha.hpp>
#include <infinicore/ops/mrope.hpp>
#include <infinicore/ops/mul.hpp>
#include <infinicore/ops/rearrange.hpp>
#include <infinicore/ops/silu.hpp>
#include <infinicore/ops/swiglu.hpp>
#include <infinicore/ops/timestep_embedding.hpp>
#include <infinicore/context/context.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace infinilm::models::minimax_h3 {
namespace {

constexpr size_t H3_MODALITY_COUNT = 3;
thread_local int trace_block = -1;
// Sequence parallel: number of real (unpadded) tokens of the packed sequence,
// set by MiniMaxH3Transformer::forward for its rank thread.
thread_local size_t sp_valid_tokens = 0;

using RankInfo = infinilm::engine::distributed::RankInfo;

// k100-extreme: Ulysses sequence parallelism by default on four or more
// ranks (fastest on K100 x4 over PCIe); fewer ranks keep tensor parallelism,
// whose smaller per-card weights leave room for activations (two 64 GB cards).
// H3_PARALLEL=tp|sp forces either.
MiniMaxH3Parallel block_parallel_mode(const RankInfo &rank_info, size_t num_heads) {
    const char *mode = std::getenv("H3_PARALLEL");
    if (mode != nullptr && std::strcmp(mode, "tp") != 0 && std::strcmp(mode, "sp") != 0) {
        throw std::invalid_argument("H3_PARALLEL must be tp or sp");
    }
    const bool sequence = mode != nullptr ? std::strcmp(mode, "sp") == 0 : rank_info.tp_size >= 4;
    if (!sequence || rank_info.tp_size <= 1 || num_heads % static_cast<size_t>(rank_info.tp_size) != 0) {
        return MiniMaxH3Parallel::TENSOR;
    }
    return MiniMaxH3Parallel::SEQUENCE;
}

// Rank layout for layers holding full (unsharded) weights.
RankInfo layer_rank_info(MiniMaxH3Parallel parallel) {
    RankInfo info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    if (parallel != MiniMaxH3Parallel::TENSOR) {
        info.tp_size = 1;
        info.tp_rank = 0;
    }
    return info;
}

MiniMaxH3Parallel refiner_parallel_mode(MiniMaxH3Parallel blocks) {
    return blocks == MiniMaxH3Parallel::SEQUENCE ? MiniMaxH3Parallel::REPLICATED : blocks;
}

void check_rt(infiniStatus_t status, const char *what) {
    if (status != INFINI_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("MiniMaxH3: ") + what + " failed: "
                                 + std::to_string(static_cast<int>(status)));
    }
}

// Side stream and reusable events of one device for overlapped sequence-parallel
// exchanges; owned by the rank thread driving that device.
struct SpSideStream {
    infinirtStream_t stream = nullptr;
    std::vector<infinirtEvent_t> events;
    size_t next = 0;

    infinirtEvent_t event() {
        if (next == events.size()) {
            infinirtEvent_t created = nullptr;
            check_rt(infinirtEventCreateWithFlags(&created, INFINIRT_EVENT_DISABLE_TIMING),
                     "infinirtEventCreateWithFlags");
            events.push_back(created);
        }
        return events[next++];
    }
};

SpSideStream &sp_side_stream() {
    thread_local std::unordered_map<int, SpSideStream> streams;
    auto &side = streams[infinicore::context::getDevice().getIndex()];
    if (side.stream == nullptr) {
        check_rt(infinirtStreamCreate(&side.stream), "infinirtStreamCreate");
    }
    return side;
}

// Every event is recorded once per forward. Re-recording an event while a
// stream still has an unexecuted wait on it is unsafe on DTK's HIP (events
// map to reused HSA signals, so the pending wait can latch onto the new
// record, which may sit behind the waiter itself: a device-side deadlock with
// all devices idle, seen once in about a dozen runs). The pool is rewound only
// at the start of a forward; the caller synchronizes the devices between
// forwards (the outputs are copied back), so every earlier wait has executed.
void rewind_sp_events() {
    sp_side_stream().next = 0;
}

// ---------------------------------------------------------------------------
// Peer-copy all-to-all (H3_SP_A2A=peer).
//
// The engine's ranks are threads of one process, so an all-to-all can be a set
// of device-to-device copies straight into the peers' receive buffers, run by
// the copy engines instead of RCCL kernels: no compute units are taken from the
// GEMMs and attention it overlaps, and no kernel spins while a peer is late
// (the DCUs run at their power limit). Data movement only; every byte lands
// exactly where the RCCL send/recv version puts it.
//
// Per call (all ranks call in the same order, as with any collective):
//   1. each rank records `ready` on its exchange stream (its receive buffers
//      may be written from then on) and publishes the buffers, then a host
//      barrier;
//   2. each rank copies its block for peer p into p's receive buffers on a
//      per-peer copy stream that waits for p's `ready`, joins those streams and
//      records `done`, then a second barrier;
//   3. each rank's stream waits for every peer's `done` before reading.
// Fields read by peers after a barrier are not rewritten before the next
// call's first barrier, which every rank must reach first. Events come from
// the per-forward pool, so no event is re-recorded while a wait on it pends.
class SpinBarrier {
public:
    explicit SpinBarrier(size_t count) : count_(count) {}
    void arrive_and_wait() {
        const size_t generation = generation_.load(std::memory_order_acquire);
        if (arrived_.fetch_add(1, std::memory_order_acq_rel) + 1 == count_) {
            arrived_.store(0, std::memory_order_relaxed);
            generation_.fetch_add(1, std::memory_order_acq_rel);
            return;
        }
        for (size_t spins = 0; generation_.load(std::memory_order_acquire) == generation; ++spins) {
            if (spins > 1024) {
                std::this_thread::yield();
            }
        }
    }

private:
    const size_t count_;
    std::atomic<size_t> arrived_{0};
    std::atomic<size_t> generation_{0};
};

struct PeerExchange {
    struct Slot {
        std::vector<std::byte *> recv;
        infinirtEvent_t ready = nullptr;
        infinirtEvent_t done = nullptr;
    };
    explicit PeerExchange(size_t world) : barrier(world), slots(world) {}
    SpinBarrier barrier;
    std::vector<Slot> slots;
};

// One exchange per engine: its ranks share the model configuration object.
PeerExchange &peer_exchange(const void *engine_key, size_t world) {
    static std::mutex mutex;
    static std::unordered_map<const void *, std::unique_ptr<PeerExchange>> registry;
    std::lock_guard<std::mutex> lock(mutex);
    auto &entry = registry[engine_key];
    if (!entry) {
        entry = std::make_unique<PeerExchange>(world);
    }
    return *entry;
}

PeerExchange &sp_exchange(size_t world) {
    thread_local const void *cached_key = nullptr;
    thread_local PeerExchange *cached = nullptr;
    const void *key = &infinilm::global_state::get_infinilm_config();
    if (cached_key != key) {
        cached = &peer_exchange(key, world);
        cached_key = key;
    }
    return *cached;
}

bool sp_peer_a2a_enabled() {
    const char *mode = std::getenv("H3_SP_A2A");
    if (mode == nullptr || std::strcmp(mode, "ccl") == 0) return false;
    if (std::strcmp(mode, "peer") == 0) return true;
    throw std::invalid_argument("H3_SP_A2A must be ccl or peer");
}

infinirtStream_t copy_stream(size_t index) {
    thread_local std::unordered_map<int, std::vector<infinirtStream_t>> streams;
    auto &list = streams[infinicore::context::getDevice().getIndex()];
    while (list.size() <= index) {
        infinirtStream_t created = nullptr;
        check_rt(infinirtStreamCreate(&created), "infinirtStreamCreate");
        list.push_back(created);
    }
    return list[index];
}

// recv[i] / send[i]: contiguous [world, ...]; block p of send[i] goes to rank
// p's recv[i] block `rank`, issued on `stream`.
void peer_all_to_all(PeerExchange &exchange, int rank, size_t world,
                     const std::vector<infinicore::Tensor> &recv,
                     const std::vector<infinicore::Tensor> &send, infinirtStream_t stream) {
    auto &side = sp_side_stream();
    std::vector<size_t> bytes(send.size());
    for (size_t i = 0; i < send.size(); ++i) {
        if (!recv[i]->is_contiguous() || !send[i]->is_contiguous()
            || recv[i]->numel() != send[i]->numel() || send[i]->numel() % world != 0) {
            throw std::runtime_error("peer_all_to_all: buffers must be contiguous and evenly split");
        }
        bytes[i] = send[i]->numel() / world * infinicore::dsize(send[i]->dtype());
    }
    auto &mine = exchange.slots[rank];
    mine.recv.resize(recv.size());
    for (size_t i = 0; i < recv.size(); ++i) {
        mine.recv[i] = const_cast<std::byte *>(static_cast<const std::byte *>(recv[i]->data()));
    }
    mine.ready = side.event();
    check_rt(infinirtEventRecord(mine.ready, stream), "infinirtEventRecord");
    exchange.barrier.arrive_and_wait();

    for (size_t i = 0; i < send.size(); ++i) {
        const auto *source = reinterpret_cast<const std::byte *>(send[i]->data());
        check_rt(infinirtMemcpyAsync(mine.recv[i] + rank * bytes[i], source + rank * bytes[i],
                                     bytes[i], INFINIRT_MEMCPY_D2D, stream),
                 "infinirtMemcpyAsync");
    }
    auto fork = side.event();
    check_rt(infinirtEventRecord(fork, stream), "infinirtEventRecord");
    for (size_t k = 1; k < world; ++k) {
        const size_t peer = (static_cast<size_t>(rank) + k) % world;
        auto copies = copy_stream(k - 1);
        check_rt(infinirtStreamWaitEvent(copies, fork), "infinirtStreamWaitEvent");
        check_rt(infinirtStreamWaitEvent(copies, exchange.slots[peer].ready), "infinirtStreamWaitEvent");
        for (size_t i = 0; i < send.size(); ++i) {
            const auto *source = reinterpret_cast<const std::byte *>(send[i]->data());
            check_rt(infinirtMemcpyAsync(exchange.slots[peer].recv[i] + rank * bytes[i],
                                         source + peer * bytes[i], bytes[i],
                                         INFINIRT_MEMCPY_D2D, copies),
                     "infinirtMemcpyAsync");
        }
        auto joined = side.event();
        check_rt(infinirtEventRecord(joined, copies), "infinirtEventRecord");
        check_rt(infinirtStreamWaitEvent(stream, joined), "infinirtStreamWaitEvent");
    }
    mine.done = side.event();
    check_rt(infinirtEventRecord(mine.done, stream), "infinirtEventRecord");
    exchange.barrier.arrive_and_wait();

    for (size_t peer = 0; peer < world; ++peer) {
        if (peer != static_cast<size_t>(rank)) {
            check_rt(infinirtStreamWaitEvent(stream, exchange.slots[peer].done), "infinirtStreamWaitEvent");
        }
    }
}

// `later` waits for everything issued on `earlier` so far.
void order_streams(SpSideStream &side, infinirtStream_t earlier, infinirtStream_t later) {
    auto event = side.event();
    check_rt(infinirtEventRecord(event, earlier), "infinirtEventRecord");
    check_rt(infinirtStreamWaitEvent(later, event), "infinirtStreamWaitEvent");
}

// H3_SP_OVERLAP=G (default 2) splits each rank's heads into G groups whose
// exchanges run on a side stream while other groups' attention computes;
// 1 exchanges all heads synchronously. Per-head results are identical.
size_t sp_overlap_groups(size_t local_heads) {
    const char *value = std::getenv("H3_SP_OVERLAP");
    const size_t groups = value ? static_cast<size_t>(std::stoul(value)) : 2;
    return groups > 1 && local_heads % groups == 0 ? groups : 1;
}

// H3_SP_GROUPS="a,b,..." (summing to the heads per rank) sets possibly uneven
// head groups for the pipelined path, e.g. "4,5,5": a small first group whose
// exchange hides under the remaining groups' projections. Otherwise
// H3_SP_OVERLAP equal groups. Per-head results are identical either way.
std::vector<size_t> sp_head_groups(size_t local_heads) {
    const char *spec = std::getenv("H3_SP_GROUPS");
    if (spec != nullptr && *spec != '\0') {
        std::vector<size_t> sizes;
        size_t total = 0;
        const std::string text(spec);
        for (size_t start = 0; start <= text.size();) {
            size_t end = text.find(',', start);
            if (end == std::string::npos) end = text.size();
            const size_t value = static_cast<size_t>(std::stoul(text.substr(start, end - start)));
            if (value == 0) throw std::invalid_argument("H3_SP_GROUPS entries must be positive");
            sizes.push_back(value);
            total += value;
            start = end + 1;
        }
        if (total != local_heads) {
            throw std::invalid_argument("H3_SP_GROUPS must sum to the heads per rank ("
                                        + std::to_string(local_heads) + ")");
        }
        return sizes;
    }
    const size_t groups = sp_overlap_groups(local_heads);
    return std::vector<size_t>(groups, local_heads / groups);
}

// H3_SP_CHUNKED_OUTPUT=1 (opt-in): the last head group's attention runs per
// destination rank (peers' token rows first, the local rows last) and each
// slice is sent as soon as it is computed, so only the local rows' attention
// remains after the final send. Measured on K100 x4: 0.8% slower (smaller
// attention launches, per-step send/recv) and not bitwise identical (the
// vendor flash kernel's per-row arithmetic depends on the query offset).
bool sp_chunked_output_enabled() {
    const char *value = std::getenv("H3_SP_CHUNKED_OUTPUT");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

// H3_SP_PIPELINE=1 (k100-extreme default) also splits the QKV projection by
// those head groups, so each group's exchange overlaps the next projection.
bool sp_pipeline_enabled() {
    const char *value = std::getenv("H3_SP_PIPELINE");
    return value == nullptr || std::strcmp(value, "0") != 0;
}

// H3_ADALN_BATCH=1 (default): every block's AdaLN modulation is projected at
// the start of the forward and all-gathered in one collective (instead of one
// synchronous all-gather per block), and the AdaLN operators run per segment
// of tokens sharing a modulation row on broadcast views (instead of on a
// per-token gathered [tokens, 6 * hidden] table). Same operators on the same
// values, bitwise identical; 0 restores the per-block path.
bool adaln_batch_enabled() {
    const char *value = std::getenv("H3_ADALN_BATCH");
    if (value == nullptr || std::strcmp(value, "1") == 0) return true;
    if (std::strcmp(value, "0") == 0) return false;
    throw std::invalid_argument("H3_ADALN_BATCH must be 0 or 1");
}

// H3_FUSED_QK=1 (k100-extreme default): fused q/k norm + RoPE + pack in the
// pipelined sequence-parallel path.
bool fused_qk_enabled() {
    const char *value = std::getenv("H3_FUSED_QK");
    return value == nullptr || std::strcmp(value, "0") != 0;
}

// H3_TP_OVERLAP=C (default 1, off) pipelines each tensor-parallel block's
// post-attention part over C row chunks so all-reduces overlap computation.
size_t tp_overlap_chunks() {
    const char *value = std::getenv("H3_TP_OVERLAP");
    return value ? static_cast<size_t>(std::stoul(value)) : 1;
}

// Copies `source` into rows [0, source.size(0)) of a zero tensor with `rows`
// rows (dim 0), used to pad per-token tables to ranks * local_tokens.
infinicore::Tensor pad_rows(const infinicore::Tensor &source, size_t rows) {
    auto shape = source->shape();
    if (shape[0] == rows) {
        return source;
    }
    shape[0] = rows;
    auto padded = infinicore::Tensor::zeros(shape, source->dtype(), source->device());
    padded->narrow({{0, 0, source->size(0)}})->copy_from(source);
    return padded;
}

bool pointwise_enabled(const char *operation) {
    const char *mode = std::getenv("H3_POINTWISE_NUMERICS");
    if (mode == nullptr || std::strcmp(mode, "baseline") == 0) return false;
    if (std::strcmp(mode, "exact_ref") == 0 || std::strcmp(mode, operation) == 0) return true;
    if (std::strcmp(mode, "rms") == 0 || std::strcmp(mode, "swiglu") == 0) return false;
    throw std::invalid_argument("H3_POINTWISE_NUMERICS must be baseline, exact_ref, rms, or swiglu");
}

// Diagnostic host copies only. Disabled unless explicitly requested, with no
// changes to model arithmetic or production buffers.
void trace_tensor(const char *name, const infinicore::Tensor &value) {
    const char *directory = std::getenv("H3_TRACE_DIR");
    if (directory == nullptr || *directory == '\0'
        || infinilm::global_state::get_tensor_model_parallel_rank() != 0) {
        return;
    }
    auto sampled = value;
    const char *token_limit = std::getenv("H3_TRACE_TOKENS");
    const size_t limit = token_limit ? std::stoul(token_limit) : 0;
    if (limit > 0 && value->ndim() == 3 && value->size(1) > limit) {
        sampled = value->narrow({{1, 0, limit}});
    }
    const auto device = infinicore::context::getDevice();
    infinicore::context::syncDevice();  // the traced value is final
    auto host = sampled->contiguous()->to(infinicore::Device::cpu());
    infinicore::context::setDevice(device);
    infinicore::context::syncDevice();
    host->debug(std::string(directory) + "/" + name + ".bin");
    infinicore::context::setDevice(device);
}

infinicore::Tensor cast_to(const infinicore::Tensor &input,
                           const infinicore::DataType &dtype) {
    if (input->dtype() == dtype) {
        return input;
    }
    auto output = infinicore::Tensor::empty(input->shape(), dtype, input->device());
    infinicore::op::cast_(output, input);
    return output;
}

template <typename Linear>
infinicore::Tensor linear_forward(const std::shared_ptr<Linear> &linear,
                                  const infinicore::Tensor &input) {
    auto mutable_input = input;
    return linear->forward(mutable_input);
}

infinicore::Tensor gather_rows(const infinicore::Tensor &table,
                               const infinicore::Tensor &indices) {
    return infinicore::op::embedding(indices, table);
}

infinicore::Tensor apply_adaln(const infinicore::Tensor &normalized,
                               const infinicore::Tensor &scale,
                               const infinicore::Tensor &shift) {
    const char *mode = std::getenv("H3_ADALN_NUMERICS");
    if (mode != nullptr && std::strcmp(mode, "baseline") == 0) {
        auto scaled = infinicore::op::mul(normalized, scale);
        return infinicore::op::add(infinicore::op::add(normalized, scaled), shift);
    }
    if (mode != nullptr && std::strcmp(mode, "exact_ref") != 0) {
        throw std::invalid_argument("H3_ADALN_NUMERICS must be exact_ref or baseline (diagnostic)");
    }
    // Match Diffusers' BF16 rounding boundaries: round(1 + scale), then
    // round(x * factor), then round(product + shift). x + x*scale differs
    // particularly when scale is near -1, and errors compound over 50 blocks.
    // The baseline Add descriptor requires identical shapes (no broadcast).
    auto one = infinicore::Tensor::ones(scale->shape(), scale->dtype(), scale->device());
    auto factor = infinicore::op::add(scale, one);
    return infinicore::op::add(infinicore::op::mul(normalized, factor), shift);
}

infinicore::Tensor normalize_and_modulate(
    const std::shared_ptr<MiniMaxH3RMSNorm> &norm, const infinicore::Tensor &x,
    const infinicore::Tensor &scale, const infinicore::Tensor &shift,
    const char *trace_name = nullptr) {
    const char *fused = std::getenv("H3_FUSED_ADALN");
    const bool enabled = fused && std::strcmp(fused, "1") == 0;
    if (fused && !enabled && std::strcmp(fused, "0") != 0) {
        throw std::invalid_argument("H3_FUSED_ADALN must be 0 or 1");
    }
    if (enabled) {
        const char *adaln = std::getenv("H3_ADALN_NUMERICS");
        if (!pointwise_enabled("rms") || (adaln && std::strcmp(adaln, "exact_ref") != 0)) {
            throw std::invalid_argument("H3_FUSED_ADALN requires reference RMSNorm and AdaLN numerics");
        }
        const char *directory = std::getenv("H3_TRACE_DIR");
        if (trace_name && directory && *directory) {
            trace_tensor(trace_name, norm->forward(x));
        }
        return infinicore::op::h3_rms_modulate(x, norm->weight(), scale, shift,
                                              static_cast<float>(norm->eps()));
    }
    auto normalized = norm->forward(x);
    if (trace_name) trace_tensor(trace_name, normalized);
    return apply_adaln(normalized, scale, shift);
}

// Gated residual (+ the following RMSNorm/AdaLN) as one NineToothed kernel
// with the same BF16 rounding steps as add(x, mul(gate, y)) followed by
// h3_rms_modulate, so it needs the same numerics as H3_FUSED_ADALN=1.
// H3_FUSED_RESIDUAL=0 keeps the separate operators.
bool fused_residual_enabled() {
    const char *adaln = std::getenv("H3_FUSED_ADALN");
    if (adaln == nullptr || std::strcmp(adaln, "1") != 0) return false;
    const char *mode = std::getenv("H3_FUSED_RESIDUAL");
    if (mode == nullptr || std::strcmp(mode, "1") == 0) return true;
    if (std::strcmp(mode, "0") == 0) return false;
    throw std::invalid_argument("H3_FUSED_RESIDUAL must be 0 or 1");
}

bool tracing_enabled() {
    const char *directory = std::getenv("H3_TRACE_DIR");
    return directory != nullptr && *directory != '\0';
}

void require_tensor(const std::optional<infinicore::Tensor> &tensor,
                    const char *name) {
    if (!tensor.has_value() || !tensor.value()) {
        throw std::runtime_error(
            std::string("MiniMaxH3Transformer: missing input ") + name);
    }
}

size_t config_size(const std::shared_ptr<infinilm::config::ModelConfig> &config,
                   const char *primary, const char *fallback,
                   size_t default_value) {
    const auto &json = config->get_config_json();
    if (json.contains(primary)) {
        return json.at(primary).get<size_t>();
    }
    if (fallback != nullptr && json.contains(fallback)) {
        return json.at(fallback).get<size_t>();
    }
    return default_value;
}

// Sequence-parallel exchange of several equally split buffers on `stream`:
// one peer-copy all-to-all (H3_SP_A2A=peer) or one RCCL all-to-all each.
void sp_exchange_on_stream(const std::vector<infinicore::Tensor> &recv,
                           const std::vector<infinicore::Tensor> &send, int rank, size_t world,
                           infinicclComm_t comm, infinirtStream_t stream) {
    if (sp_peer_a2a_enabled()) {
        peer_all_to_all(sp_exchange(world), rank, world, recv, send, stream);
        if (infinicore::op::distributed::synchronous_collectives()) {
            check_rt(infinirtStreamSynchronize(stream), "infinirtStreamSynchronize");
        }
        return;
    }
    for (size_t i = 0; i < send.size(); ++i) {
        infinicore::op::distributed::all_to_all_on_stream_(recv[i], send[i], rank,
                                                           static_cast<int>(world), comm, stream);
    }
}

} // namespace

infinicore::Tensor MiniMaxH3RMSNorm::forward(const infinicore::Tensor &x) const {
    if (pointwise_enabled("rms")) {
        return infinicore::op::h3_rms_norm(x, weight(), static_cast<float>(eps()));
    }
    return infinicore::nn::RMSNorm::forward(x);
}

MiniMaxH3Attention::MiniMaxH3Attention(size_t hidden_size, size_t num_heads,
                                       size_t head_dim, double qk_norm_eps,
                                       const infinicore::DataType &dtype,
                                       const infinicore::Device &device,
                                       MiniMaxH3Parallel parallel)
    : head_dim_(head_dim),
      scale_(1.0f / std::sqrt(static_cast<float>(head_dim))),
      parallel_(parallel) {
    const auto rank_info = layer_rank_info(parallel);
    if (parallel == MiniMaxH3Parallel::SEQUENCE) {
        const auto &global = infinilm::global_state::get_tensor_model_parallel_rank_info();
        sp_size_ = static_cast<size_t>(global.tp_size);
        sp_rank_ = global.tp_rank;
        sp_comm_ = global.comm;
        if (num_heads % sp_size_ != 0) {
            throw std::runtime_error(
                "MiniMaxH3Attention: num_heads must be divisible by the sequence-parallel size");
        }
    }
    if (num_heads % static_cast<size_t>(rank_info.tp_size) != 0) {
        throw std::runtime_error(
            "MiniMaxH3Attention: num_heads must be divisible by tp_size");
    }
    num_heads_ = num_heads / static_cast<size_t>(rank_info.tp_size);
    inner_dim_ = num_heads_ * head_dim_;

    auto register_fn = [this](const std::string &name,
                              infinicore::nn::Parameter parameter) {
        this->register_parameter(name, std::move(parameter));
    };
    qkv_proj_ = std::make_shared<infinilm::layers::linear::QKVParallelLinear>(
        hidden_size, head_dim_, num_heads, num_heads, "to_q", "to_k", "to_v",
        register_fn, nullptr, false, dtype, device, rank_info);
    INFINICORE_NN_MODULE_INIT(norm_q, head_dim_, qk_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(norm_k, head_dim_, qk_norm_eps, dtype, device);
    out_proj_ = this->register_module<infinilm::layers::linear::RowParallelLinear>(
        "to_out.0", num_heads * head_dim_, hidden_size, false, dtype, device,
        rank_info.tp_rank, rank_info.tp_size, rank_info.comm);
}

std::tuple<infinicore::Tensor, infinicore::Tensor, infinicore::Tensor>
MiniMaxH3Attention::project_qkv(
    const infinicore::Tensor &hidden_states,
    const std::shared_ptr<infinicore::nn::RoPE> &rope,
    const std::optional<infinicore::Tensor> &position_ids,
    const std::optional<infinicore::Tensor> &rotary_cos_sin_cache) const {
    if (hidden_states->shape().size() != 3) {
        throw std::runtime_error(
            "MiniMaxH3Attention: hidden_states must be [batch, seq, hidden]");
    }
    const size_t batch_size = hidden_states->size(0);
    const size_t sequence_length = hidden_states->size(1);

    if (!qkv_group_heads_.empty()) {
        throw std::runtime_error("MiniMaxH3Attention: QKV weight rows are grouped for the "
                                 "pipelined sequence-parallel path");
    }
    auto hidden_states_mutable = hidden_states;
    auto [query, key, value] = qkv_proj_->forward_split(hidden_states_mutable);
    query = query->view({batch_size * sequence_length, num_heads_, head_dim_});
    key = key->view({batch_size * sequence_length, num_heads_, head_dim_});
    value = value->view({batch_size, sequence_length, num_heads_, head_dim_});
    std::tie(query, key) = normalize_rotate(query, key, batch_size, sequence_length, num_heads_,
                                            rope, position_ids, rotary_cos_sin_cache);
    return {query, key, value};
}

std::pair<infinicore::Tensor, infinicore::Tensor> MiniMaxH3Attention::normalize_rotate(
    infinicore::Tensor query, infinicore::Tensor key, size_t batch_size,
    size_t sequence_length, size_t num_heads,
    const std::shared_ptr<infinicore::nn::RoPE> &rope,
    const std::optional<infinicore::Tensor> &position_ids,
    const std::optional<infinicore::Tensor> &rotary_cos_sin_cache) const {
    const char *fused_fp32 = std::getenv("H3_QK_FUSED_FP32");
    const char *numerics = std::getenv("H3_ROPE_NUMERICS");
    if (fused_fp32 && std::strcmp(fused_fp32, "1") == 0
        && (!numerics || std::strcmp(numerics, "baseline") == 0)
        && batch_size == 1 && head_dim_ == 128 && rotary_cos_sin_cache.has_value()
        && rotary_cos_sin_cache.value()->size(1) == 96 && position_ids.has_value()) {
        // Positions are the valid global row indices generated by the H3 runner.
        auto ordered = gather_rows(rotary_cos_sin_cache.value()->contiguous(), position_ids.value()->contiguous());
        auto cos = ordered->narrow({{1, 0, 48}});
        auto sin = ordered->narrow({{1, 48, 48}});
        auto q = infinicore::op::h3_qk_rms_rope_fp32(
            query->view({sequence_length, num_heads, head_dim_}), norm_q_->weight(), cos, sin, static_cast<float>(norm_q_->eps()));
        auto k = infinicore::op::h3_qk_rms_rope_fp32(
            key->view({sequence_length, num_heads, head_dim_}), norm_k_->weight(), cos, sin, static_cast<float>(norm_k_->eps()));
        return {q->unsqueeze(0), k->unsqueeze(0)};
    }
    query = norm_q_->forward(query);
    key = norm_k_->forward(key);
    query = query->view({batch_size, sequence_length, num_heads, head_dim_});
    key = key->view({batch_size, sequence_length, num_heads, head_dim_});

    if (rotary_cos_sin_cache.has_value()) {
        if (!position_ids.has_value()) {
            throw std::runtime_error(
                "MiniMaxH3Attention: position_ids are required with a rotary cache");
        }
        if (batch_size != 1) {
            throw std::runtime_error(
                "MiniMaxH3Attention: packed H3 RoPE currently supports batch size 1");
        }
        const auto &cache = rotary_cos_sin_cache.value();
        const size_t rotary_dim = cache->size(1);
        if (rotary_dim % (2 * H3_MODALITY_COUNT) != 0 || rotary_dim > head_dim_) {
            throw std::runtime_error("MiniMaxH3Attention: rotary cache width is "
                                     "incompatible with three-axis RoPE");
        }
        const size_t half_rotary_dim = rotary_dim / 2;
        const int section = static_cast<int>(rotary_dim / (2 * H3_MODALITY_COUNT));
        auto cos = cache->narrow({{1, 0, half_rotary_dim}});
        auto sin = cache->narrow({{1, half_rotary_dim, half_rotary_dim}});
        const char *rope_mode = std::getenv("H3_ROPE_NUMERICS");
        if (rope_mode && std::strcmp(rope_mode, "exact_ref") == 0) {
            // The existing embedding implementation requires contiguous rows.
            // Gather the full cache, then let NineToothed read its strided halves.
            auto ordered_cache = gather_rows(cache->contiguous(), position_ids.value()->contiguous());
            cos = ordered_cache->narrow({{1, 0, half_rotary_dim}});
            sin = ordered_cache->narrow({{1, half_rotary_dim, half_rotary_dim}});
            query = infinicore::op::h3_rope(query->squeeze(0), cos, sin)->unsqueeze(0);
            key = infinicore::op::h3_rope(key->squeeze(0), cos, sin)->unsqueeze(0);
        } else {
            if (rope_mode && std::strcmp(rope_mode, "baseline") != 0) {
                throw std::invalid_argument("H3_ROPE_NUMERICS must be baseline or exact_ref");
            }
            auto query_flat = query->squeeze(0)->view({sequence_length, num_heads * head_dim_});
            auto key_flat = key->squeeze(0)->view({sequence_length, num_heads * head_dim_});
            std::tie(query_flat, key_flat) = infinicore::op::mrope(
                query_flat, key_flat, cos, sin, position_ids.value(),
                static_cast<int>(head_dim_), static_cast<int>(rotary_dim), section,
                section, section, false);
            query = query_flat->view({batch_size, sequence_length, num_heads, head_dim_});
            key = key_flat->view({batch_size, sequence_length, num_heads, head_dim_});
        }
    } else if (rope != nullptr) {
        if (!position_ids.has_value()) {
            throw std::runtime_error(
                "MiniMaxH3Attention: position_ids are required with RoPE");
        }
        if (batch_size != 1) {
            throw std::runtime_error(
                "MiniMaxH3Attention: packed H3 RoPE currently supports batch size 1");
        }
        auto query_squeezed = query->squeeze(0);
        auto key_squeezed = key->squeeze(0);
        std::tie(query_squeezed, key_squeezed) = rope->forward(query_squeezed, key_squeezed, position_ids.value());
        query = query_squeezed->unsqueeze(0);
        key = key_squeezed->unsqueeze(0);
    }
    return {query, key};
}

infinicore::Tensor MiniMaxH3Attention::attend_rows_local(const infinicore::Tensor &query,
                                                         const infinicore::Tensor &key,
                                                         const infinicore::Tensor &value,
                                                         size_t start, size_t rows) const {
    // FlashAttention tiles queries independently, so a row range starting on a
    // tile boundary computes exactly the rows of the full-sequence call.
    auto output = infinicore::op::mha(query->narrow({{1, start, rows}}), key, value,
                                      std::nullopt, scale_, false)
                      ->view({1, rows, inner_dim_});
    return out_proj_->BaseLinear::forward(output);
}

infinicore::Tensor MiniMaxH3Attention::forward(
    const infinicore::Tensor &hidden_states,
    const std::shared_ptr<infinicore::nn::RoPE> &rope,
    const std::optional<infinicore::Tensor> &position_ids,
    const std::optional<infinicore::Tensor> &rotary_cos_sin_cache) const {
    const size_t batch_size = hidden_states->size(0);
    const size_t sequence_length = hidden_states->size(1);
    if (parallel_ == MiniMaxH3Parallel::SEQUENCE && sp_pipeline_enabled() && batch_size == 1
        && sp_valid_tokens != 0 && sp_valid_tokens <= sp_size_ * sequence_length) {
        const auto group_heads = sp_head_groups(num_heads_ / sp_size_);
        if (group_heads.size() > 1) {
            auto output = pipelined_sequence_attention(hidden_states, rope, position_ids,
                                                       rotary_cos_sin_cache, group_heads);
            return linear_forward(out_proj_, output);
        }
    }
    const char *direct_pack = std::getenv("H3_QK_DIRECT_PACK");
    const char *pack_qkv = std::getenv("H3_SP_PACK_QKV");
    const char *fused_fp32 = std::getenv("H3_QK_FUSED_FP32");
    const char *numerics = std::getenv("H3_ROPE_NUMERICS");
    if (direct_pack && std::strcmp(direct_pack, "1") == 0
        && pack_qkv && std::strcmp(pack_qkv, "1") == 0
        && fused_fp32 && std::strcmp(fused_fp32, "1") == 0
        && (!numerics || std::strcmp(numerics, "baseline") == 0)
        && parallel_ == MiniMaxH3Parallel::SEQUENCE && sp_size_ == 8
        && batch_size == 1 && num_heads_ == 56 && head_dim_ == 128
        && !sp_pipeline_enabled() && sp_overlap_groups(num_heads_ / sp_size_) == 1
        && sp_valid_tokens > 0 && sp_valid_tokens <= sp_size_ * sequence_length
        && qkv_group_heads_.empty() && rotary_cos_sin_cache.has_value()
        && rotary_cos_sin_cache.value()->size(1) == 96 && position_ids.has_value()) {
        auto [global_query, global_key, global_value] = project_direct_packed_qkv(
            hidden_states, position_ids.value(), rotary_cos_sin_cache.value());
        global_key = global_key->narrow({{1, 0, sp_valid_tokens}});
        global_value = global_value->narrow({{1, 0, sp_valid_tokens}});
        auto global_output = infinicore::op::mha(
            global_query, global_key, global_value, std::nullopt, scale_, false);
        return linear_forward(out_proj_, scatter_sequence(global_output, sequence_length));
    }
    auto [query, key, value] = project_qkv(hidden_states, rope, position_ids, rotary_cos_sin_cache);
    if (parallel_ == MiniMaxH3Parallel::SEQUENCE) {
        if (batch_size != 1 || sp_valid_tokens == 0 || sp_valid_tokens > sp_size_ * sequence_length) {
            throw std::runtime_error("MiniMaxH3Attention: invalid sequence-parallel token layout");
        }
        // Tokens -> heads: every rank now holds all tokens for heads/ranks heads.
        // Padding tokens exist only after the real sequence, so keys and values
        // are narrowed to the real length; padded queries are dropped later.
        const size_t groups = sp_overlap_groups(num_heads_ / sp_size_);
        if (groups > 1) {
            auto output = overlapped_sequence_attention(query, key, value, sequence_length, groups);
            return linear_forward(out_proj_, output);
        }
        infinicore::Tensor global_query, global_key, global_value;
        const char *pack_qkv = std::getenv("H3_SP_PACK_QKV");
        if (pack_qkv != nullptr && std::strcmp(pack_qkv, "1") == 0) {
            std::tie(global_query, global_key, global_value) = gather_qkv(query, key, value);
        } else {
            if (pack_qkv != nullptr && std::strcmp(pack_qkv, "0") != 0) {
                throw std::invalid_argument("H3_SP_PACK_QKV must be 0 or 1");
            }
            global_query = gather_sequence(query);
            global_key = gather_sequence(key);
            global_value = gather_sequence(value);
        }
        global_key = global_key->narrow({{1, 0, sp_valid_tokens}});
        global_value = global_value->narrow({{1, 0, sp_valid_tokens}});
        auto global_output = infinicore::op::mha(
            global_query, global_key, global_value, std::nullopt, scale_, false);
        auto output = scatter_sequence(global_output, sequence_length);
        return linear_forward(out_proj_, output);
    }
    auto output = infinicore::op::mha(query, key, value, std::nullopt, scale_, false);
    output = output->view({batch_size, sequence_length, inner_dim_});
    return linear_forward(out_proj_, output);
}

std::tuple<infinicore::Tensor, infinicore::Tensor, infinicore::Tensor>
MiniMaxH3Attention::project_direct_packed_qkv(
    const infinicore::Tensor &hidden_states, const infinicore::Tensor &position_ids,
    const infinicore::Tensor &rotary_cache) const {
    const size_t tokens = hidden_states->size(1);
    const size_t heads = num_heads_ / sp_size_;
    auto mutable_hidden = hidden_states;
    // Keep the original full GEMM and arithmetic; remove only Q/K pack copies.
    auto [query, key, value] = qkv_proj_->forward_split(mutable_hidden);
    auto ordered = gather_rows(rotary_cache->contiguous(), position_ids->contiguous());
    auto cos = ordered->narrow({{1, 0, 48}});
    auto sin = ordered->narrow({{1, 48, 48}});
    auto send = infinicore::Tensor::empty(
        {sp_size_, tokens, 3, heads, head_dim_}, query->dtype(), query->device());
    const char *qkv_fused=std::getenv("H3_QKV_DIRECT_PACK");
    if(qkv_fused && std::strcmp(qkv_fused,"1")==0) {
        infinicore::op::h3_qkv_norm_rope_(send,
            query->view({tokens,sp_size_,heads,head_dim_}),
            key->view({tokens,sp_size_,heads,head_dim_}),
            value->view({tokens,sp_size_,heads,head_dim_}),
            norm_q_->weight(),norm_k_->weight(),cos,sin,
            static_cast<float>(norm_q_->eps()),static_cast<float>(norm_k_->eps()));
    } else {
        if(qkv_fused && std::strcmp(qkv_fused,"0")!=0)
            throw std::invalid_argument("H3_QKV_DIRECT_PACK must be 0 or 1");
        auto q_target = send->narrow({{2, 0, 1}})->squeeze(2)->permute({1, 0, 2, 3});
        auto k_target = send->narrow({{2, 1, 1}})->squeeze(2)->permute({1, 0, 2, 3});
        infinicore::op::h3_qk_norm_rope_(q_target,
            query->view({tokens, sp_size_, heads, head_dim_}), norm_q_->weight(), cos, sin,
            static_cast<float>(norm_q_->eps()));
        infinicore::op::h3_qk_norm_rope_(k_target,
            key->view({tokens, sp_size_, heads, head_dim_}), norm_k_->weight(), cos, sin,
            static_cast<float>(norm_k_->eps()));
        auto v_target = send->narrow({{2, 2, 1}})->squeeze(2);
        auto v_source = value->view({tokens, sp_size_, heads, head_dim_})->permute({1, 0, 2, 3});
        infinicore::op::rearrange_(v_target, v_source);
    }
    auto received = infinicore::Tensor::empty(send->shape(), send->dtype(), send->device());
    infinicore::op::distributed::all_to_all_(received, send, sp_rank_,
                                             static_cast<int>(sp_size_), sp_comm_);
    auto packed = received->view({1, sp_size_ * tokens, 3, heads, head_dim_});
    return {packed->narrow({{2, 0, 1}})->squeeze(2),
            packed->narrow({{2, 1, 1}})->squeeze(2),
            packed->narrow({{2, 2, 1}})->squeeze(2)};
}

std::tuple<infinicore::Tensor, infinicore::Tensor, infinicore::Tensor>
MiniMaxH3Attention::gather_qkv(const infinicore::Tensor &query,
                             const infinicore::Tensor &key,
                             const infinicore::Tensor &value) const {
    const size_t tokens = query->size(1);
    const size_t heads = num_heads_ / sp_size_;
    // Destination-major [P,T,3,H/P,D]. One all-to-all carries every Q/K/V
    // element once. Received source blocks concatenate directly in token order;
    // selecting axis 2 gives a strided MHA input without another unpack copy.
    auto send = infinicore::Tensor::empty(
        {sp_size_, tokens, 3, heads, head_dim_}, query->dtype(), query->device());
    size_t i = 0;
    for (const auto *local : {&query, &key, &value}) {
        const auto &strides = (*local)->strides();
        auto blocks = (*local)->as_strided(
            {sp_size_, tokens, heads, head_dim_},
            {static_cast<infinicore::Stride>(heads) * strides[2], strides[1], strides[2], strides[3]});
        auto target = send->narrow({{2, i, 1}})->squeeze(2);
        infinicore::op::rearrange_(target, blocks);
        ++i;
    }
    auto received = infinicore::Tensor::empty(send->shape(), send->dtype(), send->device());
    infinicore::op::distributed::all_to_all_(received, send, sp_rank_,
                                             static_cast<int>(sp_size_), sp_comm_);
    auto packed = received->view({1, sp_size_ * tokens, 3, heads, head_dim_});
    return {packed->narrow({{2, 0, 1}})->squeeze(2),
            packed->narrow({{2, 1, 1}})->squeeze(2),
            packed->narrow({{2, 2, 1}})->squeeze(2)};
}

infinicore::Tensor MiniMaxH3Attention::gather_sequence(const infinicore::Tensor &local) const {
    // local: [1, T, H, D] with arbitrary strides (value is a view into the QKV
    // projection). Block p = heads [p*H/P, (p+1)*H/P) of all local tokens.
    const size_t tokens = local->size(1);
    const size_t heads = num_heads_ / sp_size_;
    const auto &strides = local->strides();
    auto blocks = local->as_strided(
        {sp_size_, tokens, heads, head_dim_},
        {static_cast<infinicore::Stride>(heads) * strides[2], strides[1], strides[2], strides[3]});
    auto send = blocks->contiguous();
    auto received = infinicore::Tensor::empty(send->shape(), send->dtype(), send->device());
    infinicore::op::distributed::all_to_all_(received, send, sp_rank_,
                                             static_cast<int>(sp_size_), sp_comm_);
    // Block q holds rank q's tokens, so the token axis is in global order.
    return received->view({1, sp_size_ * tokens, heads, head_dim_});
}

infinicore::Tensor MiniMaxH3Attention::overlapped_sequence_attention(
    const infinicore::Tensor &query, const infinicore::Tensor &key,
    const infinicore::Tensor &value, size_t tokens, size_t groups) const {
    // Rank p ends up with heads [p*Hl, (p+1)*Hl) of all tokens; they are split
    // into `groups` groups of Hs heads. Group g's inputs travel on the side
    // stream while earlier groups' attention runs, and its output travels back
    // while later groups compute. Exchanges keep the same issue order on every
    // rank, as collectives require.
    const size_t heads = num_heads_ / sp_size_;
    const size_t group_heads = heads / groups;
    const auto dtype = query->dtype();
    const auto device = query->device();
    auto &side = sp_side_stream();
    auto compute = infinicore::context::getStream();

    auto send_blocks = [&](const infinicore::Tensor &local, size_t group) {
        // [1, T, H, D] -> contiguous [P, T, Hs, D]; block p = heads p*Hl + group*Hs + h.
        const auto &strides = local->strides();
        auto first = local->narrow({{2, group * group_heads, heads * (sp_size_ - 1) + group_heads}});
        return first->as_strided(
                        {sp_size_, tokens, group_heads, head_dim_},
                        {static_cast<infinicore::Stride>(heads) * strides[2], strides[1], strides[2], strides[3]})
            ->contiguous();
    };
    std::vector<std::array<infinicore::Tensor, 3>> sent(groups), received(groups);
    for (size_t g = 0; g < groups; ++g) {
        size_t i = 0;
        for (const auto *local : {&query, &key, &value}) {
            sent[g][i] = send_blocks(*local, g);
            received[g][i] = infinicore::Tensor::empty(sent[g][i]->shape(), dtype, device);
            ++i;
        }
    }
    order_streams(side, compute, side.stream);
    std::vector<infinirtEvent_t> arrived(groups);
    for (size_t g = 0; g < groups; ++g) {
        sp_exchange_on_stream({received[g][0], received[g][1], received[g][2]},
                              {sent[g][0], sent[g][1], sent[g][2]}, sp_rank_, sp_size_, sp_comm_,
                              side.stream);
        arrived[g] = side.event();
        check_rt(infinirtEventRecord(arrived[g], side.stream), "infinirtEventRecord");
    }
    // `sent` stays referenced until the helper has made the compute stream
    // wait for every exchange.
    return attend_exchanged_groups(received, arrived, tokens,
                                   std::vector<size_t>(groups, group_heads));
}

infinicore::Tensor MiniMaxH3Attention::attend_exchanged_groups(
    const std::vector<std::array<infinicore::Tensor, 3>> &received,
    const std::vector<infinirtEvent_t> &arrived, size_t tokens,
    const std::vector<size_t> &group_heads) const {
    const size_t heads = num_heads_ / sp_size_;
    const size_t groups = group_heads.size();
    const auto dtype = received[0][0]->dtype();
    const auto device = received[0][0]->device();
    auto &side = sp_side_stream();
    auto compute = infinicore::context::getStream();
    std::vector<infinicore::Tensor> attended(groups), returned(groups), slices;
    for (size_t g = 0; g < groups; ++g) {
        const size_t hs = group_heads[g];
        check_rt(infinirtStreamWaitEvent(compute, arrived[g]), "infinirtStreamWaitEvent");
        const auto shape = infinicore::Shape{1, sp_size_ * tokens, hs, head_dim_};
        auto global_query = received[g][0]->view(shape);
        auto global_key = received[g][1]->view(shape)->narrow({{1, 0, sp_valid_tokens}});
        auto global_value = received[g][2]->view(shape)->narrow({{1, 0, sp_valid_tokens}});
        returned[g] = infinicore::Tensor::empty({sp_size_, tokens, hs, head_dim_}, dtype, device);
        const bool chunked = g + 1 == groups && sp_size_ > 1 && sp_chunked_output_enabled()
                          && !sp_peer_a2a_enabled();
        if (!chunked) {
            attended[g] = infinicore::op::mha(
                global_query, global_key, global_value, std::nullopt, scale_, false);
            order_streams(side, compute, side.stream);
            sp_exchange_on_stream({returned[g]}, {attended[g]->view({sp_size_, tokens, hs, head_dim_})},
                                  sp_rank_, sp_size_, sp_comm_, side.stream);
            continue;
        }
        // Step k: attend the rows of rank (r + k) and send them there while
        // receiving rank (r - k)'s rows for this rank; the local rows last.
        for (size_t k = 1; k < sp_size_; ++k) {
            const size_t dest = (static_cast<size_t>(sp_rank_) + k) % sp_size_;
            const size_t source = (static_cast<size_t>(sp_rank_) + sp_size_ - k) % sp_size_;
            auto rows = infinicore::op::mha(global_query->narrow({{1, dest * tokens, tokens}}),
                                            global_key, global_value, std::nullopt, scale_, false);
            order_streams(side, compute, side.stream);
            infinicore::op::distributed::send_recv_on_stream_(
                rows, static_cast<int>(dest), returned[g]->narrow({{0, source, 1}}),
                static_cast<int>(source), sp_comm_, side.stream);
            slices.push_back(rows);
        }
        infinicore::op::mha_(returned[g]->narrow({{0, static_cast<size_t>(sp_rank_), 1}}),
                             global_query->narrow({{1, sp_rank_ * tokens, tokens}}),
                             global_key, global_value, std::nullopt, scale_, false);
    }
    order_streams(side, side.stream, compute);
    // returned[g][q] holds heads q*Hl + offset_g + h of the local tokens.
    auto output = infinicore::Tensor::empty({1, tokens, inner_dim_}, dtype, device);
    auto by_rank = output->view({tokens, sp_size_, heads, head_dim_});
    size_t offset = 0;
    for (size_t g = 0; g < groups; ++g) {
        by_rank->narrow({{2, offset, group_heads[g]}})->copy_from(returned[g]->permute({1, 0, 2, 3}));
        offset += group_heads[g];
    }
    // Buffers above stay referenced until here, after the compute stream has
    // waited for every exchange that reads or writes them.
    return output;
}

void MiniMaxH3Attention::group_qkv_weight(const std::vector<size_t> &group_heads) const {
    if (qkv_group_heads_ == group_heads) {
        return;
    }
    if (!qkv_group_heads_.empty()) {
        throw std::runtime_error("MiniMaxH3Attention: QKV weight already grouped differently");
    }
    // Rows [q|k|v][head p*Hl + j][D] -> [g][q|k|v][p][j in group g][D], so each
    // group's projection is one contiguous block of weight rows.
    const size_t heads = num_heads_ / sp_size_;
    auto weight = qkv_proj_->weight();
    const size_t in_features = weight->size(1);
    auto by_head = weight->view({3, sp_size_, heads, head_dim_, in_features});
    auto grouped = infinicore::Tensor::empty(weight->shape(), weight->dtype(), weight->device());
    size_t row = 0, offset = 0;
    for (const size_t hs : group_heads) {
        const size_t count = 3 * sp_size_ * hs * head_dim_;
        grouped->narrow({{0, row, count}})
            ->view({3, sp_size_, hs, head_dim_, in_features})
            ->copy_from(by_head->narrow({{2, offset, hs}}));
        row += count;
        offset += hs;
    }
    weight->copy_from(grouped);
    qkv_group_heads_ = group_heads;
}

infinicore::Tensor MiniMaxH3Attention::pipelined_sequence_attention(
    const infinicore::Tensor &hidden_states,
    const std::shared_ptr<infinicore::nn::RoPE> &rope,
    const std::optional<infinicore::Tensor> &position_ids,
    const std::optional<infinicore::Tensor> &rotary_cos_sin_cache,
    const std::vector<size_t> &group_heads) const {
    group_qkv_weight(group_heads);
    const size_t groups = group_heads.size();
    const size_t tokens = hidden_states->size(1);
    const auto dtype = hidden_states->dtype();
    const auto device = hidden_states->device();
    auto &side = sp_side_stream();
    auto compute = infinicore::context::getStream();
    auto input = hidden_states->is_contiguous() ? hidden_states : hidden_states->contiguous();
    auto weight = qkv_proj_->weight();

    // [1, T, P*Hs, D] ordered (rank, head) -> contiguous [P, T, Hs, D].
    auto send_blocks = [&](const infinicore::Tensor &local, size_t hs) {
        const auto &strides = local->strides();
        return local->as_strided(
                        {sp_size_, tokens, hs, head_dim_},
                        {static_cast<infinicore::Stride>(hs) * strides[2], strides[1], strides[2], strides[3]})
            ->contiguous();
    };
    // H3_FUSED_QK (default on): q/k RMSNorm + RoPE written by one NineToothed
    // kernel straight into the send buffers, with per-token angles gathered
    // once per call (reference rounding, as H3_ROPE_NUMERICS=exact_ref).
    std::optional<std::pair<infinicore::Tensor, infinicore::Tensor>> angles;
    if (fused_qk_enabled() && rotary_cos_sin_cache.has_value() && position_ids.has_value()
        && rotary_cos_sin_cache.value()->size(1) == 96 && head_dim_ == 128) {
        auto ordered = gather_rows(rotary_cos_sin_cache.value()->contiguous(),
                                   position_ids.value()->contiguous());
        angles = std::make_pair(ordered->narrow({{1, 0, 48}}), ordered->narrow({{1, 48, 48}}));
    }
    std::vector<std::array<infinicore::Tensor, 3>> sent(groups), received(groups);
    std::vector<infinirtEvent_t> arrived(groups);
    size_t row_offset = 0;
    for (size_t g = 0; g < groups; ++g) {
        const size_t hs = group_heads[g];
        const size_t group_width = sp_size_ * hs;  // group g's heads on all ranks
        const size_t rows = 3 * group_width * head_dim_;
        auto projected = infinicore::op::linear(input, weight->narrow({{0, row_offset, rows}}), std::nullopt)
                             ->view({tokens, 3, group_width, head_dim_});
        row_offset += rows;
        auto part = [&](size_t index) { return projected->narrow({{1, index, 1}})->squeeze(1); };
        if (angles.has_value()) {
            for (size_t i = 0; i < 2; ++i) {
                auto source = part(i);  // [T, P*Hs, D], rows strided by the fused projection
                const auto &strides = source->strides();
                auto by_rank = source->as_strided(
                    {tokens, sp_size_, hs, head_dim_},
                    {strides[0], static_cast<infinicore::Stride>(hs) * strides[1], strides[1], strides[2]});
                sent[g][i] = infinicore::Tensor::empty({sp_size_, tokens, hs, head_dim_}, dtype, device);
                const auto &norm = i == 0 ? norm_q_ : norm_k_;
                infinicore::op::h3_qk_norm_rope_(sent[g][i]->permute({1, 0, 2, 3}), by_rank, norm->weight(),
                                                 angles->first, angles->second, static_cast<float>(norm->eps()));
            }
            sent[g][2] = send_blocks(part(2)->unsqueeze(0), hs);
        } else {
            auto [query, key] = normalize_rotate(part(0), part(1), 1, tokens, group_width, rope,
                                                 position_ids, rotary_cos_sin_cache);
            const std::array<infinicore::Tensor, 3> locals{query, key, part(2)->unsqueeze(0)};
            for (size_t i = 0; i < 3; ++i) {
                sent[g][i] = send_blocks(locals[i], hs);
            }
        }
        for (size_t i = 0; i < 3; ++i) {
            received[g][i] = infinicore::Tensor::empty(sent[g][i]->shape(), dtype, device);
        }
        // Group g travels while group g+1 is projected.
        order_streams(side, compute, side.stream);
        sp_exchange_on_stream({received[g][0], received[g][1], received[g][2]},
                              {sent[g][0], sent[g][1], sent[g][2]}, sp_rank_, sp_size_, sp_comm_,
                              side.stream);
        arrived[g] = side.event();
        check_rt(infinirtEventRecord(arrived[g], side.stream), "infinirtEventRecord");
    }
    return attend_exchanged_groups(received, arrived, tokens, group_heads);
}

infinicore::Tensor MiniMaxH3Attention::scatter_sequence(const infinicore::Tensor &global,
                                                        size_t local_tokens) const {
    // global: contiguous [1, P*T, H/P, D]; block p (tokens of rank p) goes home.
    const size_t heads = num_heads_ / sp_size_;
    auto send = global->view({sp_size_, local_tokens, heads, head_dim_});
    auto received = infinicore::Tensor::empty(send->shape(), send->dtype(), send->device());
    infinicore::op::distributed::all_to_all_(received, send, sp_rank_,
                                             static_cast<int>(sp_size_), sp_comm_);
    // received block q holds head group q for the local tokens.
    return received->permute({1, 0, 2, 3})->contiguous()->view({1, local_tokens, inner_dim_});
}

void MiniMaxH3Attention::process_weights_after_loading() {
    qkv_proj_->process_weights_after_loading();
}

void MiniMaxH3Attention::reset_runtime_state() const {
    qkv_proj_->reset_runtime_state();
}

MiniMaxH3MLP::MiniMaxH3MLP(size_t hidden_size, size_t ffn_hidden_size,
                           const infinicore::DataType &dtype,
                           const infinicore::Device &device,
                           MiniMaxH3Parallel parallel) {
    const auto rank_info = layer_rank_info(parallel);
    if (ffn_hidden_size % static_cast<size_t>(rank_info.tp_size) != 0) {
        throw std::runtime_error(
            "MiniMaxH3MLP: ffn_hidden_size must be divisible by tp_size");
    }
    auto register_fn = [this](const std::string &name,
                              infinicore::nn::Parameter parameter) {
        this->register_parameter(name, std::move(parameter));
    };
    // Diffusers stores [up, gate] in one projection. The synthetic names are
    // produced by the MiniMax-H3 load-time remapper so corresponding shards
    // from both halves land on every TP rank.
    gate_up_proj_ = std::make_shared<infinilm::layers::linear::GateUpParallelLinear>(
        hidden_size, ffn_hidden_size, "net.0.proj.up_proj",
        "net.0.proj.gate_proj", register_fn, nullptr, false, dtype, device,
        rank_info);
    down_proj_ = this->register_module<infinilm::layers::linear::RowParallelLinear>(
        "net.2", ffn_hidden_size, hidden_size, false, dtype, device,
        rank_info.tp_rank, rank_info.tp_size, rank_info.comm);
}

infinicore::Tensor
MiniMaxH3MLP::forward(const infinicore::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    auto [up, gate] = gate_up_proj_->forward_split(hidden_states_mutable);
    auto activated = pointwise_enabled("swiglu")
        ? infinicore::op::h3_swiglu(up, gate) : infinicore::op::swiglu(up, gate);
    return linear_forward(down_proj_, activated);
}

infinicore::Tensor
MiniMaxH3MLP::forward_local(const infinicore::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    auto [up, gate] = gate_up_proj_->forward_split(hidden_states_mutable);
    auto activated = pointwise_enabled("swiglu")
        ? infinicore::op::h3_swiglu(up, gate) : infinicore::op::swiglu(up, gate);
    return down_proj_->BaseLinear::forward(activated);
}

void MiniMaxH3MLP::process_weights_after_loading() {
    gate_up_proj_->process_weights_after_loading();
}

void MiniMaxH3MLP::reset_runtime_state() const {
    gate_up_proj_->reset_runtime_state();
}

MiniMaxH3TokenRefinerBlock::MiniMaxH3TokenRefinerBlock(
    size_t hidden_size, size_t num_heads, size_t head_dim,
    size_t ffn_hidden_size, double norm_eps, double qk_norm_eps,
    const infinicore::DataType &dtype, const infinicore::Device &device,
    MiniMaxH3Parallel parallel) {
    INFINICORE_NN_MODULE_INIT(norm1, hidden_size, norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(attn, hidden_size, num_heads, head_dim, qk_norm_eps,
                              dtype, device, parallel);
    INFINICORE_NN_MODULE_INIT(norm2, hidden_size, norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(ff, hidden_size, ffn_hidden_size, dtype, device, parallel);
}

infinicore::Tensor MiniMaxH3TokenRefinerBlock::forward(
    const infinicore::Tensor &hidden_states) const {
    auto attention_output = attn_->forward(norm1_->forward(hidden_states));
    auto output = infinicore::op::add(hidden_states, attention_output);
    auto mlp_output = ff_->forward(norm2_->forward(output));
    return infinicore::op::add(output, mlp_output);
}

MiniMaxH3TokenRefiner::MiniMaxH3TokenRefiner(
    size_t hidden_size, size_t num_heads, size_t head_dim,
    size_t ffn_hidden_size, size_t num_layers, double norm_eps,
    double qk_norm_eps, double final_norm_eps,
    const infinicore::DataType &dtype, const infinicore::Device &device,
    MiniMaxH3Parallel parallel) {
    refiner_blocks_.reserve(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        refiner_blocks_.push_back(this->register_module<MiniMaxH3TokenRefinerBlock>(
            "refiner_blocks." + std::to_string(i), hidden_size, num_heads, head_dim,
            ffn_hidden_size, norm_eps, qk_norm_eps, dtype, device, parallel));
    }
    INFINICORE_NN_MODULE_INIT(final_norm, hidden_size, final_norm_eps, dtype,
                              device);
}

infinicore::Tensor
MiniMaxH3TokenRefiner::forward(const infinicore::Tensor &hidden_states) const {
    auto output = hidden_states;
    for (const auto &block : refiner_blocks_) {
        output = block->forward(output);
    }
    return final_norm_->forward(output);
}

MiniMaxH3AdaLNProjection::MiniMaxH3AdaLNProjection(
    size_t in_features, size_t out_features, const infinicore::DataType &dtype,
    const infinicore::Device &device)
    : dtype_(dtype) {
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    if (out_features % static_cast<size_t>(rank_info.tp_size) != 0) {
        throw std::runtime_error(
            "MiniMaxH3AdaLNProjection: out_features must be divisible by tp_size");
    }
    tp_size_ = static_cast<size_t>(rank_info.tp_size);
    communicator_ = rank_info.comm;
    linear_ = this->register_module<infinilm::layers::linear::ColumnParallelLinear>(
        "linear", in_features, out_features, true, dtype, device,
        rank_info.tp_rank, rank_info.tp_size);
}

infinicore::Tensor
MiniMaxH3AdaLNProjection::forward_local(const infinicore::Tensor &activated) const {
    return linear_forward(linear_, activated);
}

infinicore::Tensor
MiniMaxH3AdaLNProjection::forward(const infinicore::Tensor &temb) const {
    auto activated = infinicore::op::silu(temb);
    activated = cast_to(activated, dtype_);
    auto local = forward_local(activated);
    if (tp_size_ == 1) {
        return local;
    }

    // allgather concatenates dim 0. Transpose the output width to dim 0 so
    // the original Diffusers projection order is reconstructed exactly.
    auto local_transposed = local->permute({1, 0})->contiguous();
    auto gathered = infinicore::op::distributed::allgather(
        local_transposed, tp_size_, communicator_);
    return gathered->permute({1, 0})->contiguous();
}

MiniMaxH3TransformerBlock::MiniMaxH3TransformerBlock(
    size_t hidden_size, size_t num_heads, size_t head_dim,
    size_t ffn_hidden_size, size_t time_embed_dim, double norm_eps,
    double qk_norm_eps, const infinicore::DataType &dtype,
    const infinicore::Device &device, MiniMaxH3Parallel parallel)
    : hidden_size_(hidden_size), parallel_(parallel) {
    if (parallel == MiniMaxH3Parallel::TENSOR) {
        const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
        tp_size_ = static_cast<size_t>(rank_info.tp_size);
        tp_comm_ = rank_info.comm;
    }
    INFINICORE_NN_MODULE_INIT(norm1, hidden_size, norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(attn, hidden_size, num_heads, head_dim, qk_norm_eps,
                              dtype, device, parallel);
    INFINICORE_NN_MODULE_INIT(norm2, hidden_size, norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(ff, hidden_size, ffn_hidden_size, dtype, device, parallel);
    // AdaLN projections stay column-parallel in every mode: each rank keeps a
    // quarter of these 13B parameters and all-gathers the (<=3-row) result.
    INFINICORE_NN_MODULE_INIT(adaln_proj, time_embed_dim,
                              H3_MODALITY_COUNT * 6 * hidden_size, dtype, device);
}

infinicore::Tensor MiniMaxH3TransformerBlock::forward(
    const infinicore::Tensor &hidden_states, const infinicore::Tensor &temb,
    const infinicore::Tensor &adaln_indices,
    const std::shared_ptr<infinicore::nn::RoPE> &rope,
    const infinicore::Tensor &position_ids,
    const std::optional<infinicore::Tensor> &rotary_cos_sin_cache) const {
    auto modulation = adaln_proj_->forward(temb)->view(
        {temb->size(0) * H3_MODALITY_COUNT, 6 * hidden_size_});
    if (trace_block == 0) trace_tensor("block0.modulation", modulation);
    auto token_modulation = gather_rows(modulation, adaln_indices)->unsqueeze(0);
    auto shift_msa = token_modulation->narrow({{2, 0, hidden_size_}});
    auto scale_msa = token_modulation->narrow({{2, hidden_size_, hidden_size_}});
    auto gate_msa = token_modulation->narrow({{2, 2 * hidden_size_, hidden_size_}});
    auto shift_mlp = token_modulation->narrow({{2, 3 * hidden_size_, hidden_size_}});
    auto scale_mlp = token_modulation->narrow({{2, 4 * hidden_size_, hidden_size_}});
    auto gate_mlp = token_modulation->narrow({{2, 5 * hidden_size_, hidden_size_}});

    const size_t chunks = tp_overlap_chunks();
    if (parallel_ == MiniMaxH3Parallel::TENSOR && tp_size_ > 1 && chunks > 1
        && hidden_states->size(0) == 1 && rotary_cos_sin_cache.has_value()) {
        return forward_overlapped(hidden_states, token_modulation, rope, position_ids,
                                  rotary_cos_sin_cache, chunks);
    }
    auto normalized = normalize_and_modulate(norm1_, hidden_states, scale_msa, shift_msa,
                                            trace_block == 0 ? "block0.norm1" : nullptr);
    if (trace_block == 0) trace_tensor("block0.modulated1", normalized);
    auto attention_output = attn_->forward(normalized, rope, position_ids, rotary_cos_sin_cache);
    if (trace_block == 0) trace_tensor("block0.attention", attention_output);
    const bool fused = fused_residual_enabled() && !(trace_block == 0 && tracing_enabled());
    infinicore::Tensor output;
    if (fused) {
        std::tie(output, normalized) = infinicore::op::h3_gated_residual_rms_modulate(
            hidden_states, gate_msa, attention_output, norm2_->weight(), scale_mlp, shift_mlp,
            static_cast<float>(norm2_->eps()));
    } else {
        output = infinicore::op::add(
            hidden_states, infinicore::op::mul(gate_msa, attention_output));
        normalized = normalize_and_modulate(norm2_, output, scale_mlp, shift_mlp,
                                           trace_block == 0 ? "block0.norm2" : nullptr);
    }
    if (trace_block == 0) trace_tensor("block0.modulated2", normalized);
    auto mlp_output = ff_->forward(normalized);
    if (trace_block == 0) trace_tensor("block0.mlp", mlp_output);
    if (fused) {
        return infinicore::op::h3_gated_residual(output, gate_mlp, mlp_output);
    }
    output = infinicore::op::add(output, infinicore::op::mul(gate_mlp, mlp_output));
    return output;
}

infinicore::Tensor MiniMaxH3TransformerBlock::forward_segmented(
    const infinicore::Tensor &hidden_states, const infinicore::Tensor &modulation,
    const std::vector<MiniMaxH3ModulationSegment> &segments,
    const std::shared_ptr<infinicore::nn::RoPE> &rope, const infinicore::Tensor &position_ids,
    const std::optional<infinicore::Tensor> &rotary_cos_sin_cache) const {
    return forward_segmented_chain(hidden_states, modulation, segments, rope,
        position_ids, rotary_cos_sin_cache, std::nullopt, nullptr, std::nullopt).first;
}

std::pair<infinicore::Tensor, std::optional<infinicore::Tensor>>
MiniMaxH3TransformerBlock::forward_segmented_chain(
    const infinicore::Tensor &hidden_states, const infinicore::Tensor &modulation,
    const std::vector<MiniMaxH3ModulationSegment> &segments,
    const std::shared_ptr<infinicore::nn::RoPE> &rope, const infinicore::Tensor &position_ids,
    const std::optional<infinicore::Tensor> &rotary_cos_sin_cache,
    const std::optional<infinicore::Tensor> &normalized_input,
    const MiniMaxH3TransformerBlock *next_block,
    const std::optional<infinicore::Tensor> &next_modulation) const {
    const size_t width = modulation->size(1);
    // Column `index` of the segment's modulation row, broadcast over its tokens.
    auto column = [&](size_t index, const MiniMaxH3ModulationSegment &segment) {
        return modulation->narrow({{0, segment.row, 1}, {1, index * hidden_size_, hidden_size_}})
            ->as_strided({1, segment.length, hidden_size_},
                         {static_cast<ptrdiff_t>(width), 0, 1});
    };
    auto rows = [](const infinicore::Tensor &tensor, const MiniMaxH3ModulationSegment &segment) {
        return tensor->narrow({{1, segment.start, segment.length}});
    };
    auto like = [&](const infinicore::Tensor &tensor) {
        return infinicore::Tensor::empty(tensor->shape(), tensor->dtype(), tensor->device());
    };
    const float eps1 = static_cast<float>(norm1_->eps());
    const float eps2 = static_cast<float>(norm2_->eps());

    // shift_msa 0, scale_msa 1, gate_msa 2, shift_mlp 3, scale_mlp 4, gate_mlp 5
    if ((next_block != nullptr) != next_modulation.has_value()) {
        throw std::invalid_argument("H3 chain: next block/modulation mismatch");
    }
    auto normalized = normalized_input.has_value() ? normalized_input.value() : like(hidden_states);
    if (!normalized_input.has_value()) {
        for (const auto &segment : segments) {
            infinicore::op::H3RMSModulate::execute(rows(normalized, segment), rows(hidden_states, segment),
                                                   norm1_->weight(), column(1, segment), column(0, segment),
                                                   eps1);
        }
    }
    auto attention_output = attn_->forward(normalized, rope, position_ids, rotary_cos_sin_cache);
    auto residual = like(hidden_states);
    auto normalized_mlp = like(hidden_states);
    for (const auto &segment : segments) {
        infinicore::op::H3GatedResidualRMSModulate::execute(
            rows(residual, segment), rows(normalized_mlp, segment), rows(hidden_states, segment),
            column(2, segment), rows(attention_output, segment), norm2_->weight(), column(4, segment),
            column(3, segment), eps2);
    }
    auto mlp_output = ff_->forward(normalized_mlp);
    auto output = like(hidden_states);
    if (next_block != nullptr) {
        // The next block consumes this exact residual immediately. Preserve
        // every BF16 rounding through the existing reference fused operator.
        if (next_block->hidden_size_ != hidden_size_) {
            throw std::invalid_argument("H3 chain: incompatible hidden width");
        }
        auto next_normalized = like(hidden_states);
        const auto &next_table = next_modulation.value();
        const size_t next_width = next_table->size(1);
        auto next_column = [&](size_t index, const MiniMaxH3ModulationSegment &segment) {
            return next_table->narrow({{0, segment.row, 1}, {1, index * hidden_size_, hidden_size_}})
                ->as_strided({1, segment.length, hidden_size_},
                             {static_cast<ptrdiff_t>(next_width), 0, 1});
        };
        for (const auto &segment : segments) {
            infinicore::op::H3GatedResidualRMSModulate::execute(
                rows(output, segment), rows(next_normalized, segment), rows(residual, segment),
                column(5, segment), rows(mlp_output, segment), next_block->norm1_->weight(),
                next_column(1, segment), next_column(0, segment),
                static_cast<float>(next_block->norm1_->eps()));
        }
        return {output, next_normalized};
    }
    for (const auto &segment : segments) {
        infinicore::op::H3GatedResidual::execute(rows(output, segment), rows(residual, segment),
                                                 column(5, segment), rows(mlp_output, segment));
    }
    return {output, std::nullopt};
}

infinicore::Tensor MiniMaxH3TransformerBlock::forward_overlapped(
    const infinicore::Tensor &hidden_states, const infinicore::Tensor &token_modulation,
    const std::shared_ptr<infinicore::nn::RoPE> &rope, const infinicore::Tensor &position_ids,
    const std::optional<infinicore::Tensor> &rotary_cos_sin_cache, size_t chunks) const {
    const size_t tokens = hidden_states->size(1);
    auto column = [&](size_t index) {
        return token_modulation->narrow({{2, index * hidden_size_, hidden_size_}});
    };
    auto shift_msa = column(0), scale_msa = column(1), gate_msa = column(2);
    auto shift_mlp = column(3), scale_mlp = column(4), gate_mlp = column(5);

    auto normalized = normalize_and_modulate(norm1_, hidden_states, scale_msa, shift_msa);
    auto [query, key, value] = attn_->project_qkv(normalized, rope, position_ids, rotary_cos_sin_cache);

    // Row chunks start on 128-row boundaries (the attention query tile).
    constexpr size_t ROW_ALIGN = 128;
    const size_t step = ((tokens + chunks - 1) / chunks + ROW_ALIGN - 1) / ROW_ALIGN * ROW_ALIGN;
    std::vector<std::pair<size_t, size_t>> ranges;
    for (size_t start = 0; start < tokens; start += step) {
        ranges.emplace_back(start, std::min(step, tokens - start));
    }
    const size_t count = ranges.size();
    auto &side = sp_side_stream();
    auto compute = infinicore::context::getStream();
    auto rows_of = [&](const infinicore::Tensor &tensor, size_t index) {
        return tensor->narrow({{1, ranges[index].first, ranges[index].second}});
    };

    // Attention and output projection per chunk; its all-reduce runs on the
    // side stream while the next chunk's attention computes.
    std::vector<infinicore::Tensor> attended(count), residual(count), mlp(count);
    std::vector<infinirtEvent_t> attended_ready(count), mlp_ready(count);
    for (size_t i = 0; i < count; ++i) {
        attended[i] = attn_->attend_rows_local(query, key, value, ranges[i].first, ranges[i].second);
        order_streams(side, compute, side.stream);
        infinicore::op::distributed::allreduce_on_stream_(attended[i], attended[i], INFINICCL_SUM,
                                                          tp_comm_, side.stream);
        attended_ready[i] = side.event();
        check_rt(infinirtEventRecord(attended_ready[i], side.stream), "infinirtEventRecord");
    }
    // Residual, norm and MLP per chunk as soon as its attention output is reduced.
    const bool fused = fused_residual_enabled();
    for (size_t i = 0; i < count; ++i) {
        check_rt(infinirtStreamWaitEvent(compute, attended_ready[i]), "infinirtStreamWaitEvent");
        infinicore::Tensor normalized_mlp;
        if (fused) {
            std::tie(residual[i], normalized_mlp) = infinicore::op::h3_gated_residual_rms_modulate(
                rows_of(hidden_states, i), rows_of(gate_msa, i), attended[i], norm2_->weight(),
                rows_of(scale_mlp, i), rows_of(shift_mlp, i), static_cast<float>(norm2_->eps()));
        } else {
            residual[i] = infinicore::op::add(rows_of(hidden_states, i),
                                              infinicore::op::mul(rows_of(gate_msa, i), attended[i]));
            normalized_mlp = normalize_and_modulate(norm2_, residual[i], rows_of(scale_mlp, i),
                                                    rows_of(shift_mlp, i));
        }
        mlp[i] = ff_->forward_local(normalized_mlp);
        order_streams(side, compute, side.stream);
        infinicore::op::distributed::allreduce_on_stream_(mlp[i], mlp[i], INFINICCL_SUM,
                                                          tp_comm_, side.stream);
        mlp_ready[i] = side.event();
        check_rt(infinirtEventRecord(mlp_ready[i], side.stream), "infinirtEventRecord");
    }
    auto output = infinicore::Tensor::empty(hidden_states->shape(), hidden_states->dtype(),
                                            hidden_states->device());
    for (size_t i = 0; i < count; ++i) {
        check_rt(infinirtStreamWaitEvent(compute, mlp_ready[i]), "infinirtStreamWaitEvent");
        if (fused) {
            infinicore::op::H3GatedResidual::execute(rows_of(output, i), residual[i],
                                                     rows_of(gate_mlp, i), mlp[i]);
        } else {
            infinicore::op::add_(rows_of(output, i), residual[i],
                                 infinicore::op::mul(rows_of(gate_mlp, i), mlp[i]));
        }
    }
    // All chunk buffers stay referenced until the compute stream has waited
    // for every all-reduce that reads or writes them.
    return output;
}

MiniMaxH3TimeEmbedder::MiniMaxH3TimeEmbedder(size_t input_dim,
                                             size_t hidden_size,
                                             size_t output_dim,
                                             const infinicore::Device &device)
    : input_dim_(input_dim) {
    INFINICORE_NN_MODULE_INIT(linear_1, input_dim, hidden_size, true,
                              infinicore::DataType::F32, device);
    INFINICORE_NN_MODULE_INIT(linear_2, hidden_size, output_dim, true,
                              infinicore::DataType::F32, device);
}

infinicore::Tensor
MiniMaxH3TimeEmbedder::forward(const infinicore::Tensor &timestep) const {
    if (input_dim_ % 2 != 0) {
        throw std::runtime_error("MiniMaxH3TimeEmbedder: input_dim must be even");
    }

    if (timestep->shape().size() == 2) {
        if (timestep->size(1) != input_dim_) {
            throw std::runtime_error(
                "MiniMaxH3TimeEmbedder: precomputed embedding has the wrong width");
        }
        auto embedding = cast_to(timestep, infinicore::DataType::F32);
        auto hidden = linear_forward(linear_1_, embedding);
        hidden = infinicore::op::silu(hidden);
        return linear_forward(linear_2_, hidden);
    }
    if (timestep->shape().size() != 1) {
        throw std::runtime_error("MiniMaxH3TimeEmbedder: timestep must be [N] or a "
                                 "precomputed [N, input_dim] embedding");
    }
    trace_tensor("temb.timestep", timestep);
    auto embedding = infinicore::op::timestep_embedding(timestep, input_dim_);
    trace_tensor("temb.frequency", embedding);
    auto hidden = linear_forward(linear_1_, embedding);
    trace_tensor("temb.linear_1", hidden);
    hidden = infinicore::op::silu(hidden);
    trace_tensor("temb.silu", hidden);
    return linear_forward(linear_2_, hidden);
}

MiniMaxH3OutputNorm::MiniMaxH3OutputNorm(size_t hidden_size,
                                         size_t time_embed_dim, double norm_eps,
                                         const infinicore::DataType &dtype,
                                         const infinicore::Device &device)
    : hidden_size_(hidden_size), dtype_(dtype) {
    INFINICORE_NN_MODULE_INIT(norm, hidden_size, norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(linear, time_embed_dim, 2 * hidden_size, true,
                              dtype, device);
}

infinicore::Tensor
MiniMaxH3OutputNorm::forward(const infinicore::Tensor &hidden_states,
                             const infinicore::Tensor &temb,
                             const infinicore::Tensor &timestep_indices) const {
    auto activated = cast_to(infinicore::op::silu(temb), dtype_);
    auto modulation = linear_forward(linear_, activated);
    auto token_modulation = gather_rows(modulation, timestep_indices)->unsqueeze(0);
    auto shift = token_modulation->narrow({{2, 0, hidden_size_}});
    auto scale = token_modulation->narrow({{2, hidden_size_, hidden_size_}});
    return normalize_and_modulate(norm_, hidden_states, scale, shift);
}

MiniMaxH3Transformer::MiniMaxH3Transformer(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device)
    : hidden_size_(config_size(model_config, "hidden_size", nullptr, 5376)),
      dtype_(model_config->get_dtype()) {
    model_config_ = model_config;
    const auto &config = model_config->get_config_json();
    const size_t num_layers = config_size(model_config, "num_layers", nullptr, 50);
    const size_t token_refiner_layers = config_size(
        model_config, "token_refiner_num_layers", "num_refiner_layers", 2);
    const size_t num_heads = config_size(model_config, "num_attention_heads", nullptr, 56);
    const size_t head_dim = config_size(model_config, "attention_head_dim", nullptr, 128);
    const size_t ffn_hidden_size = config_size(model_config, "ffn_hidden_size", "ffn_dim", 14336);
    const size_t video_latents_dim = config_size(model_config, "latents_dim", "in_channels", 24);
    const size_t audio_latents_dim = config_size(model_config, "audio_latents_dim", "audio_in_channels", 32);
    const size_t text_dim = config_size(model_config, "text_dim", nullptr, 5120);
    const size_t timestep_input_dim = config_size(model_config, "timestep_input_dim", "freq_dim", 256);
    const size_t time_embed_hidden_size = config_size(
        model_config, "time_embed_hidden_size", "time_embed_hidden_dim", 5376);
    const size_t time_embed_dim = config_size(model_config, "time_embed_dim", nullptr, 2688);
    const size_t rope_freq_dim = config_size(model_config, "rope_inv_freq_len", "rope_freq_dim", 16);
    const double norm_eps = config.value("norm_eps", 1e-5);
    const double qk_norm_eps = config.value("qk_norm_eps", 1e-5);
    const double final_norm_eps = config.value("final_norm_eps", 1e-5);

    std::vector<size_t> patch_size{1, 2, 2};
    if (config.contains("patch_size")) {
        patch_size = config.at("patch_size").get<std::vector<size_t>>();
    }
    if (patch_size.size() != 3) {
        throw std::runtime_error(
            "MiniMaxH3Transformer: patch_size must contain t, h and w");
    }
    const size_t video_patch_dim = video_latents_dim * patch_size[0] * patch_size[1] * patch_size[2];
    const size_t rotary_dim = 2 * H3_MODALITY_COUNT * rope_freq_dim;
    if (rotary_dim > head_dim) {
        throw std::runtime_error("MiniMaxH3Transformer: rotary dimensions exceed "
                                 "attention head dimension");
    }

    INFINICORE_NN_MODULE_INIT(proj_in, video_patch_dim, hidden_size_, true,
                              infinicore::DataType::F32, device);
    INFINICORE_NN_MODULE_INIT(audio_proj_in, audio_latents_dim, hidden_size_,
                              true, infinicore::DataType::F32, device);
    INFINICORE_NN_MODULE_INIT(context_embedder, text_dim, hidden_size_, true,
                              dtype_, device);
    INFINICORE_NN_MODULE_INIT(time_embedder, timestep_input_dim,
                              time_embed_hidden_size, time_embed_dim, device);
    INFINICORE_NN_MODULE_INIT(
        rope, head_dim, rotary_dim, config.value("max_position_embeddings", 8192),
        config.value("rope_theta", 10000.0), infinicore::nn::RoPE::Algo::GPT_NEOX,
        dtype_, device, nullptr,
        std::vector<int>{static_cast<int>(rope_freq_dim),
                         static_cast<int>(rope_freq_dim),
                         static_cast<int>(rope_freq_dim)},
        false);
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    const auto block_parallel = block_parallel_mode(rank_info, num_heads);
    sequence_parallel_ = block_parallel == MiniMaxH3Parallel::SEQUENCE;
    if (sequence_parallel_) {
        sp_size_ = static_cast<size_t>(rank_info.tp_size);
        sp_rank_ = rank_info.tp_rank;
        sp_comm_ = rank_info.comm;
    }
    INFINICORE_NN_MODULE_INIT(token_refiner, hidden_size_, num_heads, head_dim,
                              ffn_hidden_size, token_refiner_layers, norm_eps,
                              qk_norm_eps, final_norm_eps, dtype_, device,
                              refiner_parallel_mode(block_parallel));

    transformer_blocks_.reserve(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        transformer_blocks_.push_back(
            this->register_module<MiniMaxH3TransformerBlock>(
                "transformer_blocks." + std::to_string(i), hidden_size_, num_heads,
                head_dim, ffn_hidden_size, time_embed_dim, norm_eps, qk_norm_eps,
                dtype_, device, block_parallel));
    }
    INFINICORE_NN_MODULE_INIT(norm_out, hidden_size_, time_embed_dim,
                              final_norm_eps, dtype_, device);
    INFINICORE_NN_MODULE_INIT(proj_out, hidden_size_, video_patch_dim, true,
                              infinicore::DataType::F32, device);
    INFINICORE_NN_MODULE_INIT(audio_proj_out, hidden_size_, audio_latents_dim,
                              true, infinicore::DataType::F32, device);
}

InfinilmModel::Output MiniMaxH3Transformer::forward(const Input &input) const {
    rewind_sp_events();
    require_tensor(input.video_hidden_states, "video_hidden_states");
    require_tensor(input.audio_hidden_states, "audio_hidden_states");
    require_tensor(input.encoder_hidden_states, "encoder_hidden_states");
    require_tensor(input.timestep, "timestep");
    require_tensor(input.timestep_indices, "timestep_indices");
    require_tensor(input.token_tags, "token_tags");
    require_tensor(input.position_ids, "position_ids");
    require_tensor(input.video_indices, "video_indices");
    require_tensor(input.audio_indices, "audio_indices");
    require_tensor(input.text_indices, "text_indices");

    const auto &position_ids = input.position_ids.value();
    const bool has_rotary_cache = input.rotary_cos_sin_cache.has_value();
    if (has_rotary_cache) {
        const auto &rotary_cache = input.rotary_cos_sin_cache.value();
        if (position_ids->shape().size() != 1) {
            throw std::runtime_error("MiniMaxH3Transformer: cached RoPE position_ids "
                                     "must be [sequence_length]");
        }
        if (rotary_cache->shape().size() != 2 || rotary_cache->size(0) != position_ids->size(0)) {
            throw std::runtime_error("MiniMaxH3Transformer: rotary cache must be "
                                     "[sequence_length, rotary_dim]");
        }
    } else if (position_ids->shape().size() != 2 || position_ids->size(1) != 3) {
        throw std::runtime_error(
            "MiniMaxH3Transformer: position_ids must be [sequence_length, 3]");
    }
    const size_t sequence_length = position_ids->size(0);
    const auto &video_hidden_states = input.video_hidden_states.value();
    const auto &audio_hidden_states = input.audio_hidden_states.value();
    const auto &encoder_hidden_states = input.encoder_hidden_states.value();
    if (video_hidden_states->size(0) != 1 || audio_hidden_states->size(0) != 1 || encoder_hidden_states->size(0) != 1) {
        throw std::runtime_error("MiniMaxH3Transformer: the packed FL2VA path "
                                 "currently supports batch size 1");
    }

    auto video_input = cast_to(video_hidden_states, infinicore::DataType::F32);
    auto audio_input = cast_to(audio_hidden_states, infinicore::DataType::F32);
    auto text_input = cast_to(encoder_hidden_states, dtype_);
    auto video_embeds = cast_to(linear_forward(proj_in_, video_input), dtype_);
    auto audio_embeds = cast_to(linear_forward(audio_proj_in_, audio_input), dtype_);
    auto context_embeds = linear_forward(context_embedder_, text_input);
    auto text_embeds = token_refiner_->forward(context_embeds);
    trace_tensor("video_embeds", video_embeds);
    trace_tensor("audio_embeds", audio_embeds);
    trace_tensor("context_embeds", context_embeds);
    trace_tensor("text_embeds", text_embeds);

    // Sequence parallel: rank r owns tokens [r*T, (r+1)*T) of the sequence
    // zero-padded to P*T rows; the padding exists only after the real tokens.
    const size_t local_tokens = sequence_parallel_
        ? (sequence_length + sp_size_ - 1) / sp_size_ : sequence_length;
    const size_t padded_length = sequence_parallel_ ? local_tokens * sp_size_ : sequence_length;
    auto hidden_states = infinicore::Tensor::zeros(
        {1, padded_length, hidden_size_}, dtype_, text_embeds->device());
    hidden_states = infinicore::op::index_copy(
        hidden_states, 1, input.text_indices.value(), text_embeds);
    hidden_states = infinicore::op::index_copy(
        hidden_states, 1, input.video_indices.value(), video_embeds);
    hidden_states = infinicore::op::index_copy(
        hidden_states, 1, input.audio_indices.value(), audio_embeds);

    auto temb = time_embedder_->forward(input.timestep.value());
    trace_tensor("packed", hidden_states);
    trace_tensor("temb", temb);
    auto timestep_indices = input.timestep_indices.value();
    auto token_tags = input.token_tags.value();
    auto timestep_indices_x2 = infinicore::op::add(timestep_indices, timestep_indices);
    auto timestep_indices_x3 = infinicore::op::add(timestep_indices_x2, timestep_indices);
    auto adaln_indices = infinicore::op::add(timestep_indices_x3, token_tags);
    auto rope_positions = has_rotary_cache ? position_ids : position_ids->permute({1, 0});

    if (sequence_parallel_) {
        if (!has_rotary_cache) {
            throw std::runtime_error("MiniMaxH3Transformer: sequence parallelism requires the "
                                     "cached RoPE inputs");
        }
        // Per-token tables for the local rows; padded rows use row 0 and are
        // never used as keys/values nor returned.
        const size_t start = static_cast<size_t>(sp_rank_) * local_tokens;
        auto local_rows = [&](const infinicore::Tensor &table) {
            return pad_rows(table, padded_length)->narrow({{0, start, local_tokens}})->contiguous();
        };
        hidden_states = hidden_states->narrow({{1, start, local_tokens}})->contiguous();
        adaln_indices = local_rows(adaln_indices);
        rope_positions = local_rows(rope_positions);
        timestep_indices = local_rows(timestep_indices);
        sp_valid_tokens = sequence_length;
    }

    // Batched AdaLN (H3_ADALN_BATCH): every block's modulation up front with
    // one all-gather, and the tokens of this rank grouped into segments that
    // share a modulation row. Needs the fused reference AdaLN operators.
    const char *adaln_numerics = std::getenv("H3_ADALN_NUMERICS");
    const bool batched_adaln = adaln_batch_enabled() && !tracing_enabled()
        && fused_residual_enabled() && pointwise_enabled("rms")
        && (adaln_numerics == nullptr || std::strcmp(adaln_numerics, "exact_ref") == 0)
        && (sequence_parallel_ || tp_overlap_chunks() <= 1);
    std::vector<infinicore::Tensor> modulations;
    std::vector<MiniMaxH3ModulationSegment> segments;
    if (batched_adaln) {
        const size_t blocks = transformer_blocks_.size();
        const auto &first = transformer_blocks_[0]->adaln_projection();
        const size_t rows = temb->size(0);
        // The same silu + cast + column-parallel projection as
        // MiniMaxH3AdaLNProjection::forward, for all blocks.
        auto activated = cast_to(infinicore::op::silu(temb), first->dtype());
        std::vector<infinicore::Tensor> locals(blocks);
        for (size_t i = 0; i < blocks; ++i) {
            locals[i] = transformer_blocks_[i]->adaln_projection()->forward_local(activated);
        }
        modulations.resize(blocks);
        const size_t tp = first->tp_size();
        if (tp == 1) {
            for (size_t i = 0; i < blocks; ++i) modulations[i] = locals[i];
        } else {
            // Rows [i * n, (i + 1) * n) of the send buffer hold block i's local
            // columns transposed; the all-gather concatenates ranks on dim 0, so
            // block i's full projection is column r * n + j = gathered[r][i][j].
            const size_t n = locals[0]->size(1);
            auto send = infinicore::Tensor::empty({blocks * n, rows}, locals[0]->dtype(),
                                                  locals[0]->device());
            // rearrange_ directly: Tensor::copy_from would switch this rank
            // thread's InfiniCore context device (and with it its stream).
            for (size_t i = 0; i < blocks; ++i) {
                infinicore::op::rearrange_(send->narrow({{0, i * n, n}}), locals[i]->permute({1, 0}));
            }
            auto gathered = infinicore::op::distributed::allgather(send, tp, first->communicator())
                                ->view({tp, blocks, n, rows});
            for (size_t i = 0; i < blocks; ++i) {
                modulations[i] = gathered->narrow({{1, i, 1}})->permute({1, 3, 0, 2})
                                     ->contiguous()->view({rows, tp * n});
            }
        }
        for (auto &modulation : modulations) {
            modulation = modulation->view({rows * H3_MODALITY_COUNT, 6 * hidden_size_});
        }
        // Run-length segments of this rank's modulation rows: the few thousand
        // indices copied to the host once per forward with plain runtime calls
        // (Tensor::to / copy_from would switch this rank thread's InfiniCore
        // context device, and with it the stream the following operators use).
        const auto indices = adaln_indices->contiguous();
        const size_t count = indices->numel();
        const bool wide = indices->dtype() == infinicore::DataType::I64;
        if (!wide && indices->dtype() != infinicore::DataType::I32) {
            throw std::runtime_error("MiniMaxH3Transformer: AdaLN indices must be int32 or int64");
        }
        std::vector<int64_t> wide_values(wide ? count : 0);
        std::vector<int32_t> narrow_values(wide ? 0 : count);
        check_rt(infinirtStreamSynchronize(infinicore::context::getStream()), "infinirtStreamSynchronize");
        check_rt(infinirtMemcpy(wide ? static_cast<void *>(wide_values.data())
                                     : static_cast<void *>(narrow_values.data()),
                                indices->data(), count * (wide ? 8 : 4), INFINIRT_MEMCPY_D2H),
                 "infinirtMemcpy");
        auto value_at = [&](size_t index) -> size_t {
            return static_cast<size_t>(wide ? wide_values[index] : narrow_values[index]);
        };
        for (size_t index = 0; index < count; ++index) {
            const size_t row = value_at(index);
            if (!segments.empty() && segments.back().row == row) {
                ++segments.back().length;
            } else {
                segments.push_back({index, 1, row});
            }
        }
    }

    const char *chain_flag = std::getenv("H3_BW_CHAIN_RMS");
    if (chain_flag && std::strcmp(chain_flag, "0") != 0 && std::strcmp(chain_flag, "1") != 0) {
        throw std::invalid_argument("H3_BW_CHAIN_RMS must be 0 or 1");
    }
    const bool chain = chain_flag && std::strcmp(chain_flag, "1") == 0
        && batched_adaln && sequence_parallel_ && sp_size_ == 8
        && hidden_size_ == 5376 && dtype_ == infinicore::DataType::BF16;
    std::optional<infinicore::Tensor> chain_normalized;
    for (size_t i = 0; i < transformer_blocks_.size(); ++i) {
        trace_block = static_cast<int>(i);
        if (chain) {
            const auto next = i + 1 < transformer_blocks_.size() ? transformer_blocks_[i+1].get() : nullptr;
            const std::optional<infinicore::Tensor> next_modulation = next
                ? std::optional<infinicore::Tensor>(modulations[i+1]) : std::nullopt;
            std::tie(hidden_states, chain_normalized) = transformer_blocks_[i]->forward_segmented_chain(
                hidden_states, modulations[i], segments, rope_, rope_positions,
                input.rotary_cos_sin_cache, chain_normalized, next, next_modulation);
        } else if (batched_adaln) {
            hidden_states = transformer_blocks_[i]->forward_segmented(
                hidden_states, modulations[i], segments, rope_, rope_positions,
                input.rotary_cos_sin_cache);
        } else {
            hidden_states = transformer_blocks_[i]->forward(
                hidden_states, temb, adaln_indices, rope_, rope_positions,
                input.rotary_cos_sin_cache);
        }
        trace_tensor(("block" + std::to_string(i)).c_str(), hidden_states);
    }
    trace_block = -1;

    auto normalized = norm_out_->forward(hidden_states, temb, timestep_indices);
    trace_tensor("norm_out", normalized);
    auto normalized_f32 = cast_to(normalized, infinicore::DataType::F32);
    auto video_all = linear_forward(proj_out_, normalized_f32)->squeeze(0);
    auto audio_all = linear_forward(audio_proj_out_, normalized_f32)->squeeze(0);
    if (sequence_parallel_) {
        // Output heads ran on local rows only; collect all rows in token order.
        video_all = infinicore::op::distributed::allgather(video_all->contiguous(), sp_size_, sp_comm_)
                        ->narrow({{0, 0, sequence_length}});
        audio_all = infinicore::op::distributed::allgather(audio_all->contiguous(), sp_size_, sp_comm_)
                        ->narrow({{0, 0, sequence_length}});
        sp_valid_tokens = 0;
    }
    auto video_output = gather_rows(video_all, input.video_indices.value())->unsqueeze(0);
    auto audio_output = gather_rows(audio_all, input.audio_indices.value())->unsqueeze(0);
    return {video_output, audio_output};
}

void MiniMaxH3Transformer::reset_cache(const cache::CacheConfig *cache_config) {
    (void)cache_config;
    cache_config_.reset();
    infinilm::global_state::get_forward_context().kv_cache_vec.clear();
}

std::shared_ptr<infinilm::config::ModelConfig> create_minimax_h3_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    if (model_config->get<std::string>("model_type") != "minimax_h3") {
        throw std::runtime_error(
            "create_minimax_h3_model_config: model_type is not minimax_h3");
    }
    model_config->get_config_json()["skip_sampling"] = true;
    return model_config;
}

} // namespace infinilm::models::minimax_h3

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    minimax_h3, infinilm::models::minimax_h3::MiniMaxH3Transformer,
    infinilm::models::minimax_h3::create_minimax_h3_model_config);
} // namespace
