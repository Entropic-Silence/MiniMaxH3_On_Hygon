#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/linear/fused_linear.hpp"
#include "../../layers/linear/linear.hpp"
#include "../infinilm_model.hpp"

#include <infinicore/context/context.hpp>
#include <infinicore/nn/module.hpp>
#include <infinicore/nn/rmsnorm.hpp>
#include <infinicore/nn/rope.hpp>
#include <infinicore/tensor.hpp>

#include <array>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace infinilm::models::minimax_h3 {

// How a layer is distributed over the engine's ranks.
//   TENSOR:     Megatron tensor parallelism (sharded heads/FFN, all-reduce).
//   SEQUENCE:   Ulysses sequence parallelism: full weights, each rank owns a
//               contiguous token range; attention exchanges heads/tokens with
//               two all-to-alls around the full-sequence attention.
//   REPLICATED: full weights, identical computation on every rank.
enum class MiniMaxH3Parallel { TENSOR, SEQUENCE, REPLICATED };

class MiniMaxH3RMSNorm : public infinicore::nn::RMSNorm {
public:
    using infinicore::nn::RMSNorm::RMSNorm;
    infinicore::Tensor forward(const infinicore::Tensor &x) const;
};

class MiniMaxH3Attention : public infinicore::nn::Module {
public:
    MiniMaxH3Attention(size_t hidden_size,
                       size_t num_heads,
                       size_t head_dim,
                       double qk_norm_eps,
                       const infinicore::DataType &dtype,
                       const infinicore::Device &device,
                       MiniMaxH3Parallel parallel = MiniMaxH3Parallel::TENSOR);

    infinicore::Tensor forward(
        const infinicore::Tensor &hidden_states,
        const std::shared_ptr<infinicore::nn::RoPE> &rope = nullptr,
        const std::optional<infinicore::Tensor> &position_ids = std::nullopt,
        const std::optional<infinicore::Tensor> &rotary_cos_sin_cache = std::nullopt) const;

    // Q/K/V projection with per-head norms and RoPE: three [B, S, H, D] tensors.
    std::tuple<infinicore::Tensor, infinicore::Tensor, infinicore::Tensor> project_qkv(
        const infinicore::Tensor &hidden_states,
        const std::shared_ptr<infinicore::nn::RoPE> &rope,
        const std::optional<infinicore::Tensor> &position_ids,
        const std::optional<infinicore::Tensor> &rotary_cos_sin_cache) const;

    // Attention of query rows [start, start + rows) over all keys, followed by
    // this rank's output-projection contribution (not all-reduced): [1, rows, hidden].
    infinicore::Tensor attend_rows_local(const infinicore::Tensor &query,
                                         const infinicore::Tensor &key,
                                         const infinicore::Tensor &value,
                                         size_t start, size_t rows) const;

    void process_weights_after_loading() override;
    void reset_runtime_state() const override;

private:
    // Sequence parallel: [1, local_tokens, heads, D] -> [1, ranks * local_tokens, heads / ranks, D].
    infinicore::Tensor gather_sequence(const infinicore::Tensor &local) const;
    std::tuple<infinicore::Tensor, infinicore::Tensor, infinicore::Tensor> project_direct_packed_qkv(
        const infinicore::Tensor &hidden_states, const infinicore::Tensor &position_ids,
        const infinicore::Tensor &rotary_cache) const;
    // Exchange full QKV together; projection and attention arithmetic unchanged.
    std::tuple<infinicore::Tensor, infinicore::Tensor, infinicore::Tensor> gather_qkv(
        const infinicore::Tensor &query, const infinicore::Tensor &key,
        const infinicore::Tensor &value) const;
    // Inverse of gather_sequence for the attention output.
    infinicore::Tensor scatter_sequence(const infinicore::Tensor &global, size_t local_tokens) const;
    // Sequence-parallel attention with head-group exchanges overlapped on a side stream.
    infinicore::Tensor overlapped_sequence_attention(const infinicore::Tensor &query,
                                                     const infinicore::Tensor &key,
                                                     const infinicore::Tensor &value,
                                                     size_t tokens, size_t groups) const;
    // Waits for each group's exchanged q/k/v ([P, T, Hs_g, D] blocks), attends
    // and sends the output back; returns the local [1, T, hidden] output.
    // `group_heads[g]` heads per rank in group g (groups may differ in size).
    infinicore::Tensor attend_exchanged_groups(
        const std::vector<std::array<infinicore::Tensor, 3>> &received,
        const std::vector<infinirtEvent_t> &arrived, size_t tokens,
        const std::vector<size_t> &group_heads) const;
    // As above, but the QKV projection itself runs per head group, so group
    // g's exchange overlaps group g+1's projection (k100-extreme).
    infinicore::Tensor pipelined_sequence_attention(
        const infinicore::Tensor &hidden_states,
        const std::shared_ptr<infinicore::nn::RoPE> &rope,
        const std::optional<infinicore::Tensor> &position_ids,
        const std::optional<infinicore::Tensor> &rotary_cos_sin_cache,
        const std::vector<size_t> &group_heads) const;
    // Per-head q/k RMSNorm and RoPE: [B*S, H, D] views -> [B, S, H, D].
    std::pair<infinicore::Tensor, infinicore::Tensor> normalize_rotate(
        infinicore::Tensor query, infinicore::Tensor key, size_t batch_size,
        size_t sequence_length, size_t heads,
        const std::shared_ptr<infinicore::nn::RoPE> &rope,
        const std::optional<infinicore::Tensor> &position_ids,
        const std::optional<infinicore::Tensor> &rotary_cos_sin_cache) const;
    // Reorders the fused QKV weight rows to [group][q|k|v][rank][head][D] once.
    void group_qkv_weight(const std::vector<size_t> &group_heads) const;

    size_t num_heads_;
    size_t head_dim_;
    size_t inner_dim_;
    float scale_;
    MiniMaxH3Parallel parallel_;
    size_t sp_size_ = 1;
    int sp_rank_ = 0;
    infinicclComm_t sp_comm_ = nullptr;
    mutable std::vector<size_t> qkv_group_heads_;  // head groups the QKV weight rows are ordered by

    std::shared_ptr<infinilm::layers::linear::QKVParallelLinear> qkv_proj_;
    INFINICORE_NN_MODULE(MiniMaxH3RMSNorm, norm_q);
    INFINICORE_NN_MODULE(MiniMaxH3RMSNorm, norm_k);
    INFINICORE_NN_MODULE(infinilm::layers::linear::RowParallelLinear, out_proj);
};

class MiniMaxH3MLP : public infinicore::nn::Module {
public:
    MiniMaxH3MLP(size_t hidden_size,
                 size_t ffn_hidden_size,
                 const infinicore::DataType &dtype,
                 const infinicore::Device &device,
                 MiniMaxH3Parallel parallel = MiniMaxH3Parallel::TENSOR);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

    // This rank's contribution before the row-parallel all-reduce.
    infinicore::Tensor forward_local(const infinicore::Tensor &hidden_states) const;

    void process_weights_after_loading() override;
    void reset_runtime_state() const override;

private:
    std::shared_ptr<infinilm::layers::linear::GateUpParallelLinear> gate_up_proj_;
    INFINICORE_NN_MODULE(infinilm::layers::linear::RowParallelLinear, down_proj);
};

class MiniMaxH3TokenRefinerBlock : public infinicore::nn::Module {
public:
    MiniMaxH3TokenRefinerBlock(size_t hidden_size,
                               size_t num_heads,
                               size_t head_dim,
                               size_t ffn_hidden_size,
                               double norm_eps,
                               double qk_norm_eps,
                               const infinicore::DataType &dtype,
                               const infinicore::Device &device,
                               MiniMaxH3Parallel parallel = MiniMaxH3Parallel::TENSOR);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    INFINICORE_NN_MODULE(MiniMaxH3RMSNorm, norm1);
    INFINICORE_NN_MODULE(MiniMaxH3Attention, attn);
    INFINICORE_NN_MODULE(MiniMaxH3RMSNorm, norm2);
    INFINICORE_NN_MODULE(MiniMaxH3MLP, ff);
};

class MiniMaxH3TokenRefiner : public infinicore::nn::Module {
public:
    MiniMaxH3TokenRefiner(size_t hidden_size,
                          size_t num_heads,
                          size_t head_dim,
                          size_t ffn_hidden_size,
                          size_t num_layers,
                          double norm_eps,
                          double qk_norm_eps,
                          double final_norm_eps,
                          const infinicore::DataType &dtype,
                          const infinicore::Device &device,
                          MiniMaxH3Parallel parallel = MiniMaxH3Parallel::TENSOR);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    INFINICORE_NN_MODULE_VEC(MiniMaxH3TokenRefinerBlock, refiner_blocks);
    INFINICORE_NN_MODULE(MiniMaxH3RMSNorm, final_norm);
};

// Tokens [start, start + length) of a rank's rows that share one AdaLN
// modulation row (one timestep and modality).
struct MiniMaxH3ModulationSegment {
    size_t start;
    size_t length;
    size_t row;
};

class MiniMaxH3AdaLNProjection : public infinicore::nn::Module {
public:
    MiniMaxH3AdaLNProjection(size_t in_features,
                             size_t out_features,
                             const infinicore::DataType &dtype,
                             const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &temb) const;
    // This rank's output columns of the projection of silu(temb) already cast
    // to the projection dtype (the part forward() all-gathers).
    infinicore::Tensor forward_local(const infinicore::Tensor &activated) const;
    const infinicore::DataType &dtype() const { return dtype_; }
    size_t tp_size() const { return tp_size_; }
    infinicclComm_t communicator() const { return communicator_; }

private:
    infinicore::DataType dtype_;
    size_t tp_size_;
    infinicclComm_t communicator_;
    INFINICORE_NN_MODULE(infinilm::layers::linear::ColumnParallelLinear, linear);
};

class MiniMaxH3TransformerBlock : public infinicore::nn::Module {
public:
    MiniMaxH3TransformerBlock(size_t hidden_size,
                              size_t num_heads,
                              size_t head_dim,
                              size_t ffn_hidden_size,
                              size_t time_embed_dim,
                              double norm_eps,
                              double qk_norm_eps,
                              const infinicore::DataType &dtype,
                              const infinicore::Device &device,
                              MiniMaxH3Parallel parallel = MiniMaxH3Parallel::TENSOR);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &temb,
                               const infinicore::Tensor &adaln_indices,
                               const std::shared_ptr<infinicore::nn::RoPE> &rope,
                               const infinicore::Tensor &position_ids,
                               const std::optional<infinicore::Tensor> &rotary_cos_sin_cache = std::nullopt) const;

    // Same block with the modulation precomputed ([rows, 6 * hidden]) and the
    // AdaLN operators applied per segment of tokens sharing a modulation row
    // (broadcast views instead of a per-token gathered table).
    infinicore::Tensor forward_segmented(const infinicore::Tensor &hidden_states,
                                         const infinicore::Tensor &modulation,
                                         const std::vector<MiniMaxH3ModulationSegment> &segments,
                                         const std::shared_ptr<infinicore::nn::RoPE> &rope,
                                         const infinicore::Tensor &position_ids,
                                         const std::optional<infinicore::Tensor> &rotary_cos_sin_cache) const;

    // Same-forward look-ahead: carries only the next block's exact normalized
    // input. Caller owns this transient value; no cross-step/cache state.
    std::pair<infinicore::Tensor, std::optional<infinicore::Tensor>> forward_segmented_chain(
        const infinicore::Tensor &hidden_states, const infinicore::Tensor &modulation,
        const std::vector<MiniMaxH3ModulationSegment> &segments,
        const std::shared_ptr<infinicore::nn::RoPE> &rope,
        const infinicore::Tensor &position_ids,
        const std::optional<infinicore::Tensor> &rotary_cos_sin_cache,
        const std::optional<infinicore::Tensor> &normalized_input,
        const MiniMaxH3TransformerBlock *next_block,
        const std::optional<infinicore::Tensor> &next_modulation) const;

    const std::shared_ptr<MiniMaxH3AdaLNProjection> &adaln_projection() const { return adaln_proj_; }

private:
    // Tensor parallel with the post-attention part pipelined over row chunks,
    // so each chunk's all-reduce overlaps the next chunk's computation.
    infinicore::Tensor forward_overlapped(
        const infinicore::Tensor &hidden_states, const infinicore::Tensor &token_modulation,
        const std::shared_ptr<infinicore::nn::RoPE> &rope, const infinicore::Tensor &position_ids,
        const std::optional<infinicore::Tensor> &rotary_cos_sin_cache, size_t chunks) const;

    size_t hidden_size_;
    MiniMaxH3Parallel parallel_;
    size_t tp_size_ = 1;
    infinicclComm_t tp_comm_ = nullptr;
    INFINICORE_NN_MODULE(MiniMaxH3RMSNorm, norm1);
    INFINICORE_NN_MODULE(MiniMaxH3Attention, attn);
    INFINICORE_NN_MODULE(MiniMaxH3RMSNorm, norm2);
    INFINICORE_NN_MODULE(MiniMaxH3MLP, ff);
    INFINICORE_NN_MODULE(MiniMaxH3AdaLNProjection, adaln_proj);
};

class MiniMaxH3TimeEmbedder : public infinicore::nn::Module {
public:
    MiniMaxH3TimeEmbedder(size_t input_dim,
                          size_t hidden_size,
                          size_t output_dim,
                          const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &timestep) const;

private:
    size_t input_dim_;
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, linear_1);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, linear_2);
};

class MiniMaxH3OutputNorm : public infinicore::nn::Module {
public:
    MiniMaxH3OutputNorm(size_t hidden_size,
                        size_t time_embed_dim,
                        double norm_eps,
                        const infinicore::DataType &dtype,
                        const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &temb,
                               const infinicore::Tensor &timestep_indices) const;

private:
    size_t hidden_size_;
    infinicore::DataType dtype_;
    INFINICORE_NN_MODULE(MiniMaxH3RMSNorm, norm);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, linear);
};

class MiniMaxH3Transformer : public infinilm::InfinilmModel {
public:
    MiniMaxH3Transformer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                         const infinicore::Device &device);

    Output forward(const Input &input) const override;
    void reset_cache(const cache::CacheConfig *cache_config) override;

private:
    size_t hidden_size_;
    infinicore::DataType dtype_;
    // Ulysses sequence parallelism over the engine ranks (H3_PARALLEL=sp).
    bool sequence_parallel_ = false;
    size_t sp_size_ = 1;
    int sp_rank_ = 0;
    infinicclComm_t sp_comm_ = nullptr;

    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, proj_in);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, audio_proj_in);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, context_embedder);
    INFINICORE_NN_MODULE(MiniMaxH3TimeEmbedder, time_embedder);
    INFINICORE_NN_MODULE(infinicore::nn::RoPE, rope);
    INFINICORE_NN_MODULE(MiniMaxH3TokenRefiner, token_refiner);
    INFINICORE_NN_MODULE_VEC(MiniMaxH3TransformerBlock, transformer_blocks);
    INFINICORE_NN_MODULE(MiniMaxH3OutputNorm, norm_out);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, proj_out);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, audio_proj_out);
};

std::shared_ptr<infinilm::config::ModelConfig>
create_minimax_h3_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::minimax_h3
