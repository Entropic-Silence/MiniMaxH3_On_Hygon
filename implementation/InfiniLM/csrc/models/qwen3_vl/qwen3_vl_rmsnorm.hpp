#pragma once

#include <infinicore/nn/rmsnorm.hpp>
#include <infinicore/ops.hpp>
#include <infinicore/ops/mul.hpp>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace infinilm::models::qwen3_vl {

// Qwen3VLTextRMSNorm rounds normalization to the input dtype before the
// learned weight multiply. Its residual additions also materialize BF16
// before the RMS reduction. Compose existing InfiniCore operations to keep
// both boundaries explicit; no new compute kernel or checkpoint parameter.
class Qwen3VLReferenceRMSNorm : public infinicore::nn::RMSNorm {
public:
    Qwen3VLReferenceRMSNorm(size_t size, double eps,
                          const infinicore::DataType &dtype,
                          const infinicore::Device &device)
        : infinicore::nn::RMSNorm(size, eps, dtype, device) {
        const char *value = std::getenv("H3_QWEN_RMS");
        const std::string mode = value ? value : "baseline";
        if (mode != "baseline" && mode != "unfused" && mode != "reference") {
            throw std::invalid_argument("H3_QWEN_RMS must be baseline, unfused, or reference");
        }
        if (dtype == infinicore::DataType::BF16) {
            unfused_ = mode != "baseline";
            split_weight_ = mode == "reference";
        }
        if (split_weight_) {
            unit_weight_ = infinicore::Tensor::ones({size}, dtype, device);
        }
    }

    infinicore::Tensor forward(const infinicore::Tensor &x) const {
        if (!split_weight_) return infinicore::nn::RMSNorm::forward(x);
        auto normalized = infinicore::op::rms_norm(x, unit_weight_, static_cast<float>(eps()));
        infinicore::Strides broadcast_strides(normalized->ndim(), 0);
        broadcast_strides.back() = 1;
        auto broadcast_weight = weight()->as_strided(normalized->shape(), broadcast_strides);
        return infinicore::op::mul(normalized, broadcast_weight);
    }

    void forward_inplace(infinicore::Tensor &x, infinicore::Tensor &residual) const {
        if (!unfused_) {
            infinicore::nn::RMSNorm::forward_inplace(x, residual);
            return;
        }
        residual = residual ? infinicore::op::add(residual, x) : x;
        x = forward(residual);
    }

private:
    bool unfused_{false};
    bool split_weight_{false};
    infinicore::Tensor unit_weight_;
};
} // namespace infinilm::models::qwen3_vl
