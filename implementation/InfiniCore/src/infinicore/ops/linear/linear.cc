#include "infinicore/ops/linear.hpp"
#include "infinicore/ops/gemm.hpp"
#include "infinicore/ops/rearrange.hpp"

#if defined(ENABLE_HYGON_API) && defined(ENABLE_ATEN)
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <cstdlib>
#include <cstring>
#endif

namespace infinicore::op {

#if defined(ENABLE_HYGON_API) && defined(ENABLE_ATEN)
namespace {
// k100-extreme, opt-in (INFINICORE_HYGON_LINEAR=lt): BF16 linear layers
// through ATen with hipBLASLt, which picks faster kernels than the rocBLAS
// path for H3's SP out/down projections on gfx928 (about 2% end to end).
// Off by default: hipBLASLt loads kernel modules lazily per shape inside the
// rank threads, and module loading concurrent with kernel launches in other
// threads was seen to crash the DTK HIP runtime.
bool aten_linear_enabled() {
    static const bool enabled = [] {
        const char *mode = std::getenv("INFINICORE_HYGON_LINEAR");
        if (mode == nullptr || std::strcmp(mode, "lt") != 0) {
            return false;
        }
        at::globalContext().setBlasPreferredBackend(at::BlasBackend::Cublaslt);
        return true;
    }();
    return enabled;
}
} // namespace
#endif

Tensor linear(Tensor input,
              Tensor weight,
              std::optional<Tensor> bias,
              float alpha) {

    Size ndim = input->ndim();
    Size out_features = weight->shape()[0];

    // Assign memory to out variables
    auto output_shape = input->shape();
    output_shape[ndim - 1] = out_features;
    auto out = Tensor::empty(output_shape, input->dtype(), input->device());

    // Inplace Calculate
    linear_(out, input, weight, bias, alpha);
    return out;
}

void linear_(Tensor out,
             Tensor input,
             Tensor weight,
             std::optional<Tensor> bias,
             float alpha) {

    auto weight_shape = weight->shape();
    Size out_features = weight_shape[0];
    Size in_features = weight_shape[1];

    Size ndim = input->ndim();
    assert(out->ndim() == ndim);

    // Calculate the number of features
    Size N = 1;
    auto input_shape = input->shape();
    for (size_t i = 0; i < ndim - 1; ++i) {
        N *= input_shape[i];
    }

    // linear transformation
    Tensor out_view = out->view({N, out_features});
#if defined(ENABLE_HYGON_API) && defined(ENABLE_ATEN)
    if (!bias.has_value() && alpha == 1.0f && out->dtype() == DataType::BF16
        && out->device().getType() == Device::Type::HYGON && aten_linear_enabled()) {
#if INFINICORE_TORCH_VERSION_GE_2_11
        c10::cuda::CUDAStreamGuard guard(adaptor::get_hip_stream());
#else
        c10::hip::HIPStreamGuard guard(adaptor::get_hip_stream());
#endif
        auto result = adaptor::to_aten_tensor(out_view);
        at::mm_out(result, adaptor::to_aten_tensor(input->view({N, in_features})),
                   adaptor::to_aten_tensor(weight).t());
        return;
    }
#endif
    // Add bias
    float beta = 0.0f;
    if (bias.has_value()) {
        rearrange_(out_view,
                   bias.value()->as_strided({N, out_features}, {0, 1}));
        beta = 1.0f;
    }

    gemm_(out_view,
          input->view({N, in_features}),
          weight->permute({1, 0}), alpha, beta);
}

// --- Pre-packed weight variants ---
// packed_weight layout: [in_features, out_features] (already transposed from [OC,IC])

Tensor linear_packed(Tensor input,
                     Tensor packed_weight,
                     std::optional<Tensor> bias,
                     float alpha) {

    Size ndim = input->ndim();
    // packed_weight shape is [IC, OC], so OC is dim 1
    Size out_features = packed_weight->shape()[1];

    auto output_shape = input->shape();
    output_shape[ndim - 1] = out_features;
    auto out = Tensor::empty(output_shape, input->dtype(), input->device());

    linear_packed_(out, input, packed_weight, bias, alpha);
    return out;
}

void linear_packed_(Tensor out,
                    Tensor input,
                    Tensor packed_weight,
                    std::optional<Tensor> bias,
                    float alpha) {

    auto weight_shape = packed_weight->shape();
    // packed_weight is [IC, OC] — already the layout GEMM expects
    Size in_features = weight_shape[0];
    Size out_features = weight_shape[1];

    Size ndim = input->ndim();
    assert(out->ndim() == ndim);

    Size N = 1;
    auto input_shape = input->shape();
    for (size_t i = 0; i < ndim - 1; ++i) {
        N *= input_shape[i];
    }

    Tensor out_view = out->view({N, out_features});

    // Add bias (same logic as linear_)
    float beta = 0.0f;
    if (bias.has_value()) {
        rearrange_(out_view,
                   bias.value()->as_strided({N, out_features}, {0, 1}));
        beta = 1.0f;
    }

    // KEY: no weight->permute({1, 0}) here!
    // packed_weight is already [IC, OC] contiguous.
    gemm_(out_view,
          input->view({N, in_features}),
          packed_weight, alpha, beta);
}

} // namespace infinicore::op
