#ifndef __SITU_AND_MUL_CUDA_H__
#define __SITU_AND_MUL_CUDA_H__

#include <cmath>

namespace op::situ_and_mul::cuda {

struct SituAndMulOp {
    static constexpr size_t num_inputs = 2;

    template <typename T>
    __device__ __forceinline__ T operator()(
        const T &gate,
        const T &up,
        float beta,
        float linear_beta) const {
        float gate_f;
        float up_f;
        if constexpr (std::is_same_v<T, half>) {
            gate_f = __half2float(gate);
            up_f = __half2float(up);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            gate_f = __bfloat162float(gate);
            up_f = __bfloat162float(up);
        } else {
            gate_f = static_cast<float>(gate);
            up_f = static_cast<float>(up);
        }

        float sigmoid_gate;
        if (gate_f >= 0.0f) {
            sigmoid_gate = 1.0f / (1.0f + expf(-gate_f));
        } else {
            float exp_gate = expf(gate_f);
            sigmoid_gate = exp_gate / (1.0f + exp_gate);
        }
        float result = beta * tanhf(gate_f / beta) * sigmoid_gate
                     * linear_beta * tanhf(up_f / linear_beta);

        if constexpr (std::is_same_v<T, half>) {
            return __float2half_rn(result);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            return __float2bfloat16_rn(result);
        } else {
            return static_cast<T>(result);
        }
    }
};

} // namespace op::situ_and_mul::cuda

#endif
