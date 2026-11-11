import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from framework import (
    BaseOperatorTest,
    GenericTestRunner,
    TensorInitializer,
    TensorSpec,
    TestCase,
)

import infinicore

_TEST_CASES_DATA = [
    # batch, seq_len, intermediate, state_size
    (1, 1, 4, 2),
    (1, 5, 8, 4),
    (2, 7, 16, 4),
    (4, 3, 12, 8),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 3e-2, "rtol": 3e-2},
    infinicore.bfloat16: {"atol": 8e-2, "rtol": 8e-2},
    infinicore.float32: {"atol": 5e-4, "rtol": 5e-4},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def torch_mamba_selective_scan_ref(x, dt, b, c, a_log, d, gate, dt_bias, state):
    x_f = x.float()
    dt_f = dt.float()
    b_f = b.float()
    c_f = c.float()
    a_log_f = a_log.float()
    d_f = d.float()
    gate_f = gate.float()
    dt_bias_f = dt_bias.float()
    state_f = state.float().clone()

    batch, seq_len, intermediate = x.shape
    state_size = state.shape[-1]
    out = torch.empty_like(x_f)

    for batch_idx in range(batch):
        for ch in range(intermediate):
            s = state_f[batch_idx, ch]
            for t in range(seq_len):
                xt = x_f[batch_idx, t, ch]
                dtv = torch.nn.functional.softplus(
                    dt_f[batch_idx, t, ch] + dt_bias_f[ch]
                )
                a = -torch.exp(a_log_f[ch])
                discrete_a = torch.exp(a * dtv)
                s = discrete_a * s + dtv * b_f[batch_idx, t, :state_size] * xt
                state_f[batch_idx, ch] = s
                y = (s * c_f[batch_idx, t, :state_size]).sum() + xt * d_f[ch]
                y = y * torch.nn.functional.silu(gate_f[batch_idx, t, ch])
                out[batch_idx, t, ch] = y

    state.copy_(state_f.to(state.dtype))
    return out.to(x.dtype)


def _tensor(shape, dtype, name, scale=0.2, bias=-0.1):
    return TensorSpec.from_tensor(
        shape,
        None,
        dtype,
        name=name,
        init_mode=TensorInitializer.RANDOM,
        scale=scale,
        bias=bias,
    )


def parse_test_cases():
    tests = []
    for batch, seq_len, intermediate, state_size in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            inputs = [
                _tensor((batch, seq_len, intermediate), dtype, "x"),
                _tensor((batch, seq_len, intermediate), dtype, "dt"),
                _tensor((batch, seq_len, state_size), dtype, "b"),
                _tensor((batch, seq_len, state_size), dtype, "c"),
                _tensor((intermediate, state_size), dtype, "a_log"),
                _tensor((intermediate,), dtype, "d"),
                _tensor((batch, seq_len, intermediate), dtype, "gate"),
                _tensor((intermediate,), dtype, "dt_bias"),
                _tensor((batch, intermediate, state_size), infinicore.float32, "state"),
            ]
            tests.append(
                TestCase(
                    inputs=inputs,
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=_TOLERANCE_MAP[dtype],
                    description="MambaSelectiveScan - output and state",
                    output_count=2,
                )
            )
    return tests


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("MambaSelectiveScan")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, x, dt, b, c, a_log, d, gate, dt_bias, state, **kwargs):
        out = torch_mamba_selective_scan_ref(
            x, dt, b, c, a_log, d, gate, dt_bias, state
        )
        return out, state

    def infinicore_operator(
        self, x, dt, b, c, a_log, d, gate, dt_bias, state, **kwargs
    ):
        out = infinicore.nn.functional.mamba_selective_scan(
            x, dt, b, c, a_log, d, gate, dt_bias, state
        )
        return out, state


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
