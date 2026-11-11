import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from framework import BaseOperatorTest, GenericTestRunner, TensorSpec, TestCase

import infinicore

_TEST_CASES_DATA = [
    ((2, 4), None, None, None),
    ((128, 64), None, None, None),
    ((2, 4, 8), None, None, None),
    ((13, 4), (10, 1), (10, 1), (10, 1)),
    ((13, 4, 4), (20, 4, 1), (20, 4, 1), (20, 4, 1)),
    ((16, 5632), None, None, None),
]

_BETAS = [(4.0, 25.0), (2.0, 8.0)]

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 2e-3, "rtol": 2e-3},
    infinicore.bfloat16: {"atol": 2e-2, "rtol": 2e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-5},
}


def build_test_cases():
    test_cases = []
    for shape, gate_strides, up_strides, output_strides in _TEST_CASES_DATA:
        for beta, linear_beta in _BETAS:
            for dtype in _TENSOR_DTYPES:
                gate_spec = TensorSpec.from_tensor(shape, gate_strides, dtype)
                up_spec = TensorSpec.from_tensor(shape, up_strides, dtype)
                output_spec = TensorSpec.from_tensor(shape, output_strides, dtype)
                kwargs = {"beta": beta, "linear_beta": linear_beta}
                tolerance = _TOLERANCE_MAP[dtype]

                test_cases.append(
                    TestCase(
                        inputs=[gate_spec, up_spec],
                        kwargs=kwargs,
                        output_spec=None,
                        comparison_target=None,
                        tolerance=tolerance,
                        description="SiTUAndMul - OUT_OF_PLACE",
                    )
                )
                test_cases.append(
                    TestCase(
                        inputs=[gate_spec, up_spec],
                        kwargs=kwargs,
                        output_spec=output_spec,
                        comparison_target="out",
                        tolerance=tolerance,
                        description="SiTUAndMul - EXPLICIT_OUTPUT",
                    )
                )
    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("SiTUAndMul")

    def get_test_cases(self):
        return build_test_cases()

    def torch_operator(self, gate, up, beta=4.0, linear_beta=25.0, out=None):
        result = (
            beta
            * torch.tanh(gate / beta)
            * torch.sigmoid(gate)
            * linear_beta
            * torch.tanh(up / linear_beta)
        )
        if out is not None:
            out.copy_(result)
            return out
        return result

    def infinicore_operator(
        self,
        gate,
        up,
        beta=4.0,
        linear_beta=25.0,
        out=None,
    ):
        return infinicore.situ_and_mul(
            gate,
            up,
            beta,
            linear_beta,
            out=out,
        )


def main():
    GenericTestRunner(OpTest).run_and_exit()


if __name__ == "__main__":
    main()
