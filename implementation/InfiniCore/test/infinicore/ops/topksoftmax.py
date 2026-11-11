import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
import torch.nn.functional as F
from framework import (
    BaseOperatorTest,
    GenericTestRunner,
    TensorInitializer,
    TensorSpec,
    TestCase,
    is_broadcast,
)
from infinicore.lib import _infinicore

# (input_shape, input_strides, topk, norm) — norm is 0/1 for C++ binding (infiniop bool).
# Strides None only: kernel path matches contiguous layouts as in test/infiniop/topksoftmax.py.
_TEST_CASES_DATA = [
    ((1, 10), None, 7, 1),
    ((8, 20), None, 4, 1),
    ((2, 64), None, 6, 1),
    ((4, 16), None, 3, 0),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-3},
    infinicore.float32: {"atol": 1e-3, "rtol": 1e-3},
    infinicore.bfloat16: {"atol": 1e-3, "rtol": 1e-3},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def torch_topksoftmax(router_logits, top_k, norm_topk_prob=False):
    """Reference implementation aligned with test/infiniop/topksoftmax.py."""
    routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    if norm_topk_prob:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    return routing_weights, selected_experts.to(torch.int32)


def parse_test_cases():
    test_cases = []
    for shape, in_strides, topk, norm in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-3, "rtol": 1e-3})
            base = (
                torch.arange(0, shape[0] * shape[1], dtype=torch.float32).reshape(shape)
                * 0.5
            )
            input_spec = TensorSpec.from_tensor(
                shape,
                in_strides,
                dtype,
                init_mode=TensorInitializer.MANUAL,
                set_tensor=base,
            )
            n = shape[0]
            out_shape = (n, topk)

            desc_parts = [f"topk={topk}", f"norm={norm}"]
            if in_strides:
                desc_parts.append(f"strides={in_strides}")
            suffix = ", ".join(desc_parts)

            kwargs = {"topk": topk, "norm": norm}

            test_cases.append(
                TestCase(
                    inputs=[input_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description=f"topksoftmax - OUT_OF_PLACE - {suffix}",
                    output_count=2,
                )
            )

            values_spec = TensorSpec.from_tensor(out_shape, None, infinicore.float32)
            indices_spec = TensorSpec.from_tensor(out_shape, None, infinicore.int32)

            if not is_broadcast(values_spec.strides) and not is_broadcast(
                indices_spec.strides
            ):
                test_cases.append(
                    TestCase(
                        inputs=[input_spec],
                        kwargs=kwargs.copy(),
                        output_specs=[values_spec, indices_spec],
                        comparison_target="out",
                        tolerance=tol,
                        description=f"topksoftmax - INPLACE(out) - {suffix}",
                        output_count=2,
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("topksoftmax")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, x, topk, norm=0, out=None, **kwargs):
        norm_bool = norm != 0
        values, indices = torch_topksoftmax(x, topk, norm_bool)
        if out is not None:
            out_v, out_i = out
            out_v.copy_(values)
            out_i.copy_(indices)
        return values, indices

    def infinicore_operator(self, x, topk, norm=0, out=None, **kwargs):
        n = x.shape[0]
        if out is None:
            values = infinicore.empty(
                (n, topk), dtype=infinicore.float32, device=x.device
            )
            indices = infinicore.empty(
                (n, topk), dtype=infinicore.int32, device=x.device
            )
        else:
            values, indices = out[0], out[1]

        _infinicore.topksoftmax(
            values._underlying, indices._underlying, x._underlying, topk, int(norm)
        )
        return values, indices


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
