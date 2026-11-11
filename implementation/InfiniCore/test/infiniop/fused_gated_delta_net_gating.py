import ctypes
from ctypes import c_float, c_uint64

import torch
import torch.nn.functional as F
from libinfiniop import (
    LIBINFINIOP,
    InfiniDeviceNames,
    InfiniDtype,
    InfiniDtypeNames,
    TestTensor,
    TestWorkspace,
    check_error,
    debug,
    get_args,
    get_test_devices,
    get_tolerance,
    infiniopOperatorDescriptor_t,
    test_operator,
)


_TEST_CASES = [
    ((2, 1, 8), None, None, None),
    ((2, 3, 17), None, None, None),
    ((2, 3, 17), (80, 20, 1), (80, 20, 1), (2,)),
]

_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 2e-3, "rtol": 2e-3},
    InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-6},
    InfiniDtype.BF16: {"atol": 2e-2, "rtol": 2e-2},
}

DEBUG = False


def torch_fused_gdn_gating(g, beta_output, A_log, a, b, dt_bias, beta, threshold):
    x = a.float() + dt_bias.float().view(1, 1, -1)
    softplus_x = torch.where(
        beta * x <= threshold,
        F.softplus(x, beta=beta, threshold=threshold),
        x,
    )
    g.copy_(-A_log.float().exp().view(1, 1, -1) * softplus_x)
    beta_output.copy_(b.float().sigmoid())


def test(
    handle,
    device,
    shape,
    tensor_stride=None,
    out_stride=None,
    hidden_stride=None,
    dtype=torch.float16,
    sync=None,
):
    beta = 1.0
    threshold = 20.0
    hidden = shape[-1]

    a = TestTensor(shape, tensor_stride, dtype, device)
    b = TestTensor(shape, tensor_stride, dtype, device)
    A_log = TestTensor((hidden,), hidden_stride, dtype, device)
    dt_bias = TestTensor((hidden,), hidden_stride, dtype, device)
    g = TestTensor(shape, out_stride, InfiniDtype.F32, device, mode="ones")
    beta_output = TestTensor(shape, out_stride, InfiniDtype.F32, device, mode="ones")

    if g.is_broadcast() or beta_output.is_broadcast():
        return

    print(
        f"Testing FusedGatedDeltaNetGating on {InfiniDeviceNames[device]} "
        f"shape:{shape} dtype:{InfiniDtypeNames[dtype]}"
    )

    torch_fused_gdn_gating(
        g.torch_tensor(),
        beta_output.torch_tensor(),
        A_log.torch_tensor(),
        a.torch_tensor(),
        b.torch_tensor(),
        dt_bias.torch_tensor(),
        beta,
        threshold,
    )
    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateFusedGatedDeltaNetGatingDescriptor(
            handle,
            ctypes.byref(descriptor),
            g.descriptor,
            beta_output.descriptor,
            A_log.descriptor,
            a.descriptor,
            b.descriptor,
            dt_bias.descriptor,
            c_float(beta),
            c_float(threshold),
        )
    )

    for tensor in [g, beta_output, A_log, a, b, dt_bias]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetFusedGatedDeltaNetGatingWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    check_error(
        LIBINFINIOP.infiniopFusedGatedDeltaNetGating(
            descriptor,
            workspace.data(),
            workspace.size(),
            g.data(),
            beta_output.data(),
            A_log.data(),
            a.data(),
            b.data(),
            dt_bias.data(),
            None,
        )
    )

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(g.actual_tensor(), g.torch_tensor(), atol=atol, rtol=rtol)
        debug(
            beta_output.actual_tensor(),
            beta_output.torch_tensor(),
            atol=atol,
            rtol=rtol,
        )

    assert torch.allclose(g.actual_tensor(), g.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(
        beta_output.actual_tensor(), beta_output.torch_tensor(), atol=atol, rtol=rtol
    )
    check_error(
        LIBINFINIOP.infiniopDestroyFusedGatedDeltaNetGatingDescriptor(descriptor)
    )


if __name__ == "__main__":
    args = get_args()
    DEBUG = args.debug
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)
    print("\033[92m  Test passed!  \033[0m")
