import torch
import ctypes
from ctypes import c_uint64
from libinfiniop import *
from enum import Enum, auto
from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    get_test_devices,
    check_error,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    TestWorkspace,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
)

_TEST_CASES_ = [
    ((13, 4), None, None),
    ((13, 4), (10, 1), (10, 1)),
    ((13, 4, 4), None, None),
]

_TEST_CASES = _TEST_CASES_
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]

_TOLERANCE_MAP = {
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-4, "rtol": 1e-4},
}


def test(
    handle,
    device,
    shape,
    input_stride=None,
    output_stride=None,
    dtype=torch.float16,
    sync=None,
):
    print(
        f"Testing Erf on {InfiniDeviceNames[device]} with shape:{shape} input_stride:{input_stride} output_stride:{output_stride} "
        f"dtype:{InfiniDtypeNames[dtype]}"
    )
    input = TestTensor(shape, input_stride, dtype, device)
    output = TestTensor(shape, output_stride, dtype, device)

    output.update_torch_tensor(torch.erf(input.torch_tensor()))

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateErfDescriptor(
            handle, ctypes.byref(descriptor), output.descriptor, input.descriptor
        )
    )

    input.destroy_desc()
    output.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetErfWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    def lib_f():
        check_error(
            LIBINFINIOP.infiniopErf(
                descriptor,
                workspace.data(),
                workspace.size(),
                output.data(),
                input.data(),
                None,
            )
        )

    lib_f()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(output.actual_tensor(), output.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(
        output.actual_tensor(), output.torch_tensor(), atol=atol, rtol=rtol
    )

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: torch.erf(input.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_f(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyErfDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
