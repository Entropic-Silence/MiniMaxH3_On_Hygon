import ctypes
from ctypes import c_uint64

import torch
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
    profile_operation,
    test_operator,
)


# ==============================================================================
#  Reference Implementation
# ==============================================================================
def ref_nsa_compress_paged_cache(
    k_cmp,
    v_cmp,
    k_cache,
    v_cache,
    block_tables,
    seq_lens,
    nsa_block_size,
    update_last_only,
):
    k_ref = k_cmp.clone()
    v_ref = v_cmp.clone()
    page_block_size = k_cache.shape[2]
    subblocks_per_page = page_block_size // nsa_block_size

    for seq in range(block_tables.shape[0]):
        seq_len = int(seq_lens[seq].item())
        if seq_len <= 0:
            continue
        if update_last_only:
            nsa_blocks = [(seq_len - 1) // nsa_block_size]
        else:
            nsa_blocks = range((seq_len + nsa_block_size - 1) // nsa_block_size)

        for nsa_block in nsa_blocks:
            tok_begin = nsa_block * nsa_block_size
            if tok_begin >= seq_len:
                continue
            tok_end = min(tok_begin + nsa_block_size, seq_len)
            logical_page = tok_begin // page_block_size
            subblock = (tok_begin % page_block_size) // nsa_block_size
            physical = int(block_tables[seq, logical_page].item())
            cmp_block = physical * subblocks_per_page + subblock
            rows = (
                torch.arange(tok_begin, tok_end, device=k_cache.device)
                % page_block_size
            )
            k_ref[cmp_block] = (
                k_cache[physical, :, rows, :].float().mean(dim=1).to(k_ref.dtype)
            )
            v_ref[cmp_block] = (
                v_cache[physical, :, rows, :].float().mean(dim=1).to(v_ref.dtype)
            )
    return k_ref, v_ref


# ==============================================================================
#  Test Configuration (Internal Use Only)
# ==============================================================================
_TEST_CASES_ = [
    # (num_seqs, max_seq_len, num_kv_heads, page_block_size, nsa_block_size, update_last_only)
    (2, 192, 2, 128, 64, False),
    (3, 191, 1, 128, 64, True),
]

_TENSOR_DTYPES = [InfiniDtype.BF16, InfiniDtype.F16]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-2},
    InfiniDtype.BF16: {"atol": 5e-3, "rtol": 5e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 100


def _seq_lens(num_seqs, max_seq_len, nsa_block_size):
    values = []
    for i in range(num_seqs):
        values.append(max(nsa_block_size // 2 + 1, max_seq_len - i * 37))
    return torch.tensor(values, dtype=torch.int64)


def test(
    handle,
    device,
    num_seqs,
    max_seq_len,
    num_kv_heads,
    page_block_size,
    nsa_block_size,
    update_last_only,
    dtype,
    sync,
):
    print(
        f"Testing NsaCompressPagedCache on {InfiniDeviceNames[device]} with "
        f"num_seqs={num_seqs}, max_seq_len={max_seq_len}, num_kv_heads={num_kv_heads}, "
        f"page_block_size={page_block_size}, nsa_block_size={nsa_block_size}, "
        f"update_last_only={update_last_only}, dtype={InfiniDtypeNames[dtype]}"
    )

    head_size = 128
    max_blocks_per_seq = (max_seq_len + page_block_size - 1) // page_block_size
    num_blocks = num_seqs * max_blocks_per_seq
    subblocks_per_page = page_block_size // nsa_block_size

    k_cache = TestTensor(
        (num_blocks, num_kv_heads, page_block_size, head_size),
        None,
        dtype,
        device,
        scale=0.1,
    )
    v_cache = TestTensor(
        (num_blocks, num_kv_heads, page_block_size, head_size),
        None,
        dtype,
        device,
        scale=0.1,
    )
    k_cmp = TestTensor(
        (num_blocks * subblocks_per_page, num_kv_heads, head_size),
        None,
        dtype,
        device,
        mode="zeros",
    )
    v_cmp = TestTensor(
        (num_blocks * subblocks_per_page, num_kv_heads, head_size),
        None,
        dtype,
        device,
        mode="zeros",
    )

    block_tables_torch = torch.arange(num_blocks, dtype=torch.int64).view(
        num_seqs, max_blocks_per_seq
    )
    seq_lens_torch = _seq_lens(num_seqs, max_seq_len, nsa_block_size)
    block_tables = TestTensor.from_torch(block_tables_torch, InfiniDtype.I64, device)
    seq_lens = TestTensor.from_torch(seq_lens_torch, InfiniDtype.I64, device)

    ans_k, ans_v = ref_nsa_compress_paged_cache(
        k_cmp.torch_tensor(),
        v_cmp.torch_tensor(),
        k_cache.torch_tensor(),
        v_cache.torch_tensor(),
        block_tables.torch_tensor(),
        seq_lens.torch_tensor(),
        nsa_block_size,
        update_last_only,
    )

    if sync:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateNsaCompressPagedCacheDescriptor(
            handle,
            ctypes.byref(descriptor),
            k_cmp.descriptor,
            v_cmp.descriptor,
            k_cache.descriptor,
            v_cache.descriptor,
            block_tables.descriptor,
            seq_lens.descriptor,
            nsa_block_size,
            int(update_last_only),
        )
    )

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetNsaCompressPagedCacheWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    k_cmp.destroy_desc()
    v_cmp.destroy_desc()
    k_cache.destroy_desc()
    v_cache.destroy_desc()
    block_tables.destroy_desc()
    seq_lens.destroy_desc()

    def lib_nsa_compress_paged_cache():
        check_error(
            LIBINFINIOP.infiniopNsaCompressPagedCache(
                descriptor,
                workspace.data(),
                workspace_size.value,
                k_cmp.data(),
                v_cmp.data(),
                k_cache.data(),
                v_cache.data(),
                block_tables.data(),
                seq_lens.data(),
                None,
            )
        )

    lib_nsa_compress_paged_cache()

    if sync:
        sync()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        print("Verifying compressed K cache...")
        debug(k_cmp.actual_tensor(), ans_k, atol=atol, rtol=rtol)
        print("Verifying compressed V cache...")
        debug(v_cmp.actual_tensor(), ans_v, atol=atol, rtol=rtol)
    assert torch.allclose(k_cmp.actual_tensor(), ans_k, atol=atol, rtol=rtol)
    assert torch.allclose(v_cmp.actual_tensor(), ans_v, atol=atol, rtol=rtol)

    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: ref_nsa_compress_paged_cache(
                k_cmp.torch_tensor(),
                v_cmp.torch_tensor(),
                k_cache.torch_tensor(),
                v_cache.torch_tensor(),
                block_tables.torch_tensor(),
                seq_lens.torch_tensor(),
                nsa_block_size,
                update_last_only,
            ),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib", lib_nsa_compress_paged_cache, device, NUM_PRERUN, NUM_ITERATIONS
        )

    check_error(LIBINFINIOP.infiniopDestroyNsaCompressPagedCacheDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES_, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
