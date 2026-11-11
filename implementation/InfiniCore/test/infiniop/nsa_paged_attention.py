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
    k_cache,
    v_cache,
    block_tables,
    seq_lens,
    nsa_block_size,
):
    page_block_size = k_cache.shape[2]
    subblocks_per_page = page_block_size // nsa_block_size
    k_cmp = torch.zeros(
        (k_cache.shape[0] * subblocks_per_page, k_cache.shape[1], k_cache.shape[3]),
        dtype=k_cache.dtype,
        device=k_cache.device,
    )
    v_cmp = torch.zeros_like(k_cmp)

    for seq in range(block_tables.shape[0]):
        seq_len = int(seq_lens[seq].item())
        for nsa_block in range((seq_len + nsa_block_size - 1) // nsa_block_size):
            tok_begin = nsa_block * nsa_block_size
            tok_end = min(tok_begin + nsa_block_size, seq_len)
            logical_page = tok_begin // page_block_size
            subblock = (tok_begin % page_block_size) // nsa_block_size
            physical = int(block_tables[seq, logical_page].item())
            cmp_block = physical * subblocks_per_page + subblock
            rows = (
                torch.arange(tok_begin, tok_end, device=k_cache.device)
                % page_block_size
            )
            k_cmp[cmp_block] = (
                k_cache[physical, :, rows, :].float().mean(dim=1).to(k_cmp.dtype)
            )
            v_cmp[cmp_block] = (
                v_cache[physical, :, rows, :].float().mean(dim=1).to(v_cmp.dtype)
            )
    return k_cmp, v_cmp


def _attention_one(q, keys, values, scale):
    if keys.numel() == 0:
        return torch.zeros((values.shape[-1],), dtype=torch.float32, device=q.device)
    scores = torch.sum(keys.float() * q.float().view(1, -1), dim=-1) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.sum(probs.view(-1, 1) * values.float(), dim=0)


def ref_nsa_paged_attention(
    q,
    k_cmp,
    v_cmp,
    k_cache,
    v_cache,
    block_tables,
    seq_lens,
    gates,
    scale,
    nsa_block_size,
    window_size,
    select_blocks,
):
    num_seqs, num_heads, head_size = q.shape
    num_kv_heads = k_cache.shape[1]
    page_block_size = k_cache.shape[2]
    subblocks_per_page = page_block_size // nsa_block_size
    out = torch.empty_like(q, dtype=torch.float32)

    for seq in range(num_seqs):
        seq_len = int(seq_lens[seq].item())
        nsa_blocks = (seq_len + nsa_block_size - 1) // nsa_block_size
        for head in range(num_heads):
            kv_head = head // (num_heads // num_kv_heads)
            comp_keys = []
            comp_values = []
            block_scores = []
            for nsa_block in range(nsa_blocks):
                tok_begin = nsa_block * nsa_block_size
                logical_page = tok_begin // page_block_size
                subblock = (tok_begin % page_block_size) // nsa_block_size
                physical = int(block_tables[seq, logical_page].item())
                cmp_block = physical * subblocks_per_page + subblock
                key = k_cmp[cmp_block, kv_head]
                value = v_cmp[cmp_block, kv_head]
                comp_keys.append(key)
                comp_values.append(value)
                block_scores.append(
                    torch.sum(q[seq, head].float() * key.float()) * scale
                )

            comp_out = _attention_one(
                q[seq, head], torch.stack(comp_keys), torch.stack(comp_values), scale
            )

            top_count = min(select_blocks, nsa_blocks)
            top_blocks = torch.topk(
                torch.stack(block_scores), k=top_count
            ).indices.tolist()
            selected_keys = []
            selected_values = []
            for nsa_block in top_blocks:
                tok_begin = nsa_block * nsa_block_size
                tok_end = min(tok_begin + nsa_block_size, seq_len)
                for tok in range(tok_begin, tok_end):
                    logical_page = tok // page_block_size
                    row = tok % page_block_size
                    physical = int(block_tables[seq, logical_page].item())
                    selected_keys.append(k_cache[physical, kv_head, row])
                    selected_values.append(v_cache[physical, kv_head, row])
            sel_out = _attention_one(
                q[seq, head],
                torch.stack(selected_keys),
                torch.stack(selected_values),
                scale,
            )

            win_begin = max(0, seq_len - window_size) if window_size > 0 else seq_len
            window_keys = []
            window_values = []
            for tok in range(win_begin, seq_len):
                logical_page = tok // page_block_size
                row = tok % page_block_size
                physical = int(block_tables[seq, logical_page].item())
                window_keys.append(k_cache[physical, kv_head, row])
                window_values.append(v_cache[physical, kv_head, row])
            win_out = _attention_one(
                q[seq, head],
                torch.stack(window_keys),
                torch.stack(window_values),
                scale,
            )

            g = gates[seq, :, head].float()
            out[seq, head] = g[0] * comp_out + g[1] * sel_out + g[2] * win_out
    return out.to(q.dtype)


# ==============================================================================
#  Test Configuration (Internal Use Only)
# ==============================================================================
_TEST_CASES_ = [
    # (num_seqs, num_heads, num_kv_heads, page_block_size, max_seq_len, nsa_block_size, window_size, select_blocks)
    (2, 4, 2, 128, 192, 64, 64, 2),
    (3, 8, 2, 128, 255, 64, 128, 4),
]

_TENSOR_DTYPES = [InfiniDtype.BF16, InfiniDtype.F16]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 2e-2, "rtol": 2e-2},
    InfiniDtype.BF16: {"atol": 8e-2, "rtol": 8e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 100


def _seq_lens(num_seqs, max_seq_len, nsa_block_size):
    values = []
    for i in range(num_seqs):
        values.append(max(nsa_block_size + 1, max_seq_len - i * 31))
    return torch.tensor(values, dtype=torch.int64)


def test(
    handle,
    device,
    num_seqs,
    num_heads,
    num_kv_heads,
    page_block_size,
    max_seq_len,
    nsa_block_size,
    window_size,
    select_blocks,
    dtype,
    sync,
):
    print(
        f"Testing NsaPagedAttention on {InfiniDeviceNames[device]} with "
        f"num_seqs={num_seqs}, num_heads={num_heads}, num_kv_heads={num_kv_heads}, "
        f"page_block_size={page_block_size}, max_seq_len={max_seq_len}, "
        f"nsa_block_size={nsa_block_size}, window_size={window_size}, "
        f"select_blocks={select_blocks}, dtype={InfiniDtypeNames[dtype]}"
    )

    head_size = 128
    scale = 1.0 / (head_size**0.5)
    max_blocks_per_seq = (max_seq_len + page_block_size - 1) // page_block_size
    num_blocks = num_seqs * max_blocks_per_seq
    q = TestTensor((num_seqs, num_heads, head_size), None, dtype, device, scale=0.1)
    out = TestTensor(
        (num_seqs, num_heads, head_size), None, dtype, device, mode="zeros"
    )
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

    block_tables_torch = torch.arange(num_blocks, dtype=torch.int64).view(
        num_seqs, max_blocks_per_seq
    )
    seq_lens_torch = _seq_lens(num_seqs, max_seq_len, nsa_block_size)
    block_tables = TestTensor.from_torch(block_tables_torch, InfiniDtype.I64, device)
    seq_lens = TestTensor.from_torch(seq_lens_torch, InfiniDtype.I64, device)

    k_cmp_torch, v_cmp_torch = ref_nsa_compress_paged_cache(
        k_cache.torch_tensor(),
        v_cache.torch_tensor(),
        block_tables.torch_tensor(),
        seq_lens.torch_tensor(),
        nsa_block_size,
    )
    k_cmp = TestTensor.from_torch(k_cmp_torch, dtype, device)
    v_cmp = TestTensor.from_torch(v_cmp_torch, dtype, device)

    gates_torch = torch.rand((num_seqs, 3, num_heads), dtype=torch.float32) * 0.8 + 0.1
    gates = TestTensor.from_torch(gates_torch, InfiniDtype.F32, device)

    ans = ref_nsa_paged_attention(
        q.torch_tensor(),
        k_cmp.torch_tensor(),
        v_cmp.torch_tensor(),
        k_cache.torch_tensor(),
        v_cache.torch_tensor(),
        block_tables.torch_tensor(),
        seq_lens.torch_tensor(),
        gates.torch_tensor(),
        scale,
        nsa_block_size,
        window_size,
        select_blocks,
    )

    if sync:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateNsaPagedAttentionDescriptor(
            handle,
            ctypes.byref(descriptor),
            out.descriptor,
            q.descriptor,
            k_cmp.descriptor,
            v_cmp.descriptor,
            k_cache.descriptor,
            v_cache.descriptor,
            block_tables.descriptor,
            seq_lens.descriptor,
            gates.descriptor,
            scale,
            nsa_block_size,
            window_size,
            select_blocks,
        )
    )

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetNsaPagedAttentionWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    out.destroy_desc()
    q.destroy_desc()
    k_cmp.destroy_desc()
    v_cmp.destroy_desc()
    k_cache.destroy_desc()
    v_cache.destroy_desc()
    block_tables.destroy_desc()
    seq_lens.destroy_desc()
    gates.destroy_desc()

    def lib_nsa_paged_attention():
        check_error(
            LIBINFINIOP.infiniopNsaPagedAttention(
                descriptor,
                workspace.data(),
                workspace_size.value,
                out.data(),
                q.data(),
                k_cmp.data(),
                v_cmp.data(),
                k_cache.data(),
                v_cache.data(),
                block_tables.data(),
                seq_lens.data(),
                gates.data(),
                None,
            )
        )

    lib_nsa_paged_attention()

    if sync:
        sync()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(out.actual_tensor(), ans, atol=atol, rtol=rtol)
    assert torch.allclose(out.actual_tensor(), ans, atol=atol, rtol=rtol)

    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: ref_nsa_paged_attention(
                q.torch_tensor(),
                k_cmp.torch_tensor(),
                v_cmp.torch_tensor(),
                k_cache.torch_tensor(),
                v_cache.torch_tensor(),
                block_tables.torch_tensor(),
                seq_lens.torch_tensor(),
                gates.torch_tensor(),
                scale,
                nsa_block_size,
                window_size,
                select_blocks,
            ),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib", lib_nsa_paged_attention, device, NUM_PRERUN, NUM_ITERATIONS
        )

    check_error(LIBINFINIOP.infiniopDestroyNsaPagedAttentionDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES_, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
