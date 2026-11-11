"""Run on a HIP machine: PYTHONPATH=ninetoothed/src:tools pytest -q tests/test_aot_hip.py."""
from concurrent.futures import ThreadPoolExecutor
import pytest
import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor
from h3_attention import make_attention

pytestmark = pytest.mark.skipif(not torch.version.hip, reason="requires HIP")


def add_arrange(x, y, out):
    return tuple(t.tile((256,)) for t in (x, y, out))


def add_apply(x, y, out):
    out = x + y  # noqa: F841


def softmax_arrange(x, out):
    return tuple(t.tile((1, 256)) for t in (x, out))


def softmax_apply(x, out):
    exp = ntl.exp(x - ntl.max(x))
    out = exp / ntl.sum(exp)  # noqa: F841


@pytest.mark.parametrize("op", ["add", "softmax", "matmul", "attention", "vision_attention"])
def test_jit_aot_multidevice_streams(tmp_path, op):
    options = dict(num_warps=4, num_stages=1)
    head_dim = 72 if op == "vision_attention" else 128
    if op in ("attention", "vision_attention"):
        jit = make_attention(head_dim=head_dim, **options)
        aot = make_attention(caller="hip", output_dir=tmp_path, head_dim=head_dim, **options)
    else:
        if op == "add":
            arrange, apply = add_arrange, add_apply
            tensors = tuple(Tensor(1, dtype=ninetoothed.float32) for _ in range(3))
        elif op == "softmax":
            arrange, apply = softmax_arrange, softmax_apply
            tensors = (Tensor(shape=(None, 256), dtype=ninetoothed.float32),
                       Tensor(shape=(None, 256), dtype=ninetoothed.float32))
        else:
            arrange, apply = matmul_arrange, matmul_apply
            tensors = tuple(Tensor(2, dtype=ninetoothed.float16) for _ in range(3))
        jit = ninetoothed.make(arrange, apply, tensors, **options)
        aot = ninetoothed.make(arrange, apply, tensors, caller="hip", kernel_name=op,
                               output_dir=tmp_path, **options)

    def run(device):
        with torch.cuda.device(device), torch.cuda.stream(torch.cuda.Stream(device=device)):
            torch.manual_seed(123)
            if op == "add":
                x, y = (torch.randn(1027, device="cuda") for _ in range(2))
                args, reference = (x, y), x + y
            elif op == "softmax":
                x = torch.randn(7, 256, device="cuda")
                args, reference = (x,), x.softmax(-1)
            elif op == "matmul":
                x = torch.randn(65, 96, device="cuda", dtype=torch.float16)
                y = torch.randn(96, 67, device="cuda", dtype=torch.float16)
                args, reference = (x, y), x @ y
            else:
                # Packed strided views, odd sequence and masked last KV block.
                packed = torch.randn(1, 257, 3, 2, head_dim, device="cuda", dtype=torch.bfloat16)
                q, k, v = (packed[:, :, i].transpose(1, 2) for i in range(3))
                args = (q, k, v, head_dim**-0.5)
                # Independent full-attention math; the reference must not require
                # an optional vendor FlashAttention shared library.
                scores = (q.float() @ k.float().transpose(-2,-1)) * head_dim**-0.5
                reference = (scores.softmax(-1) @ v.float()).to(q.dtype)
            ja, aa = torch.empty_like(reference), torch.empty_like(reference)
            jit(*args, ja)
            aot(*args, aa)
            torch.cuda.synchronize(device)
            assert torch.equal(ja, aa), f"{op} JIT and AOT differ on device {device}"
            torch.testing.assert_close(aa, reference, atol=0.002 if "attention" in op else 0.01,
                                       rtol=0.01)
    # Serial first to isolate module caches, then concurrent callers on both devices.
    for device in range(torch.cuda.device_count()):
        run(device)
    with ThreadPoolExecutor(max_workers=torch.cuda.device_count()) as pool:
        list(pool.map(run, range(torch.cuda.device_count())))


def matmul_arrange(lhs, rhs, out):
    tiled = out.tile((32, 32))
    a = lhs.tile((32, 32)).tile((1, -1)).expand((-1, tiled.shape[1]))
    a.dtype = a.dtype.squeeze(0)
    b = rhs.tile((32, 32)).tile((-1, 1)).expand((tiled.shape[0], -1))
    b.dtype = b.dtype.squeeze(1)
    return a, b, tiled


def matmul_apply(lhs, rhs, out):
    acc = ntl.zeros(out.shape, dtype=ntl.float32)
    for j in range(lhs.shape[0]):
        acc += ntl.dot(lhs[j], rhs[j])
    out = acc.to(ntl.float16)  # noqa: F841


def test_attention_unaligned_inner_stride(tmp_path):
    """Exercise the generic dispatch, including nonunit D stride and B > 1."""
    jit = make_attention(num_warps=4, num_stages=1)
    aot = make_attention(caller="hip", output_dir=tmp_path, num_warps=4, num_stages=1)
    for device in range(torch.cuda.device_count()):
        with torch.cuda.device(device):
            q, k, v = (torch.randn(2, 3, 33, 257, device="cuda", dtype=torch.bfloat16)[..., 1::2]
                       for _ in range(3))
            aa = torch.empty(2, 3, 33, 256, device="cuda", dtype=torch.bfloat16)[..., ::2]
            ja = torch.empty_like(aa)
            jit(q, k, v, 128**-0.5, ja)
            aot(q, k, v, 128**-0.5, aa)
            torch.cuda.synchronize()
            assert torch.equal(ja, aa)
            scores = (q.float() @ k.float().transpose(-2,-1)) * 128**-0.5
            ref = (scores.softmax(-1) @ v.float()).to(q.dtype)
            torch.testing.assert_close(aa, ref, atol=.005, rtol=.01)
