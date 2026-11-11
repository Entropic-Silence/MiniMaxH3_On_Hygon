"""Reference rounding and HIP AOT dispatch on packed/strided H3 tensors."""
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from h3_pointwise import (make_gated_residual, make_gated_residual_rms_modulate, make_rms,
                          make_rms_modulate, make_rope, make_swiglu)

pytestmark = pytest.mark.skipif(not torch.version.hip, reason="requires HIP")


def _gfx936():
    return torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.startswith("gfx936")


def _fp64_rms_bf16(x, weight, eps):
    value = x.double()
    return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps) * weight.double()).to(x.dtype)


def _assert_norm_rounding(actual, exact):
    # gfx936 uses FP32 norm accumulation. Different valid reduction orders
    # can round to opposite sides of one BF16 midpoint. Check each result
    # independently against FP64, rather than allowing an arbitrary rtol.
    def ordered(value):
        bits = value.view(torch.int16).to(torch.int32)
        return torch.where(bits < 0, -32768 - bits, bits)
    assert torch.isfinite(actual).all()
    assert int((ordered(actual) - ordered(exact)).abs().max()) <= 1


def _assert_modulate_rounding(actual, norm, scale, shift):
    # Propagate the one-ULP norm interval through the unchanged BF16
    # multiply/add stages; cancellation can amplify final-output ULPs.
    factor = 1 + scale
    lo = torch.nextafter(norm, torch.full_like(norm, -float("inf"))) * factor + shift
    hi = torch.nextafter(norm, torch.full_like(norm, float("inf"))) * factor + shift
    lower, upper = torch.minimum(lo, hi), torch.maximum(lo, hi)
    assert torch.isfinite(actual).all()
    assert ((actual >= lower) & (actual <= upper)).all()


@pytest.mark.parametrize("operation", ["rms128", "rms5376", "swiglu"])
def test_reference_rounding_aot(tmp_path, operation):
    is_rms = operation.startswith("rms")
    width = int(operation[3:]) if is_rms else 7168
    factory = make_rms if is_rms else make_swiglu
    options = {"width": width} if is_rms else {}
    jit = factory(**options)
    aot = factory(caller="hip", output_dir=tmp_path, **options)

    def run(device):
        with torch.cuda.device(device), torch.cuda.stream(torch.cuda.Stream(device=device)):
            for scale in (.01, 1., 100.):
                packed = torch.randn(2, 17, width * 2 + 1, device="cuda", dtype=torch.bfloat16) * scale
                x = packed[..., 1:width+1]
                y_jit, y_aot = torch.empty_like(x), torch.empty_like(x)
                if is_rms:
                    w = torch.randn(width * 2, device="cuda", dtype=torch.bfloat16)[::2]
                    arguments = (x, w, 1e-5)
                    reference = torch.nn.functional.rms_norm(x, (width,), w, eps=1e-5)
                else:
                    gate = packed[..., width+1:]
                    arguments = (x, gate)
                    reference = x * torch.nn.functional.silu(gate)
                jit(*arguments, y_jit)
                aot(*arguments, y_aot)
                torch.cuda.synchronize(device)
                if is_rms and width == 5376 and _gfx936():
                    exact = _fp64_rms_bf16(x, w, 1e-5)
                    for output in (y_jit, y_aot, reference):
                        _assert_norm_rounding(output, exact)
                else:
                    assert torch.equal(y_jit, y_aot), "JIT and AOT differ"
                    assert torch.equal(y_aot, reference), operation
    for device in range(torch.cuda.device_count()):
        run(device)
    with ThreadPoolExecutor(max_workers=torch.cuda.device_count()) as pool:
        list(pool.map(run, range(torch.cuda.device_count())))


def test_fused_rms_modulate(tmp_path):
    jit = make_rms_modulate()
    aot = make_rms_modulate(caller="hip", output_dir=tmp_path)
    for device in range(torch.cuda.device_count()):
        with torch.cuda.device(device), torch.cuda.stream(torch.cuda.Stream(device=device)):
            # Match the large six-way modulation gather's noncontiguous views.
            x = torch.randn(2, 33, 5376, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(5376, device="cuda", dtype=torch.bfloat16)
            table = torch.randn(2, 33, 5376 * 6, device="cuda", dtype=torch.bfloat16)
            shift, scale = table[..., :5376], table[..., 5376:10752]
            scale[:, :3].fill_(-1.)  # The rounding of (1 + scale) matters here.
            reference = torch.nn.functional.rms_norm(x, (5376,), w, eps=1e-5) * (1 + scale) + shift
            ja, aa = torch.empty_like(x), torch.empty_like(x)
            jit(x, w, 1e-5, scale, shift, ja)
            aot(x, w, 1e-5, scale, shift, aa)
            torch.cuda.synchronize(device)
            if _gfx936():
                norm = _fp64_rms_bf16(x, w, 1e-5)
                for output in (ja, aa, reference):
                    _assert_modulate_rounding(output, norm, scale, shift)
            else:
                assert torch.equal(ja, aa)
                assert torch.equal(aa, reference)


def test_rope_reference_rounding(tmp_path):
    jit = make_rope()
    aot = make_rope(caller="hip", output_dir=tmp_path)

    def run(device):
        with torch.cuda.device(device), torch.cuda.stream(torch.cuda.Stream(device=device)):
            for seq, heads in ((17, 28), (257, 56), (2304, 28)):
                x = torch.randn(seq, heads, 385, device="cuda", dtype=torch.bfloat16)[..., 1:129]
                angles = torch.randn(seq, 97, device="cuda", dtype=torch.bfloat16)
                c, s = angles[:, 1:49], angles[:, 49:97]
                c[0].fill_(1.)
                s[0].fill_(1.)
                cosine = torch.cat((c, c), -1)[:, None]
                sine = torch.cat((s, s), -1)[:, None]
                rotated = torch.cat((-x[..., 48:96], x[..., :48]), -1)
                reference = torch.cat((x[..., :96] * cosine + rotated * sine, x[..., 96:]), -1)
                ja, aa = torch.empty_like(x), torch.empty_like(x)
                jit(x, c, s, ja)
                aot(x, c, s, aa)
                torch.cuda.synchronize(device)
                assert torch.equal(ja, aa)
                assert torch.equal(aa, reference)
    for device in range(torch.cuda.device_count()):
        run(device)
    with ThreadPoolExecutor(max_workers=torch.cuda.device_count()) as pool:
        list(pool.map(run, range(torch.cuda.device_count())))


def test_fused_gated_residual(tmp_path):
    """x + gate * y (BF16 product, BF16 sum), alone and followed by the fused
    RMSNorm/AdaLN, equal the separate torch ops and h3_rms_modulate bitwise."""
    residual_jit = make_gated_residual()
    residual_aot = make_gated_residual(caller="hip", output_dir=tmp_path)
    fused_jit = make_gated_residual_rms_modulate()
    fused_aot = make_gated_residual_rms_modulate(caller="hip", output_dir=tmp_path)
    modulate = make_rms_modulate()
    for device in range(torch.cuda.device_count()):
        with torch.cuda.device(device), torch.cuda.stream(torch.cuda.Stream(device=device)):
            for rows, scale in ((33, 1.), (1031, 8.), (4099, 100.)):
                # Gate/scale/shift are strided columns of the six-way modulation table.
                x = torch.randn(1, rows, 5376, device="cuda", dtype=torch.bfloat16) * scale
                y = torch.randn(1, rows, 5376, device="cuda", dtype=torch.bfloat16)
                table = torch.randn(1, rows, 5376 * 6, device="cuda", dtype=torch.bfloat16)
                shift, mod_scale, gate = table[..., :5376], table[..., 5376:10752], table[..., 10752:16128]
                mod_scale[:, :3].fill_(-1.)
                w = torch.randn(5376, device="cuda", dtype=torch.bfloat16)
                expected_residual = torch.add(x, torch.mul(gate, y))
                expected = torch.empty_like(x)
                modulate(expected_residual, w, 1e-6, mod_scale, shift, expected)
                for kernel in (residual_jit, residual_aot):
                    out = torch.empty_like(x)
                    kernel(x, gate, y, out)
                    torch.cuda.synchronize(device)
                    assert torch.equal(out, expected_residual)
                for kernel in (fused_jit, fused_aot):
                    residual, out = torch.empty_like(x), torch.empty_like(x)
                    kernel(x, gate, y, w, 1e-6, mod_scale, shift, residual, out)
                    torch.cuda.synchronize(device)
                    assert torch.equal(residual, expected_residual)
                    assert torch.equal(out, expected)
