"""Default runtime switches of the native MiniMax-H3 path.

The unified entry (``examples/minimax_h3.py``) and the organizers' comparison
script start the InfiniLM backend without any environment setup, so the
validated configuration has to be the default rather than something a wrapper
script exports. Every value can still be overridden from the environment; the
native code reads these variables when it runs, after this module set them.

Numerics: reference (Diffusers) BF16 rounding boundaries for RMSNorm, AdaLN,
the fused gated residual and SwiGLU; deterministic BLAS (no atomics); vendor
flash attention for the DiT and Qwen3-VL attention through InfiniCore.
"""

import os
from pathlib import Path

# Offline-measured hipBLASLt candidate per GEMM shape and device name
# (tools/tune_gemm_lt.py); shapes or cards not listed keep the library's
# first heuristic choice.
GEMM_LT_TABLE = Path(__file__).with_name("h3_gemm_lt.txt")

DEFAULTS = {
    # BW/gfx936 candidate profile; explicit environment overrides remain supported.
    "H3_BW_GATE_PREPACK": "1",
    "H3_QK_FUSED_FP32": "1",
    "H3_BW_GEMM_SELECTIVE": "1",
    "H3_BW_GATE_BLAS": "1",
    "H3_BW_SWIGLU_WIDTH": "1",
    "H3_BW_GEMM_LENGTHS": "1",
    # One packed sequence-parallel exchange for Q/K/V; arithmetic unchanged.
    "H3_SP_PACK_QKV": "1",
    # Q/K normalization and RoPE write directly into the packed SP send layout.
    "H3_QK_DIRECT_PACK": "1",
    # InfiniCore attention on Hygon: vendor flash attention through ATen SDPA
    # ("aten"); "ninetoothed" selects the NineToothed flash attention kernel.
    "INFINICORE_HYGON_MHA": "aten",
    # InfiniCore BF16 GEMMs through cuBLASLt (DTK maps it onto hipBLASLt),
    # with the best heuristic algorithm chosen once per shape: 3% faster DiT
    # steps than rocBLAS, deterministic, and no PyTorch involvement (the ATen
    # route, INFINICORE_HYGON_LINEAR=lt, faults in the co-resident pipeline).
    "INFINICORE_HYGON_GEMM": "lt",
    # Reference BF16 rounding in the NineToothed pointwise kernels.
    "H3_POINTWISE_NUMERICS": "exact_ref",
    "H3_ADALN_NUMERICS": "exact_ref",
    "H3_FUSED_ADALN": "1",
    "H3_ROPE_NUMERICS": "baseline",
    "H3_DETERMINISTIC_BLAS": "1",
    # Qwen3-VL vision: FP32 rotary angles as the reference, NineToothed d72
    # attention kernel available.
    "H3_QWEN_FP32_ROPE": "1",
    "H3_QWEN_NINETOOTHED": "1",
    # Video VAE decode: same-size tiles of a replica decoded 4 at a time
    # (batched convolutions; each tile is still decoded on its own). 1.0 s
    # faster per 124-frame 768x768 decode; outputs differ from tile-by-tile
    # decoding at FP16 rounding level (rel. L2 6e-4).
    "H3_VAE_TILE_BATCH": "4",
}


def apply_defaults() -> None:
    # DAS PyTorch's Cutlass adapter otherwise searches only its system paths.
    # Select the installed extension without importing or altering its kernels.
    if "FA_SO_PATH" not in os.environ:
        from importlib.util import find_spec
        flash_extension = find_spec("flash_attn_2_cuda")
        if flash_extension is not None and flash_extension.origin:
            os.environ["FA_SO_PATH"] = flash_extension.origin
    for name, value in DEFAULTS.items():
        os.environ.setdefault(name, value)
    # The AOT kernel is installed beside the InfiniLM extension or in a native library directory.
    dense_library = os.environ.get("H3_DENSE_TILE_LIBRARY")
    if dense_library is None:
        name = "libh3_dense_attention.so"
        candidates = [Path(__file__).parents[1] / "lib" / name]
        candidates += [Path(p) / name for p in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p]
        for candidate in candidates:
            if candidate.is_file():
                dense_library = str(candidate.resolve())
                os.environ["H3_DENSE_TILE_LIBRARY"] = dense_library
                break
    # The measured packed direct-QK branch requires an unsplit projection.
    # K100 retains its pipelined default; explicit overrides remain respected.
    if any(k not in os.environ for k in ("H3_SP_PIPELINE", "H3_QKV_DIRECT_PACK", "H3_BW_CHAIN_RMS")) or (dense_library and os.environ.get("H3_DENSE_TILE_ENABLE") != "0" and ("H3_DENSE_TILE_ENABLE" not in os.environ or "FLASH_ATTENTION_FAST_SOFTMAX" not in os.environ)):
        import torch
        if torch.cuda.is_available():
            arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
            if arch.split(":", 1)[0] == "gfx936":
                os.environ.setdefault("H3_SP_PIPELINE", "0")
                os.environ.setdefault("H3_QKV_DIRECT_PACK", "1")
                os.environ.setdefault("H3_BW_CHAIN_RMS", "1")
                if dense_library and os.environ.get("H3_DENSE_TILE_ENABLE") != "0":
                    os.environ.setdefault("FLASH_ATTENTION_FAST_SOFTMAX", "0")
                    os.environ.setdefault("H3_DENSE_TILE_ENABLE", "1")
    if GEMM_LT_TABLE.is_file():
        os.environ.setdefault("INFINICORE_GEMM_LT_TABLE", str(GEMM_LT_TABLE))


apply_defaults()
