"""Load-time warmup of the native MiniMax-H3 path.

The first use of every component pays one-time costs that are not part of
generating a video: HIP loads kernel code objects lazily at first launch,
collectives of a new engine run synchronously on its first forward (see
INFINILM_SYNC_FIRST_FORWARD), the SP DiT regroups its QKV weight rows once,
BLAS libraries pick and load kernels, and MIOpen sets up the VAE convolutions.

``warmup_pipeline`` runs the unchanged pipeline once on a small synthetic
input (448x448 grey first frame, 124 frames -- the shortest length the
pipeline accepts --, 2 scheduler steps = 1 DiT forward)
while the models load, so these costs land in model loading. Its output is
discarded; nothing it computes is kept or reused (there is no cache of any
input-dependent value), and the generation that follows recomputes every
step. The 448x448 canvas makes the VAE use its released 256-pixel tiles, four
per clip, so every parallel decode replica runs. H3_WARMUP=0 disables it.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from infinilm.multimodal._h3_residency import trim_device_caches

WARMUP_SIZE = 448   # 2 x 2 VAE tiles of 256 pixels
WARMUP_FRAMES = 124  # 17 * n + 5 within the accepted 5-15 s
WARMUP_STEPS = 2    # one transformer forward


def warmup_enabled() -> bool:
    mode = os.environ.get("H3_WARMUP", "1")
    if mode not in ("0", "1"):
        raise ValueError(f"Unsupported H3_WARMUP={mode!r}; use 0 or 1")
    return mode == "1"


def _residencies(*components: Any) -> list:
    found = []
    for component in components:
        for owner in (component, getattr(component, "runner", None)):
            residency = getattr(owner, "_residency", None)
            if residency is not None:
                found.append(residency)
    return found


def warmup_pipeline(pipe: Any, components: Sequence[Any], devices: Sequence[int],
                    task: str = "fl2va") -> bool:
    """Run one small synthetic generation; returns whether it ran."""
    if not warmup_enabled() or task != "fl2va":
        return False
    residencies = _residencies(*components)
    if any(r.sequential and not r.coexist for r in residencies):
        # Staged engines evict each other: a warmup would leave the DiT
        # loaded and make the generation reload both models.
        return False
    image = Image.fromarray(np.full((WARMUP_SIZE, WARMUP_SIZE, 3), 127, dtype=np.uint8))
    with torch.inference_mode():
        # Same flow as a generation: a co-resident conditioner is released
        # after encoding (the staged VAE decoder and the DiT, loaded on its
        # first forward here, use that memory) ...
        pipe(
            prompt="warmup",
            image=image,
            num_frames=WARMUP_FRAMES,
            num_inference_steps=WARMUP_STEPS,
            height=WARMUP_SIZE,
            width=WARMUP_SIZE,
            generator=torch.Generator("cpu").manual_seed(0),
            output=["videos", "audio", "sampling_rate"],
        )
    trim_device_caches(devices)
    # ... and reloaded from its host copy, so the generation starts with both
    # models on the devices.
    for residency in residencies:
        if residency.coexist and residency.engine is None:
            with residency.held():
                with residency.use():
                    pass
    return True
