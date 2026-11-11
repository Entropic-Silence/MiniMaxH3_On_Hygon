"""Multi-device tile-parallel decoding for the unchanged Diffusers H3 video VAE.

The Diffusers ``AutoencoderKLMiniMaxH3`` decodes every temporal clip as a grid
of independent spatial tiles (``decoder(post_quant_conv(tile))``) and blends
them afterwards. Only the scheduling of those independent tile decodes changes
here: each device holds an identical replica of ``post_quant_conv``/``decoder``
(same weights and dtypes), tiles are dealt round-robin to the devices, and the
results are stitched on the execution device with the original
``_stitch_tiles`` in the original order. Tile geometry, overlaps, blending,
temporal chunking and the autocast policy are all left exactly as released.
"""

from __future__ import annotations

import copy
import os
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

import torch


def _autocast_state(device_type: str) -> tuple[bool, torch.dtype, bool]:
    if hasattr(torch, "get_autocast_dtype"):
        dtype = torch.get_autocast_dtype(device_type)
    else:  # torch < 2.4
        dtype = torch.get_autocast_gpu_dtype()
    try:
        enabled = torch.is_autocast_enabled(device_type)
    except TypeError:  # torch < 2.4
        enabled = torch.is_autocast_enabled()
    return enabled, dtype, torch.is_autocast_cache_enabled()


def _copy_to(module: torch.nn.Module, device: torch.device,
             non_blocking: bool = False) -> torch.nn.Module:
    """Deep-copy ``module`` with every parameter and buffer placed on ``device``.

    Tensors are copied straight to the target device (no staging copy on the
    source device); values, dtypes and module structure are identical.
    ``non_blocking`` copies from pinned host memory asynchronously on the
    current stream (the caller synchronizes it).
    """
    memo = {}
    with torch.no_grad():
        for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
            if id(tensor) in memo:
                continue
            copied = tensor.detach().to(device, copy=True, non_blocking=non_blocking)
            memo[id(tensor)] = (torch.nn.Parameter(copied, requires_grad=False)
                                if isinstance(tensor, torch.nn.Parameter) else copied)
        replica = copy.deepcopy(module, memo)
    return replica.requires_grad_(False).eval()


def _pin_module(module: torch.nn.Module) -> torch.nn.Module:
    """Move ``module`` to host memory, page-locked for fast uploads."""
    module.to("cpu")
    with torch.no_grad():
        for tensor in list(module.parameters()) + list(module.buffers()):
            tensor.data = tensor.data.pin_memory()
    return module


def _create_miopen_handle(device: torch.device) -> None:
    """Create the calling thread's MIOpen handle for ``device`` (a 1x1x1
    convolution; PyTorch caches the handle per thread and device)."""
    with torch.cuda.device(device), torch.inference_mode():
        x = torch.zeros(1, 1, 1, 1, 1, device=device)
        torch.nn.functional.conv3d(x, torch.zeros(1, 1, 1, 1, 1, device=device))
        torch.cuda.synchronize(device)


class _Replica:
    def __init__(self, vae: torch.nn.Module, device: torch.device, main: bool,
                 staged: bool = False) -> None:
        self.device = device
        self.main = main
        if main:
            self.post_quant_conv = vae.post_quant_conv
            self.decoder = None if staged else vae.decoder
        else:
            self.post_quant_conv = _copy_to(vae.post_quant_conv, device)
            self.decoder = None if staged else _copy_to(vae.decoder, device)
        self.stream = torch.cuda.Stream(device=device)

    def stage_in(self, host_decoder: torch.nn.Module) -> None:
        """Upload the decoder from pinned host memory on this replica's
        (non-blocking) stream, so the copies never serialize with other
        streams computing on the device meanwhile."""
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            decoder = _copy_to(host_decoder, self.device, non_blocking=True)
        self.stream.synchronize()
        self.decoder = decoder


class ParallelTileDecoder:
    """Replaces ``vae._decode_clip`` on one VAE instance; see module docstring."""

    def __init__(self, vae: torch.nn.Module, devices: Sequence[int], staged: bool = False) -> None:
        main = next(vae.decoder.parameters()).device
        if main.type != "cuda":
            raise ValueError("parallel VAE decode requires the VAE on a CUDA/HIP device")
        ordered = [main.index] + [d for d in dict.fromkeys(devices) if d != main.index]
        self.vae = vae
        self.main = main
        # One worker thread per replica, always the same one: PyTorch keeps a
        # MIOpen handle per (thread, device), and creating one loads HIP code
        # objects lazily -- under DTK 25.04.2 that can hang (or fault) while
        # other threads launch kernels. With fixed threads every handle is
        # created here, one device at a time, and reused by the warmup and
        # every decode; a shared pool could hand a replica to a thread that
        # has no handle for its device in the middle of a decode.
        self.pools = [ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"h3-vae-tile{d}")
                      for d in ordered]
        # Replicas are copied to their devices concurrently, each by its thread.
        futures = [pool.submit(_Replica, vae, torch.device("cuda", d), d == main.index, staged)
                   for pool, d in zip(self.pools, ordered)]
        self.replicas = [future.result() for future in futures]
        for pool, replica in zip(self.pools, self.replicas):
            pool.submit(_create_miopen_handle, replica.device).result()
        # Staged (co-resident models): the decoder weights live in pinned host
        # memory and are uploaded to every device only for decoding, into the
        # memory the released conditioner leaves (stage_in_async), then freed.
        self.staged = staged
        self.host_decoder = _pin_module(vae.decoder) if staged else None
        self._stager = ThreadPoolExecutor(max_workers=len(ordered) + 1,
                                          thread_name_prefix="h3-vae-stage") if staged else None
        self._staging = None
        self._stage_lock = threading.Lock()
        self._original = vae._decode_clip
        vae._decode_clip = types.MethodType(self._decode_clip, vae)

    def stage_in_async(self) -> None:
        """Start uploading the decoder to every device (idempotent)."""
        if not self.staged:
            return
        with self._stage_lock:
            if self._staging is None:
                host = self.host_decoder
                self._staging = self._stager.submit(
                    lambda: [f.result() for f in [self._stager.submit(r.stage_in, host)
                                                  for r in self.replicas]])

    def _ensure_staged(self) -> None:
        if self.staged:
            self.stage_in_async()
            self._staging.result()

    def stage_out(self) -> None:
        """Drop the device copies of the decoder (staged mode)."""
        if not self.staged:
            return
        with self._stage_lock:
            if self._staging is not None:
                self._staging.result()
            for replica in self.replicas:
                replica.decoder = None
            self._staging = None

    def _run_replica(self, replica: _Replica, z: torch.Tensor, windows, main_stream,
                     grad: bool, autocast) -> list[torch.Tensor]:
        """Decode this replica's tiles in order; every result lands on the main device.

        Each replica computes on its own stream. Current streams are set on
        both devices so cross-device copies follow PyTorch's stream semantics
        (two-way barrier between the source and destination current streams);
        copies are issued only after all of this replica's tiles, so the
        barriers never serialize one replica's compute behind another's.
        """
        enabled, dtype, cache = autocast
        with torch.cuda.stream(main_stream), torch.inference_mode(not grad):
            if replica.main:
                replica.stream.wait_stream(main_stream)
            with torch.cuda.device(replica.device), torch.cuda.stream(replica.stream):
                local = z if replica.main else z.to(replica.device)
                with torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled,
                                    cache_enabled=cache):
                    decoded = self._decode_windows(replica, local, windows)
                if replica.main:
                    main_stream.wait_stream(replica.stream)
                    for output in decoded:
                        output.record_stream(main_stream)
                    return decoded
                outputs = [output.to(self.main) for output in decoded]
                # Source buffers are recycled on this stream after the copies.
                replica.stream.synchronize()
        return outputs

    @staticmethod
    def _decode_windows(replica: _Replica, local: torch.Tensor, windows) -> list[torch.Tensor]:
        """Decode this replica's tiles, in order. H3_VAE_TILE_BATCH=N stacks up
        to N tiles of the same size along the batch axis and decodes them in
        one decoder call (each tile is still decoded on its own: the decoder
        attends within a sample); 1 decodes tile by tile."""
        batch = int(os.environ.get("H3_VAE_TILE_BATCH", "1"))
        if batch <= 1:
            return [replica.decoder(replica.post_quant_conv(local[..., top:bottom, left:right]))
                    for top, bottom, left, right in windows]
        outputs = [None] * len(windows)
        by_size = {}
        for index, (top, bottom, left, right) in enumerate(windows):
            by_size.setdefault((bottom - top, right - left), []).append(index)
        for indices in by_size.values():
            for start in range(0, len(indices), batch):
                part = indices[start:start + batch]
                tiles = torch.cat([local[..., windows[i][0]:windows[i][1], windows[i][2]:windows[i][3]]
                                   for i in part], dim=0)
                decoded = replica.decoder(replica.post_quant_conv(tiles))
                for offset, index in enumerate(part):
                    outputs[index] = decoded[offset:offset + 1]
        return outputs

    def _decode_clip(self, vae: torch.nn.Module, z: torch.Tensor) -> torch.Tensor:
        self._ensure_staged()
        if self.staged and not vae.use_tiling:
            # The released untiled path, with the staged main-device decoder.
            main = self.replicas[0]
            return main.decoder(main.post_quant_conv(z))
        if not self.staged and (not vae.use_tiling or len(self.replicas) == 1):
            return self._original(z)
        height = z.shape[-2] * vae.spatial_compression_ratio
        width = z.shape[-1] * vae.spatial_compression_ratio
        y_indices, y_lengths, y_overlaps = vae._split_tiles(
            height, vae.tile_sample_min_height, vae.tile_sample_min_overlap_height)
        x_indices, x_lengths, x_overlaps = vae._split_tiles(
            width, vae.tile_sample_min_width, vae.tile_sample_min_overlap_width)
        ratio = vae.spatial_compression_ratio
        windows = [(i_pos // ratio, i_pos // ratio + i_len // ratio,
                    j_pos // ratio, j_pos // ratio + j_len // ratio)
                   for i_pos, i_len in zip(y_indices, y_lengths)
                   for j_pos, j_len in zip(x_indices, x_lengths)]
        count = len(self.replicas)
        main_stream = torch.cuda.current_stream(self.main)
        grad = torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
        autocast = _autocast_state("cuda")
        futures = [self.pools[r].submit(self._run_replica, replica, z, windows[r::count],
                                        main_stream, grad, autocast)
                   for r, replica in enumerate(self.replicas)]
        results = [future.result() for future in futures]
        decoded = [results[n % count][n // count] for n in range(len(windows))]
        columns = len(x_indices)
        rows = [decoded[r * columns:(r + 1) * columns] for r in range(len(y_indices))]
        return vae._stitch_tiles(rows, y_overlaps, x_overlaps)

    def close(self) -> None:
        self.vae._decode_clip = self._original
        for pool in self.pools:
            pool.shutdown(wait=True)


def vae_parallel_devices(devices: Sequence[int]) -> list[int]:
    """``H3_VAE_PARALLEL``: ``auto`` (all given devices, default), ``0``/``1``
    (disabled) or ``N`` (first N devices)."""
    mode = os.environ.get("H3_VAE_PARALLEL", "auto").strip().lower()
    devices = list(dict.fromkeys(devices))
    if mode in ("", "auto"):
        return devices
    count = int(mode)
    if count < 0:
        raise ValueError("H3_VAE_PARALLEL must be auto or a non-negative integer")
    return devices[:count] if count > 1 else []


_lock = threading.Lock()


def _trim_before_decode(vae: torch.nn.Module, devices: Sequence[int]) -> None:
    """Release denoising scratch cached by the native allocator before decoding.

    The DiT keeps its weights resident; only freed blocks are returned, so the
    VAE (and its replicas) can use the memory the last denoising step left.
    """
    if getattr(vae, "_h3_trim_before_decode", False):
        return
    from infinilm.multimodal._h3_residency import trim_device_caches
    decode = vae.decode

    def decode_after_trim(*args, **kwargs):
        trim_device_caches(devices)
        return decode(*args, **kwargs)

    vae.decode = decode_after_trim
    vae._h3_trim_before_decode = True


def _stage_out_after_decode(vae: torch.nn.Module, decoder: ParallelTileDecoder) -> None:
    decode = vae.decode

    def decode_then_stage_out(*args, **kwargs):
        try:
            return decode(*args, **kwargs)
        finally:
            decoder.stage_out()

    vae.decode = decode_then_stage_out


def install_parallel_vae_decode(vae: torch.nn.Module, devices: Sequence[int],
                                conditioner: torch.nn.Module | None = None
                                ) -> ParallelTileDecoder | None:
    """Tile-parallel decode over ``devices``. When ``conditioner`` is a
    co-resident Qwen3-VL (H3_TEXT_ENCODER=coresident), the decoder weights are
    staged: kept in pinned host memory, uploaded to the devices as soon as the
    conditioner releases its weights after encoding (overlapping the
    denoising), and freed again after each decode."""
    if torch.cuda.is_available():
        _trim_before_decode(vae, devices)
    selected = vae_parallel_devices(devices)
    if len(selected) < 2 or not torch.cuda.is_available():
        return None
    residency = getattr(conditioner, "_residency", None)
    staged = bool(getattr(residency, "coexist", False))
    with _lock:
        existing = getattr(vae, "_h3_parallel_decoder", None)
        if existing is not None:
            return existing
        decoder = ParallelTileDecoder(vae, selected, staged=staged)
        vae._h3_parallel_decoder = decoder
        if staged:
            residency.on_release.append(decoder.stage_in_async)
            _stage_out_after_decode(vae, decoder)
    print(f"[model] H3 VAE tile-parallel decode devices: "
          f"{[r.device.index for r in decoder.replicas]}" + (" (staged)" if staged else ""))
    return decoder


def install_video_postprocess(video_processor, frames_per_chunk: int = 16) -> None:
    """PIL output of ``VideoProcessor.postprocess_video`` with the pixel
    conversion on the device. The released path copies the FP32 frames to the
    host, then ``numpy_to_pil`` computes ``(x * 255).round().astype(uint8)``
    on a permuted (non-contiguous) array, which ``Image.fromarray`` copies
    again frame by frame (about 4 s for 124 768x768 frames). Here the same FP32
    multiply, round-half-to-even and cast run on the device in chunks of
    frames, only contiguous uint8 frames cross to the host, and the PIL images
    (same pixels, same order) are built on a few threads. Other output types, extra arguments and host tensors
    keep the original method; H3_VIDEO_POSTPROCESS=0 disables it."""
    if video_processor is None or os.environ.get("H3_VIDEO_POSTPROCESS", "1") == "0":
        return
    if getattr(video_processor, "_h3_postprocess", False):
        return
    from PIL import Image

    original = video_processor.postprocess_video

    def postprocess_video(video, output_type="np", **kwargs):
        if output_type != "pil" or kwargs or not video.is_cuda or video.ndim != 5:
            return original(video, output_type, **kwargs)
        outputs = []
        for batch_idx in range(video.shape[0]):
            frames = video[batch_idx].permute(1, 0, 2, 3)  # [T, C, H, W], as postprocess_video
            frames = video_processor._denormalize_conditionally(frames)
            count, channels, height, width = frames.shape
            pixels = torch.empty((count, height, width, channels), dtype=torch.uint8)
            for start in range(0, count, frames_per_chunk):
                chunk = frames[start:start + frames_per_chunk].permute(0, 2, 3, 1).float()  # pt_to_numpy
                chunk = (chunk * 255).round().to(torch.uint8)  # numpy_to_pil
                pixels[start:start + chunk.shape[0]].copy_(chunk.contiguous())
            pixels = pixels.numpy()
            if channels == 1:
                to_pil = lambda image: Image.fromarray(image.squeeze(), mode="L")  # noqa: E731
            else:
                to_pil = Image.fromarray
            with ThreadPoolExecutor(max_workers=8) as pool:
                outputs.append(list(pool.map(to_pil, pixels)))
        return outputs

    video_processor.postprocess_video = postprocess_video
    video_processor._h3_postprocess = True
