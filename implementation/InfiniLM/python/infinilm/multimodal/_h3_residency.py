"""Exact H3 weight residency; no output cache and no changes to model arithmetic."""
from contextlib import contextmanager, nullcontext
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import logging
import os
import threading
import time
import weakref

import infinicore
import torch
from infinilm.infer_engine import InferEngine
from infinilm.modeling_utils import (
    RawShardCache,
    checkpoint_shard_files,
    load_model_state_dict_by_file,
    pin_checkpoint_raw,
    prefetch_checkpoint_raw,
    read_checkpoint_raw,
    take_prefetched_checkpoint_raw,
    unpin_checkpoint_raw,
)

logger = logging.getLogger(__name__)
_lock = threading.RLock()
_STAGE_READER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="h3-stage-read")
_sessions = weakref.WeakSet()


def trim_device_caches(devices):
    """Return unused cached blocks of the native and PyTorch allocators.

    Weights of resident engines are untouched; only blocks already freed by
    finished forwards are released, e.g. denoising scratch before VAE decode.
    """
    original = infinicore.device._from_infinicore_device(infinicore.get_device())
    try:
        for index in sorted(devices):
            torch.cuda.synchronize(index)
            infinicore.set_device(infinicore.device("cuda", index))
            infinicore.trim_memory()
            with torch.cuda.device(index):
                torch.cuda.empty_cache()
    finally:
        infinicore.set_device(original)


def checkpoint_bytes(path):
    """Total size of a checkpoint directory's safetensors shards."""
    return sum(os.path.getsize(f) for f in checkpoint_shard_files(str(path)))


def _host_memory_available():
    """Bytes this process can still allocate: MemAvailable, capped by the
    cgroup (v2 or v1) limit minus its usage when one is set. Inactive file
    cache is reclaimable, so it is not counted as used (a checkpoint that
    was just read, e.g. from a local disk, otherwise looks like usage)."""
    available = None
    try:
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
    except OSError:
        pass
    for root, limit_name, usage_name, inactive_key in (
            ("/sys/fs/cgroup", "memory.max", "memory.current", "inactive_file"),
            ("/sys/fs/cgroup/memory", "memory.limit_in_bytes", "memory.usage_in_bytes",
             "total_inactive_file")):
        try:
            with open(os.path.join(root, limit_name)) as f:
                limit = f.read().strip()
            with open(os.path.join(root, usage_name)) as f:
                usage = int(f.read().strip())
        except (OSError, ValueError):
            continue
        inactive = 0
        try:
            with open(os.path.join(root, "memory.stat")) as f:
                for line in f:
                    key, _, value = line.partition(" ")
                    if key == inactive_key:
                        inactive = int(value)
        except (OSError, ValueError):
            pass
        if limit.isdigit():
            room = int(limit) - max(usage - inactive, 0)
            available = room if available is None else min(available, room)
        break
    return available


_host_cache_budget = None
_host_cache_reserved = {}


def host_cache_enabled(path):
    """Whether a host copy of the checkpoint at ``path`` may be kept.

    H3_HOST_CACHE=1 always, 0 never (safetensors are re-read on every
    sequential (re)load). The default ``auto`` keeps copies while their total
    stays within 60% of the host memory that was available at the first
    request, so machines with less RAM fall back to reading from disk. A
    checkpoint is counted once, whether it is prefetched, cached or both.
    """
    global _host_cache_budget
    mode = os.environ.get("H3_HOST_CACHE", "auto")
    if mode not in ("auto", "0", "1"):
        raise ValueError(f"Unsupported H3_HOST_CACHE={mode!r}; use auto, 0 or 1")
    if mode != "auto":
        return mode == "1"
    key = os.path.realpath(str(path))
    with _lock:
        if key in _host_cache_reserved:
            return True
        if _host_cache_budget is None:
            available = _host_memory_available()
            _host_cache_budget = float("inf") if available is None else 0.6 * available
        nbytes = checkpoint_bytes(key)
        reserved = sum(_host_cache_reserved.values())
        if reserved + nbytes > _host_cache_budget:
            logger.info("H3 host cache skipped for %s (%d bytes; budget %.0f, reserved %d)",
                        key, nbytes, _host_cache_budget, reserved)
            return False
        _host_cache_reserved[key] = nbytes
        return True


def host_cache_pipelined():
    """How a preloaded stage fills its host cache. ``upfront`` (default)
    reads every shard first and then uploads, which lets other background
    reads overlap the upload (better on slow shared storage); ``pipelined``
    uploads each shard while the next is read (better on fast local disks)."""
    mode = os.environ.get("H3_HOST_CACHE_FILL", "upfront")
    if mode not in ("upfront", "pipelined"):
        raise ValueError(f"Unsupported H3_HOST_CACHE_FILL={mode!r}; use upfront or pipelined")
    return mode == "pipelined"


def host_cache_pinned():
    """H3_HOST_CACHE_PINNED=0 keeps the host weight cache in pageable memory."""
    return os.environ.get("H3_HOST_CACHE_PINNED", "1") != "0"


def prefetch_host_cache(path, devices):
    """Begin reading a later stage's checkpoint into (pinned) host memory while
    an earlier stage loads. Staged residency keeps it as the host cache;
    resident mode consumes it once at load time, then unpins and frees it."""
    if not host_cache_enabled(path):
        return
    # k100-extreme: read unpinned (no HIP calls while other stages compute);
    # the stage page-locks the buffers on its own thread right before upload.
    prefetch_checkpoint_raw(path, pinned=False)


def prefetch_component_shards(*paths):
    """Queue background reads of Diffusers component checkpoints (the VAEs)
    for ``serve_prefetched_safetensors``. They are converted on the host
    before moving to a device, so the buffers stay pageable."""
    for path in paths:
        if host_cache_enabled(path):
            prefetch_checkpoint_raw(path, pinned=False)


@contextmanager
def serve_prefetched_safetensors(*paths):
    """Let ``safetensors.torch.load_file`` return the shards of ``paths`` that
    ``prefetch_checkpoint_raw`` already read (the file bytes, viewed as the
    same tensors) instead of reading them again through mmap.

    Used around the Diffusers component loads (VAEs), whose single-threaded
    mmap reads are slow on network storage. Other files and devices, and
    everything after the block, use the original loader.
    """
    import safetensors.torch as safetensors_torch

    cache = {}
    for path in paths:
        cache.update(take_prefetched_checkpoint_raw(path) or {})
    served = dict.fromkeys(cache)
    original = safetensors_torch.load_file

    def load_file(filename, device="cpu"):
        raw = cache.pop(os.path.realpath(str(filename)), None) if str(device) == "cpu" else None
        if raw is None:
            return original(filename, device=device)
        return dict(raw[1])

    safetensors_torch.load_file = load_file
    try:
        yield
    finally:
        safetensors_torch.load_file = original
        unpin_checkpoint_raw(served)


_warmed_devices = set()
_peer_devices = None


def enable_peer_access(devices):
    """Enable device-to-device access between every pair of ``devices`` (once
    per process, before the engines' rank threads exist) so the sequence-
    parallel exchange can copy straight into peer buffers with the copy
    engines (H3_SP_A2A=peer, opt-in). If any pair cannot access its peer, the
    exchange falls back to RCCL (H3_SP_A2A=ccl, the default). On K100 x4 the
    copies are bitwise identical to RCCL but slower: DCU0's 11 GB/s link bounds
    both, and RCCL pipelines it better."""
    global _peer_devices
    devices = sorted(set(devices))
    if os.environ.get("H3_SP_A2A", "ccl") != "peer":
        return  # RCCL exchange (default): leave device peer mappings alone
    if _peer_devices is not None and set(devices) <= _peer_devices:
        return
    ok = len(devices) > 1 and torch.cuda.is_available()
    if ok:
        try:
            import ctypes
            hip = ctypes.CDLL("libamdhip64.so")
            for i in devices:
                for j in devices:
                    if i == j:
                        continue
                    can = ctypes.c_int(0)
                    if hip.hipDeviceCanAccessPeer(ctypes.byref(can), i, j) != 0 or not can.value:
                        ok = False
                        continue
                    if hip.hipSetDevice(i) != 0:
                        ok = False
                        continue
                    status = hip.hipDeviceEnablePeerAccess(j, 0)
                    if status not in (0, 704):  # 704: already enabled
                        ok = False
            hip.hipSetDevice(torch.cuda.current_device())
        except OSError:
            ok = False
    if not ok:
        os.environ["H3_SP_A2A"] = "ccl"  # some pair cannot map its peer
    _peer_devices = set(devices) if ok else set()


def warm_vendor_attention(devices):
    """k100-extreme: load the vendor flash-attention code objects on every
    device from this (single) thread before the engines' rank threads first
    use them. HIP loads code objects lazily at first launch; doing that in
    several threads while others launch kernels was seen to crash the DTK
    runtime. Covers the H3 DiT (d128), Qwen3-VL text (causal GQA) and vision
    (d72) variants; results are discarded."""
    import torch.nn.functional as F
    for index in sorted(set(devices) - _warmed_devices):
        with torch.cuda.device(index):
            def qkv(heads, kv_heads, tokens, dim):
                return (torch.randn(1, heads, tokens, dim, device="cuda", dtype=torch.bfloat16),
                        torch.randn(1, kv_heads, tokens, dim, device="cuda", dtype=torch.bfloat16),
                        torch.randn(1, kv_heads, tokens, dim, device="cuda", dtype=torch.bfloat16))
            for heads, kv_heads, tokens, dim, causal in ((14, 14, 4099, 128, False), (64, 8, 611, 128, True),
                                                         (16, 16, 1031, 72, False), (7, 7, 257, 128, False)):
                q, k, v = qkv(heads, kv_heads, tokens, dim)
                F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=kv_heads != heads)
            if os.environ.get("INFINICORE_HYGON_LINEAR") == "lt":
                # InfiniCore's BF16 linear layers then run through hipBLASLt,
                # whose gfx928 kernels sit in one code object loaded per device
                # at first use: load it here, before any rank thread exists.
                previous = torch.backends.cuda.preferred_blas_library()
                torch.backends.cuda.preferred_blas_library("cublaslt")
                try:
                    a = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
                    torch.mm(a, a.t())
                finally:
                    torch.backends.cuda.preferred_blas_library(previous)
            torch.cuda.synchronize()
        _warmed_devices.add(index)


def dit_parallel_mode(num_devices, num_heads=56):
    """Mirror of the native choice: H3_PARALLEL=sp|tp; by default (k100-extreme)
    sp on four or more devices, tp below."""
    mode = os.environ.get("H3_PARALLEL", "sp" if num_devices >= 4 else "tp")
    if mode not in ("tp", "sp"):
        raise ValueError(f"Unsupported H3_PARALLEL={mode!r}; use tp or sp")
    if num_devices <= 1 or num_heads % num_devices != 0:
        return "tp"
    return mode


# Device memory kept free beside the resident weights of both models for DiT
# activations, attention scratch and the VAE (K100 TP4 peak: 53.7 GB with
# 33 GB of weights per card).
RESIDENT_RESERVE_BYTES = 24 << 30


def _fits_resident(model_root, device_ids):
    """Qwen3-VL and the TP-sharded DiT both fit on every device."""
    if model_root is None or not device_ids or not torch.cuda.is_available():
        return True
    weights = sum(checkpoint_bytes(os.path.join(str(model_root), name))
                  for name in ("text_encoder", "transformer"))
    capacity = min(torch.cuda.get_device_properties(d).total_memory for d in device_ids)
    return weights / len(device_ids) + RESIDENT_RESERVE_BYTES <= capacity


def _tensor_bytes(path, keep=lambda name: True):
    """Bytes of the tensors of a safetensors checkpoint selected by ``keep``,
    from the shard headers only."""
    import struct
    total = 0
    for shard in checkpoint_shard_files(str(path)):
        with open(shard, "rb") as f:
            header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        header.pop("__metadata__", None)
        total += sum(info["data_offsets"][1] - info["data_offsets"][0]
                     for name, info in header.items() if keep(name))
    return total


def _qwen_loaded_tensor(name, layers=50):
    """Qwen3-VL tensors the condition encoder loads: everything but the LM head
    and the decoder layers past the 50th, whose hidden state H3 consumes."""
    if "lm_head" in name:
        return False
    parts = name.split(".layers.")
    if len(parts) == 2 and "language_model" in parts[0]:
        return int(parts[1].split(".", 1)[0]) < layers
    return True


# Device memory beside both models' resident weights in co-resident mode:
# VAE decoder replica (4.5 GiB BF16), runtime contexts, RCCL and BLAS
# workspaces, and the conditioner's activations.
CORESIDENT_RESERVE_BYTES = 7 << 30


def _fits_coresident(model_root, device_ids):
    """Qwen3-VL (TP shards) and the sequence-parallel DiT (full non-AdaLN
    weights plus an AdaLN shard) both fit on every device."""
    if model_root is None or not device_ids or not torch.cuda.is_available():
        return False
    ranks = len(device_ids)
    dit = os.path.join(str(model_root), "transformer")
    adaln = _tensor_bytes(dit, lambda n: "adaln_proj" in n)
    dit_bytes = _tensor_bytes(dit) - adaln + adaln / ranks
    qwen_bytes = _tensor_bytes(os.path.join(str(model_root), "text_encoder"),
                               _qwen_loaded_tensor) / ranks
    capacity = min(torch.cuda.get_device_properties(d).total_memory for d in device_ids)
    return dit_bytes + qwen_bytes + CORESIDENT_RESERVE_BYTES <= capacity


def residency_mode(num_devices=1, model_root=None, device_ids=None):
    """``H3_TEXT_ENCODER`` = tp_all (both models resident, TP DiT), coresident
    (both loaded at preload with the SP DiT; Qwen3-VL is released after each
    encode and reloaded from host memory only for a later case) or sequential
    (Qwen3-VL and DiT staged, the DiT uploaded at its first forward).

    By default the SP DiT is co-resident when Qwen3-VL's TP shards fit beside
    its full non-AdaLN weights (e.g. K100 x4: 43.6 + 12 GiB of 64), otherwise
    sequential; TP is resident when both TP shards plus a working reserve fit
    in the smallest device, else sequential (e.g. two 64 GB cards)."""
    mode = os.environ.get("H3_TEXT_ENCODER")
    if mode is None:
        if dit_parallel_mode(num_devices) == "sp":
            mode = "coresident" if _fits_coresident(model_root, device_ids) else "sequential"
        else:
            mode = "tp_all" if _fits_resident(model_root, device_ids) else "sequential"
    if mode not in ("tp_all", "sequential", "coresident"):
        raise ValueError(f"Unsupported H3_TEXT_ENCODER={mode!r}; use tp_all, coresident or sequential")
    return mode


class H3EngineResidency:
    """Lazily load engines and evict overlapping sequential engines between stages.

    A process lock covers each forward and its output copy, so another H3 stage
    cannot evict an engine in use. DiT stays resident across denoising steps;
    the next conditioner evicts it before loading its own weights. With
    H3_HOST_CACHE (default on) the shard bytes are read once at construction
    and every (re)load only copies them to the devices; otherwise weights are
    re-read from safetensors. ``preload`` loads the engine at construction for
    the stage that runs first (the conditioner), like the resident mode.
    """

    def __init__(self, path, *, devices, sequential, release_after=False, load_weights=True,
                 preload=False, coexist=False, keep_host_cache=True, **engine_kwargs):
        self.path = str(path)
        self.devices = frozenset(devices)
        self.sequential = sequential
        self.release_after = release_after
        # Co-resident engines (H3_TEXT_ENCODER=coresident) never evict each
        # other; a stage that is never reloaded drops its host copy after load.
        self.coexist = coexist
        self.keep_host_cache = keep_host_cache
        self.hold = 0  # > 0: release_after is suspended (warmup keeps engines loaded)
        # Called after each release, e.g. to stage the VAE decoder into the
        # memory the conditioner just returned.
        self.on_release = []
        self.load_weights = load_weights
        self.engine_kwargs = engine_kwargs
        self.engine = None
        self.raw_cache = None
        self._raw_future = None
        if sequential:
            if not hasattr(infinicore, "trim_memory"):
                raise RuntimeError("sequential H3 requires InfiniCore with trim_memory Python binding")
            if load_weights and host_cache_enabled(self.path):
                if preload and host_cache_pipelined():
                    # Filled shard by shard while the preload reads and uploads.
                    self.raw_cache = RawShardCache(pinned=host_cache_pinned())
                elif preload and not coexist:
                    # Read (ahead of background prefetches) while the devices
                    # and the engine are set up; the preload waits for it.
                    self._raw_future = _STAGE_READER.submit(self._read_host_cache, True)
                elif preload and not keep_host_cache:
                    # Co-resident DiT: take the background prefetch (unpinned),
                    # pinned on this thread right before the upload.
                    self._raw_future = _STAGE_READER.submit(self._read_host_cache, False)
                elif preload:
                    self._raw_future = _STAGE_READER.submit(self._read_host_cache, True)
                else:
                    # Waited for on first use, so the (unpinned) read overlaps
                    # the conditioning stage and the other components.
                    self._raw_future = _STAGE_READER.submit(self._read_host_cache, False)
            with _lock:
                _sessions.add(self)
            if preload:
                with _lock:
                    self.ensure_loaded()
        else:
            # Shards read in the background while an earlier engine loaded are
            # used once and then released; resident engines never reload.
            if load_weights:
                self.raw_cache = take_prefetched_checkpoint_raw(self.path)
            try:
                self.ensure_loaded()
            finally:
                unpin_checkpoint_raw(self.raw_cache, background=True)
                self.raw_cache = None

    def _read_host_cache(self, priority):
        # The stage loaded first pins while it reads (nothing else runs on the
        # devices yet); later stages pin on their own thread before uploading.
        started = time.perf_counter()
        self.raw_cache = read_checkpoint_raw(self.path, pinned=host_cache_pinned() and priority,
                                             priority=priority)
        print("[h3.residency] " + json.dumps({"event": "host_cache", "path": self.path,
              "bytes": sum(sum(t.numel() * t.element_size() for t in raw[1].values())
                           for raw in self.raw_cache.values()),
              "seconds": time.perf_counter() - started}), flush=True)

    def ensure_loaded(self):
        if self.engine is not None:
            return self.engine
        if self.sequential:
            if not self.coexist:
                for other in list(_sessions):
                    if other is not self and self.devices.intersection(other.devices):
                        other.release()
            # Blocks cached by earlier stages (e.g. the first-frame VAE encode)
            # would otherwise be unavailable to the engine about to load.
            trim_device_caches(self.devices)
        logger.info("H3 loading %s on devices %s", self.path, sorted(self.devices))
        started = time.perf_counter()
        if os.environ.get("INFINICORE_HYGON_MHA", "aten") == "aten":
            # Before the engine's rank threads exist; overlaps a staged read.
            warm_vendor_attention(self.devices)
        enable_peer_access(self.devices)
        self.engine = InferEngine(self.path, **self.engine_kwargs)
        try:
            if self._raw_future is not None:
                future, self._raw_future = self._raw_future, None
                future.result()
            if self.raw_cache and host_cache_pinned():
                started_pin = time.perf_counter()
                pin_checkpoint_raw(self.raw_cache)
                logger.info("H3 pinned host cache of %s in %.2f s", self.path,
                            time.perf_counter() - started_pin)
            if self.load_weights:
                load_model_state_dict_by_file(self.engine, self.path, dtype=self.engine.dtype,
                                              raw_cache=self.raw_cache)
        except BaseException:
            self.release()
            raise
        if not self.keep_host_cache and self.raw_cache is not None:
            # Never reloaded: return the host copy once the uploads finished.
            # Unregistered here, synchronously: hipHostUnregister on a helper
            # thread while the engine's first forward launches kernels was
            # followed by GPU memory faults on DTK 25.04.2 (3 of 5 runs).
            for index in self.devices:
                torch.cuda.synchronize(index)
            unpin_checkpoint_raw(self.raw_cache, background=False)
            self.raw_cache = None
        if self.sequential and os.environ.get("H3_TRIM_AFTER_LOAD", "1") == "1":
            # Loading frees temporaries (e.g. unfused projections) into the
            # native allocator's cache; return them before the first forward,
            # whose collectives allocate their RCCL buffers outside that cache.
            trim_device_caches(self.devices)
            print("[h3.residency] " + json.dumps({"event": "load", "path": self.path,
                  "devices": sorted(self.devices), "seconds": time.perf_counter() - started}), flush=True)
        return self.engine

    def release(self):
        if self.engine is None:
            return
        logger.info("H3 releasing %s", self.path)
        started = time.perf_counter()
        for index in self.devices:
            torch.cuda.synchronize(index)
        # InferEngine's destructor joins all TP workers before freeing weights.
        self.engine = None
        gc.collect()
        trim_device_caches(self.devices)
        if self.sequential:
            print("[h3.residency] " + json.dumps({"event": "release", "path": self.path,
                  "devices": sorted(self.devices), "seconds": time.perf_counter() - started}), flush=True)
        for callback in list(self.on_release):
            callback()

    @contextmanager
    def use(self):
        with _lock if self.sequential else nullcontext():
            self.ensure_loaded()
            try:
                # Yield the owner, not the engine: no contextmanager local may
                # retain the C++ engine while release() trims its weight blocks.
                yield self
            except BaseException:
                if self.sequential:
                    self.release()
                raise
            finally:
                if self.sequential and self.release_after and not self.hold:
                    self.release()

    @contextmanager
    def held(self):
        """Keep the engine loaded across uses inside the block (warmup)."""
        self.hold += 1
        try:
            yield self
        finally:
            self.hold -= 1
