from __future__ import annotations

import itertools
import os
from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import infinilm.multimodal._h3_defaults  # noqa: F401  (validated defaults first)
import infinicore
import numpy as np
import torch
from infinicore.lib import _infinicore
from infinilm.cache import StaticKVCacheConfig
from infinilm.distributed import DistConfig
from infinilm.infer_engine import read_hf_config
from infinilm.multimodal._h3_residency import H3EngineResidency, residency_mode
from transformers import AutoConfig, Qwen3VLForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPast

_DEVICE_TYPE_MAP = {
    "nvidia": "cuda",
    "hygon": "cuda",
    "metax": "cuda",
    "qy": "cuda",
    "cambricon": "mlu",
    "ascend": "npu",
    "moore": "musa",
}


def _runtime_device_type(device: str) -> str:
    return _DEVICE_TYPE_MAP.get(device.lower(), device.lower())


@dataclass(frozen=True)
class MiniMaxH3TransformerInputs:
    """Packed FL2VA inputs expected by the native MiniMax-H3 transformer."""

    video_hidden_states: torch.Tensor
    audio_hidden_states: torch.Tensor
    encoder_hidden_states: torch.Tensor
    timestep: torch.Tensor
    timestep_indices: torch.Tensor
    token_tags: torch.Tensor
    position_ids: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor

    def validate(self) -> None:
        sequence_length = self.position_ids.shape[0]
        if self.position_ids.ndim != 2 or self.position_ids.shape[1] != 3:
            raise ValueError("position_ids must have shape [sequence_length, 3]")
        if self.timestep.ndim != 1:
            raise ValueError("timestep must have shape [num_timesteps]")
        for name in ("timestep_indices", "token_tags"):
            tensor = getattr(self, name)
            if tensor.shape != (sequence_length,):
                raise ValueError(
                    f"{name} must have shape [{sequence_length}], got {tuple(tensor.shape)}"
                )
        for name in ("video_indices", "audio_indices", "text_indices"):
            tensor = getattr(self, name)
            if tensor.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")
            if tensor.dtype not in (torch.int32, torch.int64):
                raise TypeError(f"{name} must use int32 or int64 indices")
        for name in (
            "timestep_indices",
            "token_tags",
        ):
            tensor = getattr(self, name)
            if tensor.dtype not in (torch.int32, torch.int64):
                raise TypeError(f"{name} must use int32 or int64 indices")
        if not (
            self.position_ids.is_floating_point()
            or self.position_ids.dtype in (torch.int32, torch.int64)
        ):
            raise TypeError("position_ids must use a floating-point or integer dtype")
        for name in (
            "video_hidden_states",
            "audio_hidden_states",
            "encoder_hidden_states",
        ):
            tensor = getattr(self, name)
            if tensor.ndim != 3 or tensor.shape[0] != 1:
                raise ValueError(f"{name} must have shape [1, tokens, channels]")


@dataclass(frozen=True)
class MiniMaxH3TransformerOutput:
    sample: torch.Tensor | infinicore.Tensor
    audio_sample: torch.Tensor | infinicore.Tensor

    @property
    def video(self) -> torch.Tensor | infinicore.Tensor:
        return self.sample

    @property
    def audio(self) -> torch.Tensor | infinicore.Tensor:
        return self.audio_sample


def minimax_h3_rotary_cache(
    position_ids: torch.Tensor,
    rope_freq_dim: int = 16,
    rope_theta: float = 10000.0,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Build H3's continuous 3-axis NeoX RoPE cache on-device (PyTorch form).

    Reference only (diagnostics); the DiT gets the same table, bitwise, from
    InfiniCore (MiniMaxH3TransformerRunner._rotary_cache)."""

    if position_ids.ndim != 2 or position_ids.shape[1] != 3:
        raise ValueError("position_ids must have shape [sequence_length, 3]")
    inv_freq = 1.0 / (
        rope_theta
        ** (
            torch.arange(
                0,
                2 * rope_freq_dim,
                2,
                dtype=torch.float32,
                device=position_ids.device,
            )
            / (2 * rope_freq_dim)
        )
    )
    frequencies = (
        position_ids.float().unsqueeze(-1) * inv_freq.view(1, 1, -1)
    ).flatten(1)
    return torch.cat((frequencies.cos(), frequencies.sin()), dim=-1).to(dtype)


class _MiniMaxH3NativeQwen3VLModel(torch.nn.Module):
    def __init__(self, encoder: "MiniMaxH3ConditionEncoder") -> None:
        super().__init__()
        object.__setattr__(self, "_encoder", encoder)

    def forward(self, **model_inputs: Any) -> BaseModelOutputWithPast:
        hidden_states = self._encoder.encode(model_inputs)
        all_hidden_states = [None] * (self._encoder.num_hidden_layers + 1)
        all_hidden_states[self._encoder.num_hidden_layers] = hidden_states
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=tuple(all_hidden_states),
        )


class MiniMaxH3ConditionEncoder(Qwen3VLForConditionalGeneration):
    """Diffusers-compatible Qwen3-VL facade backed by native InfiniLM kernels."""

    num_hidden_layers = 50

    def __init__(
        self,
        text_encoder_path: str | Path,
        *,
        device: str = "nvidia",
        device_id: int = 0,
        tp_size: int = 1,
        tp_device_ids: Sequence[int] | None = None,
        dtype: torch.dtype = torch.bfloat16,
        load_weights: bool = True,
        max_cache_len: int | None = None,
        **_: Any,
    ) -> None:
        torch.nn.Module.__init__(self)
        # The conditioner runs one prefill of the prompt and the first/last
        # frame tokens (about 1-2k); its static KV cache covers all 50 layers,
        # so 32k tokens cost 1.6 GiB per device that the co-resident DiT needs.
        if max_cache_len is None:
            max_cache_len = int(os.environ.get("H3_QWEN_MAX_CACHE_LEN", "8192"))
        self.max_cache_len = max_cache_len
        self.text_encoder_path = Path(text_encoder_path)
        if (self.text_encoder_path / "text_encoder" / "config.json").is_file():
            self.text_encoder_path = self.text_encoder_path / "text_encoder"

        self.config = AutoConfig.from_pretrained(
            str(self.text_encoder_path), local_files_only=True
        )
        if not hasattr(self.config, "text_config"):
            raise RuntimeError("MiniMax-H3 requires a Qwen3-VL text_config")
        if self.config.text_config.num_hidden_layers <= self.num_hidden_layers:
            raise RuntimeError(
                "MiniMax-H3 requires a hidden state before the Qwen3-VL final norm"
            )

        if tp_device_ids is not None:
            tp_device_ids = list(tp_device_ids)
            if not tp_device_ids:
                raise ValueError("tp_device_ids must not be empty")
            if tp_size != 1 and tp_size != len(tp_device_ids):
                raise ValueError("tp_size must match tp_device_ids when both are set")
            distributed_config = DistConfig(tp_device_ids=tp_device_ids)
            device_id = tp_device_ids[0]
        else:
            distributed_config = DistConfig(tp_size=tp_size)

        self.runtime_device = infinicore.device(_runtime_device_type(device), device_id)
        self.torch_device = torch.device("cuda", device_id)
        self._dtype = dtype
        # Validated on gfx936: match the reference Qwen BF16 RMS boundaries.
        # Keep other architectures and explicit user overrides unchanged.
        if "H3_QWEN_RMS" not in os.environ and torch.version.hip:
            arch = getattr(torch.cuda.get_device_properties(self.torch_device), "gcnArchName", "").split(":")[0]
            if arch == "gfx936":
                os.environ.setdefault("H3_QWEN_RMS", "reference")
                print("[h3.conditioner] Qwen RMS: reference (gfx936 default)", flush=True)
        mode = residency_mode(len(distributed_config.tp_device_ids),
                              self.text_encoder_path.parent,
                              distributed_config.tp_device_ids)
        self._residency = H3EngineResidency(
            str(self.text_encoder_path),
            devices=distributed_config.tp_device_ids,
            sequential=mode != "tp_all",
            coexist=mode == "coresident",
            release_after=True,
            preload=True,
            load_weights=load_weights,
            device=self.runtime_device,
            distributed_config=distributed_config,
            cache_config=StaticKVCacheConfig(1, max_cache_len),
            hf_config_overrides={
                "condition_encoder_mode": True,
                "condition_encoder_num_hidden_layers": self.num_hidden_layers,
                "condition_encoder_skip_final_norm": True,
                "load_only_model_keys": True,
                "skip_sampling": True,
            },
        )
        self.model = _MiniMaxH3NativeQwen3VLModel(self)
        self.lm_head = None

    @property
    def engine(self):
        return self._residency.engine

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self.torch_device

    @staticmethod
    def _split_pixels(
        pixels: torch.Tensor | None,
        grids: torch.Tensor | None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if pixels is None or grids is None:
            return []
        items = []
        offset = 0
        for grid in grids:
            grid_values = [int(value) for value in grid.tolist()]
            grid_t, grid_h, grid_w = grid_values
            patch_count = grid_t * grid_h * grid_w
            items.append((pixels.narrow(0, offset, patch_count), grid))
            offset += patch_count
        if offset != pixels.shape[0]:
            raise ValueError("Qwen3-VL pixel rows do not match grid_thw")
        return items

    def _prepare_native_inputs(self, model_inputs: Mapping[str, Any]) -> dict[str, Any]:
        input_ids = model_inputs["input_ids"]
        token_types = model_inputs["mm_token_type_ids"]
        attention_mask = model_inputs.get("attention_mask")
        if input_ids.shape[0] != 1:
            raise NotImplementedError(
                "MiniMax-H3 native Qwen3-VL currently supports batch size 1"
            )

        image_items = self._split_pixels(
            model_inputs.get("pixel_values"),
            model_inputs.get("image_grid_thw"),
        )
        video_items = self._split_pixels(
            model_inputs.get("pixel_values_videos"),
            model_inputs.get("video_grid_thw"),
        )
        merge_size = int(self.config.vision_config.spatial_merge_size)
        sequence_length = input_ids.shape[1]
        if sequence_length > self.max_cache_len:
            raise ValueError(
                f"Qwen3-VL input has {sequence_length} tokens, more than its KV cache "
                f"({self.max_cache_len}); set H3_QWEN_MAX_CACHE_LEN to at least that"
            )
        # Qwen3-VL's 3-axis position ids (integer bookkeeping, built on the
        # host and copied once; no device computation outside InfiniCore).
        position_ids = np.zeros((3, sequence_length), dtype=np.int64)
        pixels = []
        grids = []
        bounds = []
        current_position = 0
        image_index = 0
        video_index = 0
        active_video = None

        token_type_values = token_types[0].tolist()
        for modality, group in itertools.groupby(
            enumerate(token_type_values), key=lambda item: item[1]
        ):
            group = list(group)
            start = group[0][0]
            end = group[-1][0] + 1
            if modality == 0:
                position_ids[:, start:end] = (
                    np.arange(end - start, dtype=np.int64) + current_position
                )
                current_position += end - start
                continue

            if modality == 1:
                if image_index >= len(image_items):
                    raise ValueError("Qwen3-VL image spans do not match pixel inputs")
                media_pixels, media_grid = image_items[image_index]
                image_index += 1
                grid_t, grid_h, grid_w = [int(value) for value in media_grid.tolist()]
                media_bounds = [(start, end)]
            elif modality == 2:
                if active_video is None:
                    if video_index >= len(video_items):
                        raise ValueError(
                            "Qwen3-VL video spans do not match pixel inputs"
                        )
                    media_pixels, media_grid = video_items[video_index]
                    video_index += 1
                    original_grid = [int(value) for value in media_grid.tolist()]
                    active_video = {
                        "pixels": media_pixels,
                        "grid": media_grid,
                        "remaining": original_grid[0],
                        "bounds": [],
                    }
                media_pixels = active_video["pixels"]
                media_grid = active_video["grid"]
                _, grid_h, grid_w = [int(value) for value in media_grid.tolist()]
                grid_t = 1
                active_video["bounds"].append((start, end))
                active_video["remaining"] -= 1
                media_bounds = None
            else:
                raise ValueError(f"Unsupported Qwen3-VL modality type {modality}")

            llm_t = grid_t
            llm_h = grid_h // merge_size
            llm_w = grid_w // merge_size
            temporal = np.arange(llm_t, dtype=np.int64)
            height = np.arange(llm_h, dtype=np.int64) + current_position
            width = np.arange(llm_w, dtype=np.int64) + current_position
            grid_positions = np.stack(
                np.meshgrid(temporal, height, width, indexing="ij"), axis=0
            ).reshape(3, -1)
            grid_positions[0] += current_position
            if grid_positions.shape[1] != end - start:
                raise ValueError(
                    "Qwen3-VL visual token span does not match the merged grid"
                )
            position_ids[:, start:end] = grid_positions
            current_position += max(grid_h, grid_w) // merge_size

            if modality == 2:
                if active_video["remaining"] < 0:
                    raise ValueError("Qwen3-VL video has too many visual spans")
                if active_video["remaining"] == 0:
                    media_bounds = active_video["bounds"]
                    active_video = None
                else:
                    continue

            pixels.append(media_pixels.to(self.torch_device, dtype=self.dtype))
            grids.append(media_grid.to(self.torch_device))
            bounds.append(
                torch.tensor(media_bounds, dtype=torch.int64, device=self.torch_device)
            )

        if active_video is not None:
            raise ValueError("Qwen3-VL video has fewer visual spans than frames")
        if image_index != len(image_items) or video_index != len(video_items):
            raise ValueError("Qwen3-VL pixel inputs have no matching token spans")

        if attention_mask is not None and not bool(attention_mask.all()):
            raise NotImplementedError(
                "MiniMax-H3 native Qwen3-VL does not support padded conditioner batches"
            )

        def to_native(tensor):
            return infinicore.from_torch(tensor.contiguous())

        return {
            "input_ids": to_native(input_ids.to(self.torch_device)),
            "position_ids": to_native(torch.from_numpy(position_ids).to(self.torch_device)),
            "input_offsets": to_native(
                torch.tensor(
                    [0, sequence_length],
                    dtype=torch.int32,
                    device=self.torch_device,
                )
            ),
            "past_kv_lengths": to_native(
                torch.tensor([0], dtype=torch.int32, device=self.torch_device)
            ),
            "total_kv_lengths": to_native(
                torch.tensor(
                    [sequence_length],
                    dtype=torch.int32,
                    device=self.torch_device,
                )
            ),
            "pixel_values": [to_native(value) for value in pixels],
            "image_grid_thw": [to_native(value) for value in grids],
            "image_bound": [to_native(value) for value in bounds],
            "image_req_ids": [0] * len(pixels),
        }

    def encode(self, model_inputs: Mapping[str, Any]) -> torch.Tensor:
        with self._residency.use():
            native_inputs = self._prepare_native_inputs(model_inputs)
            # from_torch shares storage; the native worker streams do not inherit
            # dependencies from PyTorch's producer stream.
            torch.cuda.synchronize(self.torch_device)
            input_ids = native_inputs.pop("input_ids")
            output = self.engine.forward_raw(input_ids, **native_inputs)
            return MiniMaxH3TransformerRunner._to_torch(output["logits"], self.torch_device)

    def forward(self, **model_inputs: Any) -> BaseModelOutputWithPast:
        return self.model(**model_inputs)


class MiniMaxH3TransformerRunner:
    """Bridge between Python FL2VA preprocessing/codecs and the native DiT.

    The video/audio VAEs remain Python components. Their packed tensors cross
    this boundary once per denoising step.
    """

    def __init__(
        self,
        transformer_path: str | Path,
        *,
        device: str = "nvidia",
        device_id: int = 0,
        tp_size: int = 1,
        tp_device_ids: Sequence[int] | None = None,
        load_weights: bool = True,
    ) -> None:
        self.transformer_path = Path(transformer_path)
        if (self.transformer_path / "transformer" / "config.json").is_file():
            self.transformer_path = self.transformer_path / "transformer"
        if tp_device_ids is not None:
            tp_device_ids = list(tp_device_ids)
            if not tp_device_ids:
                raise ValueError("tp_device_ids must not be empty")
            if tp_size != 1 and tp_size != len(tp_device_ids):
                raise ValueError("tp_size must match tp_device_ids when both are set")
            distributed_config = DistConfig(tp_device_ids=tp_device_ids)
            device_id = tp_device_ids[0]
        else:
            if tp_size < 1:
                raise ValueError("tp_size must be positive")
            distributed_config = DistConfig(tp_size=tp_size)

        self.device = infinicore.device(_runtime_device_type(device), device_id)
        self.hf_config = read_hf_config(str(self.transformer_path))
        mode = residency_mode(len(distributed_config.tp_device_ids),
                              self.transformer_path.parent,
                              distributed_config.tp_device_ids)
        self._residency = H3EngineResidency(
            str(self.transformer_path),
            devices=distributed_config.tp_device_ids,
            sequential=mode != "tp_all",
            # Co-resident: loaded at its first use (the load-time warmup) beside
            # Qwen3-VL and never released, so its host copy is dropped after
            # the upload.
            coexist=mode == "coresident",
            keep_host_cache=mode != "coresident",
            load_weights=load_weights,
            device=self.device,
            distributed_config=distributed_config,
        )
        if self.engine is not None:
            self.hf_config = self.engine.hf_config
        self.rope_freq_dim = int(self.hf_config.get("rope_freq_dim", 16))
        self.rope_theta = float(self.hf_config.get("rope_theta", 10000.0))
        self._rope_power: tuple[torch.device, torch.Tensor] | None = None

    @property
    def engine(self):
        return self._residency.engine

    @staticmethod
    def _to_infinicore(tensor: torch.Tensor) -> infinicore.Tensor:
        return infinicore.from_torch(tensor.contiguous())

    @staticmethod
    def _to_torch(tensor: infinicore.Tensor, device: torch.device) -> torch.Tensor:
        output = torch.empty(
            tuple(tensor.shape),
            dtype=infinicore.utils.to_torch_dtype(tensor.dtype),
            device=device,
        )
        infinicore.from_torch(output).copy_(tensor)
        infinicore.sync_device()
        return output

    def _rotary_cache(self, position_ids: torch.Tensor) -> infinicore.Tensor:
        """H3's 3-axis RoPE table [S, 96] (cat(cos, sin), BF16) computed by
        InfiniCore: the powers theta ** (2k / 2d) with float_power (the device
        powf, once per device) and the per-token cos/sin with the NineToothed
        h3_rope_cache kernel. Bitwise equal to minimax_h3_rotary_cache."""
        if self.rope_freq_dim != 16:
            raise NotImplementedError("the native H3 RoPE table covers rope_freq_dim 16")
        device = position_ids.device
        previous = _infinicore.get_device()
        infinicore.set_device(infinicore.device(device.type, device.index or 0))
        try:
            if self._rope_power is None or self._rope_power[0] != device:
                d = self.rope_freq_dim
                # Exact constants: arange(0, 2d, 2) / 2d and theta.
                exponent = torch.tensor([2 * k / (2 * d) for k in range(d)],
                                        dtype=torch.float32, device=device)
                base = torch.full((d,), self.rope_theta, dtype=torch.float32, device=device)
                power = torch.empty_like(exponent)
                torch.cuda.synchronize(device)
                _infinicore.float_power_(*(infinicore.from_torch(t)._underlying
                                           for t in (power, base, exponent)))
                infinicore.sync_stream()
                self._rope_power = (device, power)
            positions = position_ids.to(torch.float64).contiguous()
            # The native kernel runs on InfiniCore's stream, not PyTorch's.
            torch.cuda.synchronize(device)
            cache = infinicore.Tensor(_infinicore.h3_rope_cache(
                infinicore.from_torch(positions)._underlying,
                infinicore.from_torch(self._rope_power[1])._underlying))
            infinicore.sync_stream()
            return cache
        finally:
            _infinicore.set_device(previous)

    def __call__(
        self,
        inputs: MiniMaxH3TransformerInputs,
        *,
        return_torch: bool = True,
    ) -> MiniMaxH3TransformerOutput:
        inputs.validate()
        with self._residency.use():
            return self._forward(inputs, return_torch=return_torch)

    def _forward(
        self,
        inputs: MiniMaxH3TransformerInputs,
        *,
        return_torch: bool,
    ) -> MiniMaxH3TransformerOutput:
        torch_inputs = {
            field.name: getattr(inputs, field.name) for field in fields(inputs)
        }
        if infinicore.utils.to_torch_dtype(self.engine.dtype) != torch.bfloat16:
            raise NotImplementedError("the native H3 RoPE table is BF16")
        rotary_cache = self._rotary_cache(inputs.position_ids)
        # Row indices into the table (built on the host, copied once).
        torch_inputs["position_ids"] = torch.from_numpy(
            np.arange(inputs.position_ids.shape[0], dtype=np.int64)
        ).to(inputs.position_ids.device)
        native_inputs = {
            name: self._to_infinicore(tensor) for name, tensor in torch_inputs.items()
        }
        native_inputs["rotary_cos_sin_cache"] = rotary_cache
        # Include the contiguous copies. Without this boundary, subsequent
        # denoising calls can read unfinished inputs.
        for device in {tensor.device for tensor in torch_inputs.values() if tensor.is_cuda}:
            torch.cuda.synchronize(device)
        output = self.engine.forward_raw(None, **native_inputs)
        video = output["logits"]
        audio = output["hidden_states"]
        if return_torch:
            torch_device = inputs.video_hidden_states.device
            video = self._to_torch(video, torch_device)
            audio = self._to_torch(audio, torch_device)
        return MiniMaxH3TransformerOutput(sample=video, audio_sample=audio)


class MiniMaxH3DiffusersAdapter(torch.nn.Module):
    """Diffusers-style facade around :class:`MiniMaxH3TransformerRunner`."""

    def __init__(
        self,
        runner: MiniMaxH3TransformerRunner,
        *,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.runner = runner
        self._dtype = dtype
        self.config = SimpleNamespace(**runner.hf_config)

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
        attention_kwargs: Mapping[str, Any] | None = None,
        return_dict: bool = True,
    ) -> MiniMaxH3TransformerOutput | tuple[torch.Tensor, torch.Tensor]:
        del attention_kwargs
        output = self.runner(
            MiniMaxH3TransformerInputs(
                video_hidden_states=hidden_states,
                audio_hidden_states=audio_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                timestep_indices=timestep_indices,
                token_tags=token_tags,
                position_ids=position_ids,
                video_indices=video_indices,
                audio_indices=audio_indices,
                text_indices=text_indices,
            )
        )
        if return_dict:
            return output
        return output.sample, output.audio_sample
