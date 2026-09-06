"""Llama 3.2 compatibility implementation for transformers==4.35.2.

transformers 4.35.2 already contains the original Llama decoder, but its
LlamaConfig only accepts the old linear/dynamic RoPE scaling schema and cannot
load Llama 3.2's ``rope_type='llama3'`` configuration.  The decoder itself is
structurally the same rotary/GQA/RMSNorm/SwiGLU stack that this project already
uses for its local Qwen2 compatibility implementation, so reuse that stack and
only provide Llama 3.x RoPE plus Llama-specific configuration defaults.
"""

from __future__ import annotations

import json
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers.utils import cached_file

from .qwen2_compat import (
    Qwen2Attention,
    Qwen2Config,
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
    Qwen2MLP,
    Qwen2Model,
    Qwen2PreTrainedModel,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
)


class Llama32Config(Qwen2Config):
    """Config compatible with the text-only Llama 3.2 decoder checkpoints."""

    model_type = "llama"

    def __init__(
        self,
        vocab_size: int = 128256,
        hidden_size: int = 3072,
        intermediate_size: int = 8192,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 24,
        num_key_value_heads: Optional[int] = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 131072,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        tie_word_embeddings: bool = True,
        rope_theta: float = 500000.0,
        rope_scaling: Optional[Dict[str, float]] = None,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        pretraining_tp: int = 1,
        **kwargs,
    ) -> None:
        if mlp_bias:
            raise ValueError("Llama32 compatibility currently supports mlp_bias=False only.")
        if int(pretraining_tp) != 1:
            raise ValueError("Llama32 compatibility currently supports pretraining_tp=1 only.")

        self.head_dim = int(head_dim)
        self.rope_scaling = dict(rope_scaling) if rope_scaling is not None else None
        self.mlp_bias = bool(mlp_bias)
        self.pretraining_tp = int(pretraining_tp)
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            attention_dropout=attention_dropout,
            attention_bias=attention_bias,
            **kwargs,
        )


class Llama32RotaryEmbedding(Qwen2RotaryEmbedding):
    """RoPE used by Llama 3.1/3.2, with the legacy Qwen2-compatible API."""

    def __init__(self, config: Llama32Config) -> None:
        super().__init__(
            dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        rope_scaling = config.rope_scaling
        if rope_scaling is None:
            return
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
        if rope_type != "llama3":
            raise ValueError(
                "Llama32 compatibility expects rope_scaling.rope_type='llama3', "
                f"got {rope_type!r}."
            )

        factor = float(rope_scaling["factor"])
        low_freq_factor = float(rope_scaling["low_freq_factor"])
        high_freq_factor = float(rope_scaling["high_freq_factor"])
        old_context_len = int(rope_scaling["original_max_position_embeddings"])
        if factor < 1.0 or low_freq_factor <= 0.0 or high_freq_factor <= low_freq_factor:
            raise ValueError(f"Invalid Llama3 rope_scaling configuration: {rope_scaling}")

        inv_freq = self.inv_freq.to(dtype=torch.float32)
        wavelen = (2.0 * math.pi) / inv_freq
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        # This is the Llama 3.x frequency interpolation used by modern
        # Transformers. High frequencies stay unchanged, low frequencies are
        # divided by ``factor``, and the middle band is smoothly interpolated.
        scaled_inv_freq = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smoothed_inv_freq = (
            (1.0 - smooth_factor) * scaled_inv_freq / factor
            + smooth_factor * scaled_inv_freq
        )
        is_medium_freq = (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen)
        self.inv_freq = torch.where(is_medium_freq, smoothed_inv_freq, scaled_inv_freq)


class Llama32Attention(Qwen2Attention):
    def __init__(self, config: Llama32Config, layer_idx: Optional[int] = None) -> None:
        super().__init__(config=config, layer_idx=layer_idx)
        self.rotary_emb = Llama32RotaryEmbedding(config)
        self.scaling = self.head_dim ** -0.5


class Llama32DecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config: Llama32Config, layer_idx: int) -> None:
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = Llama32Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Llama32Model(Qwen2Model):
    config_class = Llama32Config
    _no_split_modules = ["Llama32DecoderLayer"]

    def __init__(self, config: Llama32Config) -> None:
        Qwen2PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Llama32DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()


class Llama32ForCausalLM(Qwen2ForCausalLM):
    config_class = Llama32Config
    _no_split_modules = ["Llama32DecoderLayer"]

    def __init__(self, config: Llama32Config) -> None:
        Qwen2PreTrainedModel.__init__(self, config)
        self.model = Llama32Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()


def load_llama32_compat_model(model_id: str, *, torch_dtype: torch.dtype):
    """Load a Llama 3.2 checkpoint without invoking old LlamaConfig parsing."""
    config_path = cached_file(model_id, "config.json")
    if config_path is None:
        raise FileNotFoundError(f"config.json was not found for {model_id}")
    with open(config_path, "r", encoding="utf-8") as handle:
        config_dict = json.load(handle)

    rope_scaling = config_dict.get("rope_scaling")
    rope_type = None
    if isinstance(rope_scaling, dict):
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
    if str(config_dict.get("model_type", "")).lower() != "llama" or rope_type != "llama3":
        raise ValueError(
            f"{model_id} does not look like a Llama 3.x text checkpoint with llama3 RoPE."
        )

    config = Llama32Config(**config_dict)
    return Llama32ForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
