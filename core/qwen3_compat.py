"""Compatibility implementation for loading Qwen3 on old Transformers.

Qwen3 keeps the Qwen2-style decoder/MLP/RoPE layout used by this project, but
adds per-head RMSNorm to projected queries and keys.  transformers==4.35.2
predates Qwen3, so this module reuses the local Qwen2 compatibility stack and
only supplies the Qwen3-specific configuration and attention normalization.
"""

from __future__ import annotations

from typing import Optional

import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .qwen2_compat import (
    Qwen2Attention,
    Qwen2Config,
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
    Qwen2MLP,
    Qwen2Model,
    Qwen2PreTrainedModel,
    Qwen2RMSNorm,
)


class Qwen3Config(Qwen2Config):
    model_type = "qwen3"

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 40960,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 1000000.0,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        sliding_window: Optional[int] = None,
        max_window_layers: Optional[int] = None,
        use_sliding_window: bool = False,
        **kwargs,
    ) -> None:
        self.head_dim = int(head_dim)
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
            sliding_window=sliding_window,
            max_window_layers=max_window_layers,
            use_sliding_window=use_sliding_window,
            **kwargs,
        )


class Qwen3Attention(Qwen2Attention):
    def __init__(self, config: Qwen3Config, layer_idx: Optional[int] = None) -> None:
        super().__init__(config=config, layer_idx=layer_idx)
        # Qwen3 normalizes q/k independently on each attention head after the
        # linear projection and before RoPE.  Qwen2/Qwen2.5 do not have these.
        self.q_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.scaling = self.head_dim ** -0.5


class Qwen3DecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int) -> None:
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen3Model(Qwen2Model):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DecoderLayer"]

    def __init__(self, config: Qwen3Config) -> None:
        Qwen2PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()


class Qwen3ForCausalLM(Qwen2ForCausalLM):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DecoderLayer"]

    def __init__(self, config: Qwen3Config) -> None:
        Qwen2PreTrainedModel.__init__(self, config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()


def register_qwen3_compat() -> None:
    """Register local Qwen3 classes with Auto* mappings on old Transformers."""
    try:
        AutoConfig.register("qwen3", Qwen3Config)
    except ValueError:
        pass
    try:
        AutoModel.register(Qwen3Config, Qwen3Model)
    except ValueError:
        pass
    try:
        AutoModelForCausalLM.register(Qwen3Config, Qwen3ForCausalLM)
    except ValueError:
        pass
