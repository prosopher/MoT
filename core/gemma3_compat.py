"""Text-only Gemma 3 compatibility for transformers==4.35.2.

The pinned Transformers version predates Gemma 3.  This module implements the
small subset MoT needs for ``google/gemma-3-1b-it`` while preserving the
checkpoint's native module/state-dict layout:

* Gemma RMSNorm semantics (``1 + weight``),
* scaled token embeddings,
* GeGLU MLP,
* per-head q/k RMSNorm,
* query-pre-attention scaling,
* alternating local/global RoPE bases and sliding-window attention.

It supports both standalone ``gemma3_text`` checkpoints and multimodal
``gemma3`` checkpoints.  For multimodal checkpoints MoT loads only the nested
``language_model`` weights; the vision tower and projector are intentionally
not materialized because train/eval/AgentRunner are text-only.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationConfig, PretrainedConfig
from transformers.utils import cached_file

from .qwen2_compat import (
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2Model,
    Qwen2PreTrainedModel,
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)


class Gemma3TextConfig(Qwen2Config):
    model_type = "gemma3_text"

    def __init__(
        self,
        vocab_size: int = 262208,
        hidden_size: int = 2304,
        intermediate_size: int = 9216,
        num_hidden_layers: int = 26,
        num_attention_heads: int = 8,
        num_key_value_heads: Optional[int] = 4,
        head_dim: int = 256,
        hidden_activation: str = "gelu_pytorch_tanh",
        hidden_act: Optional[str] = None,
        max_position_embeddings: int = 131072,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: Optional[int] = 0,
        eos_token_id: Optional[int] = 1,
        bos_token_id: Optional[int] = 2,
        tie_word_embeddings: bool = True,
        rope_theta: float = 1000000.0,
        rope_local_base_freq: float = 10000.0,
        rope_scaling=None,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        query_pre_attn_scalar: float = 256.0,
        sliding_window: Optional[int] = 4096,
        sliding_window_pattern: int = 6,
        final_logit_softcapping: Optional[float] = None,
        attn_logit_softcapping: Optional[float] = None,
        use_bidirectional_attention: bool = False,
        **kwargs,
    ) -> None:
        if use_bidirectional_attention:
            raise ValueError("MoT Gemma3 compatibility supports causal text attention only.")
        if rope_scaling not in (None, {}):
            rope_type = rope_scaling.get("rope_type", rope_scaling.get("type")) if isinstance(rope_scaling, dict) else None
            if rope_type != "linear" or float(rope_scaling.get("factor", 0.0)) < 1.0:
                raise ValueError(
                    "MoT Gemma3 compatibility supports default RoPE or linear full-attention RoPE scaling only."
                )
        if int(sliding_window_pattern) < 1:
            raise ValueError("sliding_window_pattern must be >= 1")

        self.head_dim = int(head_dim)
        self.hidden_activation = str(hidden_activation if hidden_act is None else hidden_act)
        self.rope_local_base_freq = float(rope_local_base_freq)
        self.rope_scaling = rope_scaling
        self.query_pre_attn_scalar = float(query_pre_attn_scalar)
        self.sliding_window_pattern = int(sliding_window_pattern)
        self.final_logit_softcapping = final_logit_softcapping
        self.attn_logit_softcapping = attn_logit_softcapping
        self.use_bidirectional_attention = bool(use_bidirectional_attention)
        # Keep the same per-layer convention as modern Transformers: five local
        # layers followed by one global layer when pattern=6.
        self.layer_types = [
            "sliding_attention" if (i + 1) % self.sliding_window_pattern else "full_attention"
            for i in range(int(num_hidden_layers))
        ]

        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=self.hidden_activation,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            attention_dropout=attention_dropout,
            attention_bias=attention_bias,
            sliding_window=sliding_window,
            **kwargs,
        )


class Gemma3RMSNorm(nn.Module):
    """Gemma RMSNorm, whose checkpoint weight is an offset from one."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = float(eps)
        # Some replay helpers / kernels look for the Qwen-style attribute.
        self.variance_epsilon = self.eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        x_normed = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_normed * (1.0 + self.weight.float())).to(input_dtype)


class Gemma3ScaledWordEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int], embed_scale: float) -> None:
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.scalar_embed_scale = float(embed_scale)
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


class Gemma3MLP(nn.Module):
    def __init__(self, config: Gemma3TextConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        if config.hidden_activation not in {"gelu_pytorch_tanh", "gelu_new", "gelu"}:
            raise ValueError(f"Unsupported Gemma3 activation: {config.hidden_activation}")
        self.hidden_activation = config.hidden_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        if self.hidden_activation in {"gelu_pytorch_tanh", "gelu_new"}:
            gate = F.gelu(gate, approximate="tanh")
        else:
            gate = F.gelu(gate)
        return self.down_proj(gate * self.up_proj(x))


class Gemma3RotaryEmbedding(Qwen2RotaryEmbedding):
    def __init__(self, config: Gemma3TextConfig, *, layer_type: str) -> None:
        if layer_type == "sliding_attention":
            base = config.rope_local_base_freq
        elif layer_type == "full_attention":
            base = config.rope_theta
        else:
            raise ValueError(f"Unknown Gemma3 attention layer type: {layer_type}")
        super().__init__(
            dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=float(base),
        )
        # Modern Gemma 3 applies serialized ``rope_scaling`` only to global
        # (full-attention) layers.  Linear scaling is equivalent to dividing
        # inverse frequencies by the configured factor; local/sliding layers
        # keep the unscaled local RoPE base.
        if layer_type == "full_attention" and config.rope_scaling not in (None, {}):
            factor = float(config.rope_scaling["factor"])
            self.inv_freq = self.inv_freq / factor
        self.layer_type = layer_type


class Gemma3Attention(nn.Module):
    def __init__(self, config: Gemma3TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)
        self.layer_type = config.layer_types[layer_idx]
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = int(config.head_dim)
        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.q_norm = Gemma3RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3RotaryEmbedding(config, layer_type=self.layer_type)
        self.scaling = config.query_pre_attn_scalar ** -0.5
        self.attention_dropout = config.attention_dropout
        self.attn_logit_softcapping = config.attn_logit_softcapping
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None
        self.is_sliding = self.sliding_window is not None
        self.is_causal = True

    def _apply_sliding_window_mask(
        self,
        attn_weights: torch.Tensor,
        *,
        q_len: int,
        kv_len: int,
    ) -> torch.Tensor:
        if self.sliding_window is None or int(self.sliding_window) <= 0:
            return attn_weights
        # Queries occupy the final q_len positions after concatenating any
        # cached prefix. Gemma's window is exclusive: distance < window.
        query_positions = torch.arange(kv_len - q_len, kv_len, device=attn_weights.device)[:, None]
        key_positions = torch.arange(kv_len, device=attn_weights.device)[None, :]
        outside_window = key_positions <= (query_positions - int(self.sliding_window))
        return attn_weights.masked_fill(
            outside_window.view(1, 1, q_len, kv_len),
            torch.finfo(attn_weights.dtype).min,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        query_states = self.q_norm(query_states).transpose(1, 2)
        key_states = self.k_norm(key_states).transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids=position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat((past_key, key_states), dim=2)
            value_states = torch.cat((past_value, value_states), dim=2)
        present_key_value = (key_states, value_states) if use_cache else None

        expanded_key = repeat_kv(key_states, self.num_key_value_groups)
        expanded_value = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, expanded_key.transpose(2, 3)) * self.scaling
        attn_weights = self._apply_sliding_window_mask(
            attn_weights, q_len=q_len, kv_len=expanded_key.shape[-2]
        )
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : expanded_key.shape[-2]]
        if self.attn_logit_softcapping is not None:
            softcap = float(self.attn_logit_softcapping)
            attn_weights = torch.tanh(attn_weights / softcap) * softcap

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, expanded_value)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(
            bsz, q_len, self.num_heads * self.head_dim
        )
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights if output_attentions else None, present_key_value


class Gemma3DecoderLayer(nn.Module):
    def __init__(self, config: Gemma3TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)
        self.hidden_size = config.hidden_size
        self.self_attn = Gemma3Attention(config, layer_idx)
        self.mlp = Gemma3MLP(config)
        self.input_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights, present_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


def _gemma3_init_weights(self, module: nn.Module) -> None:
    # Match Gemma's zero-centered RMSNorm parameterization. Linear and
    # embedding initialization only matters before checkpoint weights load.
    if isinstance(module, Gemma3RMSNorm):
        module.weight.data.zero_()
        return
    Qwen2PreTrainedModel._init_weights(self, module)


class Gemma3TextModel(Qwen2Model):
    config_class = Gemma3TextConfig
    _no_split_modules = ["Gemma3DecoderLayer"]

    _init_weights = _gemma3_init_weights

    def __init__(self, config: Gemma3TextConfig) -> None:
        Qwen2PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = Gemma3ScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            embed_scale=config.hidden_size ** 0.5,
        )
        self.layers = nn.ModuleList(
            [Gemma3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()


class Gemma3ForCausalLM(Qwen2ForCausalLM):
    config_class = Gemma3TextConfig
    _no_split_modules = ["Gemma3DecoderLayer"]
    _tied_weights_keys = ["lm_head.weight"]

    _init_weights = _gemma3_init_weights

    def __init__(self, config: Gemma3TextConfig) -> None:
        Qwen2PreTrainedModel.__init__(self, config)
        self.model = Gemma3TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

class Gemma3MultimodalConfig(PretrainedConfig):
    """Minimal outer config used only to stream ``language_model.*`` weights."""

    model_type = "gemma3"

    def __init__(self, text_config: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        if isinstance(text_config, Gemma3TextConfig):
            self.text_config = text_config
        else:
            self.text_config = Gemma3TextConfig(**(text_config or {}))
        kwargs.setdefault("tie_word_embeddings", self.text_config.tie_word_embeddings)
        super().__init__(**kwargs)


class Gemma3TextOnlyFromMultimodal(Qwen2PreTrainedModel):
    """Checkpoint-loading shell matching Gemma 3 multimodal weight prefixes.

    The multimodal checkpoints omit ``language_model.lm_head.weight`` because
    Gemma 3 ties it to ``language_model.model.embed_tokens.weight``.  Exposing
    the nested input/output embeddings here is essential: Transformers 4.35.2
    calls ``tie_weights()`` on the *outer* object during ``from_pretrained``.
    Without these delegates the inner LM head stays randomly initialized.
    """

    config_class = Gemma3MultimodalConfig
    _no_split_modules = ["Gemma3DecoderLayer"]

    def __init__(self, config: Gemma3MultimodalConfig) -> None:
        super().__init__(config)
        self.language_model = Gemma3ForCausalLM(config.text_config)
        tied_keys = getattr(self.language_model, "_tied_weights_keys", None)
        if tied_keys is not None:
            self._tied_weights_keys = [f"language_model.{key}" for key in tied_keys]

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, value):
        self.language_model.set_output_embeddings(value)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)


def _gemma3_text_config_from_outer(config_dict: Dict[str, Any]) -> Gemma3TextConfig:
    # Gemma3Config constructs its nested Gemma3TextConfig directly from
    # ``text_config``.  In particular, the multimodal outer eos_token_id=106
    # must not replace the text model's eos_token_id=1; generation_config.json
    # carries the valid [1, 106] generation terminators.
    return Gemma3TextConfig(**dict(config_dict.get("text_config") or {}))


def load_gemma3_compat_model(model_id: str, *, torch_dtype: torch.dtype):
    """Load Gemma 3 text decoding weights on transformers==4.35.2.

    ``google/gemma-3-1b-it`` is a standalone ``gemma3_text`` checkpoint.
    Larger instruction checkpoints such as ``google/gemma-3-4b-it`` are
    multimodal ``gemma3`` checkpoints whose causal LM lives under
    ``language_model.*``.  MoT is text-only, so the latter are loaded through a
    lightweight shell that matches that prefix and omits vision modules.
    """
    config_path = cached_file(model_id, "config.json")
    if config_path is None:
        raise FileNotFoundError(f"config.json was not found for {model_id}")
    with open(config_path, "r", encoding="utf-8") as handle:
        config_dict = json.load(handle)

    model_type = str(config_dict.get("model_type", "")).lower()
    if model_type == "gemma3_text":
        config = Gemma3TextConfig(**config_dict)
        return Gemma3ForCausalLM.from_pretrained(
            model_id,
            config=config,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

    if model_type != "gemma3" or not isinstance(config_dict.get("text_config"), dict):
        raise ValueError(
            f"{model_id} is not a supported Gemma 3 text or multimodal checkpoint."
        )

    text_config = _gemma3_text_config_from_outer(config_dict)
    outer_kwargs = {key: value for key, value in config_dict.items() if key != "text_config"}
    outer_config = Gemma3MultimodalConfig(text_config=text_config, **outer_kwargs)
    shell = Gemma3TextOnlyFromMultimodal.from_pretrained(
        model_id,
        config=outer_config,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    language_model = shell.language_model
    if language_model.config.tie_word_embeddings:
        # ``from_pretrained`` should already have tied these through the outer
        # delegates above.  Re-apply and verify so a future loader change cannot
        # silently leave Gemma's output head randomly initialized again.
        language_model.tie_weights()
        input_weight = language_model.get_input_embeddings().weight
        output_weight = language_model.get_output_embeddings().weight
        if input_weight.data_ptr() != output_weight.data_ptr():
            raise RuntimeError("Gemma 3 input embeddings and LM head were not tied after checkpoint loading.")

    # Preserve generation metadata loaded from the outer repository (notably
    # Gemma 3's [<eos>, <end_of_turn>] EOS list) on the returned causal LM.
    try:
        language_model.generation_config = GenerationConfig.from_pretrained(model_id)
    except (OSError, ValueError, TypeError):
        if hasattr(shell, "generation_config"):
            language_model.generation_config = shell.generation_config
    return language_model
