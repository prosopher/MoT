"""Subset of Interlat's hidden-state modules.

This vendorized file keeps the hidden-state processor used by our local
Interlat implementation. The structure mirrors the upstream
`core_training/hidden_model/custom_model.py` components that stabilize and
project communicated hidden states before they are inserted into the receiver.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class AdaptiveProjection(nn.Module):
    """Adaptive numerical range projection layer from the upstream project."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.2))
        self.output_scale = nn.Parameter(torch.tensor(0.1))
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.proj[0].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.proj[0].bias)
        nn.init.xavier_uniform_(self.proj[3].weight, gain=1e-2)
        nn.init.zeros_(self.proj[3].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x * self.scale
        projected = self.proj(residual)
        return (residual + projected) * self.output_scale


class HiddenStateProcessor(nn.Module):
    """Compact processor derived from Interlat's ModelWithInsertedHiddenState."""

    def __init__(
        self,
        hidden_size: int,
        *,
        num_heads: int = 8,
        input_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.input_projector = None
        if input_dim is not None and input_dim != hidden_size:
            self.input_projector = nn.Linear(input_dim, hidden_size, bias=True)

        self.hidden_mha = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1,
        )
        self.pre_ln = nn.LayerNorm(hidden_size, eps=1e-6)
        self.post_ln = nn.LayerNorm(hidden_size, eps=1e-6)
        self.adaptive_proj = AdaptiveProjection(hidden_size)
        self._init_mha_weights()

    def _init_mha_weights(self) -> None:
        registered_params = self.hidden_mha._parameters.keys()
        for param_name in ("q_proj_weight", "k_proj_weight", "v_proj_weight"):
            if param_name in registered_params:
                param = getattr(self.hidden_mha, param_name)
                if param is not None:
                    nn.init.xavier_uniform_(param, gain=1.0 / math.sqrt(3))
        if "in_proj_weight" in registered_params:
            param = self.hidden_mha._parameters["in_proj_weight"]
            if param is not None:
                nn.init.xavier_uniform_(param, gain=1.0 / math.sqrt(3))
        for bias_name in ("q_proj_bias", "k_proj_bias", "v_proj_bias", "in_proj_bias"):
            if bias_name in registered_params:
                param = getattr(self.hidden_mha, bias_name)
                if param is not None:
                    nn.init.constant_(param, 0.0)
        if hasattr(self.hidden_mha, "out_proj"):
            nn.init.xavier_uniform_(self.hidden_mha.out_proj.weight, gain=1.0)
            if self.hidden_mha.out_proj.bias is not None:
                nn.init.constant_(self.hidden_mha.out_proj.bias, 0.0)

    def process_hidden_states(self, x: torch.Tensor) -> torch.Tensor:
        dev = self.pre_ln.weight.device
        dtyp = self.pre_ln.weight.dtype
        x = x.to(device=dev, dtype=dtyp, non_blocking=True).contiguous()
        if self.input_projector is not None:
            x = self.input_projector(x)
        normed = self.pre_ln(x).contiguous()

        weight_dtype = None
        if isinstance(getattr(self.hidden_mha, "in_proj_weight", None), torch.Tensor):
            weight_dtype = self.hidden_mha.in_proj_weight.dtype
        elif isinstance(getattr(self.hidden_mha.out_proj, "weight", None), torch.Tensor):
            weight_dtype = self.hidden_mha.out_proj.weight.dtype
        if weight_dtype is None:
            weight_dtype = dtyp

        autocast_enabled = torch.cuda.is_available() and dev.type == "cuda"
        with torch.cuda.amp.autocast(enabled=False if autocast_enabled else False):
            q = normed.to(dtype=weight_dtype).contiguous()
            k = normed.to(dtype=weight_dtype).contiguous()
            v = normed.to(dtype=weight_dtype).contiguous()
            attn_out, _ = self.hidden_mha(q, k, v, need_weights=False)
        attn_out = attn_out.to(dtyp)
        out = self.post_ln(normed + attn_out)
        return self.adaptive_proj(out)

    def forward(self, input_tensors: torch.Tensor) -> torch.Tensor:
        processed = self.process_hidden_states(input_tensors)
        return torch.clamp(processed, -10.0, 10.0)
