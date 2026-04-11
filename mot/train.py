from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from train_util import *


@dataclass(frozen=True)
class LayerMapping:
    reference_target_node_id: str
    reference_target_num_layers: int
    src_layer_start_idx: int
    src_layer_end_idx: int
    dst_layer_start_idx: int
    dst_layer_end_idx: int
    translated_num_layers: int
    src_num_layers: int
    dst_num_layers: int


@dataclass
class TrainConfig:
    alg: str
    timestamp: Optional[str]
    output_path: Optional[str]

    model_ids: str
    model_directions: str
    max_steps: int
    batch_size: int
    grad_accum_steps: int
    total_tokens: int
    prefix_tokens: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    grad_clip_norm: float
    log_every: int
    seed: int
    shuffle_buffer: int
    injection_layer_start_idx: int
    injection_window_size: int
    translator_dim: int
    translator_heads: int
    translator_depth: int
    translator_mlp_ratio: int
    device: str
    dtype: str
    translator: str
    mot_num_translators: int
    mot_top_k: int
    orthogonal_damping_rank: int
    orthogonal_damping_min_eta: float
    orthogonal_damping_max_eta: float
    orthogonal_damping_max_start_fraction: float

    def __post_init__(self) -> None:
        self.device = resolve_device(self.device)
        parse_model_ids_csv(self.model_ids)
        if self.injection_layer_start_idx < 0:
            raise ValueError("injection_layer_start_idx must be >= 0")
        if self.injection_window_size < 1:
            raise ValueError("injection_window_size must be >= 1")
        if self.translator_dim % self.translator_heads != 0:
            raise ValueError("translator_dim must be divisible by translator_heads")
        if self.translator not in {"single", "mot", "mot-r", "mot-rod"}:
            raise ValueError("translator must be one of {'single', 'mot', 'mot-r', 'mot-rod'}")
        if self.mot_num_translators < 1:
            raise ValueError("mot_num_translators must be >= 1")
        if self.mot_top_k < 1:
            raise ValueError("mot_top_k must be >= 1")
        if self.mot_top_k > self.mot_num_translators:
            raise ValueError("mot_top_k must be <= mot_num_translators")
        if self.orthogonal_damping_rank < 0:
            raise ValueError("orthogonal_damping_rank must be >= 0")
        if not (0.0 <= self.orthogonal_damping_min_eta <= 1.0):
            raise ValueError("orthogonal_damping_min_eta must be in [0, 1]")
        if not (0.0 <= self.orthogonal_damping_max_eta <= 1.0):
            raise ValueError("orthogonal_damping_max_eta must be in [0, 1]")
        if self.orthogonal_damping_min_eta > self.orthogonal_damping_max_eta:
            raise ValueError("orthogonal_damping_min_eta must be <= orthogonal_damping_max_eta")
        if not (0.0 <= self.orthogonal_damping_max_start_fraction <= 1.0):
            raise ValueError("orthogonal_damping_max_start_fraction must be in [0, 1]")
        initialize_train_output_paths(self)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 2) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, hidden: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.query_norm(hidden)
        kv = self.context_norm(context)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        hidden = hidden + attn_out
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        return hidden


class CrossLayerWindowTranslator(nn.Module):
    """
    Recurrent cross-attention translator following the LSC translator pattern.

    Input:  [batch, seq, num_layers, src_hidden]
    Output: [batch, seq, num_layers, dst_hidden]
    """

    def __init__(
        self,
        src_hidden_size: int,
        dst_hidden_size: int,
        num_layers: int,
        translator_dim: int,
        translator_heads: int,
        translator_depth: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if translator_depth < 1:
            raise ValueError("translator_depth must be >= 1")
        self.num_layers = num_layers
        self.translator_depth = translator_depth
        self.input_norm = nn.LayerNorm(src_hidden_size)
        self.input_proj = nn.Linear(src_hidden_size, translator_dim)
        self.recurrent_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        CrossAttentionBlock(
                            dim=translator_dim,
                            num_heads=translator_heads,
                            mlp_ratio=mlp_ratio,
                        )
                        for _ in range(num_layers)
                    ]
                )
                for _ in range(translator_depth)
            ]
        )
        self.output_norm = nn.LayerNorm(num_layers * translator_dim)
        self.output_proj = nn.Linear(num_layers * translator_dim, num_layers * dst_hidden_size)

    def forward(self, layer_window_cache: torch.Tensor) -> torch.Tensor:
        if layer_window_cache.ndim != 4:
            raise ValueError(
                "CrossLayerWindowTranslator expects [batch, seq, num_layers, hidden], "
                f"got {tuple(layer_window_cache.shape)}"
            )
        if layer_window_cache.shape[2] != self.num_layers:
            raise ValueError(
                f"CrossLayerWindowTranslator expected {self.num_layers} layers, got {layer_window_cache.shape[2]}"
            )

        batch_size, seq_len, _, _ = layer_window_cache.shape
        projected = F.gelu(self.input_proj(self.input_norm(layer_window_cache)))

        hidden = projected[:, :, 0, :]
        collected = []
        for stage_blocks in self.recurrent_blocks:
            stage_hidden = hidden
            stage_collected = []
            for layer_idx, block in enumerate(stage_blocks):
                stage_hidden = block(stage_hidden, projected[:, :, layer_idx, :])
                stage_collected.append(stage_hidden)
            hidden = stage_hidden
            collected = stage_collected

        fused = torch.cat(collected, dim=-1)
        translated = F.gelu(self.output_proj(self.output_norm(fused)))
        return translated.view(batch_size, seq_len, self.num_layers, -1)


class MixtureOfTranslators(nn.Module):
    def __init__(
        self,
        src_hidden_size: int,
        dst_hidden_size: int,
        num_layers: int,
        translator_dim: int,
        translator_heads: int,
        translator_depth: int,
        mlp_ratio: int,
        num_translators: int,
        top_k: int,
    ) -> None:
        super().__init__()
        if num_translators < 1:
            raise ValueError("num_translators must be >= 1")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if top_k > num_translators:
            raise ValueError("top_k must be <= num_translators")
        self.num_translators = num_translators
        self.top_k = top_k
        self.translators = nn.ModuleList(
            [
                CrossLayerWindowTranslator(
                    src_hidden_size=src_hidden_size,
                    dst_hidden_size=dst_hidden_size,
                    num_layers=num_layers,
                    translator_dim=translator_dim,
                    translator_heads=translator_heads,
                    translator_depth=translator_depth,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(num_translators)
            ]
        )
        router_input_dim = num_layers * src_hidden_size
        router_hidden_dim = max(64, min(translator_dim, router_input_dim))
        self.router = nn.Sequential(
            nn.LayerNorm(router_input_dim),
            nn.Linear(router_input_dim, router_hidden_dim),
            nn.GELU(),
            nn.Linear(router_hidden_dim, num_translators),
        )

    def _compute_mixture_weights(self, layer_window_cache: torch.Tensor) -> torch.Tensor:
        router_input = layer_window_cache.reshape(layer_window_cache.shape[0], layer_window_cache.shape[1], -1)
        router_logits = self.router(router_input)
        if self.top_k < self.num_translators:
            topk_indices = torch.topk(router_logits, k=self.top_k, dim=-1).indices
            topk_mask = torch.zeros_like(router_logits, dtype=torch.bool)
            topk_mask.scatter_(-1, topk_indices, True)
            router_logits = router_logits.masked_fill(~topk_mask, float("-inf"))
        return torch.softmax(router_logits, dim=-1)

    def forward(self, layer_window_cache: torch.Tensor) -> torch.Tensor:
        expert_outputs = [translator(layer_window_cache) for translator in self.translators]
        if len(expert_outputs) == 1:
            return expert_outputs[0]
        mixture_weights = self._compute_mixture_weights(layer_window_cache)
        stacked_outputs = torch.stack(expert_outputs, dim=2)
        return (stacked_outputs * mixture_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)


class ResidualTranslatorExpert(nn.Module):
    def __init__(
        self,
        src_hidden_size: int,
        dst_hidden_size: int,
        num_layers: int,
        translator_dim: int,
        translator_heads: int,
        translator_depth: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        self.translator = CrossLayerWindowTranslator(
            src_hidden_size=src_hidden_size,
            dst_hidden_size=dst_hidden_size,
            num_layers=num_layers,
            translator_dim=translator_dim,
            translator_heads=translator_heads,
            translator_depth=translator_depth,
            mlp_ratio=mlp_ratio,
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, layer_window_cache: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.residual_scale) * self.translator(layer_window_cache)


class ResidualMixtureOfTranslators(nn.Module):
    def __init__(
        self,
        src_hidden_size: int,
        dst_hidden_size: int,
        num_layers: int,
        translator_dim: int,
        translator_heads: int,
        translator_depth: int,
        mlp_ratio: int,
        num_translators: int,
        top_k: int,
        orthogonal_damping_rank: int = 0,
        orthogonal_damping_min_eta: float = 1.0,
        orthogonal_damping_max_eta: float = 1.0,
        orthogonal_damping_max_start_fraction: float = 0.3,
        dst_start_layer_idx: int = 0,
        dst_total_num_layers: int = 1,
    ) -> None:
        super().__init__()
        if num_translators < 1:
            raise ValueError("num_translators must be >= 1")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if top_k > num_translators:
            raise ValueError("top_k must be <= num_translators")
        self.num_translators = num_translators
        self.top_k = top_k
        self.num_layers = num_layers
        self.dst_hidden_size = dst_hidden_size
        self.orthogonal_damping_rank = orthogonal_damping_rank
        start_fraction = 0.0 if dst_total_num_layers <= 0 else float(dst_start_layer_idx) / float(dst_total_num_layers)
        self.orthogonal_damping_enabled = (
            orthogonal_damping_rank > 0
            and (orthogonal_damping_min_eta < 1.0 or orthogonal_damping_max_eta < 1.0)
            and start_fraction <= float(orthogonal_damping_max_start_fraction)
        )
        if not self.orthogonal_damping_enabled:
            layer_eta = torch.ones((num_layers,), dtype=torch.float32)
        elif dst_total_num_layers <= 1:
            layer_eta = torch.full((num_layers,), float(orthogonal_damping_max_eta), dtype=torch.float32)
        else:
            eta_values = []
            for local_layer_idx in range(num_layers):
                global_layer_idx = dst_start_layer_idx + local_layer_idx
                progress = float(global_layer_idx) / float(dst_total_num_layers - 1)
                eta_values.append(
                    float(orthogonal_damping_min_eta)
                    + (float(orthogonal_damping_max_eta) - float(orthogonal_damping_min_eta)) * progress
                )
            layer_eta = torch.tensor(eta_values, dtype=torch.float32)
        self.register_buffer("orthogonal_damping_eta", layer_eta, persistent=False)
        self.base_translator = CrossLayerWindowTranslator(
            src_hidden_size=src_hidden_size,
            dst_hidden_size=dst_hidden_size,
            num_layers=num_layers,
            translator_dim=translator_dim,
            translator_heads=translator_heads,
            translator_depth=translator_depth,
            mlp_ratio=mlp_ratio,
        )
        self.residual_experts = nn.ModuleList(
            [
                ResidualTranslatorExpert(
                    src_hidden_size=src_hidden_size,
                    dst_hidden_size=dst_hidden_size,
                    num_layers=num_layers,
                    translator_dim=translator_dim,
                    translator_heads=translator_heads,
                    translator_depth=translator_depth,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(max(0, num_translators - 1))
            ]
        )
        router_input_dim = num_layers * src_hidden_size
        router_hidden_dim = max(64, min(translator_dim, router_input_dim))
        self.router = None
        if len(self.residual_experts) > 0:
            self.router = nn.Sequential(
                nn.LayerNorm(router_input_dim),
                nn.Linear(router_input_dim, router_hidden_dim),
                nn.GELU(),
                nn.Linear(router_hidden_dim, len(self.residual_experts)),
            )
            final_linear = self.router[-1]
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)

    def _compute_mixture_weights(self, layer_window_cache: torch.Tensor) -> Optional[torch.Tensor]:
        if self.router is None:
            return None
        router_input = layer_window_cache.reshape(layer_window_cache.shape[0], layer_window_cache.shape[1], -1)
        router_logits = self.router(router_input)
        if (not self.training) and self.top_k < len(self.residual_experts):
            topk_indices = torch.topk(router_logits, k=self.top_k, dim=-1).indices
            topk_mask = torch.zeros_like(router_logits, dtype=torch.bool)
            topk_mask.scatter_(-1, topk_indices, True)
            router_logits = router_logits.masked_fill(~topk_mask, float("-inf"))
        return torch.softmax(router_logits, dim=-1)

    def _compute_principal_basis(self, features: torch.Tensor, rank: int) -> Optional[torch.Tensor]:
        if rank <= 0:
            return None
        flattened = features.reshape(-1, features.shape[-1]).detach().to(dtype=torch.float32)
        if flattened.shape[0] < 2 or flattened.shape[1] < 1:
            return None
        q = min(rank, flattened.shape[0], flattened.shape[1])
        if q < 1:
            return None
        centered = flattened - flattened.mean(dim=0, keepdim=True)
        if torch.count_nonzero(centered).item() == 0:
            return None
        with torch.no_grad():
            try:
                _, _, basis = torch.pca_lowrank(centered, q=q, center=False)
            except RuntimeError:
                return None
        return basis[:, :q].to(device=features.device, dtype=features.dtype)

    def _apply_orthogonal_damping(self, base_output: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if not self.orthogonal_damping_enabled:
            return residual
        damped_layers: List[torch.Tensor] = []
        for local_layer_idx in range(self.num_layers):
            layer_eta = float(self.orthogonal_damping_eta[local_layer_idx].item())
            if layer_eta >= 1.0:
                damped_layers.append(residual[:, :, local_layer_idx, :])
                continue
            basis = self._compute_principal_basis(base_output[:, :, local_layer_idx, :], self.orthogonal_damping_rank)
            layer_residual = residual[:, :, local_layer_idx, :]
            if basis is None:
                damped_layers.append(layer_residual)
                continue
            coeff = torch.einsum('bsd,dq->bsq', layer_residual, basis)
            parallel = torch.einsum('bsq,dq->bsd', coeff, basis)
            orthogonal = layer_residual - parallel
            damped_layers.append(parallel + (layer_eta * orthogonal))
        return torch.stack(damped_layers, dim=2)

    def forward(self, layer_window_cache: torch.Tensor) -> torch.Tensor:
        base_output = self.base_translator(layer_window_cache)
        if len(self.residual_experts) == 0:
            return base_output
        mixture_weights = self._compute_mixture_weights(layer_window_cache)
        residual_outputs = [expert(layer_window_cache) for expert in self.residual_experts]
        stacked_outputs = torch.stack(residual_outputs, dim=2)
        residual = (stacked_outputs * mixture_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)
        residual = self._apply_orthogonal_damping(base_output, residual)
        return base_output + residual


def build_window_translator(
    *,
    translator: str,
    src_hidden_size: int,
    dst_hidden_size: int,
    num_layers: int,
    translator_dim: int,
    translator_heads: int,
    translator_depth: int,
    mlp_ratio: int,
    mot_num_translators: int,
    mot_top_k: int,
    orthogonal_damping_rank: int,
    orthogonal_damping_min_eta: float,
    orthogonal_damping_max_eta: float,
    orthogonal_damping_max_start_fraction: float,
    dst_start_layer_idx: int,
    dst_total_num_layers: int,
) -> nn.Module:
    if translator == "single":
        return CrossLayerWindowTranslator(
            src_hidden_size=src_hidden_size,
            dst_hidden_size=dst_hidden_size,
            num_layers=num_layers,
            translator_dim=translator_dim,
            translator_heads=translator_heads,
            translator_depth=translator_depth,
            mlp_ratio=mlp_ratio,
        )
    if translator == "mot":
        return MixtureOfTranslators(
            src_hidden_size=src_hidden_size,
            dst_hidden_size=dst_hidden_size,
            num_layers=num_layers,
            translator_dim=translator_dim,
            translator_heads=translator_heads,
            translator_depth=translator_depth,
            mlp_ratio=mlp_ratio,
            num_translators=mot_num_translators,
            top_k=mot_top_k,
        )
    if translator in {"mot-r", "mot-rod"}:
        damping_rank = orthogonal_damping_rank if translator == "mot-rod" else 0
        damping_min_eta = orthogonal_damping_min_eta if translator == "mot-rod" else 1.0
        damping_max_eta = orthogonal_damping_max_eta if translator == "mot-rod" else 1.0
        damping_max_start_fraction = (
            orthogonal_damping_max_start_fraction if translator == "mot-rod" else 0.0
        )
        return ResidualMixtureOfTranslators(
            src_hidden_size=src_hidden_size,
            dst_hidden_size=dst_hidden_size,
            num_layers=num_layers,
            translator_dim=translator_dim,
            translator_heads=translator_heads,
            translator_depth=translator_depth,
            mlp_ratio=mlp_ratio,
            num_translators=mot_num_translators,
            top_k=mot_top_k,
            orthogonal_damping_rank=damping_rank,
            orthogonal_damping_min_eta=damping_min_eta,
            orthogonal_damping_max_eta=damping_max_eta,
            orthogonal_damping_max_start_fraction=damping_max_start_fraction,
            dst_start_layer_idx=dst_start_layer_idx,
            dst_total_num_layers=dst_total_num_layers,
        )
    raise ValueError(f"Unsupported translator type: {translator}")


class LayerWindowDirectionalTranslator(nn.Module):
    def __init__(
        self,
        src_hidden_size: int,
        dst_hidden_size: int,
        translated_num_layers: int,
        translator_dim: int,
        translator_heads: int,
        translator_depth: int,
        mlp_ratio: int,
        translator: str,
        mot_num_translators: int,
        mot_top_k: int,
        orthogonal_damping_rank: int,
        orthogonal_damping_min_eta: float,
        orthogonal_damping_max_eta: float,
        orthogonal_damping_max_start_fraction: float,
        dst_start_layer_idx: int,
        dst_total_num_layers: int,
    ) -> None:
        super().__init__()
        if translated_num_layers < 1:
            raise ValueError("translated_num_layers must be >= 1")
        self.translated_num_layers = translated_num_layers
        self.translator = translator
        self.key_translator = build_window_translator(
            translator=translator,
            src_hidden_size=src_hidden_size,
            dst_hidden_size=dst_hidden_size,
            num_layers=translated_num_layers,
            translator_dim=translator_dim,
            translator_heads=translator_heads,
            translator_depth=translator_depth,
            mlp_ratio=mlp_ratio,
            mot_num_translators=mot_num_translators,
            mot_top_k=mot_top_k,
            orthogonal_damping_rank=orthogonal_damping_rank,
            orthogonal_damping_min_eta=orthogonal_damping_min_eta,
            orthogonal_damping_max_eta=orthogonal_damping_max_eta,
            orthogonal_damping_max_start_fraction=orthogonal_damping_max_start_fraction,
            dst_start_layer_idx=dst_start_layer_idx,
            dst_total_num_layers=dst_total_num_layers,
        )
        self.value_translator = build_window_translator(
            translator=translator,
            src_hidden_size=src_hidden_size,
            dst_hidden_size=dst_hidden_size,
            num_layers=translated_num_layers,
            translator_dim=translator_dim,
            translator_heads=translator_heads,
            translator_depth=translator_depth,
            mlp_ratio=mlp_ratio,
            mot_num_translators=mot_num_translators,
            mot_top_k=mot_top_k,
            orthogonal_damping_rank=orthogonal_damping_rank,
            orthogonal_damping_min_eta=orthogonal_damping_min_eta,
            orthogonal_damping_max_eta=orthogonal_damping_max_eta,
            orthogonal_damping_max_start_fraction=orthogonal_damping_max_start_fraction,
            dst_start_layer_idx=dst_start_layer_idx,
            dst_total_num_layers=dst_total_num_layers,
        )

    def forward(self, key_block: torch.Tensor, value_block: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if key_block.shape != value_block.shape:
            raise ValueError(
                "Layer-window key/value shapes must match, "
                f"got {tuple(key_block.shape)} vs {tuple(value_block.shape)}"
            )
        if key_block.ndim != 4:
            raise ValueError(
                "Layer-window tensors must have shape [batch, seq, num_layers, hidden], "
                f"got {tuple(key_block.shape)}"
            )
        if key_block.shape[2] != self.translated_num_layers:
            raise ValueError(
                f"Expected {self.translated_num_layers} layers in the translation window, got {key_block.shape[2]}"
            )
        translated_key = self.key_translator(key_block)
        translated_value = self.value_translator(value_block)
        return translated_key, translated_value


class LayerWindowTranslatorPool(nn.Module):
    def __init__(
        self,
        model_specs: Dict[str, ModelSpec],
        edges: List[Edge],
        layer_mappings: Dict[str, LayerMapping],
        translator_dim: int,
        translator_heads: int,
        translator_depth: int,
        mlp_ratio: int,
        active_directions: List[str],
        translator: str,
        mot_num_translators: int,
        mot_top_k: int,
        orthogonal_damping_rank: int,
        orthogonal_damping_min_eta: float,
        orthogonal_damping_max_eta: float,
        orthogonal_damping_max_start_fraction: float,
    ) -> None:
        super().__init__()
        if not active_directions:
            raise ValueError("active_directions must contain at least one direction.")

        self.model_specs = model_specs
        self.layer_mappings = layer_mappings
        self.active_directions = tuple(active_directions)
        self.edges_by_id = build_edge_map(edges)

        adapters = {}
        for direction in self.active_directions:
            if direction not in self.edges_by_id:
                raise ValueError(f"Unknown direction: {direction}")
            mapping = self.layer_mappings[direction]
            edge = self.edges_by_id[direction]
            adapters[direction] = LayerWindowDirectionalTranslator(
                src_hidden_size=model_specs[edge.src_id].hidden_size,
                dst_hidden_size=model_specs[edge.dst_id].hidden_size,
                translated_num_layers=mapping.translated_num_layers,
                translator_dim=translator_dim,
                translator_heads=translator_heads,
                translator_depth=translator_depth,
                mlp_ratio=mlp_ratio,
                translator=translator,
                mot_num_translators=mot_num_translators,
                mot_top_k=mot_top_k,
                orthogonal_damping_rank=orthogonal_damping_rank,
                orthogonal_damping_min_eta=orthogonal_damping_min_eta,
                orthogonal_damping_max_eta=orthogonal_damping_max_eta,
                orthogonal_damping_max_start_fraction=orthogonal_damping_max_start_fraction,
                dst_start_layer_idx=mapping.dst_layer_start_idx,
                dst_total_num_layers=mapping.dst_num_layers,
            )
        self.adapters = nn.ModuleDict(adapters)

    def translate_layer_window(
        self,
        past_key_values: PastKeyValues,
        src_name: str,
        dst_name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, LayerMapping]:
        direction = f"{src_name}_to_{dst_name}"
        if direction not in self.adapters:
            raise ValueError(
                f"Translator direction {direction} is not available. "
                f"Active directions: {list(self.active_directions)}"
            )
        mapping = self.layer_mappings[direction]
        key_block, value_block = extract_layer_window_blocks(
            past_key_values=past_key_values,
            start_layer_idx=mapping.src_layer_start_idx,
            num_layers=mapping.translated_num_layers,
        )
        translated_key, translated_value = self.adapters[direction](key_block, value_block)
        return translated_key, translated_value, mapping

    def build_replayed_target_past(
        self,
        *,
        source_past_key_values: PastKeyValues,
        prefix_input_ids: torch.Tensor,
        target_model: PreTrainedModel,
        src_name: str,
        dst_name: str,
        dst_spec: ModelSpec,
    ) -> Tuple[PastKeyValues, PastKeyValues, LayerMapping]:
        translated_key, translated_value, mapping = self.translate_layer_window(
            past_key_values=source_past_key_values,
            src_name=src_name,
            dst_name=dst_name,
        )
        translated_window_past = blocks_to_partial_past_key_values(
            key_block=translated_key,
            value_block=translated_value,
            num_heads=dst_spec.num_heads,
            head_dim=dst_spec.head_dim,
        )
        mixed_target_past = replay_target_prefill_with_injected_window(
            target_model=target_model,
            prefix_input_ids=prefix_input_ids,
            target_start_layer_idx=mapping.dst_layer_start_idx,
            injected_key_block=translated_key,
            injected_value_block=translated_value,
            dst_spec=dst_spec,
        )
        return mixed_target_past, translated_window_past, mapping


class SimpleNamespaceConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def resolve_direction_metadata(
    model_ids: str,
    model_directions: str,
) -> Tuple[List[Node], List[Edge], List[str], Edge]:
    nodes, edges = build_nodes_and_edges(model_ids, model_directions)
    active_directions = [edge.id for edge in edges]
    if not active_directions:
        raise ValueError("No active directions were resolved from model_directions")
    edge_map = build_edge_map(edges)
    return nodes, edges, active_directions, edge_map[active_directions[0]]


def build_layer_mappings(
    config: TrainConfig,
    model_specs: Dict[str, ModelSpec],
    edges: List[Edge],
    active_directions: List[str],
    reference_edge: Edge,
) -> Dict[str, LayerMapping]:
    reference_target_spec = model_specs[reference_edge.dst_id]
    requested_window_size = int(config.injection_window_size)
    injection_layer_start_idx = int(config.injection_layer_start_idx)
    injection_layer_end_idx = injection_layer_start_idx + requested_window_size - 1
    if injection_layer_end_idx >= reference_target_spec.num_layers:
        raise ValueError(
            "injection_layer_start_idx with the requested injection_window_size would exceed the reference target stack: "
            f"start={injection_layer_start_idx}, end={injection_layer_end_idx}, "
            f"last_layer={reference_target_spec.num_layers - 1}"
        )

    edge_map = build_edge_map(edges)
    mappings: Dict[str, LayerMapping] = {}
    for direction in active_directions:
        edge = edge_map[direction]
        src_spec = model_specs[edge.src_id]
        dst_spec = model_specs[edge.dst_id]

        dst_layer_start_idx = injection_layer_start_idx
        dst_layer_end_idx = dst_layer_start_idx + requested_window_size - 1
        if dst_layer_end_idx >= dst_spec.num_layers:
            raise ValueError(
                f"direction={direction} cannot use injection_layer_start_idx={dst_layer_start_idx} "
                f"with injection_window_size={requested_window_size}: target end layer {dst_layer_end_idx} exceeds "
                f"target last layer {dst_spec.num_layers - 1}"
            )

        dst_depth_from_top = dst_spec.num_layers - 1 - dst_layer_start_idx
        src_layer_start_idx = src_spec.num_layers - 1 - dst_depth_from_top
        if not (0 <= src_layer_start_idx < src_spec.num_layers):
            raise ValueError(
                f"direction={direction} cannot align source window to target top-depth {dst_depth_from_top}: "
                f"computed src_layer_start_idx={src_layer_start_idx} is outside [0, {src_spec.num_layers - 1}]"
            )

        src_layer_end_idx = src_layer_start_idx + requested_window_size - 1
        if src_layer_end_idx >= src_spec.num_layers:
            raise ValueError(
                f"direction={direction} cannot use injection_window_size={requested_window_size} after top-depth alignment: "
                f"source end layer {src_layer_end_idx} exceeds source last layer {src_spec.num_layers - 1}"
            )

        mappings[direction] = LayerMapping(
            reference_target_node_id=reference_edge.dst_id,
            reference_target_num_layers=reference_target_spec.num_layers,
            src_layer_start_idx=src_layer_start_idx,
            src_layer_end_idx=src_layer_end_idx,
            dst_layer_start_idx=dst_layer_start_idx,
            dst_layer_end_idx=dst_layer_end_idx,
            translated_num_layers=requested_window_size,
            src_num_layers=src_spec.num_layers,
            dst_num_layers=dst_spec.num_layers,
        )
    return mappings


def extract_layer_window_blocks(
    past_key_values: PastKeyValues,
    start_layer_idx: int,
    num_layers: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    end_layer_idx = start_layer_idx + num_layers
    if not (0 <= start_layer_idx < len(past_key_values)):
        raise ValueError(f"start_layer_idx={start_layer_idx} must be in [0, {len(past_key_values) - 1}]")
    if end_layer_idx > len(past_key_values):
        raise ValueError(
            f"Cannot extract layers [{start_layer_idx}, {end_layer_idx - 1}] from cache with {len(past_key_values)} layers"
        )
    return past_key_values_to_blocks(past_key_values[start_layer_idx:end_layer_idx])


def blocks_to_partial_past_key_values(
    key_block: torch.Tensor,
    value_block: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> PastKeyValues:
    batch_size, seq_len, num_layers, hidden_size = key_block.shape
    expected_hidden = num_heads * head_dim
    if hidden_size != expected_hidden:
        raise ValueError(f"Hidden mismatch: block has {hidden_size}, expected {expected_hidden}.")

    past_key_values = []
    for layer_idx in range(num_layers):
        key_layer = key_block[:, :, layer_idx, :]
        value_layer = value_block[:, :, layer_idx, :]
        key_layer = key_layer.view(batch_size, seq_len, num_heads, head_dim)
        value_layer = value_layer.view(batch_size, seq_len, num_heads, head_dim)
        key_layer = key_layer.permute(0, 2, 1, 3).contiguous()
        value_layer = value_layer.permute(0, 2, 1, 3).contiguous()
        past_key_values.append((key_layer, value_layer))
    return tuple(past_key_values)


def require_gpt2_transformer(model: PreTrainedModel):
    transformer = getattr(model, "transformer", None)
    if transformer is None or not hasattr(transformer, "h"):
        raise ValueError(
            "mot currently supports GPT-2 style decoder stacks only "
            "(expected model.transformer.h to exist)."
        )
    return transformer


def build_gpt2_input_hidden_states(model: PreTrainedModel, input_ids: torch.Tensor) -> torch.Tensor:
    transformer = require_gpt2_transformer(model)
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape [batch, seq], got {tuple(input_ids.shape)}")
    batch_size, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    hidden_states = transformer.wte(input_ids) + transformer.wpe(position_ids)
    drop = getattr(transformer, "drop", None)
    if drop is not None:
        hidden_states = drop(hidden_states)
    return hidden_states


def unpack_block_outputs(outputs: Any) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if isinstance(outputs, tuple):
        if len(outputs) < 2:
            raise ValueError("Expected GPT-2 block outputs to include present key/value cache.")
        hidden_states = outputs[0]
        present = outputs[1]
    else:
        hidden_states = getattr(outputs, "last_hidden_state", None)
        if hidden_states is None:
            hidden_states = getattr(outputs, "hidden_states", None)
        present = getattr(outputs, "past_key_value", None)
        if hidden_states is None or present is None:
            raise ValueError("Unsupported block output type for GPT-2 layer replay.")
    if not isinstance(present, tuple) or len(present) != 2:
        raise ValueError("Expected present cache to be a (key, value) tuple.")
    return hidden_states, present


def run_gpt2_block_with_cache(block: nn.Module, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    outputs = block(
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=True,
        output_attentions=False,
    )
    return unpack_block_outputs(outputs)


def run_gpt2_block_with_injected_layer(
    block: nn.Module,
    hidden_states: torch.Tensor,
    injected_key: torch.Tensor,
    injected_value: torch.Tensor,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if injected_key.shape != injected_value.shape:
        raise ValueError(
            "Injected key/value must have identical shapes, "
            f"got {tuple(injected_key.shape)} vs {tuple(injected_value.shape)}"
        )

    attn = block.attn
    residual = hidden_states
    attn_input = block.ln_1(hidden_states)

    qkv = attn.c_attn(attn_input)
    split_size = getattr(attn, "split_size", qkv.shape[-1] // 3)
    query, _, _ = qkv.split(split_size, dim=2)

    batch_size, seq_len, _ = query.shape
    num_heads = attn.num_heads
    head_dim = attn.head_dim
    expected_cache_shape = (batch_size, num_heads, seq_len, head_dim)
    if tuple(injected_key.shape) != expected_cache_shape:
        raise ValueError(
            "Injected cache shape mismatch for GPT-2 layer replay: "
            f"expected {expected_cache_shape}, got {tuple(injected_key.shape)}"
        )

    query = query.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()

    if getattr(attn, "reorder_and_upcast_attn", False) and hasattr(attn, "_upcast_and_reordered_attn"):
        attn_output, _ = attn._upcast_and_reordered_attn(
            query,
            injected_key,
            injected_value,
            attention_mask=None,
            head_mask=None,
        )
    else:
        attn_output, _ = attn._attn(
            query,
            injected_key,
            injected_value,
            attention_mask=None,
            head_mask=None,
        )

    attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, num_heads * head_dim)
    attn_output = attn.c_proj(attn_output)
    attn_output = attn.resid_dropout(attn_output)
    hidden_states = residual + attn_output

    residual = hidden_states
    hidden_states = hidden_states + block.mlp(block.ln_2(hidden_states))
    return hidden_states, (injected_key, injected_value)


def replay_target_prefill_with_injected_window(
    target_model: PreTrainedModel,
    prefix_input_ids: torch.Tensor,
    target_start_layer_idx: int,
    injected_key_block: torch.Tensor,
    injected_value_block: torch.Tensor,
    dst_spec: ModelSpec,
) -> PastKeyValues:
    injected_window = blocks_to_partial_past_key_values(
        key_block=injected_key_block,
        value_block=injected_value_block,
        num_heads=dst_spec.num_heads,
        head_dim=dst_spec.head_dim,
    )
    translated_num_layers = len(injected_window)

    transformer = require_gpt2_transformer(target_model)
    target_end_layer_idx = target_start_layer_idx + translated_num_layers - 1
    if not (0 <= target_start_layer_idx < len(transformer.h)):
        raise ValueError(f"target_start_layer_idx={target_start_layer_idx} must be in [0, {len(transformer.h) - 1}]")
    if target_end_layer_idx >= len(transformer.h):
        raise ValueError(
            f"Injected window ending at layer {target_end_layer_idx} exceeds target stack with {len(transformer.h)} layers"
        )

    rebuilt_past: List[Tuple[torch.Tensor, torch.Tensor]] = []

    if torch.is_grad_enabled():
        with torch.no_grad():
            hidden_states = build_gpt2_input_hidden_states(target_model, prefix_input_ids)
            for lower_idx in range(target_start_layer_idx):
                hidden_states, present = run_gpt2_block_with_cache(transformer.h[lower_idx], hidden_states)
                rebuilt_past.append((present[0].detach(), present[1].detach()))
        hidden_states = hidden_states.detach()
    else:
        hidden_states = build_gpt2_input_hidden_states(target_model, prefix_input_ids)
        for lower_idx in range(target_start_layer_idx):
            hidden_states, present = run_gpt2_block_with_cache(transformer.h[lower_idx], hidden_states)
            rebuilt_past.append(present)

    for offset, injected_present in enumerate(injected_window):
        layer_idx = target_start_layer_idx + offset
        hidden_states, present = run_gpt2_block_with_injected_layer(
            transformer.h[layer_idx],
            hidden_states,
            injected_present[0],
            injected_present[1],
        )
        rebuilt_past.append(present)

    for upper_idx in range(target_end_layer_idx + 1, len(transformer.h)):
        hidden_states, present = run_gpt2_block_with_cache(transformer.h[upper_idx], hidden_states)
        rebuilt_past.append(present)

    return tuple(rebuilt_past)


def build_translator_pool(
    models: Dict[str, PreTrainedModel],
    config: TrainConfig,
) -> Tuple[LayerWindowTranslatorPool, Dict[str, ModelSpec], List[Node], List[Edge], Dict[str, LayerMapping]]:
    nodes, edges, active_directions, reference_edge = resolve_direction_metadata(
        config.model_ids,
        config.model_directions,
    )
    model_specs = {
        node.id: get_model_spec(models[node.id])
        for node in nodes
    }
    layer_mappings = build_layer_mappings(config, model_specs, edges, active_directions, reference_edge)
    translator_pool = LayerWindowTranslatorPool(
        model_specs=model_specs,
        edges=edges,
        layer_mappings=layer_mappings,
        translator_dim=config.translator_dim,
        translator_heads=config.translator_heads,
        translator_depth=config.translator_depth,
        mlp_ratio=config.translator_mlp_ratio,
        active_directions=active_directions,
        translator=config.translator,
        mot_num_translators=config.mot_num_translators,
        mot_top_k=config.mot_top_k,
        orthogonal_damping_rank=config.orthogonal_damping_rank,
        orthogonal_damping_min_eta=config.orthogonal_damping_min_eta,
        orthogonal_damping_max_eta=config.orthogonal_damping_max_eta,
        orthogonal_damping_max_start_fraction=config.orthogonal_damping_max_start_fraction,
    )
    translator_pool.to(config.device)
    return translator_pool, model_specs, nodes, edges, layer_mappings


def load_translator_pool_from_checkpoint(
    checkpoint_path: str,
    device_override: Optional[str] = None,
) -> Tuple[
    TrainConfig,
    LayerWindowTranslatorPool,
    Dict[str, ModelSpec],
    Dict[str, PreTrainedModel],
    PreTrainedTokenizerBase,
    List[Node],
    List[Edge],
    Dict[str, LayerMapping],
]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    config = TrainConfig(**payload["train_config"])
    if device_override is not None:
        config.device = device_override
    models, tokenizer, nodes, edges = build_models_and_tokenizer(config)
    translator_pool, model_specs, _, _, layer_mappings = build_translator_pool(models, config)
    translator_pool.load_state_dict(payload["translator_pool"])
    translator_pool.to(config.device)
    translator_pool.eval()
    return config, translator_pool, model_specs, models, tokenizer, nodes, edges, layer_mappings


def run_train(config: TrainConfig) -> Path:
    if config.output_path is None:
        raise ValueError("TrainConfig.output_path must be initialized before run_train.")

    set_seed(config.seed)
    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    nodes, edges, active_directions, _ = resolve_direction_metadata(
        config.model_ids,
        config.model_directions,
    )
    edge_map = build_edge_map(edges)

    config_path = get_train_config_path(output_path)
    write_json(str(config_path), asdict(config))

    log_path = get_train_log_path(output_path)
    logger = setup_logger(f"{config.alg}_train", log_path)
    logger.info("Starting training")
    logger.info("train_config=%s", asdict(config))

    logger.info("nodes=%s", [asdict(node) for node in nodes])
    logger.info("active_directions=%s", active_directions)
    logger.info("[Setup] device=%s", config.device)
    logger.info("[Setup] loading models: %s", {node.id: node.model_id for node in nodes})
    models, tokenizer, _, _ = build_models_and_tokenizer(config)
    translator_pool, model_specs, _, _, layer_mappings = build_translator_pool(models, config)
    translator_pool.train()

    logger.info("[Setup] full model specs")
    for node in nodes:
        spec = model_specs[node.id]
        logger.info(
            "  %s (%s): layers=%d, hidden=%d, heads=%d",
            node.id,
            node.model_id,
            spec.num_layers,
            spec.hidden_size,
            spec.num_heads,
        )
    anchor_direction = active_directions[0]
    anchor_mapping = layer_mappings[anchor_direction]
    logger.info(
        "[Setup] injection_window = L%d-%d on target depth anchored to %s",
        anchor_mapping.dst_layer_start_idx,
        anchor_mapping.dst_layer_end_idx,
        anchor_direction,
    )
    logger.info("[Setup] trainable translator params = %s", f"{count_trainable_parameters(translator_pool):,}")

    dataloader = build_training_dataloader(tokenizer, config)

    optimizer = torch.optim.AdamW(
        translator_pool.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=config.max_steps,
    )

    gpu_memory_tracker = GPUMemoryTracker(config.device)

    running_loss = 0.0
    progress_bar = tqdm(range(1, config.max_steps + 1), desc="Training")

    for step in progress_bar:
        optimizer.zero_grad(set_to_none=True)
        step_loss_value = 0.0

        for _ in range(config.grad_accum_steps):
            input_ids = next(dataloader).to(config.device)
            prefix_cache_ids, lm_input_ids, lm_labels = split_prefix_and_suffix_for_exact_next_token_loss(
                input_ids=input_ids,
                prefix_tokens=config.prefix_tokens,
            )

            with torch.no_grad():
                past_by_node_id = {
                    node.id: extract_past_key_values(models[node.id], prefix_cache_ids)
                    for node in nodes
                }

            total_direction_loss = 0.0
            for direction in active_directions:
                edge = edge_map[direction]
                mixed_target_past, _, mapping = translator_pool.build_replayed_target_past(
                    source_past_key_values=past_by_node_id[edge.src_id],
                    prefix_input_ids=prefix_cache_ids,
                    target_model=models[edge.dst_id],
                    src_name=edge.src_id,
                    dst_name=edge.dst_id,
                    dst_spec=model_specs[edge.dst_id],
                )
                direction_loss = compute_prefix_correction_and_suffix_lm_loss(
                    target_model=models[edge.dst_id],
                    past_key_values=mixed_target_past,
                    lm_input_ids=lm_input_ids,
                    lm_labels=lm_labels,
                    native_target_past_key_values=past_by_node_id[edge.dst_id],
                    target_start_layer_idx=mapping.dst_layer_start_idx,
                )
                total_direction_loss = total_direction_loss + direction_loss

            loss = total_direction_loss / config.grad_accum_steps
            loss.backward()
            step_loss_value += loss.item()

        torch.nn.utils.clip_grad_norm_(translator_pool.parameters(), config.grad_clip_norm)
        optimizer.step()
        scheduler.step()
        gpu_memory_tracker.update()

        running_loss += step_loss_value
        if step % config.log_every == 0:
            avg_loss = running_loss / config.log_every
            progress_bar.set_postfix(
                loss=f"{avg_loss:.4f}",
                lr=f"{scheduler.lr:.2e}",
            )
            gpu_memory = gpu_memory_tracker.summary()
            logger.info(
                "[Step %04d] window_suffix_lm_loss=%.4f | lr=%.2e | gpu_mem_avg=%s | gpu_mem_peak=%s",
                step,
                avg_loss,
                scheduler.lr,
                gpu_memory["avg_allocated_pretty"],
                gpu_memory["peak_allocated_pretty"],
            )
            running_loss = 0.0

    final_path = get_train_checkpoint_path(output_path)
    save_checkpoint(
        output_path=str(final_path),
        translator_pool=translator_pool,
        optimizer=optimizer,
        scheduler=scheduler,
        train_config=config,
        step=config.max_steps,
        extra={
            "note": "Final checkpoint trained with translated-window injection and target replay.",
            "model_ids": config.model_ids,
            "injection_layer_start_idx": config.injection_layer_start_idx,
            "injection_window_size": config.injection_window_size,
            "model_directions": config.model_directions,
            "layer_mappings": {direction: asdict(mapping) for direction, mapping in layer_mappings.items()},
        },
    )
    final_gpu_memory = gpu_memory_tracker.summary()
    logger.info(
        "[Memory] avg_gpu_mem=%s | peak_gpu_mem=%s | samples=%d",
        final_gpu_memory["avg_allocated_pretty"],
        final_gpu_memory["peak_allocated_pretty"],
        final_gpu_memory["num_samples"],
    )
    logger.info("[Done] final checkpoint saved to %s", final_path)
    logger.info("Saved train log to %s", log_path)
    return final_path
