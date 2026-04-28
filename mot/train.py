from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from core.config import Config
from core.channel_manager import (
    Channel,
    ChannelManager,
    build_resolved_channels_path,
    has_resolved_channels,
    load_resolved_channels,
    save_resolved_channels,
)
from core.channel_profiler import ChannelProfiler, load_channel_profile_config
from core.context import Context
from core.model_manager import ModelManager
from core.model_spec import ModelSpec
from core.train_util import *


MOT_VARIANTS = {"single", "mot"}


CHANNEL_ALIGNED_LAYER_ALIGNMENTS = {"terminal", "depth-ratio"}



def uses_channel_alignment(layer_alignment: str) -> bool:
    return layer_alignment in CHANNEL_ALIGNED_LAYER_ALIGNMENTS


def require_channel_profiler(ctx: Context) -> ChannelProfiler:
    if ctx.cp is None:
        raise ValueError("Channel profiler is required when layer_alignment is 'terminal' or 'depth-ratio'.")
    return ctx.cp


@dataclass
class TrainConfig(Config):
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
    layer_alignment: str
    min_window_size_ratio: float
    max_window_size_ratio: float
    translator_dim: int
    translator_heads: int
    translator_depth: int
    translator_mlp_ratio: int
    dtype: str
    variant: str
    mot_num_translators: int
    mot_top_k: int
    topk_sparse_attn: int
    num_bottom_full_attn: int

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.layer_alignment not in {"injection", "terminal", "depth-ratio"}:
            raise ValueError("layer_alignment must be one of {'injection', 'terminal', 'depth-ratio'}")
        if self.min_window_size_ratio <= 0.0:
            raise ValueError("min_window_size_ratio must be > 0")
        if self.max_window_size_ratio <= 0.0:
            raise ValueError("max_window_size_ratio must be > 0")
        if self.max_window_size_ratio < self.min_window_size_ratio:
            raise ValueError("max_window_size_ratio must be >= min_window_size_ratio")
        if self.translator_dim % self.translator_heads != 0:
            raise ValueError("translator_dim must be divisible by translator_heads")
        if self.variant not in MOT_VARIANTS:
            raise ValueError(f"variant must be one of {sorted(MOT_VARIANTS)}")
        if self.mot_num_translators < 1:
            raise ValueError("mot_num_translators must be >= 1")
        if self.mot_top_k < 1:
            raise ValueError("mot_top_k must be >= 1")
        if self.mot_top_k > self.mot_num_translators:
            raise ValueError("mot_top_k must be <= mot_num_translators")
        if self.topk_sparse_attn < 1:
            raise ValueError("topk_sparse_attn must be >= 1")
        if self.num_bottom_full_attn < 0:
            raise ValueError("num_bottom_full_attn must be >= 0")
        initialize_train_output_paths(self)


class CrossLayerWindowTranslator(nn.Module):
    """
    Recurrent cross-attention translator following the LSC translator pattern.

    Input:  [batch, seq, num_layers, src_hidden]
    Output: [batch, seq, num_layers, tgt_hidden]
    """

    def __init__(
        self,
        src_hidden_size: int,
        tgt_hidden_size: int,
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
        self.output_proj = nn.Linear(num_layers * translator_dim, num_layers * tgt_hidden_size)

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
        tgt_hidden_size: int,
        num_layers: int,
        translator_dim: int,
        translator_heads: int,
        translator_depth: int,
        mlp_ratio: int,
        num_translators: int,
        top_k: int,
        translator_cls: Type[nn.Module] = CrossLayerWindowTranslator,
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
                translator_cls(
                    src_hidden_size=src_hidden_size,
                    tgt_hidden_size=tgt_hidden_size,
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
        self.last_mixture_weights: Optional[torch.Tensor] = None
        self.last_router_logits: Optional[torch.Tensor] = None

    def _compute_mixture_weights(self, layer_window_cache: torch.Tensor) -> torch.Tensor:
        router_input = layer_window_cache.reshape(layer_window_cache.shape[0], layer_window_cache.shape[1], -1)
        router_logits = self.router(router_input)
        if self.top_k < self.num_translators:
            topk_indices = torch.topk(router_logits, k=self.top_k, dim=-1).indices
            topk_mask = torch.zeros_like(router_logits, dtype=torch.bool)
            topk_mask.scatter_(-1, topk_indices, True)
            router_logits = router_logits.masked_fill(~topk_mask, float("-inf"))
        mixture_weights = torch.softmax(router_logits, dim=-1)
        self.last_router_logits = router_logits.detach()
        self.last_mixture_weights = mixture_weights.detach()
        return mixture_weights

    def forward(self, layer_window_cache: torch.Tensor) -> torch.Tensor:
        expert_outputs = [translator(layer_window_cache) for translator in self.translators]
        if len(expert_outputs) == 1:
            return expert_outputs[0]
        mixture_weights = self._compute_mixture_weights(layer_window_cache)
        stacked_outputs = torch.stack(expert_outputs, dim=2)
        return (stacked_outputs * mixture_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)

    def get_balance_metrics(self) -> Optional[Dict[str, float]]:
        if self.last_mixture_weights is None:
            return None
        weights = self.last_mixture_weights
        token_importance = weights.sum(dim=(0, 1))
        token_load = (weights > 0).to(weights.dtype).sum(dim=(0, 1))
        eps = torch.finfo(weights.dtype).eps

        def squared_cv(values: torch.Tensor) -> torch.Tensor:
            mean = values.mean()
            variance = ((values - mean) ** 2).mean()
            return variance / (mean.square() + eps)

        return {
            "gate_importance_cv2": float(squared_cv(token_importance).item()),
            "gate_load_cv2": float(squared_cv(token_load).item()),
            "gate_importance_entropy": float((-(token_importance / token_importance.sum().clamp_min(eps)) * (token_importance / token_importance.sum().clamp_min(eps)).clamp_min(eps).log()).sum().item()),
        }


def collect_mot_balance_metrics(module: nn.Module) -> Dict[str, float]:
    summed_metrics: Dict[str, float] = {}
    num_mot_modules = 0
    for submodule in module.modules():
        if not isinstance(submodule, MixtureOfTranslators):
            continue
        metrics = submodule.get_balance_metrics()
        if metrics is None:
            continue
        num_mot_modules += 1
        for name, value in metrics.items():
            summed_metrics[name] = summed_metrics.get(name, 0.0) + value
    if num_mot_modules == 0:
        return {}
    return {name: value / num_mot_modules for name, value in summed_metrics.items()}


def build_window_translator(
    *,
    variant: str,
    src_hidden_size: int,
    tgt_hidden_size: int,
    num_layers: int,
    translator_dim: int,
    translator_heads: int,
    translator_depth: int,
    mlp_ratio: int,
    mot_num_translators: int,
    mot_top_k: int,
    translator_cls: Type[nn.Module] = CrossLayerWindowTranslator,
) -> nn.Module:
    if variant == "single":
        return translator_cls(
            src_hidden_size=src_hidden_size,
            tgt_hidden_size=tgt_hidden_size,
            num_layers=num_layers,
            translator_dim=translator_dim,
            translator_heads=translator_heads,
            translator_depth=translator_depth,
            mlp_ratio=mlp_ratio,
        )
    if variant == "mot":
        return MixtureOfTranslators(
            src_hidden_size=src_hidden_size,
            tgt_hidden_size=tgt_hidden_size,
            num_layers=num_layers,
            translator_dim=translator_dim,
            translator_heads=translator_heads,
            translator_depth=translator_depth,
            mlp_ratio=mlp_ratio,
            num_translators=mot_num_translators,
            top_k=mot_top_k,
            translator_cls=translator_cls,
        )
    raise ValueError(f"Unsupported MOT variant: {variant}")


class LayerWindowDirectionalTranslator(nn.Module):
    def __init__(
        self,
        src_hidden_size: int,
        tgt_hidden_size: int,
        num_layers: int,
        translator_dim: int,
        translator_heads: int,
        translator_depth: int,
        mlp_ratio: int,
        variant: str,
        mot_num_translators: int,
        mot_top_k: int,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.num_layers = num_layers
        self.variant = variant
        self.key_translator = build_window_translator(
            variant=variant,
            src_hidden_size=src_hidden_size,
            tgt_hidden_size=tgt_hidden_size,
            num_layers=num_layers,
            translator_dim=translator_dim,
            translator_heads=translator_heads,
            translator_depth=translator_depth,
            mlp_ratio=mlp_ratio,
            mot_num_translators=mot_num_translators,
            mot_top_k=mot_top_k,
            translator_cls=CrossLayerWindowTranslator,
        )
        self.value_translator = build_window_translator(
            variant=variant,
            src_hidden_size=src_hidden_size,
            tgt_hidden_size=tgt_hidden_size,
            num_layers=num_layers,
            translator_dim=translator_dim,
            translator_heads=translator_heads,
            translator_depth=translator_depth,
            mlp_ratio=mlp_ratio,
            mot_num_translators=mot_num_translators,
            mot_top_k=mot_top_k,
            translator_cls=CrossLayerWindowTranslator,
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
        if key_block.shape[2] != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} layers in the translation window, got {key_block.shape[2]}"
            )
        translated_key = self.key_translator(key_block)
        translated_value = self.value_translator(value_block)
        return translated_key, translated_value


class LayerWindowTranslatorPool(nn.Module):
    def __init__(
        self,
        ctx: Context,
        translator_dim: int,
        translator_heads: int,
        translator_depth: int,
        mlp_ratio: int,
        variant: str,
        mot_num_translators: int,
        mot_top_k: int,
    ) -> None:
        super().__init__()

        self.ctx = ctx
        self.mm = ctx.mm
        self.cm = ctx.cm
        self.edges = tuple(ctx.edges)
        self.edge_ids = tuple(edge.id for edge in ctx.edges)
        self.edges_by_id = build_edge_map(ctx.edges)
        self.node_model_ids = {node.id: node.model_id for node in ctx.nodes}

        adapters = {}
        for edge in self.edges:
            channels = self.cm.get_channels(edge.id)
            adapters[edge.id] = LayerWindowDirectionalTranslator(
                src_hidden_size=self.mm.get_model_spec(edge.src_id).hidden_size,
                tgt_hidden_size=self.mm.get_model_spec(edge.tgt_id).hidden_size,
                num_layers=len(channels),
                translator_dim=translator_dim,
                translator_heads=translator_heads,
                translator_depth=translator_depth,
                mlp_ratio=mlp_ratio,
                variant=variant,
                mot_num_translators=mot_num_translators,
                mot_top_k=mot_top_k,
            )
        self.adapters = nn.ModuleDict(adapters)

    def _extract_channel_blocks(
        self,
        past_key_values: PastKeyValues,
        edge_id: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        channels = self.cm.get_channels(edge_id)
        selected_past = tuple(past_key_values[channel.src_layer_idx] for channel in channels)
        return past_key_values_to_blocks(selected_past)

    def translate_layer_window(
        self,
        past_key_values: PastKeyValues,
        src_node_id: str,
        tgt_node_id: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_id = f"{src_node_id}_to_{tgt_node_id}"
        if edge_id not in self.adapters:
            raise ValueError(
                f"Translator edge {edge_id} is not available. "
                f"Active edges: {list(self.edge_ids)}"
            )
        key_block, value_block = self._extract_channel_blocks(
            past_key_values=past_key_values,
            edge_id=edge_id,
        )
        translated_key, translated_value = self.adapters[edge_id](key_block, value_block)
        return translated_key, translated_value

    def build_replayed_target_past(
        self,
        *,
        source_past_key_values: PastKeyValues,
        prefix_input_ids: torch.Tensor,
        source_model: Optional[PreTrainedModel] = None,
        target_model: PreTrainedModel,
        src_node_id: str,
        tgt_node_id: str,
        tgt_spec: ModelSpec,
    ) -> Tuple[PastKeyValues, PastKeyValues]:
        edge_id = f"{src_node_id}_to_{tgt_node_id}"
        translated_key, translated_value = self.translate_layer_window(
            past_key_values=source_past_key_values,
            src_node_id=src_node_id,
            tgt_node_id=tgt_node_id,
        )
        translated_window_past = blocks_to_partial_past_key_values(
            key_block=translated_key,
            value_block=translated_value,
            num_heads=tgt_spec.num_heads,
            head_dim=tgt_spec.head_dim,
        )
        sparse_attention_indices = None
        if source_model is not None:
            src_spec = self.mm.get_model_spec(src_node_id)
            sparse_attention_indices = build_extrapolated_sparse_attention_indices(
                source_model,
                prefix_input_ids,
                source_layer_indices=self.cm.get_src_layer_indices(edge_id),
                target_layer_indices=self.cm.get_tgt_layer_indices(edge_id),
                num_source_layers=src_spec.num_layers,
                num_target_layers=tgt_spec.num_layers,
                source_model_id=self.node_model_ids.get(src_node_id),
                top_k=self.ctx.config.topk_sparse_attn,
            )
        mixed_target_past = replay_target_prefill_with_injected_window(
            target_model=target_model,
            target_model_id=self.node_model_ids.get(tgt_node_id),
            prefix_input_ids=prefix_input_ids,
            target_layer_indices=self.cm.get_tgt_layer_indices(edge_id),
            injected_key_block=translated_key,
            injected_value_block=translated_value,
            tgt_spec=tgt_spec,
            sparse_attention_indices=sparse_attention_indices,
            num_bottom_full_attn=self.ctx.config.num_bottom_full_attn,
        )
        return mixed_target_past, translated_window_past


def build_channel_map(
    ctx: Context,
    edges: List[Edge],
) -> None:
    config = ctx.config
    requested_window_size = config.injection_window_size
    injection_layer_start_idx = config.injection_layer_start_idx

    for edge in edges:
        src_spec = ctx.mm.get_model_spec(edge.src_id)
        tgt_spec = ctx.mm.get_model_spec(edge.tgt_id)

        tgt_layer_start_idx = injection_layer_start_idx
        tgt_layer_end_idx = tgt_layer_start_idx + requested_window_size - 1
        if tgt_layer_end_idx >= tgt_spec.num_layers:
            raise ValueError(
                f"edge={edge.id} cannot use injection_layer_start_idx={tgt_layer_start_idx} "
                f"with injection_window_size={requested_window_size}: target end layer {tgt_layer_end_idx} exceeds "
                f"target last layer {tgt_spec.num_layers - 1}"
            )

        tgt_depth_from_top = tgt_spec.num_layers - 1 - tgt_layer_start_idx
        src_layer_start_idx = src_spec.num_layers - 1 - tgt_depth_from_top
        if not (0 <= src_layer_start_idx < src_spec.num_layers):
            raise ValueError(
                f"edge={edge.id} cannot align source window to target top-depth {tgt_depth_from_top}: "
                f"computed src_layer_start_idx={src_layer_start_idx} is outside [0, {src_spec.num_layers - 1}]"
            )

        src_layer_end_idx = src_layer_start_idx + requested_window_size - 1
        if src_layer_end_idx >= src_spec.num_layers:
            raise ValueError(
                f"edge={edge.id} cannot use injection_window_size={requested_window_size} after top-depth alignment: "
                f"source end layer {src_layer_end_idx} exceeds source last layer {src_spec.num_layers - 1}"
            )

        for offset in range(requested_window_size):
            ctx.cm.add_channel(
                edge.id,
                src_layer_idx=src_layer_start_idx + offset,
                dst_layer_idx=tgt_layer_start_idx + offset,
            )


def resolve_channels(ctx: Context) -> None:
    config = ctx.config
    if has_resolved_channels(ctx.cm, ctx.edges):
        return
    if config.layer_alignment == "injection":
        build_channel_map(ctx, ctx.edges)
        return

    profiler = require_channel_profiler(ctx)
    profiler.profile_all_edges()


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


def normalize_model_family(model_id: str) -> Optional[str]:
    normalized = str(model_id).strip().lower()
    if "facebook/opt" in normalized or "/opt-" in normalized or normalized.startswith("opt-"):
        return "opt"
    if "gpt2" in normalized:
        return "gpt2"
    return None


def resolve_target_model_family(
    target_model: PreTrainedModel,
    *,
    target_model_id: Optional[str] = None,
) -> str:
    model_family = normalize_model_family(target_model_id or "")
    if model_family is not None:
        return model_family

    if getattr(target_model, "transformer", None) is not None and hasattr(target_model.transformer, "h"):
        return "gpt2"
    model_wrapper = getattr(target_model, "model", None)
    decoder = getattr(model_wrapper, "decoder", None)
    if decoder is not None and hasattr(decoder, "layers"):
        return "opt"
    decoder = getattr(target_model, "decoder", None)
    if decoder is not None and hasattr(decoder, "layers"):
        return "opt"

    raise ValueError(
        "mot target-model replay supports GPT-2 and OPT decoder stacks only "
        f"(target_model_id={target_model_id!r})."
    )


def require_gpt2_transformer(model: PreTrainedModel):
    transformer = getattr(model, "transformer", None)
    if transformer is None or not hasattr(transformer, "h"):
        raise ValueError(
            "mot currently supports GPT-2 style decoder stacks only "
            "(expected model.transformer.h to exist)."
        )
    return transformer


def require_opt_decoder(model: PreTrainedModel):
    model_wrapper = getattr(model, "model", None)
    decoder = getattr(model_wrapper, "decoder", None)
    if decoder is None:
        decoder = getattr(model, "decoder", None)
    if decoder is None or not hasattr(decoder, "layers"):
        raise ValueError(
            "mot currently supports OPT style decoder stacks only "
            "(expected model.model.decoder.layers or model.decoder.layers to exist)."
        )
    return decoder


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


def build_opt_input_hidden_states(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    decoder = require_opt_decoder(model)
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape [batch, seq], got {tuple(input_ids.shape)}")
    batch_size, seq_len = input_ids.shape
    flat_input_ids = input_ids.view(batch_size, seq_len)
    token_attention_mask = torch.ones(batch_size, seq_len, device=input_ids.device, dtype=torch.long)

    hidden_states = decoder.embed_tokens(flat_input_ids)
    project_in = getattr(decoder, "project_in", None)
    if project_in is not None:
        hidden_states = project_in(hidden_states)

    try:
        pos_embeds = decoder.embed_positions(token_attention_mask, past_key_values_length=0)
    except TypeError:
        pos_embeds = decoder.embed_positions(token_attention_mask)
    hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)

    dropout_p = float(getattr(decoder, "dropout", 0.0))
    hidden_states = F.dropout(hidden_states, p=dropout_p, training=decoder.training)

    prepare_mask = getattr(decoder, "_prepare_decoder_attention_mask", None)
    if prepare_mask is not None:
        attention_mask = prepare_mask(
            token_attention_mask,
            (batch_size, seq_len),
            hidden_states,
            0,
        )
    else:
        attention_mask = build_causal_attention_mask(hidden_states)
    return hidden_states, token_attention_mask, attention_mask


def extract_source_attention_topk_indices(
    source_model: PreTrainedModel,
    prefix_input_ids: torch.Tensor,
    layer_indices: List[int],
    *,
    source_model_id: Optional[str] = None,
    top_k: int,
) -> List[torch.Tensor]:
    if len(layer_indices) == 0:
        return []
    model_family = resolve_target_model_family(source_model, target_model_id=source_model_id)
    model_kwargs: Dict[str, Any] = {
        "input_ids": prefix_input_ids,
        "use_cache": False,
        "output_attentions": True,
        "return_dict": True,
    }
    if model_family == "opt":
        model_kwargs["attention_mask"] = torch.ones_like(prefix_input_ids)
    with torch.no_grad():
        outputs = source_model(**model_kwargs)
    attentions = getattr(outputs, "attentions", None)
    if attentions is None:
        raise ValueError("Source model did not return attentions for sparse injected replay.")

    sparse_indices: List[torch.Tensor] = []
    for layer_idx in layer_indices:
        layer_attn = attentions[layer_idx]
        if layer_attn is None:
            raise ValueError(f"Attention for source layer {layer_idx} is unavailable.")
        shared_attn = layer_attn.detach().mean(dim=1, keepdim=True)
        seq_len = shared_attn.shape[-1]
        k = max(1, min(int(top_k), seq_len))
        sparse_indices.append(torch.topk(shared_attn, k=k, dim=-1).indices)
    return sparse_indices


def extrapolate_source_layer_alignment(
    *,
    source_layer_indices: List[int],
    target_layer_indices: List[int],
    num_source_layers: int,
    num_target_layers: int,
) -> List[int]:
    if len(source_layer_indices) != len(target_layer_indices):
        raise ValueError(
            "source_layer_indices and target_layer_indices must have the same length for extrapolation, "
            f"got {len(source_layer_indices)} vs {len(target_layer_indices)}"
        )
    if len(source_layer_indices) == 0:
        return []
    if len(source_layer_indices) == 1:
        only = max(0, min(num_source_layers - 1, int(source_layer_indices[0])))
        return [only for _ in range(num_target_layers)]

    aligned_source = [int(x) for x in source_layer_indices]
    aligned_target = [int(x) for x in target_layer_indices]

    def interpolate(target_layer_idx: int) -> int:
        if target_layer_idx <= aligned_target[0]:
            left = 0
            right = 1
        elif target_layer_idx >= aligned_target[-1]:
            left = len(aligned_target) - 2
            right = len(aligned_target) - 1
        else:
            left = 0
            right = 1
            for idx in range(len(aligned_target) - 1):
                if aligned_target[idx] <= target_layer_idx <= aligned_target[idx + 1]:
                    left = idx
                    right = idx + 1
                    break
        src_left = aligned_source[left]
        src_right = aligned_source[right]
        tgt_left = aligned_target[left]
        tgt_right = aligned_target[right]
        if tgt_right == tgt_left:
            mapped = src_left
        else:
            ratio = float(target_layer_idx - tgt_left) / float(tgt_right - tgt_left)
            mapped = int(round(src_left + ratio * (src_right - src_left)))
        return max(0, min(num_source_layers - 1, mapped))

    return [interpolate(target_layer_idx) for target_layer_idx in range(num_target_layers)]


def build_extrapolated_sparse_attention_indices(
    source_model: PreTrainedModel,
    prefix_input_ids: torch.Tensor,
    *,
    source_layer_indices: List[int],
    target_layer_indices: List[int],
    num_source_layers: int,
    num_target_layers: int,
    source_model_id: Optional[str] = None,
    top_k: int,
) -> List[torch.Tensor]:
    aligned_source_by_target = extrapolate_source_layer_alignment(
        source_layer_indices=source_layer_indices,
        target_layer_indices=target_layer_indices,
        num_source_layers=num_source_layers,
        num_target_layers=num_target_layers,
    )
    unique_source_layers = sorted(set(aligned_source_by_target))
    unique_sparse_indices = extract_source_attention_topk_indices(
        source_model,
        prefix_input_ids,
        unique_source_layers,
        source_model_id=source_model_id,
        top_k=top_k,
    )
    sparse_by_source_layer = {layer_idx: sparse_idx for layer_idx, sparse_idx in zip(unique_source_layers, unique_sparse_indices)}
    return [sparse_by_source_layer[source_layer_idx] for source_layer_idx in aligned_source_by_target]


def expand_sparse_attention_indices(sparse_attention_indices: torch.Tensor, num_heads: int) -> torch.Tensor:
    if sparse_attention_indices.ndim != 4:
        raise ValueError(
            "sparse_attention_indices must have shape [batch, heads|1, seq, k], "
            f"got {tuple(sparse_attention_indices.shape)}"
        )
    if sparse_attention_indices.size(1) == num_heads:
        return sparse_attention_indices
    if sparse_attention_indices.size(1) == 1:
        return sparse_attention_indices.expand(-1, num_heads, -1, -1)
    raise ValueError(
        "Unable to broadcast sparse attention indices across heads: "
        f"indices heads={sparse_attention_indices.size(1)} vs target heads={num_heads}"
    )


def gather_sparse_sequence_vectors(sequence: torch.Tensor, sparse_attention_indices: torch.Tensor) -> torch.Tensor:
    batch_size, num_heads, seq_len, head_dim = sequence.shape
    sparse_attention_indices = sparse_attention_indices.to(device=sequence.device, dtype=torch.long).clamp(0, seq_len - 1)
    sparse_attention_indices = expand_sparse_attention_indices(sparse_attention_indices, num_heads)
    _, _, target_len, top_k = sparse_attention_indices.shape
    flat_sequence = sequence.reshape(batch_size * num_heads, seq_len, head_dim)
    flat_indices = sparse_attention_indices.reshape(batch_size * num_heads, target_len, top_k)
    flat_batch = torch.arange(batch_size * num_heads, device=sequence.device).view(-1, 1, 1)
    gathered = flat_sequence[flat_batch, flat_indices]
    return gathered.view(batch_size, num_heads, target_len, top_k, head_dim)


def build_sparse_query_mask(sparse_attention_indices: torch.Tensor, seq_len: int, num_heads: int) -> torch.Tensor:
    sparse_attention_indices = expand_sparse_attention_indices(sparse_attention_indices, num_heads)
    query_positions = torch.arange(seq_len, device=sparse_attention_indices.device).view(1, 1, seq_len, 1)
    return sparse_attention_indices > query_positions


def run_gpt2_block(
    block: nn.Module,
    hidden_states: torch.Tensor,
    *,
    sparse_attention_indices: Optional[torch.Tensor] = None,
    injected_key: Optional[torch.Tensor] = None,
    injected_value: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if (injected_key is None) != (injected_value is None):
        raise ValueError("injected_key and injected_value must be provided together.")
    if injected_key is not None and injected_key.shape != injected_value.shape:
        raise ValueError(
            "Injected key/value must have identical shapes, "
            f"got {tuple(injected_key.shape)} vs {tuple(injected_value.shape)}"
        )

    attn = block.attn
    residual = hidden_states
    attn_input = block.ln_1(hidden_states)

    qkv = attn.c_attn(attn_input)
    split_size = getattr(attn, "split_size", qkv.shape[-1] // 3)
    query, native_key, native_value = qkv.split(split_size, dim=2)

    batch_size, seq_len, _ = query.shape
    num_heads = attn.num_heads
    head_dim = attn.head_dim
    expected_cache_shape = (batch_size, num_heads, seq_len, head_dim)

    query = query.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    native_key = native_key.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    native_value = native_value.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()

    attention_key = native_key if injected_key is None else injected_key
    attention_value = native_value if injected_value is None else injected_value
    if tuple(attention_key.shape) != expected_cache_shape:
        raise ValueError(
            "Attention cache shape mismatch for GPT-2 layer replay: "
            f"expected {expected_cache_shape}, got {tuple(attention_key.shape)}"
        )

    if sparse_attention_indices is not None:
        sparse_attention_indices = expand_sparse_attention_indices(sparse_attention_indices, num_heads)
        selected_key = gather_sparse_sequence_vectors(attention_key, sparse_attention_indices)
        selected_value = gather_sparse_sequence_vectors(attention_value, sparse_attention_indices)
        attn_scores = (query.unsqueeze(-2) * selected_key).sum(dim=-1)
        invalid_mask = build_sparse_query_mask(sparse_attention_indices, seq_len=seq_len, num_heads=num_heads)
        attn_scores = attn_scores.masked_fill(invalid_mask, torch.finfo(attn_scores.dtype).min)
        attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_dropout = getattr(attn, "attn_dropout", None)
        if isinstance(attn_dropout, nn.Dropout):
            attn_weights = attn_dropout(attn_weights)
        else:
            attn_weights = F.dropout(attn_weights, p=float(getattr(attn, "attn_pdrop", 0.0)), training=block.training)
        attn_output = (attn_weights.unsqueeze(-1) * selected_value).sum(dim=-2)
    else:
        attn_output, _ = attn._attn(
            query,
            attention_key,
            attention_value,
            attention_mask=None,
            head_mask=None,
        )

    attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, num_heads * head_dim)
    attn_output = attn.c_proj(attn_output)
    attn_output = attn.resid_dropout(attn_output)
    hidden_states = residual + attn_output
    hidden_states = hidden_states + block.mlp(block.ln_2(hidden_states))
    return hidden_states, (native_key, native_value)


def run_opt_block(
    block: nn.Module,
    hidden_states: torch.Tensor,
    *,
    attention_mask: torch.Tensor,
    token_attention_mask: torch.Tensor,
    sparse_attention_indices: Optional[torch.Tensor] = None,
    injected_key: Optional[torch.Tensor] = None,
    injected_value: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    del token_attention_mask

    if (injected_key is None) != (injected_value is None):
        raise ValueError("injected_key and injected_value must be provided together.")
    if injected_key is not None and injected_key.shape != injected_value.shape:
        raise ValueError(
            "Injected key/value must have identical shapes, "
            f"got {tuple(injected_key.shape)} vs {tuple(injected_value.shape)}"
        )

    attn = block.self_attn
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_heads = getattr(attn, "num_heads", None)
    if num_heads is None:
        raise ValueError("Unable to determine OPT attention head count.")
    head_dim = getattr(attn, "head_dim", hidden_size // num_heads)
    expected_cache_shape = (batch_size, num_heads, seq_len, head_dim)

    residual = hidden_states
    if getattr(block, "do_layer_norm_before", False):
        hidden_states = block.self_attn_layer_norm(hidden_states)

    query_states = attn.q_proj(hidden_states) * float(getattr(attn, "scaling", head_dim ** -0.5))
    native_key = attn.k_proj(hidden_states)
    native_value = attn.v_proj(hidden_states)
    query_states = query_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()
    native_key = native_key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()
    native_value = native_value.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()

    attention_key = native_key if injected_key is None else injected_key
    attention_value = native_value if injected_value is None else injected_value
    if tuple(attention_key.shape) != expected_cache_shape:
        raise ValueError(
            "Attention cache shape mismatch for OPT layer replay: "
            f"expected {expected_cache_shape}, got {tuple(attention_key.shape)}"
        )

    if sparse_attention_indices is not None:
        sparse_attention_indices = expand_sparse_attention_indices(sparse_attention_indices, num_heads)
        selected_key = gather_sparse_sequence_vectors(attention_key, sparse_attention_indices)
        selected_value = gather_sparse_sequence_vectors(attention_value, sparse_attention_indices)
        attn_weights = (query_states.unsqueeze(-2) * selected_key).sum(dim=-1)
        invalid_mask = build_sparse_query_mask(sparse_attention_indices, seq_len=seq_len, num_heads=num_heads)
        attn_weights = attn_weights.masked_fill(invalid_mask, torch.finfo(attn_weights.dtype).min)
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=float(getattr(attn, "dropout", 0.0)), training=block.training)
        attn_output = (attn_weights.unsqueeze(-1) * selected_value).sum(dim=-2)
    else:
        attn_weights = torch.matmul(query_states, attention_key.transpose(-1, -2))
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=float(getattr(attn, "dropout", 0.0)), training=block.training)
        attn_output = torch.matmul(attn_weights, attention_value)
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, num_heads * head_dim)
    attn_output = attn.out_proj(attn_output)
    attn_output = F.dropout(attn_output, p=float(getattr(block, "dropout", 0.0)), training=block.training)
    hidden_states = residual + attn_output

    if not getattr(block, "do_layer_norm_before", False):
        hidden_states = block.self_attn_layer_norm(hidden_states)

    hidden_states_shape = hidden_states.shape
    hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
    residual = hidden_states
    if getattr(block, "do_layer_norm_before", False):
        hidden_states = block.final_layer_norm(hidden_states)
    hidden_states = block.fc1(hidden_states)
    hidden_states = block.activation_fn(hidden_states)
    hidden_states = block.fc2(hidden_states)
    hidden_states = F.dropout(hidden_states, p=float(getattr(block, "dropout", 0.0)), training=block.training)
    hidden_states = (residual + hidden_states).view(hidden_states_shape)
    if not getattr(block, "do_layer_norm_before", False):
        hidden_states = block.final_layer_norm(hidden_states)
    return hidden_states, (native_key, native_value)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if position_ids is not None and cos.ndim == 4:
        gather_indices = position_ids[:, None, :, None].repeat(1, cos.shape[1], 1, cos.shape[3])
        cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
        sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    elif cos.ndim == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.ndim == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    query_rot, query_pass = query[..., :rotary_dim], query[..., rotary_dim:]
    key_rot, key_pass = key[..., :rotary_dim], key[..., rotary_dim:]
    query_embed = torch.cat([(query_rot * cos) + (rotate_half(query_rot) * sin), query_pass], dim=-1)
    key_embed = torch.cat([(key_rot * cos) + (rotate_half(key_rot) * sin), key_pass], dim=-1)
    return query_embed, key_embed


def build_causal_attention_mask(hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, _ = hidden_states.shape
    mask = torch.full(
        (seq_len, seq_len),
        fill_value=torch.finfo(hidden_states.dtype).min,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)


def replay_target_prefill_with_injected_window(
    target_model: PreTrainedModel,
    prefix_input_ids: torch.Tensor,
    target_layer_indices: List[int],
    injected_key_block: torch.Tensor,
    injected_value_block: torch.Tensor,
    tgt_spec: ModelSpec,
    target_model_id: Optional[str] = None,
    sparse_attention_indices: Optional[List[torch.Tensor]] = None,
    num_bottom_full_attn: int = 3,
) -> PastKeyValues:
    injected_window = blocks_to_partial_past_key_values(
        key_block=injected_key_block,
        value_block=injected_value_block,
        num_heads=tgt_spec.num_heads,
        head_dim=tgt_spec.head_dim,
    )
    translated_num_layers = len(injected_window)

    # zip(target_layer_indices, injected_window) would silently truncate on mismatch,
    # so this remains a correctness guard rather than a mere runtime-prevention check.
    if len(target_layer_indices) != translated_num_layers:
        raise ValueError(
            "Number of target_layer_indices must match translated window size, "
            f"got {len(target_layer_indices)} vs {translated_num_layers}"
        )
    if sparse_attention_indices is not None and len(sparse_attention_indices) != tgt_spec.num_layers:
        raise ValueError(
            "Number of sparse_attention_indices entries must match the total number of target layers, "
            f"got {len(sparse_attention_indices)} vs {tgt_spec.num_layers}"
        )

    model_family = resolve_target_model_family(target_model, target_model_id=target_model_id)
    if model_family == "gpt2":
        transformer = require_gpt2_transformer(target_model)
        target_blocks = transformer.h

        def build_initial_hidden_states() -> Tuple[torch.Tensor, None, None, None]:
            return build_gpt2_input_hidden_states(target_model, prefix_input_ids), None, None, None

        def run_block(
            block: nn.Module,
            hidden_states: torch.Tensor,
            _: Any,
            __: Any,
            ___: Any,
            sparse_attention_indices: Optional[torch.Tensor],
            injected_key: Optional[torch.Tensor] = None,
            injected_value: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            return run_gpt2_block(
                block,
                hidden_states,
                sparse_attention_indices=sparse_attention_indices,
                injected_key=injected_key,
                injected_value=injected_value,
            )

    elif model_family == "opt":
        decoder = require_opt_decoder(target_model)
        target_blocks = decoder.layers

        def build_initial_hidden_states() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
            hidden_states, token_attention_mask, attention_mask = build_opt_input_hidden_states(target_model, prefix_input_ids)
            return hidden_states, token_attention_mask, attention_mask, None

        def run_block(
            block: nn.Module,
            hidden_states: torch.Tensor,
            token_attention_mask: torch.Tensor,
            attention_mask: torch.Tensor,
            _: Any,
            sparse_attention_indices: Optional[torch.Tensor],
            injected_key: Optional[torch.Tensor] = None,
            injected_value: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            return run_opt_block(
                block,
                hidden_states,
                attention_mask=attention_mask,
                token_attention_mask=token_attention_mask,
                sparse_attention_indices=sparse_attention_indices,
                injected_key=injected_key,
                injected_value=injected_value,
            )

    else:
        raise ValueError(f"Unsupported replay target model family: {model_family}")

    rebuilt_past: List[Tuple[torch.Tensor, torch.Tensor]] = []
    target_start_layer_idx = target_layer_indices[0]
    layer_sparse_attention_indices = sparse_attention_indices if sparse_attention_indices is not None else [None] * tgt_spec.num_layers

    def native_sparse_attention_for_layer(layer_idx: int) -> Optional[torch.Tensor]:
        # Keep the lowest num_bottom_full_attn native-only layers exact: do not sparsify their attention.
        return None if layer_idx < num_bottom_full_attn else layer_sparse_attention_indices[layer_idx]

    if torch.is_grad_enabled():
        with torch.no_grad():
            hidden_states, position_ids, attention_mask, position_embeddings = build_initial_hidden_states()
            for lower_idx in range(target_start_layer_idx):
                hidden_states, present = run_block(
                    target_blocks[lower_idx],
                    hidden_states,
                    position_ids,
                    attention_mask,
                    position_embeddings,
                    native_sparse_attention_for_layer(lower_idx),
                )
                rebuilt_past.append((present[0].detach(), present[1].detach()))
        hidden_states = hidden_states.detach()
    else:
        hidden_states, position_ids, attention_mask, position_embeddings = build_initial_hidden_states()
        for lower_idx in range(target_start_layer_idx):
            hidden_states, present = run_block(
                target_blocks[lower_idx],
                hidden_states,
                position_ids,
                attention_mask,
                position_embeddings,
                native_sparse_attention_for_layer(lower_idx),
            )
            rebuilt_past.append(present)

    previous_layer_idx = target_start_layer_idx - 1
    for layer_idx, injected_present in zip(target_layer_indices, injected_window):
        for native_layer_idx in range(previous_layer_idx + 1, layer_idx):
            hidden_states, present = run_block(
                target_blocks[native_layer_idx],
                hidden_states,
                position_ids,
                attention_mask,
                position_embeddings,
                native_sparse_attention_for_layer(native_layer_idx),
            )
            rebuilt_past.append(present)
        hidden_states, present = run_block(
            target_blocks[layer_idx],
            hidden_states,
            position_ids,
            attention_mask,
            position_embeddings,
            layer_sparse_attention_indices[layer_idx],
            injected_present[0],
            injected_present[1],
        )
        rebuilt_past.append(present)
        previous_layer_idx = layer_idx

    for upper_idx in range(previous_layer_idx + 1, len(target_blocks)):
        hidden_states, present = run_block(
            target_blocks[upper_idx],
            hidden_states,
            position_ids,
            attention_mask,
            position_embeddings,
            native_sparse_attention_for_layer(upper_idx),
        )
        rebuilt_past.append(present)

    return tuple(rebuilt_past)


def build_translator_pool(
    ctx: Context,
) -> LayerWindowTranslatorPool:
    config = ctx.config
    resolve_channels(ctx)
    translator_pool = LayerWindowTranslatorPool(
        ctx=ctx,
        translator_dim=config.translator_dim,
        translator_heads=config.translator_heads,
        translator_depth=config.translator_depth,
        mlp_ratio=config.translator_mlp_ratio,
        variant=config.variant,
        mot_num_translators=config.mot_num_translators,
        mot_top_k=config.mot_top_k,
    )
    translator_pool.to(config.device)
    return translator_pool


def load_translator_pool_from_checkpoint(
    checkpoint_dir_path: str,
    nodes: List[Node],
    edges: List[Edge],
    device_override: Optional[str] = None,
) -> Tuple[
    Context,
    LayerWindowTranslatorPool,
]:
    checkpoint_dir_path_obj = Path(checkpoint_dir_path)
    if not checkpoint_dir_path_obj.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir_path_obj}")
    checkpoint_path_obj = get_train_checkpoint_path(checkpoint_dir_path_obj)
    if not checkpoint_path_obj.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path_obj}")
    train_config_path = get_train_config_path(checkpoint_dir_path_obj)
    if not train_config_path.exists():
        raise FileNotFoundError(f"Train config not found under checkpoint directory: {checkpoint_dir_path}")

    config = TrainConfig(**read_json(train_config_path))
    if device_override is not None:
        config.device = device_override
    translator_pool_state_dict = torch.load(str(checkpoint_path_obj), map_location="cpu")
    models, tokenizers = build_models_and_tokenizers(config, nodes)
    ctx = Context(
        config,
        nodes,
        edges,
        ModelManager(models, tokenizers),
        ChannelManager(edges),
    )
    if uses_channel_alignment(config.layer_alignment):
        profile_config_path = Path(checkpoint_dir_path_obj) / "channel_profile.json"
        ctx.cp = ChannelProfiler(ctx, load_channel_profile_config(profile_config_path))

    resolved_channels_path = build_resolved_channels_path(checkpoint_dir_path_obj)
    if resolved_channels_path.exists():
        load_resolved_channels(resolved_channels_path, ctx.cm, ctx.edges)
    elif uses_channel_alignment(config.layer_alignment):
        raise FileNotFoundError(
            "Resolved channel map not found under checkpoint directory: "
            f"{resolved_channels_path}. Re-run training with channel persistence enabled."
        )

    translator_pool = build_translator_pool(ctx)
    translator_pool.load_state_dict(translator_pool_state_dict)
    translator_pool.to(config.device)
    translator_pool.eval()
    return ctx, translator_pool


def run_train(
    ctx: Context,
    gpu_memory_tracker: GPUMemoryTracker,
) -> Path:
    config = ctx.config
    nodes = ctx.nodes
    edges = ctx.edges
    set_seed(config.seed)
    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    cp = None
    if uses_channel_alignment(config.layer_alignment):
        cp = require_channel_profiler(ctx)

    config_path = get_train_config_path(output_path)
    write_json(str(config_path), asdict(config))
    if cp is not None:
        from core.channel_profiler import save_channel_profile_config
        save_channel_profile_config(output_path, cp.profile_config)

    log_path = get_train_log_path(output_path)
    logging.info("Starting training")
    logging.info("train_config=%s", asdict(config))

    logging.info("nodes=%s", [asdict(node) for node in nodes])
    logging.info("edges=%s", [edge.id for edge in edges])
    logging.info("[Setup] device=%s", config.device)
    logging.info("[Setup] loading models: %s", {node.id: node.model_id for node in nodes})
    translator_pool = build_translator_pool(ctx)
    save_resolved_channels(output_path, ctx.cm, edges)
    translator_pool.train()

    logging.info("[Setup] full model specs")
    for node in nodes:
        spec = ctx.mm.get_model_spec(node.id)
        logging.info(
            "  %s (%s): layers=%d, hidden=%d, heads=%d",
            node.id,
            node.model_id,
            spec.num_layers,
            spec.hidden_size,
            spec.num_heads,
        )
    logging.info("[Setup] trainable translator params = %s", f"{count_trainable_parameters(translator_pool):,}")

    dataloaders_by_target = build_training_dataloaders_by_target(ctx)

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


    running_loss = 0.0
    running_gate_importance_cv2 = 0.0
    running_gate_load_cv2 = 0.0
    running_gate_importance_entropy = 0.0
    progress_bar = tqdm(range(1, config.max_steps + 1), desc="Training")

    for step in progress_bar:
        optimizer.zero_grad(set_to_none=True)
        step_loss_value = 0.0

        for _ in range(config.grad_accum_steps):
            target_batches = {}
            for target_node_id, dataloader in dataloaders_by_target.items():
                input_ids = next(dataloader).to(config.device)
                prefix_cache_ids, lm_input_ids, lm_labels = split_prefix_and_suffix_for_exact_next_token_loss(
                    input_ids=input_ids,
                    prefix_tokens=config.prefix_tokens,
                )
                with torch.no_grad():
                    past_by_node_id = {
                        node.id: extract_past_key_values(ctx.mm.get_model(node.id), prefix_cache_ids)
                        for node in nodes
                    }
                target_batches[target_node_id] = (prefix_cache_ids, lm_input_ids, lm_labels, past_by_node_id)

            total_direction_loss = 0.0
            for edge in edges:
                prefix_cache_ids, lm_input_ids, lm_labels, past_by_node_id = target_batches[edge.tgt_id]
                mixed_target_past, _ = translator_pool.build_replayed_target_past(
                    source_past_key_values=past_by_node_id[edge.src_id],
                    prefix_input_ids=prefix_cache_ids,
                    source_model=ctx.mm.get_model(edge.src_id),
                    target_model=ctx.mm.get_model(edge.tgt_id),
                    src_node_id=edge.src_id,
                    tgt_node_id=edge.tgt_id,
                    tgt_spec=ctx.mm.get_model_spec(edge.tgt_id),
                )
                direction_loss = compute_prefix_correction_and_suffix_lm_loss(
                    target_model=ctx.mm.get_model(edge.tgt_id),
                    past_key_values=mixed_target_past,
                    lm_input_ids=lm_input_ids,
                    lm_labels=lm_labels,
                    native_target_past_key_values=past_by_node_id[edge.tgt_id],
                    target_layer_indices=ctx.cm.get_tgt_layer_indices(edge.id),
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
        gate_metrics = collect_mot_balance_metrics(translator_pool)
        running_gate_importance_cv2 += gate_metrics.get("gate_importance_cv2", 0.0)
        running_gate_load_cv2 += gate_metrics.get("gate_load_cv2", 0.0)
        running_gate_importance_entropy += gate_metrics.get("gate_importance_entropy", 0.0)
        if step % config.log_every == 0:
            avg_loss = running_loss / config.log_every
            avg_gate_importance_cv2 = running_gate_importance_cv2 / config.log_every
            avg_gate_load_cv2 = running_gate_load_cv2 / config.log_every
            avg_gate_importance_entropy = running_gate_importance_entropy / config.log_every
            progress_bar.set_postfix(
                loss=f"{avg_loss:.4f}",
                gate_load_cv2=f"{avg_gate_load_cv2:.4f}",
                lr=f"{scheduler.lr:.2e}",
            )
            gpu_memory = gpu_memory_tracker.summary()
            logging.info(
                "[Step %04d] loss=%.4f | gate_importance_cv2=%.4f | gate_load_cv2=%.4f | gate_importance_entropy=%.4f | lr=%.2e | gpu_mem_avg=%s | gpu_mem_peak=%s",
                step,
                avg_loss,
                avg_gate_importance_cv2,
                avg_gate_load_cv2,
                avg_gate_importance_entropy,
                scheduler.lr,
                gpu_memory["avg_allocated_pretty"],
                gpu_memory["peak_allocated_pretty"],
            )
            running_loss = 0.0
            running_gate_importance_cv2 = 0.0
            running_gate_load_cv2 = 0.0
            running_gate_importance_entropy = 0.0

    final_path = get_train_checkpoint_path(output_path)
    save_checkpoint(
        output_path=final_path,
        translator_pool=translator_pool,
    )
    final_gpu_memory = gpu_memory_tracker.summary()
    logging.info(
        "[Memory] avg_gpu_mem=%s | peak_gpu_mem=%s | samples=%d",
        final_gpu_memory["avg_allocated_pretty"],
        final_gpu_memory["peak_allocated_pretty"],
        final_gpu_memory["num_samples"],
    )
    logging.info("[Done] final checkpoint saved to %s", final_path)
    logging.info("Saved train log to %s", log_path)
    return final_path
