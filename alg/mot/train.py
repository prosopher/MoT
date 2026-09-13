from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from core.common import build_step_pasts_and_batches, ensure_token_ids_model
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
from core.translator_pool import TranslatorPool
from core.model_spec import ModelSpec
from core.train_util import *
from core.topology import get_translator_id


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


def initialize_translators(
    ctx: Context,
    translator_dim: int,
    translator_heads: int,
    translator_depth: int,
    mlp_ratio: int,
    variant: str,
    mot_num_translators: int,
    mot_top_k: int,
) -> TranslatorPool:
    tp = ctx.tp
    node_model_ids = {node.id: node.model_id for node in ctx.nodes}

    for edge in ctx.edges:
        translator_id = get_translator_id(node_model_ids[edge.src_id], node_model_ids[edge.tgt_id])
        if translator_id in tp.translators:
            continue
        src_spec = tp.get_model_spec(edge.src_id)
        tgt_spec = tp.get_model_spec(edge.tgt_id)
        channels = ctx.cm.get_channels(edge.id)
        tp.add_translator(
            translator_id,
            LayerWindowDirectionalTranslator(
                src_hidden_size=src_spec.kv_hidden_size,
                tgt_hidden_size=tgt_spec.kv_hidden_size,
                num_layers=len(channels),
                translator_dim=translator_dim,
                translator_heads=translator_heads,
                translator_depth=translator_depth,
                mlp_ratio=mlp_ratio,
                variant=variant,
                mot_num_translators=mot_num_translators,
                mot_top_k=mot_top_k,
            ),
        )
    return tp


def extract_channel_blocks(
    ctx: Context,
    past_key_values: PastKeyValues,
    edge_id: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    channels = ctx.cm.get_channels(edge_id)
    selected_past = tuple(past_key_values[channel.src_layer_idx] for channel in channels)
    return past_key_values_to_blocks(selected_past)


def translate_layer_window(
    ctx: Context,
    past_key_values: PastKeyValues,
    src_node_id: str,
    tgt_node_id: str,
    *,
    token_alignment_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    edge_id = f"{src_node_id}_to_{tgt_node_id}"
    edge_ids = tuple(edge.id for edge in ctx.edges)
    if edge_id not in build_edge_map(ctx.edges):
        raise ValueError(
            f"Translator edge {edge_id} is not available. "
            f"Active edges: {list(edge_ids)}"
        )

    node_model_ids = {node.id: node.model_id for node in ctx.nodes}
    translator_id = get_translator_id(node_model_ids[src_node_id], node_model_ids[tgt_node_id])
    if translator_id not in ctx.tp.translators:
        raise ValueError(
            f"Translator {translator_id} is not available. "
            f"Active translators: {list(ctx.tp.translators.keys())}"
        )
    key_block, value_block = extract_channel_blocks(
        ctx=ctx,
        past_key_values=past_key_values,
        edge_id=edge_id,
    )
    if token_alignment_weights is not None:
        key_block, value_block = align_cache_blocks_to_target_tokens(
            key_block,
            value_block,
            alignment_weights=token_alignment_weights,
        )
    translated_key, translated_value = ctx.tp.translators[translator_id](key_block, value_block)
    return translated_key, translated_value



def _decode_token_ids(tokenizer, token_ids: List[int]) -> str:
    kwargs = {
        "skip_special_tokens": False,
        "clean_up_tokenization_spaces": False,
    }
    try:
        return tokenizer.decode(token_ids, **kwargs)
    except TypeError:
        kwargs.pop("clean_up_tokenization_spaces", None)
        return tokenizer.decode(token_ids, **kwargs)


def _common_prefix_length(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    idx = 0
    while idx < limit and left[idx] == right[idx]:
        idx += 1
    return idx


def _token_spans_for_ids(tokenizer, token_ids: List[int], text: str) -> List[Tuple[int, int]]:
    """Return character spans for exactly ``token_ids`` over ``text``.

    Fast-tokenizer offset mappings are used when the decode/encode round trip
    reproduces the supplied ids.  A decoder-prefix fallback keeps this usable
    for tokenizers that do not expose offset mappings (or for special tokens).
    """
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        encoded_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        offsets = encoded["offset_mapping"] if isinstance(encoded, dict) else encoded.offset_mapping
        if (
            list(encoded_ids) == list(token_ids)
            and len(offsets) == len(token_ids)
        ):
            return [(int(start), int(end)) for start, end in offsets]
    except (TypeError, ValueError, KeyError, AttributeError, NotImplementedError):
        pass

    spans: List[Tuple[int, int]] = []
    previous_end = 0
    for end_idx in range(1, len(token_ids) + 1):
        prefix_text = _decode_token_ids(tokenizer, token_ids[:end_idx])
        current_end = _common_prefix_length(text, prefix_text)
        current_end = max(previous_end, min(len(text), current_end))
        spans.append((previous_end, current_end))
        previous_end = current_end
    if spans and spans[-1][1] < len(text):
        spans[-1] = (spans[-1][0], len(text))
    return spans


def _nearest_span_index(span: Tuple[int, int], candidates: List[Tuple[int, int]]) -> int:
    start, end = span
    center = 0.5 * (start + end)
    best_idx = 0
    best_distance = float("inf")
    for idx, (candidate_start, candidate_end) in enumerate(candidates):
        candidate_center = 0.5 * (candidate_start + candidate_end)
        distance = abs(candidate_center - center)
        if distance < best_distance:
            best_idx = idx
            best_distance = distance
    return best_idx


def _build_overlap_weights(
    source_spans: List[Tuple[int, int]],
    target_spans: List[Tuple[int, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source_len = len(source_spans)
    target_len = len(target_spans)
    if source_len < 1 or target_len < 1:
        raise ValueError("Cross-token alignment requires non-empty source and target token sequences.")

    overlap = torch.zeros(target_len, source_len, dtype=torch.float32)
    for target_idx, (target_start, target_end) in enumerate(target_spans):
        for source_idx, (source_start, source_end) in enumerate(source_spans):
            amount = max(0, min(target_end, source_end) - max(target_start, source_start))
            if amount > 0:
                overlap[target_idx, source_idx] = float(amount)
        if not torch.any(overlap[target_idx] > 0):
            overlap[target_idx, _nearest_span_index((target_start, target_end), source_spans)] = 1.0

    row_sums = overlap.sum(dim=1, keepdim=True).clamp_min(1.0)
    target_from_source_weights = overlap / row_sums
    target_to_source = overlap.argmax(dim=1).to(dtype=torch.long)

    source_to_target = torch.empty(source_len, dtype=torch.long)
    for source_idx, source_span in enumerate(source_spans):
        column = overlap[:, source_idx]
        if torch.any(column > 0):
            source_to_target[source_idx] = int(column.argmax().item())
        else:
            source_to_target[source_idx] = _nearest_span_index(source_span, target_spans)

    return target_from_source_weights, target_to_source, source_to_target


def build_cross_token_alignment(
    source_model: Model,
    target_model: Model,
    source_context_token_ids: TokenIDs,
    target_context_token_ids: TokenIDs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build semantic token-grid alignment from decoded character spans.

    Returns:
      weights: [batch, target_seq, source_seq]
      target_to_source: [batch, target_seq]
      source_to_target: [batch, source_seq]
    """
    ensure_token_ids_model(source_model, source_context_token_ids)
    ensure_token_ids_model(target_model, target_context_token_ids)
    if source_context_token_ids.ndim != 2 or target_context_token_ids.ndim != 2:
        raise ValueError("Cross-token alignment expects rank-2 token id tensors.")
    if source_context_token_ids.shape[0] != target_context_token_ids.shape[0]:
        raise ValueError(
            "Cross-token alignment requires matching batch sizes, "
            f"got {source_context_token_ids.shape[0]} and {target_context_token_ids.shape[0]}"
        )

    source_rows = source_context_token_ids.as_tensor().detach().cpu().tolist()
    target_rows = target_context_token_ids.as_tensor().detach().cpu().tolist()
    all_weights = []
    all_target_to_source = []
    all_source_to_target = []

    for source_ids, target_ids in zip(source_rows, target_rows):
        target_text = _decode_token_ids(target_model.tokenizer, target_ids)
        source_spans = _token_spans_for_ids(source_model.tokenizer, source_ids, target_text)
        target_spans = _token_spans_for_ids(target_model.tokenizer, target_ids, target_text)
        weights, target_to_source, source_to_target = _build_overlap_weights(source_spans, target_spans)
        all_weights.append(weights)
        all_target_to_source.append(target_to_source)
        all_source_to_target.append(source_to_target)

    return (
        torch.stack(all_weights, dim=0),
        torch.stack(all_target_to_source, dim=0),
        torch.stack(all_source_to_target, dim=0),
    )


def align_cache_blocks_to_target_tokens(
    key_block: torch.Tensor,
    value_block: torch.Tensor,
    *,
    alignment_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if key_block.shape != value_block.shape:
        raise ValueError(
            "Key/value blocks must have identical shapes before token alignment, "
            f"got {tuple(key_block.shape)} vs {tuple(value_block.shape)}"
        )
    if key_block.ndim != 4:
        raise ValueError(
            "Cache blocks must have shape [batch, source_seq, layers, hidden], "
            f"got {tuple(key_block.shape)}"
        )
    if alignment_weights.ndim != 3:
        raise ValueError(
            "alignment_weights must have shape [batch, target_seq, source_seq], "
            f"got {tuple(alignment_weights.shape)}"
        )
    if key_block.shape[0] != alignment_weights.shape[0] or key_block.shape[1] != alignment_weights.shape[2]:
        raise ValueError(
            "Token-alignment shape mismatch: "
            f"cache={tuple(key_block.shape)}, weights={tuple(alignment_weights.shape)}"
        )

    weights = alignment_weights.to(device=key_block.device, dtype=key_block.dtype)
    aligned_key = torch.einsum("bts,bslh->btlh", weights, key_block)
    aligned_value = torch.einsum("bts,bslh->btlh", weights, value_block)
    return aligned_key, aligned_value


def align_sparse_attention_indices_to_target_tokens(
    sparse_attention_indices: torch.Tensor,
    *,
    target_to_source: torch.Tensor,
    source_to_target: torch.Tensor,
) -> torch.Tensor:
    if sparse_attention_indices.ndim != 4:
        raise ValueError(
            "sparse_attention_indices must have shape [batch, heads|1, source_seq, k], "
            f"got {tuple(sparse_attention_indices.shape)}"
        )
    batch_size, num_heads, source_seq_len, top_k = sparse_attention_indices.shape
    if target_to_source.shape[0] != batch_size or source_to_target.shape != (batch_size, source_seq_len):
        raise ValueError(
            "Sparse token-alignment shape mismatch: "
            f"indices={tuple(sparse_attention_indices.shape)}, "
            f"target_to_source={tuple(target_to_source.shape)}, "
            f"source_to_target={tuple(source_to_target.shape)}"
        )

    indices = sparse_attention_indices.to(dtype=torch.long)
    target_to_source = target_to_source.to(device=indices.device, dtype=torch.long)
    source_to_target = source_to_target.to(device=indices.device, dtype=torch.long)
    target_seq_len = target_to_source.shape[1]

    query_gather = target_to_source[:, None, :, None].expand(batch_size, num_heads, target_seq_len, top_k)
    selected_source_indices = torch.gather(indices, dim=2, index=query_gather)
    selected_source_indices = selected_source_indices.clamp(0, source_seq_len - 1)

    key_lookup = source_to_target[:, None, None, :].expand(batch_size, num_heads, target_seq_len, source_seq_len)
    remapped = torch.gather(key_lookup, dim=3, index=selected_source_indices)

    # Multiple source tokens can collapse onto one target token.  Keep the
    # highest-ranked occurrence and mark later duplicates with an out-of-range
    # sentinel; gather_sparse_sequence_vectors clamps it for gathering while
    # build_sparse_query_mask masks it out of the softmax.
    for topk_idx in range(1, top_k):
        duplicate = (remapped[..., topk_idx : topk_idx + 1] == remapped[..., :topk_idx]).any(dim=-1)
        remapped[..., topk_idx] = torch.where(
            duplicate,
            torch.full_like(remapped[..., topk_idx], target_seq_len),
            remapped[..., topk_idx],
        )
    return remapped


def _retokenize_target_context_row_for_source(
    source_model: Model,
    target_model: Model,
    target_context_token_ids: TokenIDs,
) -> TokenIDs:
    ensure_token_ids_model(target_model, target_context_token_ids)
    if target_context_token_ids.shape[0] != 1:
        raise ValueError("Target-context retokenization expects a single batch row.")
    target_ids = target_context_token_ids.as_tensor()[0].detach().cpu().tolist()
    context_text = _decode_token_ids(target_model.tokenizer, target_ids)
    encoded = source_model.tokenizer(
        context_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    source_token_ids = TokenIDs(encoded.input_ids, model_id=source_model.id).to(source_model.device)
    if source_token_ids.shape[1] < 1:
        raise ValueError("Retokenized source context must contain at least one token.")
    return source_token_ids


def _concat_past_batches(past_batches: List[PastKeyValues]) -> PastKeyValues:
    if not past_batches:
        raise ValueError("past_batches must contain at least one batch.")
    num_layers = len(past_batches[0])
    if any(len(past) != num_layers for past in past_batches):
        raise ValueError("All past batches must have the same number of layers.")
    return tuple(
        (
            torch.cat([past[layer_idx][0] for past in past_batches], dim=0),
            torch.cat([past[layer_idx][1] for past in past_batches], dim=0),
        )
        for layer_idx in range(num_layers)
    )


def build_replayed_target_past(
    ctx: Context,
    *,
    source_past_key_values: PastKeyValues,
    source_context_token_ids: TokenIDs,
    target_context_token_ids: TokenIDs,
    source_model: Model,
    target_model: Model,
    src_node_id: str,
    tgt_node_id: str,
    tgt_spec: ModelSpec,
) -> Tuple[PastKeyValues, PastKeyValues]:
    edge_id = f"{src_node_id}_to_{tgt_node_id}"
    node_model_ids = {node.id: node.model_id for node in ctx.nodes}
    src_spec = ctx.tp.get_model_spec(src_node_id)
    sparse_attention_indices = build_extrapolated_sparse_attention_indices(
        source_model,
        source_context_token_ids,
        source_layer_indices=ctx.cm.get_src_layer_indices(edge_id),
        target_layer_indices=ctx.cm.get_tgt_layer_indices(edge_id),
        num_source_layers=src_spec.num_layers,
        num_target_layers=tgt_spec.num_layers,
        source_model_id=node_model_ids.get(src_node_id),
        top_k=ctx.config.topk_sparse_attn,
    )

    needs_token_alignment = (
        source_model.id != target_model.id
        or source_context_token_ids.shape != target_context_token_ids.shape
        or not torch.equal(
            source_context_token_ids.as_tensor().detach().cpu(),
            target_context_token_ids.as_tensor().detach().cpu(),
        )
    )
    alignment_weights = None
    if needs_token_alignment:
        alignment_weights, target_to_source, source_to_target = build_cross_token_alignment(
            source_model=source_model,
            target_model=target_model,
            source_context_token_ids=source_context_token_ids,
            target_context_token_ids=target_context_token_ids,
        )
        sparse_attention_indices = [
            align_sparse_attention_indices_to_target_tokens(
                sparse_indices,
                target_to_source=target_to_source,
                source_to_target=source_to_target,
            )
            for sparse_indices in sparse_attention_indices
        ]

    translated_key, translated_value = translate_layer_window(
        ctx=ctx,
        past_key_values=source_past_key_values,
        src_node_id=src_node_id,
        tgt_node_id=tgt_node_id,
        token_alignment_weights=alignment_weights,
    )

    translated_window_past = blocks_to_partial_past_key_values(
        key_block=translated_key,
        value_block=translated_value,
        num_heads=tgt_spec.num_key_value_heads,
        head_dim=tgt_spec.head_dim,
    )
    mixed_target_past = replay_target_prefill_with_injected_window(
        target_model=target_model,
        target_model_id=node_model_ids.get(tgt_node_id),
        context_token_ids=target_context_token_ids,
        target_layer_indices=ctx.cm.get_tgt_layer_indices(edge_id),
        injected_key_block=translated_key,
        injected_value_block=translated_value,
        tgt_spec=tgt_spec,
        sparse_attention_indices=sparse_attention_indices,
        num_bottom_full_attn=ctx.config.num_bottom_full_attn,
    )
    return mixed_target_past, translated_window_past


def build_replayed_target_past_from_target_context(
    ctx: Context,
    *,
    target_context_token_ids: TokenIDs,
    source_model: Model,
    target_model: Model,
    src_node_id: str,
    tgt_node_id: str,
    tgt_spec: ModelSpec,
) -> Tuple[PastKeyValues, PastKeyValues]:
    """Replay a target batch using source caches built from the exact same text.

    Heterogeneous tokenizers can produce different source lengths for each row,
    so source prefill/replay is performed row-wise and the target-grid caches are
    concatenated after semantic token alignment.
    """
    ensure_token_ids_model(target_model, target_context_token_ids)
    mixed_batches: List[PastKeyValues] = []
    translated_batches: List[PastKeyValues] = []
    for batch_idx in range(target_context_token_ids.shape[0]):
        target_row = target_context_token_ids[batch_idx : batch_idx + 1]
        source_row = _retokenize_target_context_row_for_source(
            source_model=source_model,
            target_model=target_model,
            target_context_token_ids=target_row,
        )
        with torch.no_grad():
            source_past = extract_past_key_values(source_model, source_row)
        mixed_row, translated_row = build_replayed_target_past(
            ctx,
            source_past_key_values=source_past,
            source_context_token_ids=source_row,
            target_context_token_ids=target_row,
            source_model=source_model,
            target_model=target_model,
            src_node_id=src_node_id,
            tgt_node_id=tgt_node_id,
            tgt_spec=tgt_spec,
        )
        mixed_batches.append(mixed_row)
        translated_batches.append(translated_row)

    return _concat_past_batches(mixed_batches), _concat_past_batches(translated_batches)


def build_channel_map(
    ctx: Context,
    edges: List[Edge],
) -> None:
    config = ctx.config
    requested_window_size = config.injection_window_size
    injection_layer_start_idx = config.injection_layer_start_idx

    for edge in edges:
        src_spec = ctx.tp.get_model_spec(edge.src_id)
        tgt_spec = ctx.tp.get_model_spec(edge.tgt_id)

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
    if (
        "qwen2" in normalized
        or "qwen2.5" in normalized
        or "qwen/qwen2" in normalized
        or "qwen3" in normalized
        or "qwen/qwen3" in normalized
    ):
        return "qwen2"
    if "gemma-3" in normalized or "gemma3" in normalized:
        return "gemma3"
    if "llama-3.2" in normalized or "llama3.2" in normalized:
        return "llama"
    if "facebook/opt" in normalized or "/opt-" in normalized or normalized.startswith("opt-"):
        return "opt"
    if "gpt2" in normalized:
        return "gpt2"
    return None


def resolve_target_model_family(
    target_model: Model,
    *,
    target_model_id: Optional[str] = None,
) -> str:
    model_family = normalize_model_family(target_model_id or "")
    if model_family is not None:
        return model_family

    config_model_type = str(getattr(getattr(target_model, "config", None), "model_type", "")).lower()
    if config_model_type in {"qwen2", "qwen2_5", "qwen3"}:
        return "qwen2"
    if config_model_type == "gemma3_text":
        return "gemma3"
    if config_model_type == "llama":
        return "llama"

    if getattr(target_model, "transformer", None) is not None and hasattr(target_model.transformer, "h"):
        return "gpt2"
    model_wrapper = getattr(target_model, "model", None)
    if (
        model_wrapper is not None
        and hasattr(model_wrapper, "layers")
        and hasattr(model_wrapper, "embed_tokens")
        and any(hasattr(layer, "input_layernorm") for layer in getattr(model_wrapper, "layers", [])[:1])
    ):
        return "qwen2"
    decoder = getattr(model_wrapper, "decoder", None)
    if decoder is not None and hasattr(decoder, "layers"):
        return "opt"
    decoder = getattr(target_model, "decoder", None)
    if decoder is not None and hasattr(decoder, "layers"):
        return "opt"

    raise ValueError(
        "mot target-model replay supports GPT-2, OPT, Qwen2/Qwen2.5, Qwen3, Llama 3.2, and Gemma 3 text decoder stacks only "
        f"(target_model_id={target_model_id!r})."
    )


def require_gpt2_transformer(model: Model):
    transformer = getattr(model, "transformer", None)
    if transformer is None or not hasattr(transformer, "h"):
        raise ValueError(
            "mot currently supports GPT-2 style decoder stacks only "
            "(expected model.transformer.h to exist)."
        )
    return transformer


def require_opt_decoder(model: Model):
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




def _require_rotary_decoder_model(model: Model, *, family_name: str):
    # core.model.Model adds one wrapper around the Hugging Face causal LM.
    # Walk common wrapper attributes instead of assuming a fixed nesting depth.
    pending = [model]
    seen: set[int] = set()
    while pending:
        candidate = pending.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))

        if hasattr(candidate, "layers") and hasattr(candidate, "embed_tokens"):
            return candidate

        for attr_name in ("model", "base_model", "module"):
            try:
                wrapped = getattr(candidate, attr_name, None)
            except Exception:
                continue
            if wrapped is not None and wrapped is not candidate and id(wrapped) not in seen:
                pending.append(wrapped)

    raise ValueError(
        f"mot could not locate the {family_name} decoder stack "
        "(expected a wrapped decoder with layers/embed_tokens to exist)."
    )


def require_qwen2_model(model: Model):
    return _require_rotary_decoder_model(model, family_name="Qwen2/Qwen2.5/Qwen3")


def require_llama_model(model: Model):
    return _require_rotary_decoder_model(model, family_name="Llama")


def require_gemma3_model(model: Model):
    return _require_rotary_decoder_model(model, family_name="Gemma 3")


def build_gpt2_input_hidden_states(model: Model, token_ids: TokenIDs) -> torch.Tensor:
    transformer = require_gpt2_transformer(model)
    if token_ids.ndim != 2:
        raise ValueError(f"token_ids must have shape [batch, seq], got {tuple(token_ids.shape)}")
    batch_size, seq_len = token_ids.shape
    position_ids = torch.arange(seq_len, device=token_ids.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    hidden_states = transformer.wte(token_ids.as_tensor()) + transformer.wpe(position_ids)
    drop = getattr(transformer, "drop", None)
    if drop is not None:
        hidden_states = drop(hidden_states)
    return hidden_states


def build_opt_input_hidden_states(
    model: Model,
    token_ids: TokenIDs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    decoder = require_opt_decoder(model)
    if token_ids.ndim != 2:
        raise ValueError(f"token_ids must have shape [batch, seq], got {tuple(token_ids.shape)}")
    batch_size, seq_len = token_ids.shape
    flat_token_ids = token_ids.as_tensor().view(batch_size, seq_len)
    token_attention_mask = torch.ones(batch_size, seq_len, device=token_ids.device, dtype=torch.long)

    hidden_states = decoder.embed_tokens(flat_token_ids)
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



def _build_rotary_input_hidden_states(
    decoder_model: nn.Module,
    token_ids: TokenIDs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    if token_ids.ndim != 2:
        raise ValueError(f"token_ids must have shape [batch, seq], got {tuple(token_ids.shape)}")
    batch_size, seq_len = token_ids.shape
    position_ids = torch.arange(seq_len, device=token_ids.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    hidden_states = decoder_model.embed_tokens(token_ids.as_tensor())
    attention_mask = build_causal_attention_mask(hidden_states)

    position_embeddings = None
    rotary_emb = getattr(decoder_model, "rotary_emb", None)
    if rotary_emb is not None:
        try:
            position_embeddings = rotary_emb(hidden_states, position_ids)
        except TypeError:
            position_embeddings = None
    return hidden_states, position_ids, attention_mask, position_embeddings


def build_qwen2_input_hidden_states(
    model: Model,
    token_ids: TokenIDs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    return _build_rotary_input_hidden_states(require_qwen2_model(model), token_ids)


def build_llama_input_hidden_states(
    model: Model,
    token_ids: TokenIDs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    # Llama 3.2 uses the same embedding/position inputs as the Qwen rotary
    # decoder path, but keep family validation explicit.
    return _build_rotary_input_hidden_states(require_llama_model(model), token_ids)


def build_gemma3_input_hidden_states(
    model: Model,
    token_ids: TokenIDs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
    decoder = require_gemma3_model(model)
    if token_ids.ndim != 2:
        raise ValueError(f"token_ids must have shape [batch, seq], got {tuple(token_ids.shape)}")
    batch_size, seq_len = token_ids.shape
    position_ids = torch.arange(seq_len, device=token_ids.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    # Gemma3ScaledWordEmbedding applies the model-required sqrt(hidden_size)
    # scaling internally. RoPE is layer-specific (local vs global), so unlike
    # Llama/Qwen it must be computed by each attention module during replay.
    hidden_states = decoder.embed_tokens(token_ids.as_tensor())
    attention_mask = build_causal_attention_mask(hidden_states)
    return hidden_states, position_ids, attention_mask, None


def extract_source_attention_topk_indices(
    source_model: Model,
    context_token_ids: TokenIDs,
    layer_indices: List[int],
    *,
    source_model_id: Optional[str] = None,
    top_k: int,
) -> List[torch.Tensor]:
    if len(layer_indices) == 0:
        return []
    model_family = resolve_target_model_family(source_model, target_model_id=source_model_id)
    model_kwargs: Dict[str, Any] = {
        "input_ids": context_token_ids.as_tensor(),
        "use_cache": False,
        "output_attentions": True,
        "return_dict": True,
    }
    if model_family in {"opt", "qwen2", "llama", "gemma3"}:
        model_kwargs["attention_mask"] = torch.ones_like(context_token_ids.as_tensor())
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
    source_model: Model,
    context_token_ids: TokenIDs,
    *,
    source_layer_indices: List[int],
    target_layer_indices: List[int],
    num_source_layers: int,
    num_target_layers: int,
    source_model_id: Optional[str] = None,
    top_k: int,
) -> List[torch.Tensor]:
    ensure_token_ids_model(source_model, context_token_ids)
    aligned_source_by_target = extrapolate_source_layer_alignment(
        source_layer_indices=source_layer_indices,
        target_layer_indices=target_layer_indices,
        num_source_layers=num_source_layers,
        num_target_layers=num_target_layers,
    )
    unique_source_layers = sorted(set(aligned_source_by_target))
    unique_sparse_indices = extract_source_attention_topk_indices(
        source_model,
        context_token_ids,
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



def repeat_key_value_heads(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand grouped-query KV heads from [batch, kv_heads, seq, head_dim] to query heads."""
    if n_rep == 1:
        return hidden_states
    batch_size, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch_size,
        num_key_value_heads,
        n_rep,
        seq_len,
        head_dim,
    )
    return hidden_states.reshape(batch_size, num_key_value_heads * n_rep, seq_len, head_dim)


def get_qwen2_attention_shape(attn: nn.Module, hidden_size: int) -> Tuple[int, int, int, int]:
    config = getattr(attn, "config", None)
    num_query_heads = getattr(attn, "num_heads", None)
    if num_query_heads is None:
        num_query_heads = getattr(attn, "num_attention_heads", None)
    if num_query_heads is None and config is not None:
        num_query_heads = getattr(config, "num_attention_heads", None)
    if num_query_heads is None:
        raise ValueError("Unable to determine Qwen2 attention query head count.")
    num_query_heads = int(num_query_heads)

    num_key_value_heads = getattr(attn, "num_key_value_heads", None)
    if num_key_value_heads is None and config is not None:
        num_key_value_heads = getattr(config, "num_key_value_heads", num_query_heads)
    if num_key_value_heads is None:
        num_key_value_heads = num_query_heads
    num_key_value_heads = int(num_key_value_heads)

    head_dim = int(getattr(attn, "head_dim", hidden_size // num_query_heads))
    if num_query_heads % num_key_value_heads != 0:
        raise ValueError(
            "Qwen2 num_attention_heads must be divisible by num_key_value_heads, "
            f"got {num_query_heads} and {num_key_value_heads}"
        )
    return num_query_heads, num_key_value_heads, num_query_heads // num_key_value_heads, head_dim


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
    query, native_like_key, native_like_value = qkv.split(split_size, dim=2)

    batch_size, seq_len, _ = query.shape
    num_heads = attn.num_heads
    head_dim = attn.head_dim
    expected_cache_shape = (batch_size, num_heads, seq_len, head_dim)

    query = query.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    native_like_key = native_like_key.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    native_like_value = native_like_value.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()

    attention_key = native_like_key if injected_key is None else injected_key
    attention_value = native_like_value if injected_value is None else injected_value
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
    return hidden_states, (native_like_key, native_like_value)


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
    native_like_key = attn.k_proj(hidden_states)
    native_like_value = attn.v_proj(hidden_states)
    query_states = query_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()
    native_like_key = native_like_key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()
    native_like_value = native_like_value.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()

    attention_key = native_like_key if injected_key is None else injected_key
    attention_value = native_like_value if injected_value is None else injected_value
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
    return hidden_states, (native_like_key, native_like_value)



def run_qwen2_block(
    block: nn.Module,
    hidden_states: torch.Tensor,
    *,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
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

    attn = block.self_attn
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_query_heads, num_key_value_heads, num_key_value_groups, head_dim = get_qwen2_attention_shape(attn, hidden_size)
    expected_cache_shape = (batch_size, num_key_value_heads, seq_len, head_dim)

    residual = hidden_states
    attn_input = block.input_layernorm(hidden_states)

    query_states = attn.q_proj(attn_input).view(batch_size, seq_len, num_query_heads, head_dim)
    native_like_key = attn.k_proj(attn_input).view(batch_size, seq_len, num_key_value_heads, head_dim)
    native_like_value = attn.v_proj(attn_input).view(batch_size, seq_len, num_key_value_heads, head_dim)

    # Qwen3 adds RMSNorm on each projected q/k head before RoPE.  Keeping the
    # check structural lets the Qwen2/Qwen2.5 replay path remain unchanged.
    q_norm = getattr(attn, "q_norm", None)
    k_norm = getattr(attn, "k_norm", None)
    if q_norm is not None:
        query_states = q_norm(query_states)
    if k_norm is not None:
        native_like_key = k_norm(native_like_key)

    query_states = query_states.transpose(1, 2).contiguous()
    native_like_key = native_like_key.transpose(1, 2).contiguous()
    native_like_value = native_like_value.transpose(1, 2).contiguous()

    if position_embeddings is not None:
        cos, sin = position_embeddings
        query_states, native_like_key = apply_rotary_pos_emb(query_states, native_like_key, cos, sin)
    else:
        rotary_emb = getattr(attn, "rotary_emb", None)
        if rotary_emb is None:
            raise ValueError("Qwen replay requires rotary embeddings from the decoder or layer self-attention.")
        try:
            cos, sin = rotary_emb(native_like_value, seq_len=seq_len)
        except TypeError:
            cos, sin = rotary_emb(native_like_value, position_ids)
        query_states, native_like_key = apply_rotary_pos_emb(query_states, native_like_key, cos, sin, position_ids)

    attention_key = native_like_key if injected_key is None else injected_key
    attention_value = native_like_value if injected_value is None else injected_value
    if tuple(attention_key.shape) != expected_cache_shape:
        raise ValueError(
            "Attention cache shape mismatch for rotary GQA layer replay: "
            f"expected {expected_cache_shape}, got {tuple(attention_key.shape)}"
        )

    expanded_attention_key = repeat_key_value_heads(attention_key, num_key_value_groups)
    expanded_attention_value = repeat_key_value_heads(attention_value, num_key_value_groups)
    scaling = float(getattr(attn, "scaling", head_dim ** -0.5))

    if sparse_attention_indices is not None:
        sparse_attention_indices = expand_sparse_attention_indices(sparse_attention_indices, num_query_heads)
        selected_key = gather_sparse_sequence_vectors(expanded_attention_key, sparse_attention_indices)
        selected_value = gather_sparse_sequence_vectors(expanded_attention_value, sparse_attention_indices)
        attn_weights = (query_states.unsqueeze(-2) * selected_key).sum(dim=-1) * scaling
        invalid_mask = build_sparse_query_mask(sparse_attention_indices, seq_len=seq_len, num_heads=num_query_heads)
        attn_weights = attn_weights.masked_fill(invalid_mask, torch.finfo(attn_weights.dtype).min)
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        dropout_p = float(getattr(attn, "attention_dropout", getattr(attn, "dropout", 0.0)))
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=block.training)
        attn_output = (attn_weights.unsqueeze(-1) * selected_value).sum(dim=-2)
    else:
        attn_weights = torch.matmul(query_states, expanded_attention_key.transpose(-1, -2)) * scaling
        effective_attention_mask = attention_mask
        sliding_window = getattr(attn, "sliding_window", None)
        if sliding_window is not None and int(sliding_window) > 0:
            query_positions = torch.arange(seq_len, device=hidden_states.device).view(1, 1, seq_len, 1)
            key_positions = torch.arange(seq_len, device=hidden_states.device).view(1, 1, 1, seq_len)
            sliding_mask = key_positions <= (query_positions - int(sliding_window))
            sliding_bias = torch.zeros_like(attn_weights).masked_fill(sliding_mask, torch.finfo(attn_weights.dtype).min)
            effective_attention_mask = effective_attention_mask + sliding_bias if effective_attention_mask is not None else sliding_bias
        if effective_attention_mask is not None:
            attn_weights = attn_weights + effective_attention_mask
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        dropout_p = float(getattr(attn, "attention_dropout", getattr(attn, "dropout", 0.0)))
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=block.training)
        attn_output = torch.matmul(attn_weights, expanded_attention_value)

    attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, num_query_heads * head_dim)
    attn_output = attn.o_proj(attn_output)
    hidden_states = residual + attn_output

    residual = hidden_states
    hidden_states = block.post_attention_layernorm(hidden_states)
    hidden_states = block.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states, (native_like_key, native_like_value)


def run_llama_block(
    block: nn.Module,
    hidden_states: torch.Tensor,
    *,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    sparse_attention_indices: Optional[torch.Tensor] = None,
    injected_key: Optional[torch.Tensor] = None,
    injected_value: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    # Llama 3.2's attention/MLP block is structurally identical for the fields
    # consumed by the Qwen2-family replay helper. Llama-specific RoPE lives on
    # the attention module itself, so the shared replay applies it correctly.
    return run_qwen2_block(
        block,
        hidden_states,
        position_ids=position_ids,
        attention_mask=attention_mask,
        position_embeddings=position_embeddings,
        sparse_attention_indices=sparse_attention_indices,
        injected_key=injected_key,
        injected_value=injected_value,
    )


def run_gemma3_block(
    block: nn.Module,
    hidden_states: torch.Tensor,
    *,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    sparse_attention_indices: Optional[torch.Tensor] = None,
    injected_key: Optional[torch.Tensor] = None,
    injected_value: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Replay one Gemma 3 text block with an optional translated KV cache.

    Gemma 3 shares the rotary GQA projection layout with Qwen3, but its block
    normalization order is different: attention and MLP outputs are each
    post-normalized before the residual addition. Local layers also use a
    different RoPE base and a sliding attention window. Keep those semantics
    isolated here so the existing Qwen/Llama paths are unchanged.
    """
    if (injected_key is None) != (injected_value is None):
        raise ValueError("injected_key and injected_value must be provided together.")
    if injected_key is not None and injected_key.shape != injected_value.shape:
        raise ValueError(
            "Injected key/value must have identical shapes, "
            f"got {tuple(injected_key.shape)} vs {tuple(injected_value.shape)}"
        )

    attn = block.self_attn
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_query_heads, num_key_value_heads, num_key_value_groups, head_dim = get_qwen2_attention_shape(attn, hidden_size)
    expected_cache_shape = (batch_size, num_key_value_heads, seq_len, head_dim)

    residual = hidden_states
    attn_input = block.input_layernorm(hidden_states)

    query_states = attn.q_proj(attn_input).view(batch_size, seq_len, num_query_heads, head_dim)
    native_like_key = attn.k_proj(attn_input).view(batch_size, seq_len, num_key_value_heads, head_dim)
    native_like_value = attn.v_proj(attn_input).view(batch_size, seq_len, num_key_value_heads, head_dim)

    query_states = attn.q_norm(query_states).transpose(1, 2).contiguous()
    native_like_key = attn.k_norm(native_like_key).transpose(1, 2).contiguous()
    native_like_value = native_like_value.transpose(1, 2).contiguous()

    rotary_emb = getattr(attn, "rotary_emb", None)
    if rotary_emb is None:
        raise ValueError("Gemma3 replay requires a layer-local rotary embedding module.")
    try:
        cos, sin = rotary_emb(native_like_value, position_ids=position_ids)
    except TypeError:
        cos, sin = rotary_emb(native_like_value, position_ids)
    query_states, native_like_key = apply_rotary_pos_emb(
        query_states, native_like_key, cos, sin, position_ids
    )

    attention_key = native_like_key if injected_key is None else injected_key
    attention_value = native_like_value if injected_value is None else injected_value
    if tuple(attention_key.shape) != expected_cache_shape:
        raise ValueError(
            "Attention cache shape mismatch for Gemma3 replay: "
            f"expected {expected_cache_shape}, got {tuple(attention_key.shape)}"
        )

    expanded_attention_key = repeat_key_value_heads(attention_key, num_key_value_groups)
    expanded_attention_value = repeat_key_value_heads(attention_value, num_key_value_groups)
    scaling = float(getattr(attn, "scaling", head_dim ** -0.5))
    sliding_window = getattr(attn, "sliding_window", None)

    if sparse_attention_indices is not None:
        sparse_attention_indices = expand_sparse_attention_indices(sparse_attention_indices, num_query_heads)
        selected_key = gather_sparse_sequence_vectors(expanded_attention_key, sparse_attention_indices)
        selected_value = gather_sparse_sequence_vectors(expanded_attention_value, sparse_attention_indices)
        attn_weights = (query_states.unsqueeze(-2) * selected_key).sum(dim=-1) * scaling
        invalid_mask = build_sparse_query_mask(
            sparse_attention_indices, seq_len=seq_len, num_heads=num_query_heads
        )
        if sliding_window is not None and int(sliding_window) > 0:
            query_positions = torch.arange(seq_len, device=hidden_states.device).view(1, 1, seq_len, 1)
            outside_window = sparse_attention_indices <= (query_positions - int(sliding_window))
            invalid_mask = invalid_mask | outside_window
        attn_weights = attn_weights.masked_fill(invalid_mask, torch.finfo(attn_weights.dtype).min)
    else:
        attn_weights = torch.matmul(query_states, expanded_attention_key.transpose(-1, -2)) * scaling
        effective_attention_mask = attention_mask
        if sliding_window is not None and int(sliding_window) > 0:
            query_positions = torch.arange(seq_len, device=hidden_states.device).view(1, 1, seq_len, 1)
            key_positions = torch.arange(seq_len, device=hidden_states.device).view(1, 1, 1, seq_len)
            outside_window = key_positions <= (query_positions - int(sliding_window))
            sliding_bias = torch.zeros_like(attn_weights).masked_fill(
                outside_window, torch.finfo(attn_weights.dtype).min
            )
            effective_attention_mask = (
                effective_attention_mask + sliding_bias
                if effective_attention_mask is not None
                else sliding_bias
            )
        if effective_attention_mask is not None:
            attn_weights = attn_weights + effective_attention_mask

    softcap = getattr(attn, "attn_logit_softcapping", None)
    if softcap is not None:
        softcap = float(softcap)
        attn_weights = torch.tanh(attn_weights / softcap) * softcap
    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    dropout_p = float(getattr(attn, "attention_dropout", getattr(attn, "dropout", 0.0)))
    attn_weights = F.dropout(attn_weights, p=dropout_p, training=block.training)

    if sparse_attention_indices is not None:
        attn_output = (attn_weights.unsqueeze(-1) * selected_value).sum(dim=-2)
    else:
        attn_output = torch.matmul(attn_weights, expanded_attention_value)

    attn_output = attn_output.transpose(1, 2).contiguous().reshape(
        batch_size, seq_len, num_query_heads * head_dim
    )
    attn_output = attn.o_proj(attn_output)
    hidden_states = block.post_attention_layernorm(attn_output)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = block.pre_feedforward_layernorm(hidden_states)
    hidden_states = block.mlp(hidden_states)
    hidden_states = block.post_feedforward_layernorm(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states, (native_like_key, native_like_value)


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
    target_model: Model,
    context_token_ids: TokenIDs,
    target_layer_indices: List[int],
    injected_key_block: torch.Tensor,
    injected_value_block: torch.Tensor,
    tgt_spec: ModelSpec,
    target_model_id: Optional[str] = None,
    sparse_attention_indices: Optional[List[torch.Tensor]] = None,
    num_bottom_full_attn: int = 3,
    cache_injected_window: bool = False,
) -> PastKeyValues:
    ensure_token_ids_model(target_model, context_token_ids)
    injected_window = blocks_to_partial_past_key_values(
        key_block=injected_key_block,
        value_block=injected_value_block,
        num_heads=tgt_spec.num_key_value_heads,
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
            return build_gpt2_input_hidden_states(target_model, context_token_ids), None, None, None

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
            hidden_states, token_attention_mask, attention_mask = build_opt_input_hidden_states(target_model, context_token_ids)
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

    elif model_family == "qwen2":
        qwen_model = require_qwen2_model(target_model)
        target_blocks = qwen_model.layers

        def build_initial_hidden_states() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
            return build_qwen2_input_hidden_states(target_model, context_token_ids)

        def run_block(
            block: nn.Module,
            hidden_states: torch.Tensor,
            position_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
            sparse_attention_indices: Optional[torch.Tensor],
            injected_key: Optional[torch.Tensor] = None,
            injected_value: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            return run_qwen2_block(
                block,
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                sparse_attention_indices=sparse_attention_indices,
                injected_key=injected_key,
                injected_value=injected_value,
            )

    elif model_family == "llama":
        llama_model = require_llama_model(target_model)
        target_blocks = llama_model.layers

        def build_initial_hidden_states() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
            return build_llama_input_hidden_states(target_model, context_token_ids)

        def run_block(
            block: nn.Module,
            hidden_states: torch.Tensor,
            position_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
            sparse_attention_indices: Optional[torch.Tensor],
            injected_key: Optional[torch.Tensor] = None,
            injected_value: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            return run_llama_block(
                block,
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                sparse_attention_indices=sparse_attention_indices,
                injected_key=injected_key,
                injected_value=injected_value,
            )

    elif model_family == "gemma3":
        gemma_model = require_gemma3_model(target_model)
        target_blocks = gemma_model.layers

        def build_initial_hidden_states() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
            return build_gemma3_input_hidden_states(target_model, context_token_ids)

        def run_block(
            block: nn.Module,
            hidden_states: torch.Tensor,
            position_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            _: Any,
            sparse_attention_indices: Optional[torch.Tensor],
            injected_key: Optional[torch.Tensor] = None,
            injected_value: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            return run_gemma3_block(
                block,
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
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
        # run_block uses injected_key/injected_value for attention, but returns the
        # native-like KV cache that should be used by training/loss paths. Heatmap
        # analysis can opt in to caching the injected KV itself so the plotted
        # "full_mix" cache reflects the translated window rather than native-like KV.
        rebuilt_past.append(injected_present if cache_injected_window else present)
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
) -> TranslatorPool:
    config = ctx.config
    resolve_channels(ctx)
    translator_pool = initialize_translators(
        ctx=ctx,
        translator_dim=config.translator_dim,
        translator_heads=config.translator_heads,
        translator_depth=config.translator_depth,
        mlp_ratio=config.translator_mlp_ratio,
        variant=config.variant,
        mot_num_translators=config.mot_num_translators,
        mot_top_k=config.mot_top_k,
    )
    move_trainable_module_to_config_dtype(translator_pool, config)
    return translator_pool


def load_translator_pool_from_checkpoint(
    checkpoint_dir_path: str,
    device_override: Optional[str] = None,
) -> Tuple[
    Context,
    TranslatorPool,
]:
    checkpoint_dir_path_obj = Path(checkpoint_dir_path)
    if not checkpoint_dir_path_obj.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir_path_obj}")
    train_config_path = get_train_config_path(checkpoint_dir_path_obj)
    if not train_config_path.exists():
        raise FileNotFoundError(f"Train config not found under checkpoint directory: {checkpoint_dir_path}")

    config = TrainConfig(**read_json(train_config_path))
    if device_override is not None:
        config.device = device_override
    ctx = Context(config)
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
    load_translator_checkpoints(checkpoint_dir_path_obj, translator_pool)
    move_trainable_module_to_config_dtype(translator_pool, config)
    translator_pool.eval()
    return ctx, translator_pool

def run_train(ctx: Context) -> Path:
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
        spec = ctx.tp.get_model_spec(node.id)
        logging.info(
            "  %s (%s): layers=%d, hidden=%d, heads=%d",
            node.id,
            node.model_id,
            spec.num_layers,
            spec.hidden_size,
            spec.num_heads,
        )
    logging.info("[Setup] trainable translator params = %s", f"{count_trainable_parameters(translator_pool):,}")

    training_dataloaders = build_training_dataloaders(ctx)

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
            past_by_node_id, batches_by_node_id = build_step_pasts_and_batches(
                ctx,
                training_dataloaders,
            )

            total_direction_loss = 0.0
            for edge in edges:
                target_context_token_ids, prompt_token_ids, label_token_ids = batches_by_node_id[edge.tgt_id]
                mixed_target_past, _ = build_replayed_target_past_from_target_context(
                    ctx,
                    target_context_token_ids=target_context_token_ids,
                    source_model=ctx.tp.get_model(edge.src_id),
                    target_model=ctx.tp.get_model(edge.tgt_id),
                    src_node_id=edge.src_id,
                    tgt_node_id=edge.tgt_id,
                    tgt_spec=ctx.tp.get_model_spec(edge.tgt_id),
                )
                direction_loss = compute_prefix_correction_and_suffix_lm_loss(
                    target_model=ctx.tp.get_model(edge.tgt_id),
                    past_key_values=mixed_target_past,
                    prompt_token_ids=prompt_token_ids,
                    label_token_ids=label_token_ids,
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
            logging.info(
                "[Step %04d] loss=%.4f | gate_importance_cv2=%.4f | gate_load_cv2=%.4f | gate_importance_entropy=%.4f | lr=%.2e",
                step,
                avg_loss,
                avg_gate_importance_cv2,
                avg_gate_load_cv2,
                avg_gate_importance_entropy,
                scheduler.lr,
            )
            running_loss = 0.0
            running_gate_importance_cv2 = 0.0
            running_gate_load_cv2 = 0.0
            running_gate_importance_entropy = 0.0

    final_path = get_train_checkpoint_path(output_path)
    save_translator_checkpoints(output_path, translator_pool)
    logging.info("[Done] final translator checkpoints saved to %s", final_path)
    logging.info("Saved train log to %s", log_path)
    return final_path
