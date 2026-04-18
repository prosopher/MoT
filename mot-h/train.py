from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from core.channel_manager import (
    ChannelManager,
    build_resolved_channels_path,
    load_resolved_channels,
    save_resolved_channels,
)
from core.channel_profiler import ChannelProfiler, load_channel_profile_config
from core.context import Context
from core.model_manager import ModelManager
from core.model_spec import ModelSpec
from core.train_util import *
from mot.train import (
    TrainConfig,
    require_channel_profiler,
    require_gpt2_transformer,
    replay_target_prefill_with_injected_window,
    resolve_channels,
    uses_channel_alignment,
)




class CrossLayerWindowTranslator(nn.Module):
    """
    Local MOT-H copy of the recurrent cross-attention translator, specialized for
    hidden-state translation.

    Compared with mot.train.CrossLayerWindowTranslator, this variant keeps the
    output in a signed continuous hidden-state space and uses independent
    per-layer output heads for calibration.
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
        # Keep per-layer calibration heads independent so each translated target
        # attention-input slot can preserve its own scale/bias statistics.
        self.output_norms = nn.ModuleList(
            [nn.LayerNorm(translator_dim) for _ in range(num_layers)]
        )
        self.output_projs = nn.ModuleList(
            [nn.Linear(translator_dim, tgt_hidden_size) for _ in range(num_layers)]
        )

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
        projected = self.input_proj(layer_window_cache)

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

        translated_layers = []
        for layer_idx, layer_hidden in enumerate(collected):
            calibrated = self.output_norms[layer_idx](layer_hidden)
            translated_layers.append(self.output_projs[layer_idx](calibrated))
        return torch.stack(translated_layers, dim=2)


class PerLayerPerHeadKVAdapter(nn.Module):
    def __init__(self, num_layers: int, num_heads: int, head_dim: int) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if num_heads < 1:
            raise ValueError("num_heads must be >= 1")
        if head_dim < 1:
            raise ValueError("head_dim must be >= 1")
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        shape = (num_layers, num_heads, head_dim)
        self.key_scale = nn.Parameter(torch.ones(shape))
        self.key_bias = nn.Parameter(torch.zeros(shape))
        self.value_scale = nn.Parameter(torch.ones(shape))
        self.value_bias = nn.Parameter(torch.zeros(shape))

    def forward(
        self,
        key_block: torch.Tensor,
        value_block: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if key_block.shape != value_block.shape:
            raise ValueError(
                "key_block and value_block must have identical shapes, "
                f"got {tuple(key_block.shape)} vs {tuple(value_block.shape)}"
            )
        if key_block.ndim != 4:
            raise ValueError(
                "PerLayerPerHeadKVAdapter expects [batch, seq, num_layers, hidden], "
                f"got {tuple(key_block.shape)}"
            )
        batch_size, seq_len, num_layers, hidden_size = key_block.shape
        expected_hidden = self.num_heads * self.head_dim
        if num_layers != self.num_layers:
            raise ValueError(
                f"Adapter expected {self.num_layers} layers, got {num_layers}"
            )
        if hidden_size != expected_hidden:
            raise ValueError(
                f"Adapter expected hidden size {expected_hidden}, got {hidden_size}"
            )

        key = key_block.view(batch_size, seq_len, num_layers, self.num_heads, self.head_dim)
        value = value_block.view(batch_size, seq_len, num_layers, self.num_heads, self.head_dim)
        key = key * self.key_scale.unsqueeze(0).unsqueeze(0) + self.key_bias.unsqueeze(0).unsqueeze(0)
        value = value * self.value_scale.unsqueeze(0).unsqueeze(0) + self.value_bias.unsqueeze(0).unsqueeze(0)
        return (
            key.view(batch_size, seq_len, num_layers, expected_hidden),
            value.view(batch_size, seq_len, num_layers, expected_hidden),
        )


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


def build_hidden_state_window_translator(
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
) -> nn.Module:
    if variant == "single":
        return CrossLayerWindowTranslator(
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
        )
    raise ValueError(f"Unsupported MOT variant: {variant}")

class LayerWindowAttnInputTranslator(nn.Module):
    """
    Attention-input variant of the window translator.

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
        variant: str,
        mot_num_translators: int,
        mot_top_k: int,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.num_layers = num_layers
        self.attn_input_translator = build_hidden_state_window_translator(
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
        )

    def forward(self, attn_input_block: torch.Tensor) -> torch.Tensor:
        if attn_input_block.ndim != 4:
            raise ValueError(
                "Layer-window hidden states must have shape [batch, seq, num_layers, hidden], "
                f"got {tuple(attn_input_block.shape)}"
            )
        if attn_input_block.shape[2] != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} layers in the translation window, got {attn_input_block.shape[2]}"
            )
        return self.attn_input_translator(attn_input_block)


@torch.no_grad()
def extract_model_prefill_artifacts(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
) -> Tuple[PastKeyValues, Tuple[torch.Tensor, ...]]:
    outputs = model(
        input_ids=input_ids,
        use_cache=True,
        output_hidden_states=True,
    )
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise ValueError("Model forward did not return hidden_states; expected GPT-2 style outputs.")
    return outputs.past_key_values, tuple(hidden_states)


def extract_selected_layer_canonical_attn_input_block(
    model: PreTrainedModel,
    hidden_states: Sequence[torch.Tensor],
    layer_indices: Sequence[int],
) -> torch.Tensor:
    if not layer_indices:
        raise ValueError("layer_indices must contain at least one layer.")
    max_valid_layer_idx = len(hidden_states) - 2
    if max_valid_layer_idx < 0:
        raise ValueError("hidden_states must include at least embeddings and one transformer layer state.")

    transformer = require_gpt2_transformer(model)
    selected_layers = []
    for layer_idx in layer_indices:
        layer_idx = int(layer_idx)
        if not (0 <= layer_idx <= max_valid_layer_idx):
            raise ValueError(
                f"layer_idx={layer_idx} is outside [0, {max_valid_layer_idx}] for hidden_states tuple of length {len(hidden_states)}"
            )
        ln_1 = transformer.h[layer_idx].ln_1
        selected_layers.append(
            F.layer_norm(
                hidden_states[layer_idx],
                ln_1.normalized_shape,
                weight=None,
                bias=None,
                eps=ln_1.eps,
            )
        )
    return torch.stack(selected_layers, dim=2)


def decanonicalize_gpt2_attn_inputs(
    block: nn.Module,
    canonical_attn_inputs: torch.Tensor,
) -> torch.Tensor:
    if canonical_attn_inputs.ndim != 3:
        raise ValueError(
            "canonical_attn_inputs must have shape [batch, seq, hidden], "
            f"got {tuple(canonical_attn_inputs.shape)}"
        )

    ln_1 = block.ln_1
    weight = getattr(ln_1, "weight", None)
    if weight is None:
        raise ValueError("GPT-2 ln_1 is expected to expose affine weight for de-canonicalization.")
    bias = getattr(ln_1, "bias", None)
    attn_inputs = canonical_attn_inputs * weight.view(1, 1, -1)
    if bias is not None:
        attn_inputs = attn_inputs + bias.view(1, 1, -1)
    return attn_inputs


def reconstruct_gpt2_kv_from_attn_inputs(
    block: nn.Module,
    attn_inputs: torch.Tensor,
    *,
    attn_input_scale: Optional[torch.Tensor] = None,
    attn_input_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if attn_inputs.ndim != 3:
        raise ValueError(
            "attn_inputs must have shape [batch, seq, hidden], "
            f"got {tuple(attn_inputs.shape)}"
        )

    attn = block.attn
    if attn_input_scale is not None:
        if attn_input_scale.ndim != 1 or attn_input_scale.shape[0] != attn_inputs.shape[-1]:
            raise ValueError(
                "attn_input_scale must have shape [hidden], "
                f"got {tuple(attn_input_scale.shape)} for hidden={attn_inputs.shape[-1]}"
            )
        attn_inputs = attn_inputs * attn_input_scale.view(1, 1, -1)
    if attn_input_bias is not None:
        if attn_input_bias.ndim != 1 or attn_input_bias.shape[0] != attn_inputs.shape[-1]:
            raise ValueError(
                "attn_input_bias must have shape [hidden], "
                f"got {tuple(attn_input_bias.shape)} for hidden={attn_inputs.shape[-1]}"
            )
        attn_inputs = attn_inputs + attn_input_bias.view(1, 1, -1)
    qkv = attn.c_attn(attn_inputs)
    split_size = getattr(attn, "split_size", qkv.shape[-1] // 3)
    _, key, value = qkv.split(split_size, dim=2)

    batch_size, seq_len, _ = key.shape
    num_heads = attn.num_heads
    head_dim = attn.head_dim

    key = key.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    value = value.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    return key, value


def reconstruct_kv_block_from_attn_input_block(
    target_model: PreTrainedModel,
    attn_input_block: torch.Tensor,
    target_layer_indices: Sequence[int],
    tgt_spec: ModelSpec,
    *,
    kv_adapter: Optional[PerLayerPerHeadKVAdapter] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if attn_input_block.ndim != 4:
        raise ValueError(
            "attn_input_block must have shape [batch, seq, num_layers, hidden], "
            f"got {tuple(attn_input_block.shape)}"
        )
    if attn_input_block.shape[2] != len(target_layer_indices):
        raise ValueError(
            "Number of translated attention-input layers must match target_layer_indices, "
            f"got {attn_input_block.shape[2]} vs {len(target_layer_indices)}"
        )

    transformer = require_gpt2_transformer(target_model)
    reconstructed_layers = []
    for slot_idx, layer_idx in enumerate(target_layer_indices):
        target_block = transformer.h[layer_idx]
        target_attn_inputs = decanonicalize_gpt2_attn_inputs(
            target_block,
            attn_input_block[:, :, slot_idx, :],
        )
        key, value = reconstruct_gpt2_kv_from_attn_inputs(
            target_block,
            target_attn_inputs,
        )
        reconstructed_layers.append((key, value))
    key_block, value_block = past_key_values_to_blocks(tuple(reconstructed_layers))
    if kv_adapter is not None:
        key_block, value_block = kv_adapter(key_block, value_block)
    return key_block, value_block


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

        adapters = {}
        kv_adapters = {}
        for edge in self.edges:
            channels = self.cm.get_channels(edge.id)
            adapters[edge.id] = LayerWindowAttnInputTranslator(
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
            kv_adapters[edge.id] = PerLayerPerHeadKVAdapter(
                num_layers=len(channels),
                num_heads=self.mm.get_model_spec(edge.tgt_id).num_heads,
                head_dim=self.mm.get_model_spec(edge.tgt_id).head_dim,
            )
        self.adapters = nn.ModuleDict(adapters)
        self.kv_adapters = nn.ModuleDict(kv_adapters)

    def translate_layer_window(
        self,
        source_attn_input_block: torch.Tensor,
        src_node_id: str,
        tgt_node_id: str,
    ) -> torch.Tensor:
        edge_id = f"{src_node_id}_to_{tgt_node_id}"
        if edge_id not in self.adapters:
            raise ValueError(
                f"Translator edge {edge_id} is not available. "
                f"Active edges: {list(self.edge_ids)}"
            )
        return self.adapters[edge_id](source_attn_input_block)

    def build_replayed_target_past(
        self,
        *,
        source_past_key_values: PastKeyValues,
        prefix_input_ids: torch.Tensor,
        target_model: PreTrainedModel,
        src_node_id: str,
        tgt_node_id: str,
        tgt_spec: ModelSpec,
        source_canonical_attn_input_block: Optional[torch.Tensor] = None,
    ) -> Tuple[PastKeyValues, PastKeyValues]:
        del source_past_key_values  # Interface compatibility with other translator pools.

        edge_id = f"{src_node_id}_to_{tgt_node_id}"
        target_layer_indices = self.cm.get_tgt_layer_indices(edge_id)

        if source_canonical_attn_input_block is None:
            _, source_hidden_states = extract_model_prefill_artifacts(
                self.mm.get_model(src_node_id),
                prefix_input_ids,
            )
            source_canonical_attn_input_block = extract_selected_layer_canonical_attn_input_block(
                self.mm.get_model(src_node_id),
                source_hidden_states,
                self.cm.get_src_layer_indices(edge_id),
            )

        translated_attn_input_block = self.translate_layer_window(
            source_attn_input_block=source_canonical_attn_input_block,
            src_node_id=src_node_id,
            tgt_node_id=tgt_node_id,
        )
        translated_key, translated_value = reconstruct_kv_block_from_attn_input_block(
            target_model=target_model,
            attn_input_block=translated_attn_input_block,
            target_layer_indices=target_layer_indices,
            tgt_spec=tgt_spec,
            kv_adapter=self.kv_adapters[edge_id],
        )
        translated_window_past = blocks_to_partial_past_key_values(
            key_block=translated_key,
            value_block=translated_value,
            num_heads=tgt_spec.num_heads,
            head_dim=tgt_spec.head_dim,
        )
        mixed_target_past = replay_target_prefill_with_injected_window(
            target_model=target_model,
            prefix_input_ids=prefix_input_ids,
            target_layer_indices=target_layer_indices,
            injected_key_block=translated_key,
            injected_value_block=translated_value,
            tgt_spec=tgt_spec,
        )
        return mixed_target_past, translated_window_past



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
    models, tokenizer = build_models_and_tokenizer(config, nodes)
    ctx = Context(
        config,
        nodes,
        edges,
        ModelManager(models),
        tokenizer,
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
    logger = setup_logger(f"{config.alg}_train", log_path)
    logger.info("Starting canonicalized attn-input translator training")
    logger.info("train_config=%s", asdict(config))

    logger.info("nodes=%s", [asdict(node) for node in nodes])
    logger.info("edges=%s", [edge.id for edge in edges])
    logger.info("[Setup] device=%s", config.device)
    logger.info("[Setup] loading models: %s", {node.id: node.model_id for node in nodes})
    translator_pool = build_translator_pool(ctx)
    save_resolved_channels(output_path, ctx.cm, edges)
    translator_pool.train()

    logger.info("[Setup] full model specs")
    for node in nodes:
        spec = ctx.mm.get_model_spec(node.id)
        logger.info(
            "  %s (%s): layers=%d, hidden=%d, heads=%d",
            node.id,
            node.model_id,
            spec.num_layers,
            spec.hidden_size,
            spec.num_heads,
        )
    logger.info("[Setup] trainable translator params = %s", f"{count_trainable_parameters(translator_pool):,}")
    logger.info("[Setup] translation_mode=translate_canonical_attn_input_window_and_restore_target_kv")

    dataloader = build_training_dataloader(ctx)

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
                prefill_by_node_id = {
                    node.id: extract_model_prefill_artifacts(ctx.mm.get_model(node.id), prefix_cache_ids)
                    for node in nodes
                }
                past_by_node_id = {
                    node.id: prefill_by_node_id[node.id][0]
                    for node in nodes
                }
                hidden_states_by_node_id = {
                    node.id: prefill_by_node_id[node.id][1]
                    for node in nodes
                }

            total_direction_loss = 0.0
            for edge in edges:
                source_attn_input_block = extract_selected_layer_canonical_attn_input_block(
                    ctx.mm.get_model(edge.src_id),
                    hidden_states_by_node_id[edge.src_id],
                    ctx.cm.get_src_layer_indices(edge.id),
                )
                mixed_target_past, _ = translator_pool.build_replayed_target_past(
                    source_past_key_values=past_by_node_id[edge.src_id],
                    prefix_input_ids=prefix_cache_ids,
                    target_model=ctx.mm.get_model(edge.tgt_id),
                    src_node_id=edge.src_id,
                    tgt_node_id=edge.tgt_id,
                    tgt_spec=ctx.mm.get_model_spec(edge.tgt_id),
                    source_canonical_attn_input_block=source_attn_input_block,
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
        if step % config.log_every == 0:
            avg_loss = running_loss / config.log_every
            progress_bar.set_postfix(
                loss=f"{avg_loss:.4f}",
                lr=f"{scheduler.lr:.2e}",
            )
            gpu_memory = gpu_memory_tracker.summary()
            logger.info(
                "[Step %04d] attn_input_window_suffix_lm_loss=%.4f | lr=%.2e | gpu_mem_avg=%s | gpu_mem_peak=%s",
                step,
                avg_loss,
                scheduler.lr,
                gpu_memory["avg_allocated_pretty"],
                gpu_memory["peak_allocated_pretty"],
            )
            running_loss = 0.0

    final_path = get_train_checkpoint_path(output_path)
    save_checkpoint(
        output_path=final_path,
        translator_pool=translator_pool,
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
