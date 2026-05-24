from dataclasses import asdict
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
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
from alg.mot.train import (
    CrossLayerWindowTranslator,
    TrainConfig,
    build_window_translator,
    collect_mot_balance_metrics,
    require_channel_profiler,
    resolve_channels,
    uses_channel_alignment,
)



class LayerWindowIntermediateActivationTranslator(nn.Module):
    """
    Intermediate-activation variant of the window translator.

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
        self.activation_translator = build_window_translator(
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

    def forward(self, activation_block: torch.Tensor) -> torch.Tensor:
        if activation_block.ndim != 4:
            raise ValueError(
                "Layer-window intermediate activations must have shape [batch, seq, num_layers, hidden], "
                f"got {tuple(activation_block.shape)}"
            )
        if activation_block.shape[2] != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} layers in the translation window, got {activation_block.shape[2]}"
            )
        return self.activation_translator(activation_block)


_LAYER_CONTAINER_PATHS = (
    ("transformer", "h"),
    ("transformer", "layers"),
    ("transformer", "blocks"),
    ("model", "layers"),
    ("model", "decoder", "layers"),
    ("decoder", "layers"),
    ("gpt_neox", "layers"),
    ("model", "gpt_neox", "layers"),
    ("backbone", "layers"),
    ("language_model", "model", "layers"),
)


def _get_nested_attr(root: Any, path: Sequence[str]) -> Optional[Any]:
    current = root
    for name in path:
        current = getattr(current, name, None)
        if current is None:
            return None
    return current


def _as_module_sequence(candidate: Any) -> Optional[Tuple[nn.Module, ...]]:
    if isinstance(candidate, (nn.ModuleList, list, tuple)) and all(isinstance(layer, nn.Module) for layer in candidate):
        return tuple(candidate)
    return None


def resolve_transformer_layers(model: PreTrainedModel) -> Tuple[nn.Module, ...]:
    for path in _LAYER_CONTAINER_PATHS:
        layers = _as_module_sequence(_get_nested_attr(model, path))
        if layers:
            return layers

    config = getattr(model, "config", None)
    expected_num_layers = None
    for field_name in ("num_hidden_layers", "n_layer", "n_layers"):
        value = getattr(config, field_name, None)
        if value is not None:
            expected_num_layers = int(value)
            break
    if expected_num_layers is not None:
        for _, module in model.named_modules():
            layers = _as_module_sequence(module)
            if layers and len(layers) == expected_num_layers:
                return layers

    raise ValueError(
        "Unable to locate a transformer layer stack. Expected one of: "
        + ", ".join(".".join(path) for path in _LAYER_CONTAINER_PATHS)
    )


def _read_layer_hidden_from_hook_args(
    args: Tuple[Any, ...],
    kwargs: dict,
    *,
    layer_idx: int,
) -> torch.Tensor:
    if args and torch.is_tensor(args[0]):
        hidden_states = args[0]
    elif "hidden_states" in kwargs and torch.is_tensor(kwargs["hidden_states"]):
        hidden_states = kwargs["hidden_states"]
    else:
        raise ValueError(
            f"Unable to read hidden_states from transformer layer {layer_idx} hook inputs."
        )
    if hidden_states.ndim != 3:
        raise ValueError(
            "Intermediate activations must have shape [batch, seq, hidden], "
            f"got {tuple(hidden_states.shape)} at layer {layer_idx}"
        )
    return hidden_states


def _replace_layer_hidden_in_hook_args(
    args: Tuple[Any, ...],
    kwargs: dict,
    replacement: torch.Tensor,
) -> Tuple[Tuple[Any, ...], dict]:
    if args and torch.is_tensor(args[0]):
        return (replacement, *args[1:]), kwargs
    if "hidden_states" in kwargs and torch.is_tensor(kwargs["hidden_states"]):
        new_kwargs = dict(kwargs)
        new_kwargs["hidden_states"] = replacement
        return args, new_kwargs
    raise ValueError("Unable to replace hidden_states in transformer layer hook inputs.")


@torch.no_grad()
def extract_model_prefill_artifacts(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
) -> Tuple[PastKeyValues, Tuple[torch.Tensor, ...]]:
    """Return native KV cache plus per-layer intermediate activations.

    The collected activation for layer i is the tensor passed into that
    transformer layer, matching HCache's layer-input activation view rather
    than HuggingFace's optional output_hidden_states tuple.
    """
    layers = resolve_transformer_layers(model)
    intermediate_activations: List[Optional[torch.Tensor]] = [None] * len(layers)
    handles = []

    def make_hook(layer_idx: int):
        def hook(module: nn.Module, args: Tuple[Any, ...], kwargs: dict):
            del module
            hidden_states = _read_layer_hidden_from_hook_args(args, kwargs, layer_idx=layer_idx)
            intermediate_activations[layer_idx] = hidden_states.detach()
            return None

        return hook

    for layer_idx, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True))
    try:
        outputs = model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()

    past_key_values = getattr(outputs, "past_key_values", None)
    if past_key_values is None:
        raise ValueError("Model forward did not return past_key_values.")
    missing = [idx for idx, activation in enumerate(intermediate_activations) if activation is None]
    if missing:
        raise ValueError(f"Failed to collect intermediate activations for layers: {missing}")
    return tuple(past_key_values), tuple(activation for activation in intermediate_activations if activation is not None)


def extract_selected_layer_intermediate_activation_block(
    intermediate_activations: Sequence[torch.Tensor],
    layer_indices: Sequence[int],
) -> torch.Tensor:
    if not layer_indices:
        raise ValueError("layer_indices must contain at least one layer.")
    max_valid_layer_idx = len(intermediate_activations) - 1
    if max_valid_layer_idx < 0:
        raise ValueError("intermediate_activations must include at least one transformer layer input.")

    selected_layers = []
    for layer_idx in layer_indices:
        layer_idx = int(layer_idx)
        if not (0 <= layer_idx <= max_valid_layer_idx):
            raise ValueError(
                f"layer_idx={layer_idx} is outside [0, {max_valid_layer_idx}] "
                f"for intermediate activation tuple of length {len(intermediate_activations)}"
            )
        selected_layers.append(intermediate_activations[layer_idx])
    return torch.stack(selected_layers, dim=2)


def build_partial_past_from_layer_indices(
    past_key_values: PastKeyValues,
    layer_indices: Sequence[int],
    *,
    num_key_value_heads: int,
    head_dim: int,
) -> PastKeyValues:
    selected_past = tuple(past_key_values[int(layer_idx)] for layer_idx in layer_indices)
    key_block, value_block = past_key_values_to_blocks(selected_past)
    return blocks_to_partial_past_key_values(
        key_block=key_block,
        value_block=value_block,
        num_heads=num_key_value_heads,
        head_dim=head_dim,
    )


def prefill_target_with_injected_intermediate_activations(
    *,
    target_model: PreTrainedModel,
    prefix_input_ids: torch.Tensor,
    target_layer_indices: Sequence[int],
    injected_activation_block: torch.Tensor,
) -> PastKeyValues:
    if injected_activation_block.ndim != 4:
        raise ValueError(
            "injected_activation_block must have shape [batch, seq, num_layers, hidden], "
            f"got {tuple(injected_activation_block.shape)}"
        )
    if injected_activation_block.shape[2] != len(target_layer_indices):
        raise ValueError(
            "Number of injected activation layers must match target_layer_indices, "
            f"got {injected_activation_block.shape[2]} vs {len(target_layer_indices)}"
        )

    layers = resolve_transformer_layers(target_model)
    slot_by_layer_idx = {}
    for slot_idx, layer_idx in enumerate(target_layer_indices):
        layer_idx = int(layer_idx)
        if not (0 <= layer_idx < len(layers)):
            raise ValueError(
                f"target layer_idx={layer_idx} is outside [0, {len(layers) - 1}]"
            )
        slot_by_layer_idx[layer_idx] = slot_idx

    handles = []

    def make_hook(layer_idx: int, slot_idx: int):
        def hook(module: nn.Module, args: Tuple[Any, ...], kwargs: dict):
            del module
            native_hidden_states = _read_layer_hidden_from_hook_args(args, kwargs, layer_idx=layer_idx)
            injected = injected_activation_block[:, :, slot_idx, :].to(
                device=native_hidden_states.device,
                dtype=native_hidden_states.dtype,
            )
            if tuple(injected.shape) != tuple(native_hidden_states.shape):
                raise ValueError(
                    "Injected intermediate activation shape mismatch at target layer "
                    f"{layer_idx}: expected {tuple(native_hidden_states.shape)}, got {tuple(injected.shape)}"
                )
            return _replace_layer_hidden_in_hook_args(args, kwargs, injected)

        return hook

    for layer_idx, slot_idx in slot_by_layer_idx.items():
        handles.append(layers[layer_idx].register_forward_pre_hook(make_hook(layer_idx, slot_idx), with_kwargs=True))
    try:
        outputs = target_model(
            input_ids=prefix_input_ids,
            use_cache=True,
            return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()

    past_key_values = getattr(outputs, "past_key_values", None)
    if past_key_values is None:
        raise ValueError("Target model forward did not return past_key_values.")
    return tuple(past_key_values)


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
            adapters[edge.id] = LayerWindowIntermediateActivationTranslator(
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

    def translate_layer_window(
        self,
        source_intermediate_activation_block: torch.Tensor,
        src_node_id: str,
        tgt_node_id: str,
    ) -> torch.Tensor:
        edge_id = f"{src_node_id}_to_{tgt_node_id}"
        if edge_id not in self.adapters:
            raise ValueError(
                f"Translator edge {edge_id} is not available. "
                f"Active edges: {list(self.edge_ids)}"
            )
        return self.adapters[edge_id](source_intermediate_activation_block)

    def build_replayed_target_past(
        self,
        *,
        source_past_key_values: PastKeyValues,
        prefix_input_ids: torch.Tensor,
        target_model: PreTrainedModel,
        src_node_id: str,
        tgt_node_id: str,
        tgt_spec: ModelSpec,
        source_intermediate_activation_block: Optional[torch.Tensor] = None,
    ) -> Tuple[PastKeyValues, PastKeyValues]:
        del source_past_key_values  # Interface compatibility with other translator pools.

        edge_id = f"{src_node_id}_to_{tgt_node_id}"
        target_layer_indices = self.cm.get_tgt_layer_indices(edge_id)

        if source_intermediate_activation_block is None:
            _, source_intermediate_activations = extract_model_prefill_artifacts(
                self.mm.get_model(src_node_id),
                prefix_input_ids,
            )
            source_intermediate_activation_block = extract_selected_layer_intermediate_activation_block(
                source_intermediate_activations,
                self.cm.get_src_layer_indices(edge_id),
            )

        translated_activation_block = self.translate_layer_window(
            source_intermediate_activation_block=source_intermediate_activation_block,
            src_node_id=src_node_id,
            tgt_node_id=tgt_node_id,
        )
        mixed_target_past = prefill_target_with_injected_intermediate_activations(
            target_model=target_model,
            prefix_input_ids=prefix_input_ids,
            target_layer_indices=target_layer_indices,
            injected_activation_block=translated_activation_block,
        )
        translated_window_past = build_partial_past_from_layer_indices(
            mixed_target_past,
            target_layer_indices,
            num_key_value_heads=tgt_spec.num_key_value_heads,
            head_dim=tgt_spec.head_dim,
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
    move_trainable_module_to_config_dtype(translator_pool, config)
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
    move_trainable_module_to_config_dtype(translator_pool, config)
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
    logging.info("Starting intermediate-activation translator training")
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
    logging.info("[Setup] translation_mode=translate_intermediate_activation_window_and_restore_target_kv")

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
                    prefill_by_node_id = {
                        node.id: extract_model_prefill_artifacts(ctx.mm.get_model(node.id), prefix_cache_ids)
                        for node in nodes
                    }
                    past_by_node_id = {
                        node.id: prefill_by_node_id[node.id][0]
                        for node in nodes
                    }
                    intermediate_activations_by_node_id = {
                        node.id: prefill_by_node_id[node.id][1]
                        for node in nodes
                    }
                target_batches[target_node_id] = (prefix_cache_ids, lm_input_ids, lm_labels, past_by_node_id, intermediate_activations_by_node_id)

            total_direction_loss = 0.0
            for edge in edges:
                prefix_cache_ids, lm_input_ids, lm_labels, past_by_node_id, intermediate_activations_by_node_id = target_batches[edge.tgt_id]
                source_intermediate_activation_block = extract_selected_layer_intermediate_activation_block(
                    intermediate_activations_by_node_id[edge.src_id],
                    ctx.cm.get_src_layer_indices(edge.id),
                )
                mixed_target_past, _ = translator_pool.build_replayed_target_past(
                    source_past_key_values=past_by_node_id[edge.src_id],
                    prefix_input_ids=prefix_cache_ids,
                    target_model=ctx.mm.get_model(edge.tgt_id),
                    src_node_id=edge.src_id,
                    tgt_node_id=edge.tgt_id,
                    tgt_spec=ctx.mm.get_model_spec(edge.tgt_id),
                    source_intermediate_activation_block=source_intermediate_activation_block,
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
