from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from core.common import build_step_pasts_and_batches
from core.config import Config
from core.context import Context
from core.translator_pool import TranslatorPool
from core.model_spec import ModelSpec
from core.train_util import *


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
    shared_slots: int
    shared_dim: int
    translator_dim: int
    translator_heads: int
    translator_mlp_ratio: int
    dtype: str

    def __post_init__(self) -> None:
        super().__post_init__()
        initialize_train_output_paths(self)


@dataclass
class SharedCache:
    key: torch.Tensor
    value: torch.Tensor



class LocalToSharedTranslator(nn.Module):
    """
    Input:  [batch, seq, local_layers, local_hidden]
    Output: [batch, seq, shared_slots, shared_dim]
    """

    def __init__(
        self,
        local_hidden_size: int,
        local_layers: int,
        shared_slots: int,
        shared_dim: int,
        translator_dim: int,
        translator_heads: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        self.local_layers = local_layers
        self.shared_slots = shared_slots
        self.shared_dim = shared_dim
        self.input_norm = nn.LayerNorm(local_hidden_size)
        self.input_proj = nn.Linear(local_hidden_size, translator_dim)
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim=translator_dim,
                    num_heads=translator_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(local_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(local_layers * translator_dim)
        self.output_proj = nn.Linear(local_layers * translator_dim, shared_slots * shared_dim)

    def forward(self, local_block: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, _ = local_block.shape
        projected = F.gelu(self.input_proj(self.input_norm(local_block)))
        hidden = projected[:, :, 0, :]
        collected = []
        for layer_idx, block in enumerate(self.blocks):
            hidden = block(hidden, projected[:, :, layer_idx, :])
            collected.append(hidden)
        fused = torch.cat(collected, dim=-1)
        shared = F.gelu(self.output_proj(self.output_norm(fused)))
        return shared.view(batch_size, seq_len, self.shared_slots, self.shared_dim)


class SharedToLocalTranslator(nn.Module):
    """
    Input:  [batch, seq, shared_slots, shared_dim]
    Output: [batch, seq, local_layers, local_hidden]
    """

    def __init__(
        self,
        local_hidden_size: int,
        local_layers: int,
        shared_slots: int,
        shared_dim: int,
        translator_dim: int,
        translator_heads: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        self.local_layers = local_layers
        self.local_hidden_size = local_hidden_size
        self.shared_slots = shared_slots
        self.shared_dim = shared_dim
        flat_shared = shared_slots * shared_dim
        self.input_norm = nn.LayerNorm(flat_shared)
        self.input_proj = nn.Linear(flat_shared, local_layers * translator_dim)
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim=translator_dim,
                    num_heads=translator_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(local_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(local_layers * translator_dim)
        self.output_proj = nn.Linear(local_layers * translator_dim, local_layers * local_hidden_size)

    def forward(self, shared_block: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, _ = shared_block.shape
        flat_shared = shared_block.reshape(batch_size, seq_len, self.shared_slots * self.shared_dim)
        expanded = F.gelu(self.input_proj(self.input_norm(flat_shared)))
        expanded = expanded.view(batch_size, seq_len, self.local_layers, -1)
        hidden = expanded[:, :, 0, :]
        collected = []
        for layer_idx, block in enumerate(self.blocks):
            hidden = block(hidden, expanded[:, :, layer_idx, :])
            collected.append(hidden)
        fused = torch.cat(collected, dim=-1)
        local = F.gelu(self.output_proj(self.output_norm(fused)))
        return local.view(batch_size, seq_len, self.local_layers, self.local_hidden_size)


class ModelLatentTranslator(nn.Module):
    def __init__(
        self,
        local_layers: int,
        local_hidden_size: int,
        shared_slots: int,
        shared_dim: int,
        translator_dim: int,
        translator_heads: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        self.key_to_shared = LocalToSharedTranslator(
            local_hidden_size=local_hidden_size,
            local_layers=local_layers,
            shared_slots=shared_slots,
            shared_dim=shared_dim,
            translator_dim=translator_dim,
            translator_heads=translator_heads,
            mlp_ratio=mlp_ratio,
        )
        self.value_to_shared = LocalToSharedTranslator(
            local_hidden_size=local_hidden_size,
            local_layers=local_layers,
            shared_slots=shared_slots,
            shared_dim=shared_dim,
            translator_dim=translator_dim,
            translator_heads=translator_heads,
            mlp_ratio=mlp_ratio,
        )
        self.key_from_shared = SharedToLocalTranslator(
            local_hidden_size=local_hidden_size,
            local_layers=local_layers,
            shared_slots=shared_slots,
            shared_dim=shared_dim,
            translator_dim=translator_dim,
            translator_heads=translator_heads,
            mlp_ratio=mlp_ratio,
        )
        self.value_from_shared = SharedToLocalTranslator(
            local_hidden_size=local_hidden_size,
            local_layers=local_layers,
            shared_slots=shared_slots,
            shared_dim=shared_dim,
            translator_dim=translator_dim,
            translator_heads=translator_heads,
            mlp_ratio=mlp_ratio,
        )

    def to_shared(self, key_block: torch.Tensor, value_block: torch.Tensor) -> SharedCache:
        return SharedCache(
            key=self.key_to_shared(key_block),
            value=self.value_to_shared(value_block),
        )

    def from_shared(self, shared_cache: SharedCache) -> Tuple[torch.Tensor, torch.Tensor]:
        key_block = self.key_from_shared(shared_cache.key)
        value_block = self.value_from_shared(shared_cache.value)
        return key_block, value_block


def initialize_translators(
    ctx: Context,
    *,
    shared_slots: int,
    shared_dim: int,
    translator_dim: int,
    translator_heads: int,
    mlp_ratio: int,
) -> TranslatorPool:
    translator_pool = ctx.tp
    translator_pool.lsc_node_model_ids = {node.id: node.model_id for node in ctx.nodes}
    for node in ctx.nodes:
        translator_id = node.model_id
        if translator_id in translator_pool.translators:
            continue
        spec = ctx.tp.get_model_spec(node.id)
        translator_pool.add_translator(
            translator_id,
            ModelLatentTranslator(
                local_layers=spec.num_layers,
                local_hidden_size=spec.kv_hidden_size,
                shared_slots=shared_slots,
                shared_dim=shared_dim,
                translator_dim=translator_dim,
                translator_heads=translator_heads,
                mlp_ratio=mlp_ratio,
            ),
        )
    return translator_pool


def translate_blocks(
    *,
    translator_pool: TranslatorPool,
    key_block: torch.Tensor,
    value_block: torch.Tensor,
    src_node_id: str,
    tgt_node_id: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    node_model_ids = getattr(translator_pool, "lsc_node_model_ids", None)
    if node_model_ids is None:
        raise ValueError("LSC translator metadata has not been initialized.")
    src_translator = translator_pool.translators[node_model_ids[src_node_id]]
    tgt_translator = translator_pool.translators[node_model_ids[tgt_node_id]]
    shared_cache = src_translator.to_shared(key_block, value_block)
    return tgt_translator.from_shared(shared_cache)


def translate_layers(
    *,
    translator_pool: TranslatorPool,
    past_key_values: PastKeyValues,
    src_node_id: str,
    tgt_node_id: str,
    tgt_spec: ModelSpec,
) -> PastKeyValues:
    key_block, value_block = past_key_values_to_blocks(past_key_values)
    translated_key, translated_value = translate_blocks(
        translator_pool=translator_pool,
        key_block=key_block,
        value_block=value_block,
        src_node_id=src_node_id,
        tgt_node_id=tgt_node_id,
    )
    return blocks_to_past_key_values(
        key_block=translated_key,
        value_block=translated_value,
        model_spec=tgt_spec,
    )


def blocks_to_past_key_values(
    key_block: torch.Tensor,
    value_block: torch.Tensor,
    model_spec: ModelSpec,
) -> PastKeyValues:
    batch_size, seq_len, num_layers, hidden_size = key_block.shape
    if num_layers != model_spec.num_layers:
        raise ValueError(f"Layer mismatch: block has {num_layers}, model expects {model_spec.num_layers}.")
    if hidden_size != model_spec.kv_hidden_size:
        raise ValueError(
            f"Hidden mismatch: block has {hidden_size}, "
            f"model expects KV hidden {model_spec.kv_hidden_size} "
            f"({model_spec.num_key_value_heads} kv heads * {model_spec.head_dim} head dim)."
        )

    past_key_values = []
    for layer_idx in range(model_spec.num_layers):
        key_layer = key_block[:, :, layer_idx, :]
        value_layer = value_block[:, :, layer_idx, :]
        key_layer = key_layer.view(batch_size, seq_len, model_spec.num_key_value_heads, model_spec.head_dim)
        value_layer = value_layer.view(batch_size, seq_len, model_spec.num_key_value_heads, model_spec.head_dim)
        key_layer = key_layer.permute(0, 2, 1, 3).contiguous()
        value_layer = value_layer.permute(0, 2, 1, 3).contiguous()
        past_key_values.append((key_layer, value_layer))
    return tuple(past_key_values)


def build_translator_pool(
    ctx: Context,
) -> TranslatorPool:
    config = ctx.config
    translator_pool = initialize_translators(
        ctx=ctx,
        shared_slots=config.shared_slots,
        shared_dim=config.shared_dim,
        translator_dim=config.translator_dim,
        translator_heads=config.translator_heads,
        mlp_ratio=config.translator_mlp_ratio,
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
    translator_pool = build_translator_pool(ctx)
    load_translator_checkpoints(checkpoint_dir_path_obj, translator_pool)
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


    config_path = get_train_config_path(output_path)
    write_json(str(config_path), asdict(config))

    log_path = get_train_log_path(output_path)
    logging.info("Starting training")
    logging.info("train_config=%s", asdict(config))

    logging.info("nodes=%s", [asdict(node) for node in nodes])
    logging.info("edges=%s", [edge.id for edge in edges])

    logging.info("[Setup] device=%s", config.device)
    logging.info("[Setup] loading models: %s", {node.id: node.model_id for node in nodes})
    translator_pool = build_translator_pool(ctx)
    translator_pool.train()

    logging.info("[Setup] model specs used for translation")
    for node in nodes:
        spec = ctx.tp.get_model_spec(node.id)
        logging.info(
            "  %s (%s): layers=%d, hidden=%d, heads=%d, kv_heads=%d, kv_hidden=%d",
            node.id,
            node.model_id,
            spec.num_layers,
            spec.hidden_size,
            spec.num_heads,
            spec.num_key_value_heads,
            spec.kv_hidden_size,
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
                _, prompt_token_ids, label_token_ids = batches_by_node_id[edge.tgt_id]
                translated_past = translate_layers(
                    translator_pool=translator_pool,
                    past_key_values=past_by_node_id[edge.src_id],
                    src_node_id=edge.src_id,
                    tgt_node_id=edge.tgt_id,
                    tgt_spec=ctx.tp.get_model_spec(edge.tgt_id),
                )
                direction_loss = compute_suffix_lm_loss(
                    target_model=ctx.tp.get_model(edge.tgt_id),
                    past_key_values=translated_past,
                    prompt_token_ids=prompt_token_ids,
                    label_token_ids=label_token_ids,
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
            logging.info(
                "[Step %04d] loss=%.4f | lr=%.2e | gpu_mem_avg=%s | gpu_mem_peak=%s",
                step,
                avg_loss,
                scheduler.lr,
                gpu_memory["avg_allocated_pretty"],
                gpu_memory["peak_allocated_pretty"],
            )
            running_loss = 0.0

    final_path = get_train_checkpoint_path(output_path)
    save_translator_checkpoints(output_path, translator_pool)
    final_gpu_memory = gpu_memory_tracker.summary()
    logging.info(
        "[Memory] avg_gpu_mem=%s | peak_gpu_mem=%s | samples=%d",
        final_gpu_memory["avg_allocated_pretty"],
        final_gpu_memory["peak_allocated_pretty"],
        final_gpu_memory["num_samples"],
    )
    logging.info("[Done] final translator checkpoints saved to %s", final_path)
    logging.info("Saved train log to %s", log_path)
    return final_path
