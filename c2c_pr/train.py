from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from train_util import *
from c2c.train import (
    PastKeyValues,
    ModelSpec,
    extract_top_layer_blocks,
    blocks_to_partial_past_key_values,
)


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
    top_layers_to_project: int
    projector_dim: int
    projector_depth: int
    projector_mlp_ratio: int
    device: str
    dtype: str

    def __post_init__(self) -> None:
        self.device = resolve_device(self.device)
        parse_model_ids_csv(self.model_ids)
        initialize_train_output_paths(self)


# def normalize_legacy_train_config(config_dict: Dict) -> Dict:
#     normalized = dict(config_dict)
#     legacy_to_new = {
#         "top_layers_to_fuse": "top_layers_to_project",
#         "fuser_dim": "projector_dim",
#         "fuser_depth": "projector_depth",
#         "fuser_mlp_ratio": "projector_mlp_ratio",
#     }
#     for legacy_key, new_key in legacy_to_new.items():
#         if new_key not in normalized and legacy_key in normalized:
#             normalized[new_key] = normalized[legacy_key]

#     normalized.pop("fuser_heads", None)
#     normalized.pop("gate_temperature_start", None)
#     normalized.pop("gate_temperature_end", None)
#     normalized.pop("hard_gate_eval", None)
#     return normalized


class ProjectionMLP(nn.Module):
    """
    Projection-only cache adapter for the C2C ablation in Table 8 ("Project").
    It directly maps the sharer's top-layer KV cache into the receiver hidden space
    and replaces the receiver top layers.
    """

    def __init__(
        self,
        src_hidden_size: int,
        dst_hidden_size: int,
        hidden_dim: int,
        depth: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError('depth must be >= 1')
        layers: List[nn.Module] = [nn.LayerNorm(src_hidden_size)]
        in_dim = src_hidden_size
        inner_dim = max(hidden_dim, dst_hidden_size)
        for _ in range(max(depth - 1, 0)):
            layers.extend([
                nn.Linear(in_dim, inner_dim),
                nn.GELU(),
            ])
            in_dim = inner_dim
            inner_dim = max(inner_dim, dst_hidden_size * max(1, mlp_ratio))
        layers.append(nn.Linear(in_dim, dst_hidden_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DirectionalCacheProjector(nn.Module):
    def __init__(
        self,
        src_hidden_size: int,
        dst_hidden_size: int,
        top_layers_to_project: int,
        hidden_dim: int,
        depth: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        self.top_layers_to_project = top_layers_to_project
        self.key_projector = ProjectionMLP(
            src_hidden_size=src_hidden_size,
            dst_hidden_size=dst_hidden_size,
            hidden_dim=hidden_dim,
            depth=depth,
            mlp_ratio=mlp_ratio,
        )
        self.value_projector = ProjectionMLP(
            src_hidden_size=src_hidden_size,
            dst_hidden_size=dst_hidden_size,
            hidden_dim=hidden_dim,
            depth=depth,
            mlp_ratio=mlp_ratio,
        )

    def mean_gate_probability(self) -> float:
        return 1.0

    def forward(
        self,
        sharer_key_block: torch.Tensor,
        sharer_value_block: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.key_projector(sharer_key_block), self.value_projector(sharer_value_block)


class C2CProjectorPool(nn.Module):
    def __init__(
        self,
        model_specs: Dict[str, ModelSpec],
        edges: List[Edge],
        top_layers_to_project: int,
        projector_dim: int,
        projector_depth: int,
        mlp_ratio: int,
        active_directions: List[str],
    ) -> None:
        super().__init__()
        if top_layers_to_project < 1:
            raise ValueError('top_layers_to_project must be >= 1')
        if not active_directions:
            raise ValueError('active_directions must contain at least one direction')

        self.model_specs = model_specs
        self.top_layers_to_project = top_layers_to_project
        self.active_directions = tuple(active_directions)
        self.edges_by_id = build_edge_map(edges)

        adapters = {}
        for direction in self.active_directions:
            if direction not in self.edges_by_id:
                raise ValueError(f'Unknown direction: {direction}')
            edge = self.edges_by_id[direction]
            src_spec = model_specs[edge.src_id]
            dst_spec = model_specs[edge.dst_id]
            max_allowed = min(src_spec.num_layers, dst_spec.num_layers)
            if top_layers_to_project > max_allowed:
                raise ValueError(
                    f'top_layers_to_project={top_layers_to_project} exceeds min layer count {max_allowed} '
                    f'for direction {direction}.'
                )
            adapters[direction] = DirectionalCacheProjector(
                src_hidden_size=src_spec.hidden_size,
                dst_hidden_size=dst_spec.hidden_size,
                top_layers_to_project=top_layers_to_project,
                hidden_dim=projector_dim,
                depth=projector_depth,
                mlp_ratio=mlp_ratio,
            )
        self.adapters = nn.ModuleDict(adapters)

    def mean_gate_probability(self) -> float:
        if not self.adapters:
            return float('nan')
        return float(sum(module.mean_gate_probability() for module in self.adapters.values()) / len(self.adapters))

    def project_top_layers(
        self,
        sharer_past_key_values: PastKeyValues,
        src_name: str,
        dst_name: str,
        dst_spec: ModelSpec,
    ) -> PastKeyValues:
        adapter_name = f'{src_name}_to_{dst_name}'
        if adapter_name not in self.adapters:
            raise ValueError(
                f'C2C-Project direction {adapter_name} is not available. '
                f'Active directions: {list(self.active_directions)}'
            )
        sharer_key_block, sharer_value_block = extract_top_layer_blocks(
            past_key_values=sharer_past_key_values,
            top_layers_to_fuse=self.top_layers_to_project,
        )
        projected_key, projected_value = self.adapters[adapter_name](
            sharer_key_block=sharer_key_block,
            sharer_value_block=sharer_value_block,
        )
        return blocks_to_partial_past_key_values(
            key_block=projected_key,
            value_block=projected_value,
            num_heads=dst_spec.num_heads,
            head_dim=dst_spec.head_dim,
        )


def build_translator_pool(
    models: Dict[str, PreTrainedModel],
    config: TrainConfig,
) -> Tuple[C2CProjectorPool, Dict[str, ModelSpec], List[Node], List[Edge]]:
    nodes, edges = build_nodes_and_edges(config.model_ids, config.model_directions)
    model_specs = {
        node.id: get_model_spec(models[node.id])
        for node in nodes
    }
    active_directions = parse_model_directions(
        config.model_directions,
        allowed_directions=[edge.id for edge in edges],
    )
    translator_pool = C2CProjectorPool(
        model_specs=model_specs,
        edges=edges,
        top_layers_to_project=config.top_layers_to_project,
        projector_dim=config.projector_dim,
        projector_depth=config.projector_depth,
        mlp_ratio=config.projector_mlp_ratio,
        active_directions=active_directions,
    )
    translator_pool.to(config.device)
    return translator_pool, model_specs, nodes, edges



def load_translator_pool_from_checkpoint(
    checkpoint_path: str,
    device_override: Optional[str] = None,
) -> Tuple[
    TrainConfig,
    C2CProjectorPool,
    Dict[str, ModelSpec],
    Dict[str, PreTrainedModel],
    PreTrainedTokenizerBase,
    List[Node],
    List[Edge],
]:
    payload = torch.load(checkpoint_path, map_location='cpu')
    config = TrainConfig(**payload['train_config'])
    if device_override is not None:
        config.device = device_override
    models, tokenizer, nodes, edges = build_models_and_tokenizer(config)
    translator_pool, model_specs, _, _ = build_translator_pool(models, config)
    translator_pool.load_state_dict(payload['translator_pool'])
    translator_pool.to(config.device)
    translator_pool.eval()
    return config, translator_pool, model_specs, models, tokenizer, nodes, edges



def run_train(config: TrainConfig) -> Path:
    if config.output_path is None:
        raise ValueError('TrainConfig.output_path must be initialized before run_train.')

    set_seed(config.seed)
    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    nodes, edges = build_nodes_and_edges(config.model_ids, config.model_directions)
    edge_map = build_edge_map(edges)

    config_path = get_train_config_path(output_path)
    write_json(str(config_path), asdict(config))

    log_path = get_train_log_path(output_path)
    logger = setup_logger(f'{config.alg}_train', log_path)
    logger.info('Starting training')
    logger.info('train_config=%s', asdict(config))

    model_directions = parse_model_directions(
        config.model_directions,
        allowed_directions=[edge.id for edge in edges],
    )
    logger.info('nodes=%s', [asdict(node) for node in nodes])
    logger.info('model_directions=%s', model_directions)

    logger.info('[Setup] device=%s', config.device)
    logger.info('[Setup] loading models: %s', {node.id: node.model_id for node in nodes})
    models, tokenizer, _, _ = build_models_and_tokenizer(config)
    translator_pool, model_specs, _, _ = build_translator_pool(models, config)
    translator_pool.train()

    logger.info('[Setup] full model specs')
    for node in nodes:
        spec = model_specs[node.id]
        logger.info(
            '  %s (%s): layers=%d, hidden=%d, heads=%d',
            node.id,
            node.model_id,
            spec.num_layers,
            spec.hidden_size,
            spec.num_heads,
        )
    logger.info('[Setup] top_layers_to_project = %d', config.top_layers_to_project)
    logger.info('[Setup] trainable C2C-Project params = %s', f"{count_trainable_parameters(translator_pool):,}")

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
    progress_bar = tqdm(range(1, config.max_steps + 1), desc='Training')

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
            for direction in model_directions:
                edge = edge_map[direction]
                projected_top_past = translator_pool.project_top_layers(
                    sharer_past_key_values=past_by_node_id[edge.src_id],
                    src_name=edge.src_id,
                    dst_name=edge.dst_id,
                    dst_spec=model_specs[edge.dst_id],
                )
                projected_target_past = replace_top_layers(
                    base_past_key_values=past_by_node_id[edge.dst_id],
                    translated_top_past_key_values=projected_top_past,
                )
                direction_loss = compute_suffix_lm_loss(
                    target_model=models[edge.dst_id],
                    past_key_values=projected_target_past,
                    lm_input_ids=lm_input_ids,
                    lm_labels=lm_labels,
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
                loss=f'{avg_loss:.4f}',
                lr=f'{scheduler.lr:.2e}',
                proj='1.000',
            )
            gpu_memory = gpu_memory_tracker.summary()
            logger.info(
                '[Step %04d] total_suffix_lm_loss=%.4f | lr=%.2e | gpu_mem_avg=%s | gpu_mem_peak=%s',
                step,
                avg_loss,
                scheduler.lr,
                gpu_memory['avg_allocated_pretty'],
                gpu_memory['peak_allocated_pretty'],
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
            'note': 'Final checkpoint trained with C2C projection-only suffix LM loss.',
            'model_ids': config.model_ids,
            'top_layers_to_project': config.top_layers_to_project,
            'model_directions': config.model_directions,
        },
    )
    final_gpu_memory = gpu_memory_tracker.summary()
    logger.info(
        '[Memory] avg_gpu_mem=%s | peak_gpu_mem=%s | samples=%d',
        final_gpu_memory['avg_allocated_pretty'],
        final_gpu_memory['peak_allocated_pretty'],
        final_gpu_memory['num_samples'],
    )
    logger.info('[Done] final checkpoint saved to %s', final_path)
    logger.info('Saved train log to %s', log_path)
    return final_path
