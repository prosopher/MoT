from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from core.config import Config
from core.context import Context
from core.train_util import *


C2C_VARIANTS = {"c2c", "c2c-pr"}


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
    top_layers_to_fuse: int
    fuser_dim: int
    fuser_heads: int
    fuser_depth: int
    fuser_mlp_ratio: int
    gate_temperature_start: float
    gate_temperature_end: float
    hard_gate_eval: bool
    top_layers_to_project: int
    projector_dim: int
    projector_depth: int
    projector_mlp_ratio: int
    dtype: str
    variant: str

    def __post_init__(self) -> None:
        super().__post_init__()
        self.variant = validate_c2c_variant(self.variant)
        initialize_train_output_paths(self)


def validate_c2c_variant(variant: str) -> str:
    normalized = str(variant).strip().lower()
    if normalized not in C2C_VARIANTS:
        raise ValueError(f"Unsupported c2c variant: {variant}. Expected one of {sorted(C2C_VARIANTS)}")
    return normalized


def is_projection_only_variant(variant_or_config: Union[str, TrainConfig]) -> bool:
    if isinstance(variant_or_config, TrainConfig):
        return variant_or_config.variant == "c2c-pr"
    return validate_c2c_variant(variant_or_config) == "c2c-pr"


def get_top_layers_to_translate(config: TrainConfig) -> int:
    return config.top_layers_to_project if is_projection_only_variant(config) else config.top_layers_to_fuse


def get_translation_loss_name(config: TrainConfig) -> str:
    return "projected" if is_projection_only_variant(config) else "fused"


def get_translation_mode_name(config: TrainConfig) -> str:
    if is_projection_only_variant(config):
        return "project_top_layers_and_replace_target_top_layers"
    return "fuse_top_layers_after_target_forward"


def get_trainable_module_label(config: TrainConfig) -> str:
    return "C2C-Project" if is_projection_only_variant(config) else "C2C"



def translate_top_layers(
    translator_pool,
    train_config: TrainConfig,
    sharer_past_key_values: PastKeyValues,
    receiver_past_key_values: PastKeyValues,
    src_name: str,
    dst_name: str,
    dst_spec: ModelSpec,
) -> PastKeyValues:
    if is_projection_only_variant(train_config):
        return translator_pool.project_top_layers(
            sharer_past_key_values=sharer_past_key_values,
            src_name=src_name,
            dst_name=dst_name,
            dst_spec=dst_spec,
        )
    return translator_pool.fuse_top_layers(
        sharer_past_key_values=sharer_past_key_values,
        receiver_past_key_values=receiver_past_key_values,
        src_name=src_name,
        dst_name=dst_name,
        dst_spec=dst_spec,
    )


class ResidualCacheFuser(nn.Module):
    """
    Fuses top-layer receiver/sharer cache blocks following the C2C recipe:
    project -> feature fuse -> dynamic weighting -> gated residual injection.

    receiver_block: [batch, seq, num_layers, dst_hidden]
    sharer_block:   [batch, seq, num_layers, src_hidden]
    output:         [batch, seq, num_layers, dst_hidden]
    """

    def __init__(
        self,
        src_hidden_size: int,
        dst_hidden_size: int,
        num_layers: int,
        fuser_dim: int,
        fuser_heads: int,
        fuser_depth: int,
        mlp_ratio: int,
        gate_temperature_start: float,
        hard_gate_eval: bool,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if fuser_depth < 1:
            raise ValueError("fuser_depth must be >= 1")
        if gate_temperature_start <= 0.0:
            raise ValueError("gate_temperature_start must be > 0")

        self.num_layers = num_layers
        self.fuser_depth = fuser_depth
        self.hard_gate_eval = hard_gate_eval
        self.temperature = float(gate_temperature_start)

        self.receiver_norm = nn.LayerNorm(dst_hidden_size)
        self.receiver_proj = nn.Linear(dst_hidden_size, fuser_dim)
        self.sharer_norm = nn.LayerNorm(src_hidden_size)
        self.sharer_proj = nn.Linear(src_hidden_size, fuser_dim)

        self.feature_norm = nn.LayerNorm(fuser_dim * 2)
        self.feature_proj = nn.Linear(fuser_dim * 2, fuser_dim)
        self.feature_fusion = nn.Sequential(
            nn.LayerNorm(fuser_dim),
            nn.Linear(fuser_dim, fuser_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(fuser_dim * mlp_ratio, fuser_dim),
        )

        self.dynamic_weight_norm = nn.LayerNorm(fuser_dim)
        self.dynamic_weight = nn.Linear(fuser_dim, fuser_dim)

        self.recurrent_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        CrossAttentionBlock(
                            dim=fuser_dim,
                            num_heads=fuser_heads,
                            mlp_ratio=mlp_ratio,
                        )
                        for _ in range(num_layers)
                    ]
                )
                for _ in range(fuser_depth)
            ]
        )

        self.output_norm = nn.LayerNorm(num_layers * fuser_dim)
        self.output_proj = nn.Linear(num_layers * fuser_dim, num_layers * dst_hidden_size)
        # Starting all gates exactly at 0.0 makes it very easy for a short toy run to
        # learn an all-closed hard gate solution, which turns the whole C2C path into
        # an exact identity map at evaluation time. We bias the gates slightly open so
        # the fusion module actually gets used early in training.
        self.gate_logits = nn.Parameter(torch.full((num_layers,), 1.5))

    def set_temperature(self, temperature: float) -> None:
        self.temperature = max(1e-4, float(temperature))

    def set_hard_gate_eval(self, enabled: bool) -> None:
        self.hard_gate_eval = bool(enabled)

    def gate_probabilities(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logits)

    def sample_gate(self) -> torch.Tensor:
        probs = self.gate_probabilities()
        if self.training:
            uniform = torch.rand_like(probs).clamp_(1e-6, 1.0 - 1e-6)
            logistic_noise = torch.log(uniform) - torch.log1p(-uniform)
            soft_gate = torch.sigmoid((self.gate_logits + logistic_noise) / self.temperature)
            return soft_gate
        if self.hard_gate_eval:
            return (probs >= 0.5).to(probs.dtype)
        return probs

    def forward(self, receiver_block: torch.Tensor, sharer_block: torch.Tensor) -> torch.Tensor:
        if receiver_block.ndim != 4 or sharer_block.ndim != 4:
            raise ValueError(
                "ResidualCacheFuser expects [batch, seq, num_layers, hidden] tensors, "
                f"got receiver={tuple(receiver_block.shape)}, sharer={tuple(sharer_block.shape)}"
            )
        if receiver_block.shape[:3] != sharer_block.shape[:3]:
            raise ValueError(
                "receiver_block and sharer_block must agree on [batch, seq, num_layers], "
                f"got receiver={tuple(receiver_block.shape)}, sharer={tuple(sharer_block.shape)}"
            )
        if receiver_block.shape[2] != self.num_layers:
            raise ValueError(
                f"ResidualCacheFuser expected {self.num_layers} aligned layers, got {receiver_block.shape[2]}"
            )

        batch_size, seq_len, _, dst_hidden_size = receiver_block.shape

        receiver_projected = F.gelu(self.receiver_proj(self.receiver_norm(receiver_block)))
        sharer_projected = F.gelu(self.sharer_proj(self.sharer_norm(sharer_block)))

        combined = torch.cat([receiver_projected, sharer_projected], dim=-1)
        projected = F.gelu(self.feature_proj(self.feature_norm(combined)))
        fused_context = self.feature_fusion(projected) + projected

        weights = torch.sigmoid(self.dynamic_weight(self.dynamic_weight_norm(fused_context)))
        weighted_context = fused_context * weights

        hidden = receiver_projected[:, :, 0, :]
        collected = []
        for stage_blocks in self.recurrent_blocks:
            stage_hidden = hidden
            stage_collected = []
            for layer_idx, block in enumerate(stage_blocks):
                stage_hidden = block(stage_hidden, weighted_context[:, :, layer_idx, :])
                stage_collected.append(stage_hidden)
            hidden = stage_hidden
            collected = stage_collected

        residual = F.gelu(self.output_proj(self.output_norm(torch.cat(collected, dim=-1))))
        residual = residual.view(batch_size, seq_len, self.num_layers, dst_hidden_size)

        gate = self.sample_gate().view(1, 1, self.num_layers, 1)
        return receiver_block + (gate * residual)


class DirectionalCacheFuser(nn.Module):
    def __init__(
        self,
        src_hidden_size: int,
        dst_hidden_size: int,
        top_layers_to_fuse: int,
        fuser_dim: int,
        fuser_heads: int,
        fuser_depth: int,
        mlp_ratio: int,
        gate_temperature_start: float,
        hard_gate_eval: bool,
    ) -> None:
        super().__init__()
        self.top_layers_to_fuse = top_layers_to_fuse
        self.key_fuser = ResidualCacheFuser(
            src_hidden_size=src_hidden_size,
            dst_hidden_size=dst_hidden_size,
            num_layers=top_layers_to_fuse,
            fuser_dim=fuser_dim,
            fuser_heads=fuser_heads,
            fuser_depth=fuser_depth,
            mlp_ratio=mlp_ratio,
            gate_temperature_start=gate_temperature_start,
            hard_gate_eval=hard_gate_eval,
        )
        self.value_fuser = ResidualCacheFuser(
            src_hidden_size=src_hidden_size,
            dst_hidden_size=dst_hidden_size,
            num_layers=top_layers_to_fuse,
            fuser_dim=fuser_dim,
            fuser_heads=fuser_heads,
            fuser_depth=fuser_depth,
            mlp_ratio=mlp_ratio,
            gate_temperature_start=gate_temperature_start,
            hard_gate_eval=hard_gate_eval,
        )

    def set_temperature(self, temperature: float) -> None:
        self.key_fuser.set_temperature(temperature)
        self.value_fuser.set_temperature(temperature)

    def set_hard_gate_eval(self, enabled: bool) -> None:
        self.key_fuser.set_hard_gate_eval(enabled)
        self.value_fuser.set_hard_gate_eval(enabled)

    def mean_gate_probability(self) -> float:
        key_prob = float(self.key_fuser.gate_probabilities().mean().item())
        value_prob = float(self.value_fuser.gate_probabilities().mean().item())
        return 0.5 * (key_prob + value_prob)

    def forward(
        self,
        receiver_key_block: torch.Tensor,
        receiver_value_block: torch.Tensor,
        sharer_key_block: torch.Tensor,
        sharer_value_block: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fused_key = self.key_fuser(receiver_key_block, sharer_key_block)
        fused_value = self.value_fuser(receiver_value_block, sharer_value_block)
        return fused_key, fused_value


class C2CFuserPool(nn.Module):
    def __init__(
        self,
        model_specs: Dict[str, ModelSpec],
        edges: List[Edge],
        top_layers_to_fuse: int,
        fuser_dim: int,
        fuser_heads: int,
        fuser_depth: int,
        mlp_ratio: int,
        gate_temperature_start: float,
        hard_gate_eval: bool,
            ) -> None:
        super().__init__()
        if top_layers_to_fuse < 1:
            raise ValueError("top_layers_to_fuse must be >= 1")
        if not edges:
            raise ValueError("edges must contain at least one edge")

        self.model_specs = model_specs
        self.top_layers_to_fuse = top_layers_to_fuse
        self.edges = tuple(edges)
        self.edge_ids = tuple(edge.id for edge in edges)
        self.edges_by_id = build_edge_map(edges)

        adapters = {}
        for edge in self.edges:
            src_spec = model_specs[edge.src_id]
            dst_spec = model_specs[edge.dst_id]
            max_allowed = min(src_spec.num_layers, dst_spec.num_layers)
            if top_layers_to_fuse > max_allowed:
                raise ValueError(
                    f"top_layers_to_fuse={top_layers_to_fuse} exceeds min layer count {max_allowed} "
                    f"for edge {edge.id}."
                )

            adapters[edge.id] = DirectionalCacheFuser(
                src_hidden_size=src_spec.hidden_size,
                dst_hidden_size=dst_spec.hidden_size,
                top_layers_to_fuse=top_layers_to_fuse,
                fuser_dim=fuser_dim,
                fuser_heads=fuser_heads,
                fuser_depth=fuser_depth,
                mlp_ratio=mlp_ratio,
                gate_temperature_start=gate_temperature_start,
                hard_gate_eval=hard_gate_eval,
            )

        self.adapters = nn.ModuleDict(adapters)

    def set_temperature(self, temperature: float) -> None:
        for module in self.adapters.values():
            module.set_temperature(temperature)

    def set_hard_gate_eval(self, enabled: bool) -> None:
        for module in self.adapters.values():
            module.set_hard_gate_eval(enabled)

    def mean_gate_probability(self) -> float:
        if not self.adapters:
            return float("nan")
        return float(sum(module.mean_gate_probability() for module in self.adapters.values()) / len(self.adapters))

    def fuse_top_layer_blocks(
        self,
        receiver_key_block: torch.Tensor,
        receiver_value_block: torch.Tensor,
        sharer_key_block: torch.Tensor,
        sharer_value_block: torch.Tensor,
        src_name: str,
        dst_name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        adapter_name = f"{src_name}_to_{dst_name}"
        if adapter_name not in self.adapters:
            raise ValueError(
                f"C2C edge {adapter_name} is not available. Active edges: {list(self.edge_ids)}"
            )
        return self.adapters[adapter_name](
            receiver_key_block=receiver_key_block,
            receiver_value_block=receiver_value_block,
            sharer_key_block=sharer_key_block,
            sharer_value_block=sharer_value_block,
        )

    def fuse_top_layers(
        self,
        sharer_past_key_values: PastKeyValues,
        receiver_past_key_values: PastKeyValues,
        src_name: str,
        dst_name: str,
        dst_spec: ModelSpec,
    ) -> PastKeyValues:
        sharer_key_block, sharer_value_block = extract_top_layer_blocks(
            past_key_values=sharer_past_key_values,
            top_layers_to_fuse=self.top_layers_to_fuse,
        )
        receiver_key_block, receiver_value_block = extract_top_layer_blocks(
            past_key_values=receiver_past_key_values,
            top_layers_to_fuse=self.top_layers_to_fuse,
        )
        fused_key, fused_value = self.fuse_top_layer_blocks(
            receiver_key_block=receiver_key_block,
            receiver_value_block=receiver_value_block,
            sharer_key_block=sharer_key_block,
            sharer_value_block=sharer_value_block,
            src_name=src_name,
            dst_name=dst_name,
        )
        return blocks_to_partial_past_key_values(
            key_block=fused_key,
            value_block=fused_value,
            num_heads=dst_spec.num_heads,
            head_dim=dst_spec.head_dim,
        )


def extract_top_layer_blocks(
    past_key_values: PastKeyValues,
    top_layers_to_fuse: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if top_layers_to_fuse < 1:
        raise ValueError("top_layers_to_fuse must be >= 1")
    if top_layers_to_fuse > len(past_key_values):
        raise ValueError(
            f"Cannot extract {top_layers_to_fuse} layers from cache with only {len(past_key_values)} layers."
        )
    return past_key_values_to_blocks(past_key_values[-top_layers_to_fuse:])






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
            raise ValueError("depth must be >= 1")
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
            ) -> None:
        super().__init__()
        if top_layers_to_project < 1:
            raise ValueError("top_layers_to_project must be >= 1")
        if not edges:
            raise ValueError("edges must contain at least one edge")

        self.model_specs = model_specs
        self.top_layers_to_project = top_layers_to_project
        self.edges = tuple(edges)
        self.edge_ids = tuple(edge.id for edge in edges)
        self.edges_by_id = build_edge_map(edges)

        adapters = {}
        for edge in self.edges:
            src_spec = model_specs[edge.src_id]
            dst_spec = model_specs[edge.dst_id]
            max_allowed = min(src_spec.num_layers, dst_spec.num_layers)
            if top_layers_to_project > max_allowed:
                raise ValueError(
                    f"top_layers_to_project={top_layers_to_project} exceeds min layer count {max_allowed} "
                    f"for edge {edge.id}."
                )
            adapters[edge.id] = DirectionalCacheProjector(
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
            return float("nan")
        return float(sum(module.mean_gate_probability() for module in self.adapters.values()) / len(self.adapters))

    def project_top_layers(
        self,
        sharer_past_key_values: PastKeyValues,
        src_name: str,
        dst_name: str,
        dst_spec: ModelSpec,
    ) -> PastKeyValues:
        adapter_name = f"{src_name}_to_{dst_name}"
        if adapter_name not in self.adapters:
            raise ValueError(
                f"C2C-Project edge {adapter_name} is not available. "
                f"Active edges: {list(self.edge_ids)}"
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
    ctx: Context,
) -> Union[C2CFuserPool, C2CProjectorPool]:
    config = ctx.config
    model_specs = ctx.model_specs
    edges = ctx.edges
    if is_projection_only_variant(config):
        translator_pool = C2CProjectorPool(
            model_specs=model_specs,
            edges=edges,
            top_layers_to_project=config.top_layers_to_project,
            projector_dim=config.projector_dim,
            projector_depth=config.projector_depth,
            mlp_ratio=config.projector_mlp_ratio,
        )
    else:
        translator_pool = C2CFuserPool(
            model_specs=model_specs,
            edges=edges,
            top_layers_to_fuse=config.top_layers_to_fuse,
            fuser_dim=config.fuser_dim,
            fuser_heads=config.fuser_heads,
            fuser_depth=config.fuser_depth,
            mlp_ratio=config.fuser_mlp_ratio,
            gate_temperature_start=config.gate_temperature_start,
            hard_gate_eval=config.hard_gate_eval,
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
    Union[C2CFuserPool, C2CProjectorPool],
    Dict[str, PreTrainedModel],
    PreTrainedTokenizerBase,
    List[Node],
    List[Edge],
]:
    checkpoint_dir_path_obj = Path(checkpoint_dir_path)
    if not checkpoint_dir_path_obj.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir_path_obj}")
    checkpoint_path_obj = get_train_checkpoint_path(checkpoint_dir_path_obj)
    checkpoint_path = str(checkpoint_path_obj)
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
        build_model_specs_for_nodes(models, nodes),
        nodes,
        edges,
    )
    translator_pool = build_translator_pool(ctx)
    translator_pool.load_state_dict(translator_pool_state_dict)
    translator_pool.to(config.device)
    translator_pool.eval()
    return ctx, translator_pool, models, tokenizer



def compute_gate_temperature(config: TrainConfig, step: int) -> float:
    if config.max_steps <= 1:
        return float(config.gate_temperature_end)
    progress = (step - 1) / (config.max_steps - 1)
    temperature = (
        (1.0 - progress) * float(config.gate_temperature_start)
        + progress * float(config.gate_temperature_end)
    )
    return max(1e-4, temperature)



def run_train(
    ctx: Context,
    models: Dict[str, PreTrainedModel],
    tokenizer: PreTrainedTokenizerBase,
) -> Path:
    config = ctx.config
    model_specs = ctx.model_specs
    nodes = ctx.nodes
    edges = ctx.edges
    set_seed(config.seed)
    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    edge_map = build_edge_map(edges)

    config_path = get_train_config_path(output_path)
    write_json(str(config_path), asdict(config))

    log_path = get_train_log_path(output_path)
    logger = setup_logger(f"{config.alg}_train", log_path)
    logger.info("Starting training")
    logger.info("train_config=%s", asdict(config))

    logger.info("nodes=%s", [asdict(node) for node in nodes])
    logger.info("edges=%s", [edge.id for edge in edges])

    logger.info("[Setup] device=%s", config.device)
    logger.info("[Setup] loading models: %s", {node.id: node.model_id for node in nodes})
    translator_pool = build_translator_pool(ctx)
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
    logger.info("[Setup] variant = %s", config.variant)
    if is_projection_only_variant(config):
        logger.info("[Setup] top_layers_to_project = %d", config.top_layers_to_project)
    else:
        logger.info("[Setup] top_layers_to_fuse = %d", config.top_layers_to_fuse)
    logger.info("[Setup] trainable %s params = %s", get_trainable_module_label(config), f"{count_trainable_parameters(translator_pool):,}")

    dataloader = build_training_dataloader(config, tokenizer)

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
        gate_temperature = None
        if not is_projection_only_variant(config):
            gate_temperature = compute_gate_temperature(config, step)
            translator_pool.set_temperature(gate_temperature)

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
            for edge in edges:
                translated_top_past = translate_top_layers(
                    translator_pool=translator_pool,
                    train_config=config,
                    sharer_past_key_values=past_by_node_id[edge.src_id],
                    receiver_past_key_values=past_by_node_id[edge.dst_id],
                    src_name=edge.src_id,
                    dst_name=edge.dst_id,
                    dst_spec=model_specs[edge.dst_id],
                )
                translated_target_past = replace_top_layers(
                    base_past_key_values=past_by_node_id[edge.dst_id],
                    translated_top_past_key_values=translated_top_past,
                )
                direction_loss = compute_suffix_lm_loss(
                    target_model=models[edge.dst_id],
                    past_key_values=translated_target_past,
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
            postfix = {
                "loss": f"{avg_loss:.4f}",
                "lr": f"{scheduler.lr:.2e}",
            }
            if is_projection_only_variant(config):
                postfix["proj"] = "1.000"
            else:
                postfix["gate"] = f"{translator_pool.mean_gate_probability():.3f}"
            progress_bar.set_postfix(**postfix)

            gpu_memory = gpu_memory_tracker.summary()
            if is_projection_only_variant(config):
                logger.info(
                    "[Step %04d] total_suffix_lm_loss=%.4f | lr=%.2e | gpu_mem_avg=%s | gpu_mem_peak=%s",
                    step,
                    avg_loss,
                    scheduler.lr,
                    gpu_memory["avg_allocated_pretty"],
                    gpu_memory["peak_allocated_pretty"],
                )
            else:
                logger.info(
                    "[Step %04d] total_suffix_lm_loss=%.4f | lr=%.2e | gate_temp=%.4f | mean_gate_prob=%.4f | gpu_mem_avg=%s | gpu_mem_peak=%s",
                    step,
                    avg_loss,
                    scheduler.lr,
                    gate_temperature,
                    translator_pool.mean_gate_probability(),
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
