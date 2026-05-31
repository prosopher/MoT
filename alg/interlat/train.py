from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from core.channel_manager import ChannelManager
from core.common import (
    GPUMemoryTracker,
    PastKeyValues,
    count_trainable_parameters,
    extract_past_key_values,
    get_model_parameter_dtype,
    load_frozen_model,
    load_tokenizer,
    read_json,
    set_seed,
    split_prefix_and_suffix_for_exact_next_token_loss,
    write_json,
)
from core.config import Config
from core.context import Context
from core.model_manager import ModelManager
from core.train_util import (
    build_models_and_tokenizers,
    WarmupCosineScheduler,
    build_training_dataloaders_by_target,
    get_train_checkpoint_path,
    get_train_config_path,
    get_train_log_path,
    initialize_train_output_paths,
    move_trainable_module_to_config_dtype
)
from core.topology import Edge, Node, get_translator_id
from alg.interlat.vender import ModelArguments as VendorModelArguments
from alg.interlat.vender.hidden_model.custom_model import HiddenStateProcessor


_VENDOR_MODEL_ARGUMENTS = VendorModelArguments()


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
    prepended_num_heads: int
    plan_similarity_weight: float
    random_contrast_weight: float
    dtype: str

    def __post_init__(self) -> None:
        super().__post_init__()
        initialize_train_output_paths(self)
        if self.prefix_tokens < 2:
            raise ValueError("prefix_tokens must be >= 2")
        if self.prefix_tokens >= self.total_tokens:
            raise ValueError("prefix_tokens must be smaller than total_tokens")
        if self.prepended_num_heads < 1:
            raise ValueError("prepended_num_heads must be >= 1")




class InterLatEdgeTranslator(nn.Module):
    """Interlat-style hidden-state translator.

    The upstream Interlat path directly communicates last-layer hidden states and
    lets a compact hidden-state processor translate their numerical range before
    insertion into the receiver.  This module intentionally avoids the previous
    benchmark-specific query-token cross-attention compressor and hidden-state
    regression target.
    """

    def __init__(
        self,
        *,
        source_hidden_size: int,
        target_hidden_size: int,
        prepended_num_heads: int,
    ) -> None:
        super().__init__()
        self.hidden_processor = HiddenStateProcessor(
            hidden_size=target_hidden_size,
            num_heads=_candidate_num_heads(target_hidden_size, prepended_num_heads),
            input_dim=source_hidden_size,
        )

    def forward(self, source_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.hidden_processor(source_hidden_states)


class InterLatTranslatorPool(nn.Module):
    def __init__(self, ctx: Context) -> None:
        super().__init__()
        config = ctx.config
        self.node_model_ids = {node.id: node.model_id for node in ctx.nodes}
        translators = {}
        for edge in ctx.edges:
            translator_id = get_translator_id(self.node_model_ids[edge.src_id], self.node_model_ids[edge.tgt_id])
            if translator_id in translators:
                continue
            src_spec = ctx.mm.get_model_spec(edge.src_id)
            tgt_spec = ctx.mm.get_model_spec(edge.tgt_id)
            translators[translator_id] = InterLatEdgeTranslator(
                source_hidden_size=src_spec.hidden_size,
                target_hidden_size=tgt_spec.hidden_size,
                prepended_num_heads=config.prepended_num_heads,
            )
        self.translators = nn.ModuleDict(translators)
        self.translator_ids = tuple(self.translators.keys())

    def translate_hidden_states(
        self,
        *,
        source_hidden_states: torch.Tensor,
        src_node_id: str,
        tgt_node_id: str,
    ) -> torch.Tensor:
        translator_id = get_translator_id(self.node_model_ids[src_node_id], self.node_model_ids[tgt_node_id])
        if translator_id not in self.translators:
            raise ValueError(
                f"InterLat translator {translator_id} is not available. "
                f"Active translators: {list(self.translator_ids)}"
            )
        return self.translators[translator_id](source_hidden_states)


def get_model_context_limit(model) -> int:
    config = getattr(model, "config", None)
    candidates = [
        getattr(config, "n_positions", None),
        getattr(config, "max_position_embeddings", None),
        getattr(config, "n_ctx", None),
    ]
    limits = [value for value in candidates if isinstance(value, int) and value > 0]
    if not limits:
        return 1024
    return min(limits)


@torch.no_grad()
def extract_interlat_source_hidden_states(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Extract the last-layer hidden states that InterLat communicates.

    The input ids are the same prefix ids used to build the common prefix cache
    for the other algorithms.  This preserves the benchmark data split while
    keeping InterLat's algorithmic input faithful to hidden-state communication.
    """

    outputs = model(
        input_ids=input_ids,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None or len(hidden_states) == 0:
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if last_hidden_state is None:
            raise ValueError("model did not return hidden states for InterLat source extraction")
        source_hidden_states = last_hidden_state
    else:
        source_hidden_states = hidden_states[-1]
    return source_hidden_states.to(dtype=get_model_parameter_dtype(model))


def build_latent_conditioned_past(
    model,
    *,
    latent_prefix: torch.Tensor,
) -> PastKeyValues:
    """Build a target past from communicated InterLat states only.

    The communicated sequence length is the common benchmark prefix length
    (``prefix_tokens - 1``).  This avoids giving InterLat an extra
    target-prefix cache on top of the communicated source latent.
    """

    model_context_limit = get_model_context_limit(model)
    latent_tokens = int(latent_prefix.shape[1])
    if latent_tokens >= model_context_limit:
        raise ValueError(
            f"latent length ({latent_tokens}) must be smaller than model context limit ({model_context_limit})"
        )

    attention_mask = torch.ones(
        latent_prefix.shape[:2],
        dtype=torch.long,
        device=latent_prefix.device,
    )
    outputs = model(
        input_ids=None,
        inputs_embeds=latent_prefix,
        attention_mask=attention_mask,
        use_cache=True,
    )
    return outputs.past_key_values


def compute_suffix_logits_and_loss(
    *,
    target_model,
    past_key_values: PastKeyValues,
    lm_input_ids: torch.Tensor,
    lm_labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    outputs = target_model(
        input_ids=lm_input_ids,
        past_key_values=past_key_values,
        use_cache=False,
    )
    logits = outputs.logits
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        lm_labels.reshape(-1),
        reduction="mean",
    )
    return logits, loss


def _flatten_logits_for_valid_labels(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    valid = labels.reshape(-1).ne(-100)
    flat = logits.reshape(-1, logits.shape[-1])
    if not bool(valid.any()):
        return flat[:0]
    return flat[valid]


def compute_plan_similarity_loss(
    *,
    normal_logits: torch.Tensor,
    plan_logits: torch.Tensor,
    labels: torch.Tensor,
    margin_kl: float = 0.7,
    margin_cos: float = 0.3,
) -> torch.Tensor:
    """Interlat-style plan-aligned regularization on receiver logits."""

    normal = _flatten_logits_for_valid_labels(normal_logits, labels)
    plan = _flatten_logits_for_valid_labels(plan_logits, labels).detach()
    if normal.numel() == 0 or plan.numel() == 0:
        return normal_logits.new_zeros(())
    kl_loss = F.kl_div(
        F.log_softmax(normal, dim=-1),
        F.softmax(plan, dim=-1).clamp_min(1e-8),
        reduction="batchmean",
    )
    normal_prob = F.softmax(normal, dim=-1).clamp_min(1e-8).reshape(-1)
    plan_prob = F.softmax(plan, dim=-1).clamp_min(1e-8).reshape(-1)
    cosine_loss = 1.0 - F.cosine_similarity(normal_prob, plan_prob, dim=0)
    return margin_kl * kl_loss + margin_cos * cosine_loss


def compute_random_contrast_loss(
    *,
    normal_logits: torch.Tensor,
    random_logits: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.69,
) -> torch.Tensor:
    """Interlat-style separation from mismatched latent communication."""

    normal = _flatten_logits_for_valid_labels(normal_logits, labels)
    random = _flatten_logits_for_valid_labels(random_logits, labels).detach()
    if normal.numel() == 0 or random.numel() == 0:
        return normal_logits.new_zeros(())
    normal_prob = F.softmax(normal, dim=-1).clamp_min(1e-8)
    random_prob = F.softmax(random, dim=-1).clamp_min(1e-8)
    midpoint = 0.5 * (normal_prob + random_prob)
    js_div = 0.5 * F.kl_div(normal_prob.log(), midpoint, reduction="batchmean")
    js_div = js_div + 0.5 * F.kl_div(random_prob.log(), midpoint, reduction="batchmean")
    return torch.clamp(float(margin) - js_div, min=0.0)


def adjust_interlat_loss_weights(
    *,
    random_contrast_loss: torch.Tensor,
    plan_similarity_loss: torch.Tensor,
    initial_plan_weight: float,
    initial_random_weight: float,
) -> Tuple[float, float]:
    """Match upstream Interlat's dynamic auxiliary-loss weighting rule.

    The constructor/config weights are kept as fallbacks for non-finite losses,
    while normal training follows the upstream ranges: plan in [0.01, 0.05]
    and random contrast in [0.01, 0.50].
    """

    plan_value = float(plan_similarity_loss.detach().item())
    random_value = float(random_contrast_loss.detach().item())
    if not torch.isfinite(plan_similarity_loss.detach()):
        plan_weight = float(initial_plan_weight)
    else:
        plan_normalized = max(0.0, min(1.0, plan_value / 4.0))
        plan_weight = 0.01 + plan_normalized * 0.04
    if not torch.isfinite(random_contrast_loss.detach()):
        random_weight = float(initial_random_weight)
    else:
        contrast_normalized = max(0.0, min(1.0, random_value / 0.69))
        random_weight = 0.01 + contrast_normalized * 0.49
    return plan_weight, random_weight


def build_mismatched_source_hidden_states(source_hidden_states: torch.Tensor) -> torch.Tensor:
    if source_hidden_states.shape[0] < 2:
        return source_hidden_states.detach()
    return torch.roll(source_hidden_states.detach(), shifts=1, dims=0)


def _candidate_num_heads(hidden_size: int, requested_heads: int) -> int:
    for candidate in range(min(requested_heads, hidden_size), 0, -1):
        if hidden_size % candidate == 0:
            return candidate
    return 1


def build_translator_pool(ctx: Context) -> InterLatTranslatorPool:
    pool = InterLatTranslatorPool(ctx)
    move_trainable_module_to_config_dtype(pool, ctx.config)
    return pool


def load_translator_pool_from_checkpoint(
    checkpoint_dir_path: str,
    nodes: List[Node],
    edges: List[Edge],
    device_override: Optional[str] = None,
):
    checkpoint_dir_path_obj = Path(checkpoint_dir_path)
    checkpoint_path_obj = get_train_checkpoint_path(checkpoint_dir_path_obj)
    train_config_path = get_train_config_path(checkpoint_dir_path_obj)
    if not checkpoint_path_obj.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path_obj}")
    if not train_config_path.exists():
        raise FileNotFoundError(f"Train config not found: {train_config_path}")

    config = TrainConfig(**read_json(train_config_path))
    if device_override is not None:
        config.device = device_override
    models, tokenizers = build_models_and_tokenizers(config, nodes)
    ctx = Context(
        config,
        nodes,
        edges,
        ModelManager(models, tokenizers),
        ChannelManager(edges),
    )
    translator_pool = build_translator_pool(ctx)
    translator_pool.load_state_dict(torch.load(str(checkpoint_path_obj), map_location="cpu"))
    move_trainable_module_to_config_dtype(translator_pool, config)
    translator_pool.eval()
    return ctx, translator_pool


def run_train(
    ctx: Context,
    gpu_memory_tracker: GPUMemoryTracker,
) -> Path:
    config: TrainConfig = ctx.config
    set_seed(config.seed)
    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    config_path = get_train_config_path(output_path)
    write_json(str(config_path), asdict(config))
    log_path = get_train_log_path(output_path)
    logging.info("Starting Interlat training")
    logging.info("train_config=%s", asdict(config))
    logging.info("upstream_vendor_defaults.prepended_length=%d", _VENDOR_MODEL_ARGUMENTS.prepended_length)
    logging.info(
        "InterLat source communication is derived from the same common prefix cache used by the other algorithms; "
        "the target past is built from the communicated latent sequence only."
    )

    translator_pool = build_translator_pool(ctx)
    translator_pool.train()
    for edge in ctx.edges:
        src_spec = ctx.mm.get_model_spec(edge.src_id)
        tgt_spec = ctx.mm.get_model_spec(edge.tgt_id)
        logging.info(
            "edge=%s | src_hidden=%d | tgt_hidden=%d | prefix_cache_tokens=%d",
            edge.id,
            src_spec.hidden_size,
            tgt_spec.hidden_size,
            config.prefix_tokens,
        )

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
    dataloaders_by_target = build_training_dataloaders_by_target(ctx)

    running_total_loss = 0.0
    running_ce_loss = 0.0
    running_plan_loss = 0.0
    running_random_loss = 0.0
    running_positive_cosine = 0.0
    running_plan_weight = 0.0
    running_random_weight = 0.0

    progress_bar = tqdm(range(1, config.max_steps + 1), desc="Interlat Training")
    for step in progress_bar:
        optimizer.zero_grad(set_to_none=True)
        step_total_loss = 0.0
        step_ce_loss = 0.0
        step_plan_loss = 0.0
        step_random_loss = 0.0
        step_positive_cosine = 0.0
        step_plan_weight = 0.0
        step_random_weight = 0.0
        used_micro_batches = 0

        while used_micro_batches < config.grad_accum_steps:
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
                        for node in ctx.nodes
                    }
                target_batches[target_node_id] = (prefix_cache_ids, lm_input_ids, lm_labels, past_by_node_id)

            total_edge_loss = 0.0
            total_edge_ce = 0.0
            total_edge_plan = 0.0
            total_edge_random = 0.0
            total_edge_cosine = 0.0
            for edge in ctx.edges:
                tgt_prefix_ids, lm_input_ids, lm_labels, past_by_node_id = target_batches[edge.tgt_id]

                target_model = ctx.mm.get_model(edge.tgt_id)
                tgt_model_context_limit = get_model_context_limit(target_model)

                source_hidden_states = extract_interlat_source_hidden_states(
                    ctx.mm.get_model(edge.src_id),
                    tgt_prefix_ids,
                )

                translated_latents = translator_pool.translate_hidden_states(
                    src_node_id=edge.src_id,
                    tgt_node_id=edge.tgt_id,
                    source_hidden_states=source_hidden_states,
                )
                if translated_latents.shape[1] + lm_input_ids.shape[1] > tgt_model_context_limit:
                    raise ValueError(
                        "InterLat training sequence does not fit target model context window: "
                        f"target={edge.tgt_id}, model_context_limit={tgt_model_context_limit}, "
                        f"latent_tokens={translated_latents.shape[1]}, lm_input_tokens={lm_input_ids.shape[1]}"
                    )

                conditioned_past = build_latent_conditioned_past(
                    target_model,
                    latent_prefix=translated_latents,
                )
                normal_logits, ce_loss = compute_suffix_logits_and_loss(
                    target_model=target_model,
                    past_key_values=conditioned_past,
                    lm_input_ids=lm_input_ids,
                    lm_labels=lm_labels,
                )

                with torch.no_grad():
                    plan_past = past_by_node_id[edge.tgt_id]
                    plan_logits, _ = compute_suffix_logits_and_loss(
                        target_model=target_model,
                        past_key_values=plan_past,
                        lm_input_ids=lm_input_ids,
                        lm_labels=lm_labels,
                    )
                    random_latents = translator_pool.translate_hidden_states(
                        src_node_id=edge.src_id,
                        tgt_node_id=edge.tgt_id,
                        source_hidden_states=build_mismatched_source_hidden_states(source_hidden_states),
                    )
                    random_past = build_latent_conditioned_past(
                        target_model,
                        latent_prefix=random_latents,
                    )
                    random_logits, _ = compute_suffix_logits_and_loss(
                        target_model=target_model,
                        past_key_values=random_past,
                        lm_input_ids=lm_input_ids,
                        lm_labels=lm_labels,
                    )

                plan_loss = compute_plan_similarity_loss(
                    normal_logits=normal_logits,
                    plan_logits=plan_logits,
                    labels=lm_labels,
                )
                random_loss = compute_random_contrast_loss(
                    normal_logits=normal_logits,
                    random_logits=random_logits,
                    labels=lm_labels,
                )
                positive_cosine = F.cosine_similarity(
                    F.softmax(_flatten_logits_for_valid_labels(normal_logits, lm_labels), dim=-1).reshape(-1),
                    F.softmax(_flatten_logits_for_valid_labels(plan_logits, lm_labels), dim=-1).reshape(-1),
                    dim=0,
                )
                plan_weight, random_weight = adjust_interlat_loss_weights(
                    random_contrast_loss=random_loss,
                    plan_similarity_loss=plan_loss,
                    initial_plan_weight=config.plan_similarity_weight,
                    initial_random_weight=config.random_contrast_weight,
                )
                loss = ce_loss + plan_weight * plan_loss + random_weight * random_loss
                total_edge_loss = total_edge_loss + loss
                total_edge_ce = total_edge_ce + ce_loss.detach()
                total_edge_plan = total_edge_plan + plan_loss.detach()
                total_edge_random = total_edge_random + random_loss.detach()
                total_edge_cosine = total_edge_cosine + positive_cosine.detach()
                step_plan_weight += plan_weight
                step_random_weight += random_weight

            total_edge_loss = total_edge_loss / config.grad_accum_steps
            total_edge_loss.backward()
            step_total_loss += float(total_edge_loss.detach().item())
            step_ce_loss += float((total_edge_ce / max(1, len(ctx.edges))).item())
            step_plan_loss += float((total_edge_plan / max(1, len(ctx.edges))).item())
            step_random_loss += float((total_edge_random / max(1, len(ctx.edges))).item())
            step_positive_cosine += float((total_edge_cosine / max(1, len(ctx.edges))).item())
            used_micro_batches += 1

        torch.nn.utils.clip_grad_norm_(translator_pool.parameters(), config.grad_clip_norm)
        optimizer.step()
        scheduler.step()
        gpu_memory_tracker.update()

        running_total_loss += step_total_loss
        running_ce_loss += step_ce_loss
        running_plan_loss += step_plan_loss
        running_random_loss += step_random_loss
        running_positive_cosine += step_positive_cosine
        running_plan_weight += step_plan_weight / max(1, used_micro_batches * len(ctx.edges))
        running_random_weight += step_random_weight / max(1, used_micro_batches * len(ctx.edges))

        if step % config.log_every == 0:
            divisor = float(config.log_every)
            avg_total_loss = running_total_loss / divisor
            avg_ce_loss = running_ce_loss / divisor
            avg_plan_loss = running_plan_loss / divisor
            avg_random_loss = running_random_loss / divisor
            avg_positive_cosine = running_positive_cosine / divisor
            avg_plan_weight = running_plan_weight / divisor
            avg_random_weight = running_random_weight / divisor
            progress_bar.set_postfix(
                loss=f"{avg_total_loss:.4f}",
                ce=f"{avg_ce_loss:.4f}",
                cos=f"{avg_positive_cosine:.4f}",
                lr=f"{scheduler.lr:.2e}",
            )
            gpu_memory = gpu_memory_tracker.summary()
            logging.info(
                "[Step %04d] loss=%.4f | ce=%.4f | plan=%.4f | random=%.4f | positive_cosine=%.4f | plan_w=%.4f | random_w=%.4f | lr=%.2e | gpu_mem_avg=%s | gpu_mem_peak=%s",
                step,
                avg_total_loss,
                avg_ce_loss,
                avg_plan_loss,
                avg_random_loss,
                avg_positive_cosine,
                avg_plan_weight,
                avg_random_weight,
                scheduler.lr,
                gpu_memory["avg_allocated_pretty"],
                gpu_memory["peak_allocated_pretty"],
            )
            running_total_loss = 0.0
            running_ce_loss = 0.0
            running_plan_loss = 0.0
            running_random_loss = 0.0
            running_positive_cosine = 0.0
            running_plan_weight = 0.0
            running_random_weight = 0.0

    final_path = get_train_checkpoint_path(output_path)
    torch.save(translator_pool.state_dict(), final_path)
    final_gpu_memory = gpu_memory_tracker.summary()
    logging.info(
        "[Memory] avg_gpu_mem=%s | peak_gpu_mem=%s | samples=%d",
        final_gpu_memory["avg_allocated_pretty"],
        final_gpu_memory["peak_allocated_pretty"],
        final_gpu_memory["num_samples"],
    )
    logging.info("[Params] trainable_translator_params=%s", f"{count_trainable_parameters(translator_pool):,}")
    logging.info("[Done] final checkpoint saved to %s", final_path)
    logging.info("Saved train log to %s", log_path)
    return final_path
