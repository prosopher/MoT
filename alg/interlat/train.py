from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from core.common import (
    build_step_pasts_and_batches,
    build_receiver_aligned_sharer_token_ids,
    PastKeyValues,
    TokenIDs,
    count_trainable_parameters,
    ensure_token_ids_model,
    extract_past_key_values,
    get_model_parameter_dtype,
    read_json,
    set_seed,
    split_context_and_prompt_token_ids,
    write_json,
)
from core.config import Config
from core.context import Context
from core.model import Model
from core.translator_pool import TranslatorPool
from core.train_util import (
    WarmupCosineScheduler,
    build_training_dataloaders,
    get_train_checkpoint_path,
    get_train_config_path,
    get_train_log_path,
    initialize_train_output_paths,
    load_translator_checkpoints,
    move_trainable_module_to_config_dtype,
    save_translator_checkpoints,
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
    """Interlat-style hidden-state adapter.

    The upstream Interlat path directly communicates last-layer hidden states and
    lets a compact hidden-state processor adapt their numerical range before
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


def initialize_translators(ctx: Context) -> TranslatorPool:
    config = ctx.config
    translator_pool = ctx.tp
    translator_pool.interlat_node_model_ids = {node.id: node.model_id for node in ctx.nodes}
    for edge in ctx.edges:
        translator_id = get_translator_id(
            translator_pool.interlat_node_model_ids[edge.src_id],
            translator_pool.interlat_node_model_ids[edge.tgt_id],
        )
        if translator_id in translator_pool.translators:
            continue
        src_spec = ctx.tp.get_model_spec(edge.src_id)
        tgt_spec = ctx.tp.get_model_spec(edge.tgt_id)
        translator_pool.add_translator(
            translator_id,
            InterLatEdgeTranslator(
                source_hidden_size=src_spec.hidden_size,
                target_hidden_size=tgt_spec.hidden_size,
                prepended_num_heads=config.prepended_num_heads,
            ),
        )
    return translator_pool


def translate_hidden_states(
    *,
    translator_pool: TranslatorPool,
    source_hidden_states: torch.Tensor,
    src_node_id: str,
    tgt_node_id: str,
) -> torch.Tensor:
    node_model_ids = getattr(translator_pool, "interlat_node_model_ids", None)
    if node_model_ids is None:
        raise ValueError("InterLat translator metadata has not been initialized.")
    translator_id = get_translator_id(node_model_ids[src_node_id], node_model_ids[tgt_node_id])
    if translator_id not in translator_pool.translators:
        raise ValueError(f"InterLat translator {translator_id} is not available. Active translators: {list(translator_pool.translators.keys())}")
    return translator_pool.translators[translator_id](source_hidden_states)


def retokenize_agent_runner_context(
    *,
    source_model: Model,
    target_model: Model,
    source_context_token_ids: TokenIDs,
) -> TokenIDs:
    """Return the target-tokenizer representation of an AgentRunner cache.

    This intentionally matches the C2C/LSC AgentRunner contract: the logical
    context text is preserved while the target tokenizer defines the KV grid.
    """
    ensure_token_ids_model(source_model, source_context_token_ids)
    if source_context_token_ids.ndim != 2 or source_context_token_ids.shape[0] != 1:
        raise ValueError("AgentRunner InterLat retokenization expects a single batch row.")
    source_ids = source_context_token_ids.as_tensor()[0].detach().cpu().tolist()
    try:
        text = source_model.tokenizer.decode(
            source_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        text = source_model.tokenizer.decode(source_ids, skip_special_tokens=False)
    encoded = target_model.tokenizer(text, return_tensors="pt", add_special_tokens=False)
    target_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    return TokenIDs(target_ids, model_id=target_model.id).to(target_model.device)


@torch.no_grad()
def build_agent_runner_translated_past(
    *,
    translator_pool: TranslatorPool,
    target_context_token_ids: TokenIDs,
    source_model: Model,
    target_model: Model,
    src_node_id: str,
    tgt_node_id: str,
) -> PastKeyValues:
    """Build InterLat KV on the target tokenizer's token grid.

    InterLat communicates one translated hidden state per receiver KV position.
    For heterogeneous tokenizers, target tokenization is therefore authoritative.
    We map every target token position to exactly one source-token position using
    the same receiver-alignment rule as C2C/LSC, extract source hidden states on
    that aligned sequence, translate them, and build the target latent past.
    """
    ensure_token_ids_model(target_model, target_context_token_ids)
    aligned_source_token_ids = build_receiver_aligned_sharer_token_ids(
        receiver_context_token_ids=target_context_token_ids,
        receiver_model=target_model,
        sharer_model=source_model,
    ).to(source_model.device)
    source_hidden_states = extract_interlat_source_hidden_states(
        source_model,
        aligned_source_token_ids,
    )
    translated_latents = translate_hidden_states(
        translator_pool=translator_pool,
        source_hidden_states=source_hidden_states,
        src_node_id=src_node_id,
        tgt_node_id=tgt_node_id,
    )
    target_tokens = int(target_context_token_ids.shape[1])
    if int(translated_latents.shape[1]) != target_tokens:
        raise ValueError(
            f"InterLat target-grid latent length mismatch on {src_node_id}->{tgt_node_id}: "
            f"target_tokens={target_tokens} translated_latents={int(translated_latents.shape[1])}"
        )
    translated_past = build_latent_conditioned_past(
        target_model,
        latent_prefix=translated_latents,
    )
    translated_tokens = int(translated_past[0][0].shape[2]) if translated_past else 0
    if translated_tokens != target_tokens:
        raise ValueError(
            f"InterLat target-grid cache length mismatch on {src_node_id}->{tgt_node_id}: "
            f"target_tokens={target_tokens} translated_tokens={translated_tokens}"
        )
    return translated_past


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
def extract_interlat_source_hidden_states(model, token_ids: TokenIDs) -> torch.Tensor:
    """Extract the last-layer hidden states that InterLat communicates.

    The token ids are the same context token ids used to build the common context cache
    for the other algorithms.  This preserves the benchmark data split while
    keeping InterLat's algorithmic input faithful to hidden-state communication.
    """

    ensure_token_ids_model(model, token_ids)
    outputs = model(
        input_ids=token_ids.as_tensor(),
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
    target context cache on top of the communicated source latent.
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
    prompt_token_ids: TokenIDs,
    label_token_ids: TokenIDs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ensure_token_ids_model(target_model, prompt_token_ids)
    ensure_token_ids_model(target_model, label_token_ids)
    outputs = target_model(
        input_ids=prompt_token_ids.as_tensor(),
        past_key_values=past_key_values,
        use_cache=False,
    )
    logits = outputs.logits
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        label_token_ids.as_tensor().reshape(-1),
        reduction="mean",
    )
    return logits, loss


def _flatten_logits_for_valid_labels(logits: torch.Tensor, label_token_ids: TokenIDs) -> torch.Tensor:
    valid = label_token_ids.as_tensor().reshape(-1).ne(-100)
    flat = logits.reshape(-1, logits.shape[-1])
    if not bool(valid.any()):
        return flat[:0]
    return flat[valid]


def compute_plan_similarity_loss(
    *,
    normal_logits: torch.Tensor,
    plan_logits: torch.Tensor,
    label_token_ids: TokenIDs,
    margin_kl: float = 0.7,
    margin_cos: float = 0.3,
) -> torch.Tensor:
    """Interlat-style plan-aligned regularization on receiver logits."""

    normal = _flatten_logits_for_valid_labels(normal_logits, label_token_ids)
    plan = _flatten_logits_for_valid_labels(plan_logits, label_token_ids).detach()
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
    label_token_ids: TokenIDs,
    margin: float = 0.69,
) -> torch.Tensor:
    """Interlat-style separation from mismatched latent communication."""

    normal = _flatten_logits_for_valid_labels(normal_logits, label_token_ids)
    random = _flatten_logits_for_valid_labels(random_logits, label_token_ids).detach()
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


def build_translator_pool(ctx: Context) -> TranslatorPool:
    translator_pool = initialize_translators(ctx)
    move_trainable_module_to_config_dtype(translator_pool, ctx.config)
    return translator_pool


def load_translator_pool_from_checkpoint(
    checkpoint_dir_path: str,
    device_override: Optional[str] = None,
    dtype_override: Optional[str] = None,
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
    if dtype_override is not None:
        config.dtype = dtype_override
    ctx = Context(config)
    translator_pool = build_translator_pool(ctx)
    load_translator_checkpoints(checkpoint_dir_path_obj, translator_pool)
    move_trainable_module_to_config_dtype(translator_pool, config)
    translator_pool.eval()
    return ctx, translator_pool


def run_train(ctx: Context) -> Path:
    config: TrainConfig = ctx.config
    set_seed(config.seed)
    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    config_path = get_train_config_path(output_path)
    write_json(str(config_path), asdict(config))
    log_path = get_train_log_path(output_path)
    logging.info("Starting training")
    logging.info("train_config=%s", asdict(config))
    logging.info("upstream_vendor_defaults.prepended_length=%d", _VENDOR_MODEL_ARGUMENTS.prepended_length)
    logging.info(
        "InterLat source communication is derived from the same common context cache used by the other algorithms; "
        "the target past is built from the communicated latent sequence only."
    )

    translator_pool = build_translator_pool(ctx)
    translator_pool.train()
    for edge in ctx.edges:
        src_spec = ctx.tp.get_model_spec(edge.src_id)
        tgt_spec = ctx.tp.get_model_spec(edge.tgt_id)
        logging.info(
            "edge=%s | src_hidden=%d | tgt_hidden=%d | context_tokens=%d",
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
    training_dataloaders = build_training_dataloaders(ctx)

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
            past_by_node_id, batches_by_node_id = build_step_pasts_and_batches(
                ctx,
                training_dataloaders,
            )

            total_edge_loss = 0.0
            total_edge_ce = 0.0
            total_edge_plan = 0.0
            total_edge_random = 0.0
            total_edge_cosine = 0.0
            for edge in ctx.edges:
                src_prefix_ids, _, _ = batches_by_node_id[edge.src_id]
                _, prompt_token_ids, label_token_ids = batches_by_node_id[edge.tgt_id]

                target_model = ctx.tp.get_model(edge.tgt_id)
                tgt_model_context_limit = get_model_context_limit(target_model)

                source_hidden_states = extract_interlat_source_hidden_states(
                    ctx.tp.get_model(edge.src_id),
                    src_prefix_ids,
                )

                translated_latents = translate_hidden_states(
                    translator_pool=translator_pool,
                    src_node_id=edge.src_id,
                    tgt_node_id=edge.tgt_id,
                    source_hidden_states=source_hidden_states,
                )
                if translated_latents.shape[1] + prompt_token_ids.shape[1] > tgt_model_context_limit:
                    raise ValueError(
                        "InterLat training sequence does not fit target model context window: "
                        f"target={edge.tgt_id}, model_context_limit={tgt_model_context_limit}, "
                        f"latent_tokens={translated_latents.shape[1]}, prompt_tokens={prompt_token_ids.shape[1]}"
                    )

                conditioned_past = build_latent_conditioned_past(
                    target_model,
                    latent_prefix=translated_latents,
                )
                normal_logits, ce_loss = compute_suffix_logits_and_loss(
                    target_model=target_model,
                    past_key_values=conditioned_past,
                    prompt_token_ids=prompt_token_ids,
                    label_token_ids=label_token_ids,
                )

                with torch.no_grad():
                    plan_past = past_by_node_id[edge.tgt_id]
                    plan_logits, _ = compute_suffix_logits_and_loss(
                        target_model=target_model,
                        past_key_values=plan_past,
                        prompt_token_ids=prompt_token_ids,
                        label_token_ids=label_token_ids,
                    )
                    random_latents = translate_hidden_states(
                        translator_pool=translator_pool,
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
                        prompt_token_ids=prompt_token_ids,
                        label_token_ids=label_token_ids,
                    )

                plan_loss = compute_plan_similarity_loss(
                    normal_logits=normal_logits,
                    plan_logits=plan_logits,
                    label_token_ids=label_token_ids,
                )
                random_loss = compute_random_contrast_loss(
                    normal_logits=normal_logits,
                    random_logits=random_logits,
                    label_token_ids=label_token_ids,
                )
                positive_cosine = F.cosine_similarity(
                    F.softmax(_flatten_logits_for_valid_labels(normal_logits, label_token_ids), dim=-1).reshape(-1),
                    F.softmax(_flatten_logits_for_valid_labels(plan_logits, label_token_ids), dim=-1).reshape(-1),
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
            logging.info(
                "[Step %04d] loss=%.4f | ce=%.4f | plan=%.4f | random=%.4f | positive_cosine=%.4f | plan_w=%.4f | random_w=%.4f | lr=%.2e",
                step,
                avg_total_loss,
                avg_ce_loss,
                avg_plan_loss,
                avg_random_loss,
                avg_positive_cosine,
                avg_plan_weight,
                avg_random_weight,
                scheduler.lr,
            )
            running_total_loss = 0.0
            running_ce_loss = 0.0
            running_plan_loss = 0.0
            running_random_loss = 0.0
            running_positive_cosine = 0.0
            running_plan_weight = 0.0
            running_random_weight = 0.0

    final_path = get_train_checkpoint_path(output_path)
    save_translator_checkpoints(output_path, translator_pool)
    logging.info("[Params] trainable_translator_params=%s", f"{count_trainable_parameters(translator_pool):,}")
    logging.info("[Done] final translator checkpoints saved to %s", final_path)
    logging.info("Saved train log to %s", log_path)
    return final_path
