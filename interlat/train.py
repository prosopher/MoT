from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from core.channel_manager import ChannelManager
from core.config import Config
from core.context import Context
from core.model_manager import ModelManager
from core.model_spec import ModelSpec
from core.train_util import *
from mot.train import (
    build_gpt2_input_hidden_states,
    require_gpt2_transformer,
    run_gpt2_block_with_cache,
)


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
    latent_steps: int
    adapter_dim: int
    adapter_heads: int
    adapter_depth: int
    adapter_mlp_ratio: int
    curriculum_token_mix_start: float
    curriculum_token_mix_end: float
    conditional_jsd_weight: float
    plan_align_weight: float
    contrastive_margin_bits: float
    dtype: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.latent_steps < 1:
            raise ValueError("latent_steps must be >= 1")
        initialize_train_output_paths(self)


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 2) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = self.attn_norm(hidden_states)
        attn_out, _ = self.attn(normalized, normalized, normalized, need_weights=False)
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.ffn(self.ffn_norm(hidden_states))
        return hidden_states


class LatentCommunicationAdapter(nn.Module):
    """
    Interlat-style communication adapter:
    source last-layer hidden states -> lightweight self-attention -> target hidden space.
    """

    def __init__(
        self,
        src_hidden_size: int,
        tgt_hidden_size: int,
        adapter_dim: int,
        adapter_heads: int,
        adapter_depth: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        if adapter_depth < 1:
            raise ValueError("adapter_depth must be >= 1")
        self.input_norm = nn.LayerNorm(src_hidden_size)
        self.input_proj = nn.Linear(src_hidden_size, adapter_dim)
        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=adapter_dim,
                    num_heads=adapter_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(adapter_depth)
            ]
        )
        self.output_norm = nn.LayerNorm(adapter_dim)
        self.output_proj = nn.Linear(adapter_dim, tgt_hidden_size)
        self.residual_proj = nn.Linear(src_hidden_size, tgt_hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(hidden_states)
        hidden_states = self.input_proj(self.input_norm(hidden_states))
        for block in self.blocks:
            hidden_states = block(hidden_states)
        hidden_states = self.output_proj(self.output_norm(hidden_states))
        return hidden_states + residual


class InterlatTranslatorPool(nn.Module):
    def __init__(
        self,
        ctx: Context,
        latent_steps: int,
        adapter_dim: int,
        adapter_heads: int,
        adapter_depth: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        self.ctx = ctx
        self.latent_steps = latent_steps
        self.adapters = nn.ModuleDict()
        for edge in ctx.edges:
            src_spec = ctx.mm.get_model_spec(edge.src_id)
            tgt_spec = ctx.mm.get_model_spec(edge.tgt_id)
            self.adapters[edge.id] = LatentCommunicationAdapter(
                src_hidden_size=src_spec.hidden_size,
                tgt_hidden_size=tgt_spec.hidden_size,
                adapter_dim=adapter_dim,
                adapter_heads=adapter_heads,
                adapter_depth=adapter_depth,
                mlp_ratio=mlp_ratio,
            )
        self.latent_start = nn.ParameterDict(
            {
                node.id: nn.Parameter(torch.zeros(1, 1, ctx.mm.get_model_spec(node.id).hidden_size))
                for node in ctx.nodes
            }
        )
        self.latent_end = nn.ParameterDict(
            {
                node.id: nn.Parameter(torch.zeros(1, 1, ctx.mm.get_model_spec(node.id).hidden_size))
                for node in ctx.nodes
            }
        )

    def _edge_id(self, src_node_id: str, tgt_node_id: str) -> str:
        return f"{src_node_id}_to_{tgt_node_id}"

    def select_source_latents(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        seq_len = last_hidden_state.shape[1]
        steps = min(seq_len, self.latent_steps)
        return last_hidden_state[:, -steps:, :]

    def translate_latents(
        self,
        source_last_hidden_state: torch.Tensor,
        src_node_id: str,
        tgt_node_id: str,
    ) -> torch.Tensor:
        edge_id = self._edge_id(src_node_id, tgt_node_id)
        return self.adapters[edge_id](self.select_source_latents(source_last_hidden_state))

    def build_communication_embeddings(
        self,
        source_last_hidden_state: torch.Tensor,
        src_node_id: str,
        tgt_node_id: str,
        target_model: nn.Module,
        source_input_ids: Optional[torch.Tensor] = None,
        curriculum_token_mix_rate: float = 0.0,
    ) -> torch.Tensor:
        translated = self.translate_latents(source_last_hidden_state, src_node_id, tgt_node_id)
        if curriculum_token_mix_rate > 0.0 and source_input_ids is not None:
            target_token_embeds = extract_target_tail_token_embeddings(
                target_model=target_model,
                source_input_ids=source_input_ids,
                steps=translated.shape[1],
            )
            replace_steps = min(
                translated.shape[1],
                int(round(curriculum_token_mix_rate * translated.shape[1])),
            )
            if replace_steps > 0:
                translated = translated.clone()
                translated[:, :replace_steps, :] = target_token_embeds[:, :replace_steps, :]
        batch_size = translated.shape[0]
        latent_start = self.latent_start[tgt_node_id].expand(batch_size, -1, -1)
        latent_end = self.latent_end[tgt_node_id].expand(batch_size, -1, -1)
        return torch.cat([latent_start, translated, latent_end], dim=1)

    def build_target_past_from_source_latents(
        self,
        *,
        source_last_hidden_state: torch.Tensor,
        prefix_input_ids: torch.Tensor,
        target_model: nn.Module,
        src_node_id: str,
        tgt_node_id: str,
        source_input_ids_for_curriculum: Optional[torch.Tensor] = None,
        curriculum_token_mix_rate: float = 0.0,
    ) -> PastKeyValues:
        communication_embeds = self.build_communication_embeddings(
            source_last_hidden_state=source_last_hidden_state,
            src_node_id=src_node_id,
            tgt_node_id=tgt_node_id,
            target_model=target_model,
            source_input_ids=source_input_ids_for_curriculum,
            curriculum_token_mix_rate=curriculum_token_mix_rate,
        )
        return replay_target_prefill_with_communication(
            target_model=target_model,
            prefix_input_ids=prefix_input_ids,
            communication_embeds=communication_embeds,
        )


@torch.no_grad()
def extract_model_prefill_artifacts(
    model: nn.Module,
    input_ids: torch.Tensor,
) -> Tuple[PastKeyValues, torch.Tensor]:
    transformer = require_gpt2_transformer(model)
    hidden_states = build_gpt2_input_hidden_states(model, input_ids)
    presents: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for block in transformer.h:
        hidden_states, present = run_gpt2_block_with_cache(block, hidden_states)
        presents.append(present)
    ln_f = getattr(transformer, "ln_f", None)
    if ln_f is not None:
        hidden_states = ln_f(hidden_states)
    return tuple(presents), hidden_states



def extract_target_tail_token_embeddings(
    *,
    target_model: nn.Module,
    source_input_ids: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    transformer = require_gpt2_transformer(target_model)
    return transformer.wte(source_input_ids[:, -steps:])



def build_gpt2_input_hidden_states_with_communication(
    target_model: nn.Module,
    prefix_input_ids: torch.Tensor,
    communication_embeds: torch.Tensor,
) -> torch.Tensor:
    transformer = require_gpt2_transformer(target_model)
    if prefix_input_ids.ndim != 2:
        raise ValueError(f"prefix_input_ids must have shape [batch, seq], got {tuple(prefix_input_ids.shape)}")
    if communication_embeds.ndim != 3:
        raise ValueError(
            "communication_embeds must have shape [batch, comm_seq, hidden], "
            f"got {tuple(communication_embeds.shape)}"
        )
    batch_size, prefix_len = prefix_input_ids.shape
    comm_batch_size, comm_len, comm_hidden = communication_embeds.shape
    if comm_batch_size != batch_size:
        raise ValueError(f"Batch mismatch: prefix batch={batch_size}, communication batch={comm_batch_size}")
    if comm_hidden != target_model.config.n_embd:
        raise ValueError(
            f"Communication hidden size {comm_hidden} must match target hidden size {target_model.config.n_embd}"
        )

    total_len = comm_len + prefix_len
    position_ids = torch.arange(total_len, device=prefix_input_ids.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    position_embeddings = transformer.wpe(position_ids)
    comm_hidden_states = communication_embeds.to(position_embeddings.dtype) + position_embeddings[:, :comm_len, :]
    token_hidden_states = transformer.wte(prefix_input_ids) + position_embeddings[:, comm_len:, :]
    hidden_states = torch.cat([comm_hidden_states, token_hidden_states], dim=1)
    drop = getattr(transformer, "drop", None)
    if drop is not None:
        hidden_states = drop(hidden_states)
    return hidden_states



def replay_target_prefill_with_communication(
    target_model: nn.Module,
    prefix_input_ids: torch.Tensor,
    communication_embeds: torch.Tensor,
) -> PastKeyValues:
    transformer = require_gpt2_transformer(target_model)
    hidden_states = build_gpt2_input_hidden_states_with_communication(
        target_model=target_model,
        prefix_input_ids=prefix_input_ids,
        communication_embeds=communication_embeds,
    )
    rebuilt_past: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for block in transformer.h:
        hidden_states, present = run_gpt2_block_with_cache(block, hidden_states)
        rebuilt_past.append(present)
    return tuple(rebuilt_past)



def slice_past_sequence_positions(
    past_key_values: PastKeyValues,
    start_idx: int,
    end_idx: Optional[int] = None,
) -> PastKeyValues:
    sliced = []
    for key, value in past_key_values:
        sliced.append((key[:, :, start_idx:end_idx, :], value[:, :, start_idx:end_idx, :]))
    return tuple(sliced)



def trim_communication_prefix_from_past(
    past_key_values: PastKeyValues,
    communication_length: int,
) -> PastKeyValues:
    if communication_length < 0:
        raise ValueError("communication_length must be >= 0")
    if communication_length == 0:
        return past_key_values
    return slice_past_sequence_positions(past_key_values, communication_length, None)



def _masked_positions(labels: torch.Tensor) -> torch.Tensor:
    return labels.reshape(-1).ne(-100)



def compute_jsd_bits_from_logits(
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
    lm_labels: torch.Tensor,
) -> torch.Tensor:
    mask = _masked_positions(lm_labels)
    if not torch.any(mask):
        return torch.zeros((), device=positive_logits.device, dtype=positive_logits.dtype)
    positive_probs = F.softmax(positive_logits.reshape(-1, positive_logits.shape[-1])[mask], dim=-1).clamp_min(1e-8)
    negative_probs = F.softmax(negative_logits.reshape(-1, negative_logits.shape[-1])[mask], dim=-1).clamp_min(1e-8)
    mixture_probs = 0.5 * (positive_probs + negative_probs)
    js_nats = 0.5 * F.kl_div(positive_probs.log(), mixture_probs, reduction="batchmean")
    js_nats = js_nats + 0.5 * F.kl_div(negative_probs.log(), mixture_probs, reduction="batchmean")
    return js_nats / torch.log(torch.tensor(2.0, device=positive_logits.device, dtype=positive_logits.dtype))



def compute_margin_jsd_separation_loss(
    *,
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
    lm_labels: torch.Tensor,
    margin_bits: float,
) -> torch.Tensor:
    jsd_bits = compute_jsd_bits_from_logits(
        positive_logits=positive_logits,
        negative_logits=negative_logits,
        lm_labels=lm_labels,
    )
    margin_tensor = torch.tensor(margin_bits, device=positive_logits.device, dtype=positive_logits.dtype)
    return torch.clamp(margin_tensor - jsd_bits, min=0.0)



def compute_plan_alignment_loss(
    *,
    positive_logits: torch.Tensor,
    plan_logits: torch.Tensor,
    lm_labels: torch.Tensor,
    kl_weight: float = 0.7,
    cos_weight: float = 0.3,
) -> torch.Tensor:
    mask = _masked_positions(lm_labels)
    if not torch.any(mask):
        return torch.zeros((), device=positive_logits.device, dtype=positive_logits.dtype)
    positive_selected = positive_logits.reshape(-1, positive_logits.shape[-1])[mask]
    plan_selected = plan_logits.reshape(-1, plan_logits.shape[-1])[mask]
    kl_loss = F.kl_div(
        F.log_softmax(positive_selected, dim=-1),
        F.softmax(plan_selected, dim=-1).clamp_min(1e-8),
        reduction="batchmean",
    )
    positive_probs = F.softmax(positive_selected, dim=-1).clamp_min(1e-8).reshape(-1)
    plan_probs = F.softmax(plan_selected, dim=-1).clamp_min(1e-8).reshape(-1)
    cos_loss = 1.0 - F.cosine_similarity(positive_probs, plan_probs, dim=0)
    return (kl_weight * kl_loss) + (cos_weight * cos_loss)



def forward_target_logits(
    *,
    target_model: nn.Module,
    past_key_values: PastKeyValues,
    lm_input_ids: torch.Tensor,
) -> torch.Tensor:
    outputs = target_model(
        input_ids=lm_input_ids,
        past_key_values=past_key_values,
        use_cache=False,
    )
    return outputs.logits



def compute_interlat_training_losses(
    *,
    target_model: nn.Module,
    positive_past_key_values: PastKeyValues,
    negative_past_key_values: PastKeyValues,
    plan_past_key_values: PastKeyValues,
    lm_input_ids: torch.Tensor,
    lm_labels: torch.Tensor,
    conditional_jsd_weight: float,
    plan_align_weight: float,
    contrastive_margin_bits: float,
) -> Dict[str, torch.Tensor]:
    positive_logits = forward_target_logits(
        target_model=target_model,
        past_key_values=positive_past_key_values,
        lm_input_ids=lm_input_ids,
    )
    vocab_size = positive_logits.shape[-1]
    ce_loss = F.cross_entropy(
        positive_logits.reshape(-1, vocab_size),
        lm_labels.reshape(-1),
        ignore_index=-100,
        reduction="mean",
    )

    negative_logits = forward_target_logits(
        target_model=target_model,
        past_key_values=negative_past_key_values,
        lm_input_ids=lm_input_ids,
    )

    with torch.no_grad():
        plan_logits = forward_target_logits(
            target_model=target_model,
            past_key_values=plan_past_key_values,
            lm_input_ids=lm_input_ids,
        )

    random_contrast = compute_margin_jsd_separation_loss(
        positive_logits=positive_logits,
        negative_logits=negative_logits,
        lm_labels=lm_labels,
        margin_bits=contrastive_margin_bits,
    )
    plan_align = compute_plan_alignment_loss(
        positive_logits=positive_logits,
        plan_logits=plan_logits,
        lm_labels=lm_labels,
    )

    total_loss = ce_loss + (conditional_jsd_weight * random_contrast) + (plan_align_weight * plan_align)
    return {
        "total": total_loss,
        "ce": ce_loss.detach(),
        "jsd": random_contrast.detach(),
        "plan_align": plan_align.detach(),
    }



def compute_curriculum_token_mix_rate(config: TrainConfig, step: int) -> float:
    if config.max_steps <= 1:
        return float(config.curriculum_token_mix_end)
    progress = (max(1, step) - 1) / float(config.max_steps - 1)
    start = float(config.curriculum_token_mix_start)
    end = float(config.curriculum_token_mix_end)
    return max(0.0, min(1.0, start + ((end - start) * progress)))



def build_translator_pool(ctx: Context) -> InterlatTranslatorPool:
    config = ctx.config
    translator_pool = InterlatTranslatorPool(
        ctx=ctx,
        latent_steps=config.latent_steps,
        adapter_dim=config.adapter_dim,
        adapter_heads=config.adapter_heads,
        adapter_depth=config.adapter_depth,
        mlp_ratio=config.adapter_mlp_ratio,
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
    InterlatTranslatorPool,
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

    config_path = get_train_config_path(output_path)
    write_json(str(config_path), asdict(config))

    log_path = get_train_log_path(output_path)
    logging.info("Starting training")
    logging.info("train_config=%s", asdict(config))
    logging.info("nodes=%s", [asdict(node) for node in nodes])
    logging.info("edges=%s", [edge.id for edge in edges])

    translator_pool = build_translator_pool(ctx)
    translator_pool.train()

    for node in nodes:
        spec = ctx.mm.get_model_spec(node.id)
        logging.info(
            "translation_spec: %s layers=%d hidden=%d heads=%d (%s)",
            node.id,
            spec.num_layers,
            spec.hidden_size,
            spec.num_heads,
            node.model_id,
        )
    logging.info("[Setup] trainable translator params = %s", f"{count_trainable_parameters(translator_pool):,}")

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

    running_total_loss = 0.0
    running_ce_loss = 0.0
    running_jsd = 0.0
    running_align = 0.0
    progress_bar = tqdm(range(1, config.max_steps + 1), desc="Training")

    for step in progress_bar:
        optimizer.zero_grad(set_to_none=True)
        step_total_loss = 0.0
        step_ce_loss = 0.0
        step_jsd = 0.0
        step_align = 0.0
        curriculum_token_mix_rate = compute_curriculum_token_mix_rate(config, step)

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
                last_hidden_by_node_id = {
                    node.id: prefill_by_node_id[node.id][1]
                    for node in nodes
                }

            total_direction_loss = 0.0
            total_direction_ce = 0.0
            total_direction_jsd = 0.0
            total_direction_align = 0.0
            for edge in edges:
                positive_past = translator_pool.build_target_past_from_source_latents(
                    source_last_hidden_state=last_hidden_by_node_id[edge.src_id],
                    prefix_input_ids=prefix_cache_ids,
                    target_model=ctx.mm.get_model(edge.tgt_id),
                    src_node_id=edge.src_id,
                    tgt_node_id=edge.tgt_id,
                    source_input_ids_for_curriculum=prefix_cache_ids,
                    curriculum_token_mix_rate=curriculum_token_mix_rate,
                )
                negative_source_hidden = last_hidden_by_node_id[edge.src_id].roll(shifts=1, dims=0)
                negative_past = translator_pool.build_target_past_from_source_latents(
                    source_last_hidden_state=negative_source_hidden,
                    prefix_input_ids=prefix_cache_ids,
                    target_model=ctx.mm.get_model(edge.tgt_id),
                    src_node_id=edge.src_id,
                    tgt_node_id=edge.tgt_id,
                    source_input_ids_for_curriculum=None,
                    curriculum_token_mix_rate=0.0,
                )
                plan_past = translator_pool.build_target_past_from_source_latents(
                    source_last_hidden_state=last_hidden_by_node_id[edge.src_id],
                    prefix_input_ids=prefix_cache_ids,
                    target_model=ctx.mm.get_model(edge.tgt_id),
                    src_node_id=edge.src_id,
                    tgt_node_id=edge.tgt_id,
                    source_input_ids_for_curriculum=prefix_cache_ids,
                    curriculum_token_mix_rate=1.0,
                )
                losses = compute_interlat_training_losses(
                    target_model=ctx.mm.get_model(edge.tgt_id),
                    positive_past_key_values=positive_past,
                    negative_past_key_values=negative_past,
                    plan_past_key_values=plan_past,
                    lm_input_ids=lm_input_ids,
                    lm_labels=lm_labels,
                    conditional_jsd_weight=config.conditional_jsd_weight,
                    plan_align_weight=config.plan_align_weight,
                    contrastive_margin_bits=config.contrastive_margin_bits,
                )
                total_direction_loss = total_direction_loss + losses["total"]
                total_direction_ce = total_direction_ce + losses["ce"]
                total_direction_jsd = total_direction_jsd + losses["jsd"]
                total_direction_align = total_direction_align + losses["plan_align"]

            loss = total_direction_loss / config.grad_accum_steps
            loss.backward()
            step_total_loss += float(loss.detach().item())
            step_ce_loss += float((total_direction_ce / max(1, len(edges))).item())
            step_jsd += float((total_direction_jsd / max(1, len(edges))).item())
            step_align += float((total_direction_align / max(1, len(edges))).item())

        torch.nn.utils.clip_grad_norm_(translator_pool.parameters(), config.grad_clip_norm)
        optimizer.step()
        scheduler.step()
        gpu_memory_tracker.update()

        running_total_loss += step_total_loss
        running_ce_loss += step_ce_loss
        running_jsd += step_jsd
        running_align += step_align
        if step % config.log_every == 0:
            avg_total_loss = running_total_loss / config.log_every
            avg_ce_loss = running_ce_loss / config.log_every
            avg_jsd = running_jsd / config.log_every
            avg_align = running_align / config.log_every
            progress_bar.set_postfix(
                loss=f"{avg_total_loss:.4f}",
                ce=f"{avg_ce_loss:.4f}",
                jsd=f"{avg_jsd:.4f}",
                lr=f"{scheduler.lr:.2e}",
            )
            gpu_memory = gpu_memory_tracker.summary()
            logging.info(
                "[Step %04d] total_loss=%.4f | ce=%.4f | jsd_margin=%.4f | plan_align=%.4f | token_mix=%.3f | lr=%.2e | gpu_mem_avg=%s | gpu_mem_peak=%s",
                step,
                avg_total_loss,
                avg_ce_loss,
                avg_jsd,
                avg_align,
                curriculum_token_mix_rate,
                scheduler.lr,
                gpu_memory["avg_allocated_pretty"],
                gpu_memory["peak_allocated_pretty"],
            )
            running_total_loss = 0.0
            running_ce_loss = 0.0
            running_jsd = 0.0
            running_align = 0.0

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
