from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from core.channel_manager import ChannelManager
from core.common import (
    GPUMemoryTracker,
    PastKeyValues,
    compute_suffix_lm_loss,
    count_trainable_parameters,
    extract_past_key_values,
    load_frozen_model,
    load_tokenizer,
    read_json,
    set_seed,
    write_json,
)
from core.config import Config
from core.context import Context
from core.model_manager import ModelManager
from core.train_util import (
    build_models_and_tokenizers,
    WarmupCosineScheduler,
    get_train_checkpoint_path,
    get_train_config_path,
    get_train_log_path,
    initialize_train_output_paths
)
from core.topology import Edge, Node
from interlat.vender import ModelArguments as VendorModelArguments
from interlat.vender.hidden_model.custom_model import AdaptiveProjection, HiddenStateProcessor


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
    latent_tokens: int
    latent_dim: int
    translator_heads: int
    translator_layers: int
    translator_mlp_ratio: int
    plan_similarity_weight: float
    random_contrast_weight: float
    contrastive_margin: float
    dtype: str

    def __post_init__(self) -> None:
        super().__post_init__()
        initialize_train_output_paths(self)
        if self.prefix_tokens < 2:
            raise ValueError("prefix_tokens must be >= 2")
        if self.prefix_tokens >= self.total_tokens:
            raise ValueError("prefix_tokens must be smaller than total_tokens")
        if self.latent_tokens < 1:
            raise ValueError("latent_tokens must be >= 1")


class OpenWebTextRawStream(IterableDataset):
    def __init__(
        self,
        *,
        split: str,
        shuffle: bool,
        seed: int,
        shuffle_buffer: int,
    ) -> None:
        super().__init__()
        if split != "train":
            raise ValueError(f"Unsupported OpenWebText split: {split}")
        self.split = split
        self.shuffle = shuffle
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer

    def _build_stream(self):
        stream = load_dataset("openwebtext", split=self.split, streaming=True)
        if self.shuffle:
            stream = stream.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)
        return stream

    def __iter__(self) -> Iterable[str]:
        stream = self._build_stream()
        for example in stream:
            text = str(example.get("text", "") or "")
            if text and not text.isspace():
                yield text


class InfiniteTextDataLoader:
    def __init__(self, dataloader: DataLoader) -> None:
        self.dataloader = dataloader
        self.iterator: Iterator[List[str]] = iter(self.dataloader)

    def __next__(self) -> List[str]:
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)


class CrossAttentionRefiner(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(
            self.query_norm(queries),
            self.context_norm(context),
            self.context_norm(context),
            need_weights=False,
        )
        hidden = queries + attn_out
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        return hidden


class InterLatEdgeTranslator(nn.Module):
    def __init__(
        self,
        *,
        source_hidden_size: int,
        target_hidden_size: int,
        latent_tokens: int,
        latent_dim: int,
        translator_heads: int,
        translator_layers: int,
        translator_mlp_ratio: int,
    ) -> None:
        super().__init__()
        self.latent_tokens = latent_tokens
        self.source_norm = nn.LayerNorm(source_hidden_size)
        self.source_proj = nn.Linear(source_hidden_size, latent_dim)
        self.query_tokens = nn.Parameter(torch.randn(latent_tokens, latent_dim) * 0.02)
        self.layers = nn.ModuleList(
            [
                CrossAttentionRefiner(
                    dim=latent_dim,
                    num_heads=translator_heads,
                    mlp_ratio=translator_mlp_ratio,
                )
                for _ in range(translator_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(latent_dim)
        self.output_proj = nn.Linear(latent_dim, target_hidden_size)
        self.hidden_processor = HiddenStateProcessor(
            hidden_size=target_hidden_size,
            num_heads=_candidate_num_heads(target_hidden_size, translator_heads),
        )
        self.output_calibrator = AdaptiveProjection(target_hidden_size)

    def forward(self, source_hidden_states: torch.Tensor) -> torch.Tensor:
        source_hidden_states = self.source_proj(self.source_norm(source_hidden_states))
        batch_size = source_hidden_states.shape[0]
        latent = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        for layer in self.layers:
            latent = layer(latent, source_hidden_states)
        latent = self.output_proj(self.output_norm(latent))
        latent = self.hidden_processor(latent)
        latent = self.output_calibrator(latent)
        return torch.clamp(latent, -10.0, 10.0)


class InterLatTranslatorPool(nn.Module):
    def __init__(self, ctx: Context) -> None:
        super().__init__()
        config = ctx.config
        self.edge_translators = nn.ModuleDict()
        for edge in ctx.edges:
            src_spec = ctx.mm.get_model_spec(edge.src_id)
            tgt_spec = ctx.mm.get_model_spec(edge.tgt_id)
            self.edge_translators[edge.id] = InterLatEdgeTranslator(
                source_hidden_size=src_spec.hidden_size,
                target_hidden_size=tgt_spec.hidden_size,
                latent_tokens=config.latent_tokens,
                latent_dim=config.latent_dim,
                translator_heads=config.translator_heads,
                translator_layers=config.translator_layers,
                translator_mlp_ratio=config.translator_mlp_ratio,
            )

    def translate_hidden_states(self, *, edge_id: str, source_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.edge_translators[edge_id](source_hidden_states)


@torch.no_grad()
def extract_last_hidden_states(model, input_ids: torch.Tensor) -> torch.Tensor:
    outputs = model(input_ids=input_ids, use_cache=False, output_hidden_states=True)
    return outputs.hidden_states[-1]


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


def trim_input_ids_from_left(input_ids: torch.Tensor, *, max_length: int) -> torch.Tensor:
    if max_length < 1:
        raise ValueError(f"max_length must be >= 1, got {max_length}")
    if input_ids.shape[1] <= max_length:
        return input_ids
    return input_ids[:, -max_length:]


@torch.no_grad()
def build_latent_conditioned_past(
    model,
    *,
    prefix_input_ids: torch.Tensor,
    latent_prefix: torch.Tensor,
) -> PastKeyValues:
    model_context_limit = get_model_context_limit(model)
    latent_tokens = int(latent_prefix.shape[1])
    if latent_tokens >= model_context_limit:
        raise ValueError(
            f"latent_prefix length ({latent_tokens}) must be smaller than model context limit ({model_context_limit})"
        )

    max_prefix_tokens = model_context_limit - latent_tokens
    prefix_input_ids = trim_input_ids_from_left(prefix_input_ids, max_length=max_prefix_tokens)

    token_embeds = model.get_input_embeddings()(prefix_input_ids)
    combined_embeds = torch.cat([latent_prefix.to(token_embeds.dtype), token_embeds], dim=1)
    attention_mask = torch.ones(
        combined_embeds.shape[:2],
        dtype=torch.long,
        device=combined_embeds.device,
    )
    outputs = model(
        input_ids=None,
        inputs_embeds=combined_embeds,
        attention_mask=attention_mask,
        use_cache=True,
    )
    return outputs.past_key_values


def compute_alignment_losses(
    *,
    translated_latents: torch.Tensor,
    target_hidden_states: torch.Tensor,
    contrastive_margin: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    translated_summary = F.normalize(translated_latents.mean(dim=1), dim=-1)
    target_summary = F.normalize(target_hidden_states.mean(dim=1), dim=-1)
    positive_cosine = F.cosine_similarity(translated_summary, target_summary, dim=-1)
    plan_similarity_loss = 1.0 - positive_cosine.mean()

    if translated_summary.shape[0] < 2:
        random_contrast_loss = translated_summary.new_zeros(())
    else:
        negative_summary = torch.roll(target_summary, shifts=1, dims=0)
        negative_cosine = F.cosine_similarity(translated_summary, negative_summary, dim=-1)
        margin = torch.full_like(positive_cosine, float(contrastive_margin))
        random_contrast_loss = F.relu(margin + negative_cosine - positive_cosine).mean()
    return plan_similarity_loss, random_contrast_loss, positive_cosine.mean()


def _candidate_num_heads(hidden_size: int, requested_heads: int) -> int:
    for candidate in range(min(requested_heads, hidden_size), 0, -1):
        if hidden_size % candidate == 0:
            return candidate
    return 1


class NodeTokenizerPool:
    def __init__(self, nodes: Sequence[Node]):
        self.tokenizers = {
            node.id: self._load(node.model_id)
            for node in nodes
        }

    @staticmethod
    def _load(model_id: str):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    def __getitem__(self, node_id: str):
        return self.tokenizers[node_id]


class TokenizedBatch:
    def __init__(self, input_ids: torch.Tensor) -> None:
        self.input_ids = input_ids


def tokenize_valid_texts(
    *,
    tokenizer,
    texts: Sequence[str],
    total_tokens: int,
    device: str,
) -> Tuple[TokenizedBatch, torch.Tensor]:
    encoded = tokenizer(
        list(texts),
        add_special_tokens=False,
        truncation=True,
        max_length=total_tokens,
        padding="max_length",
        return_tensors="pt",
        return_length=True,
    )
    lengths = encoded["length"]
    valid_mask = lengths >= total_tokens
    input_ids = encoded["input_ids"].to(device)
    return TokenizedBatch(input_ids=input_ids), valid_mask


def build_translator_pool(ctx: Context) -> InterLatTranslatorPool:
    pool = InterLatTranslatorPool(ctx)
    pool.to(ctx.config.device)
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
    translator_pool.to(config.device)
    translator_pool.eval()
    node_tokenizers = NodeTokenizerPool(nodes)
    return ctx, translator_pool, node_tokenizers


def _build_text_dataloader(config: TrainConfig) -> InfiniteTextDataLoader:
    dataset = OpenWebTextRawStream(
        split="train",
        shuffle=True,
        seed=config.seed,
        shuffle_buffer=config.shuffle_buffer,
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, num_workers=0, collate_fn=lambda batch: batch)
    return InfiniteTextDataLoader(dataloader)


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

    translator_pool = build_translator_pool(ctx)
    translator_pool.train()
    node_tokenizers = NodeTokenizerPool(ctx.nodes)

    for edge in ctx.edges:
        src_spec = ctx.mm.get_model_spec(edge.src_id)
        tgt_spec = ctx.mm.get_model_spec(edge.tgt_id)
        logging.info(
            "edge=%s | src_hidden=%d | tgt_hidden=%d | latent_tokens=%d",
            edge.id,
            src_spec.hidden_size,
            tgt_spec.hidden_size,
            config.latent_tokens,
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
    text_loader = _build_text_dataloader(config)

    running_total_loss = 0.0
    running_ce_loss = 0.0
    running_plan_loss = 0.0
    running_random_loss = 0.0
    running_positive_cosine = 0.0

    progress_bar = tqdm(range(1, config.max_steps + 1), desc="Interlat Training")
    for step in progress_bar:
        optimizer.zero_grad(set_to_none=True)
        step_total_loss = 0.0
        step_ce_loss = 0.0
        step_plan_loss = 0.0
        step_random_loss = 0.0
        step_positive_cosine = 0.0
        used_micro_batches = 0

        while used_micro_batches < config.grad_accum_steps:
            texts = next(text_loader)
            tokenized_by_node = {}
            valid_mask = None
            for node in ctx.nodes:
                tokenized, node_valid = tokenize_valid_texts(
                    tokenizer=node_tokenizers[node.id],
                    texts=texts,
                    total_tokens=config.total_tokens,
                    device=config.device,
                )
                tokenized_by_node[node.id] = tokenized
                valid_mask = node_valid if valid_mask is None else (valid_mask & node_valid)

            if valid_mask is None or not bool(valid_mask.any()):
                continue

            for node_id, tokenized in tokenized_by_node.items():
                tokenized.input_ids = tokenized.input_ids[valid_mask.to(tokenized.input_ids.device)]
            batch_size = int(tokenized_by_node[ctx.nodes[0].id].input_ids.shape[0])
            if batch_size < 1:
                continue

            total_edge_loss = 0.0
            total_edge_ce = 0.0
            total_edge_plan = 0.0
            total_edge_random = 0.0
            total_edge_cosine = 0.0
            for edge in ctx.edges:
                source_input_ids = tokenized_by_node[edge.src_id].input_ids
                target_input_ids = tokenized_by_node[edge.tgt_id].input_ids

                src_prefix_ids = source_input_ids[:, : config.prefix_tokens - 1]
                tgt_prefix_ids = target_input_ids[:, : config.prefix_tokens - 1]
                lm_input_ids = target_input_ids[:, config.prefix_tokens - 1 : -1]
                lm_labels = target_input_ids[:, config.prefix_tokens:]

                tgt_model_context_limit = get_model_context_limit(ctx.mm.get_model(edge.tgt_id))
                translated_prefix_budget = tgt_model_context_limit - config.latent_tokens - lm_input_ids.shape[1]
                if translated_prefix_budget < 1:
                    raise ValueError(
                        "InterLat training sequence does not fit target model context window: "
                        f"target={edge.tgt_id}, model_context_limit={tgt_model_context_limit}, "
                        f"latent_tokens={config.latent_tokens}, lm_input_tokens={lm_input_ids.shape[1]}"
                    )
                tgt_prefix_ids = trim_input_ids_from_left(tgt_prefix_ids, max_length=translated_prefix_budget)

                with torch.no_grad():
                    source_hidden_states = extract_last_hidden_states(
                        ctx.mm.get_model(edge.src_id),
                        src_prefix_ids,
                    )
                    target_hidden_states = extract_last_hidden_states(
                        ctx.mm.get_model(edge.tgt_id),
                        tgt_prefix_ids,
                    )

                translated_latents = translator_pool.translate_hidden_states(
                    edge_id=edge.id,
                    source_hidden_states=source_hidden_states,
                )
                conditioned_past = build_latent_conditioned_past(
                    ctx.mm.get_model(edge.tgt_id),
                    prefix_input_ids=tgt_prefix_ids,
                    latent_prefix=translated_latents,
                )
                ce_loss = compute_suffix_lm_loss(
                    target_model=ctx.mm.get_model(edge.tgt_id),
                    past_key_values=conditioned_past,
                    lm_input_ids=lm_input_ids,
                    lm_labels=lm_labels,
                )
                plan_loss, random_loss, positive_cosine = compute_alignment_losses(
                    translated_latents=translated_latents,
                    target_hidden_states=target_hidden_states,
                    contrastive_margin=config.contrastive_margin,
                )
                loss = (
                    ce_loss
                    + config.plan_similarity_weight * plan_loss
                    + config.random_contrast_weight * random_loss
                )
                total_edge_loss = total_edge_loss + loss
                total_edge_ce = total_edge_ce + ce_loss.detach()
                total_edge_plan = total_edge_plan + plan_loss.detach()
                total_edge_random = total_edge_random + random_loss.detach()
                total_edge_cosine = total_edge_cosine + positive_cosine.detach()

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

        if step % config.log_every == 0:
            divisor = float(config.log_every)
            avg_total_loss = running_total_loss / divisor
            avg_ce_loss = running_ce_loss / divisor
            avg_plan_loss = running_plan_loss / divisor
            avg_random_loss = running_random_loss / divisor
            avg_positive_cosine = running_positive_cosine / divisor
            progress_bar.set_postfix(
                loss=f"{avg_total_loss:.4f}",
                ce=f"{avg_ce_loss:.4f}",
                cos=f"{avg_positive_cosine:.4f}",
                lr=f"{scheduler.lr:.2e}",
            )
            gpu_memory = gpu_memory_tracker.summary()
            logging.info(
                "[Step %04d] loss=%.4f | ce=%.4f | plan=%.4f | random=%.4f | positive_cosine=%.4f | lr=%.2e | gpu_mem_avg=%s | gpu_mem_peak=%s",
                step,
                avg_total_loss,
                avg_ce_loss,
                avg_plan_loss,
                avg_random_loss,
                avg_positive_cosine,
                scheduler.lr,
                gpu_memory["avg_allocated_pretty"],
                gpu_memory["peak_allocated_pretty"],
            )
            running_total_loss = 0.0
            running_ce_loss = 0.0
            running_plan_loss = 0.0
            running_random_loss = 0.0
            running_positive_cosine = 0.0

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
