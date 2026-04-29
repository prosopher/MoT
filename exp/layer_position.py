#!/usr/bin/env python3
import csv
import math
import sys

from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.common import *
from core.channel_manager import ChannelManager
from core.context import Context
from core.model_manager import ModelManager
from core.model_spec import ModelSpec, infer_model_spec_from_config
from core.eval_util import *
from mot.train import *
from core.train_util import *
from transformers import AutoConfig
from exp.compare_tables import (
    AI_PAPER_PALETTE,
    AI_PAPER_MARKERS,
    apply_ai_paper_style,
    style_axes_common,
)


ACCENT_RED = AI_PAPER_PALETTE[0]
ACCENT_AQUA = AI_PAPER_PALETTE[1]
ACCENT_PURPLE = AI_PAPER_PALETTE[2]
ACCENT_BLUE = AI_PAPER_PALETTE[3]
ACCENT_GREEN = AI_PAPER_PALETTE[4]
ACCENT_ORANGE = AI_PAPER_PALETTE[5]
ACCENT_BLACK = AI_PAPER_PALETTE[6]


@dataclass
class LayerPositionConfig(TrainConfig):
    injection_layer_start_idx: int
    injection_window_size: int
    study_id: Optional[str]

    eval_batch_size: int
    eval_num_workers: int
    eval_max_examples_per_dataset: int
    eval_shuffle_stream: bool
    benchmark_mode: str
    generation_max_new_tokens: int
    kv_similarity_token_group_size: int = 8

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")
        if self.prefix_tokens < 2 or self.prefix_tokens >= self.total_tokens:
            raise ValueError("prefix_tokens must satisfy 2 <= prefix_tokens < total_tokens")
        if self.injection_layer_start_idx < 0:
            raise ValueError("injection_layer_start_idx must be >= 0")
        if self.injection_window_size < 1:
            raise ValueError("injection_window_size must be >= 1")
        if self.benchmark_mode not in {"logit_qa", "gen_qa"}:
            raise ValueError("benchmark_mode must be one of {'logit_qa', 'gen_qa'}")
        if self.translator_dim % self.translator_heads != 0:
            raise ValueError("translator_dim must be divisible by translator_heads")
        if self.kv_similarity_token_group_size < 1:
            raise ValueError("kv_similarity_token_group_size must be >= 1")


class AccuracyMeter:
    def __init__(self) -> None:
        self.accuracy_sum = 0.0
        self.native_accuracy_sum = 0.0
        self.count = 0

    def update(self, accuracy_value: float, native_accuracy_value: float, n: int = 1) -> None:
        self.accuracy_sum += float(accuracy_value) * n
        self.native_accuracy_sum += float(native_accuracy_value) * n
        self.count += n

    def summary(self) -> Dict[str, float]:
        if self.count == 0:
            return {
                "accuracy": float("nan"),
                "native_accuracy": float("nan"),
                "count": 0,
            }
        return {
            "accuracy": self.accuracy_sum / self.count,
            "native_accuracy": self.native_accuracy_sum / self.count,
            "count": self.count,
        }


class F1Meter:
    def __init__(self) -> None:
        self.f1_sum = 0.0
        self.native_f1_sum = 0.0
        self.count = 0

    def update(self, f1_value: float, native_f1_value: float, n: int = 1) -> None:
        self.f1_sum += float(f1_value) * n
        self.native_f1_sum += float(native_f1_value) * n
        self.count += n

    def summary(self) -> Dict[str, float]:
        if self.count == 0:
            return {
                "f1": float("nan"),
                "native_f1": float("nan"),
                "count": 0,
            }
        return {
            "f1": self.f1_sum / self.count,
            "native_f1": self.native_f1_sum / self.count,
            "count": self.count,
        }


class ControlMetricMeter:
    def __init__(self, metric_name: str) -> None:
        self.metric_name = metric_name
        self.native_sum = 0.0
        self.dir_only_sum = 0.0
        self.mag_only_sum = 0.0
        self.full_mix_sum = 0.0
        self.count = 0

    def update(
        self,
        native_value: float,
        dir_only_value: float,
        mag_only_value: float,
        full_mix_value: float,
        n: int = 1,
    ) -> None:
        self.native_sum += float(native_value) * n
        self.dir_only_sum += float(dir_only_value) * n
        self.mag_only_sum += float(mag_only_value) * n
        self.full_mix_sum += float(full_mix_value) * n
        self.count += n

    def summary(self) -> Dict[str, float]:
        if self.count == 0:
            return {
                self.metric_name: float("nan"),
                f"native_{self.metric_name}": float("nan"),
                f"dir_only_{self.metric_name}": float("nan"),
                f"mag_only_{self.metric_name}": float("nan"),
                f"full_mix_{self.metric_name}": float("nan"),
                "delta_dir_only": float("nan"),
                "delta_mag_only": float("nan"),
                "delta_full_mix": float("nan"),
                "count": 0,
            }
        native_value = self.native_sum / self.count
        dir_only_value = self.dir_only_sum / self.count
        mag_only_value = self.mag_only_sum / self.count
        full_mix_value = self.full_mix_sum / self.count
        delta_dir_only = dir_only_value - native_value
        delta_mag_only = mag_only_value - native_value
        delta_full_mix = full_mix_value - native_value
        return {
            self.metric_name: full_mix_value,
            f"native_{self.metric_name}": native_value,
            f"dir_only_{self.metric_name}": dir_only_value,
            f"mag_only_{self.metric_name}": mag_only_value,
            f"full_mix_{self.metric_name}": full_mix_value,
            "delta_dir_only": delta_dir_only,
            "delta_mag_only": delta_mag_only,
            "delta_full_mix": delta_full_mix,
            "count": self.count,
        }


class LogitKLMeter:
    def __init__(self) -> None:
        self.native_to_dir_only_sum = 0.0
        self.native_to_mag_only_sum = 0.0
        self.native_to_full_mix_sum = 0.0
        self.full_mix_to_dir_only_sum = 0.0
        self.full_mix_to_mag_only_sum = 0.0
        self.count = 0

    def update(
        self,
        native_to_dir_only: float,
        native_to_mag_only: float,
        native_to_full_mix: float,
        full_mix_to_dir_only: float,
        full_mix_to_mag_only: float,
        n: int = 1,
    ) -> None:
        self.native_to_dir_only_sum += float(native_to_dir_only) * n
        self.native_to_mag_only_sum += float(native_to_mag_only) * n
        self.native_to_full_mix_sum += float(native_to_full_mix) * n
        self.full_mix_to_dir_only_sum += float(full_mix_to_dir_only) * n
        self.full_mix_to_mag_only_sum += float(full_mix_to_mag_only) * n
        self.count += n

    def summary(self) -> Dict[str, float]:
        if self.count == 0:
            return {
                "native_to_dir_only_logit_kl": float("nan"),
                "native_to_mag_only_logit_kl": float("nan"),
                "native_to_full_mix_logit_kl": float("nan"),
                "full_mix_to_dir_only_logit_kl": float("nan"),
                "full_mix_to_mag_only_logit_kl": float("nan"),
                "count": 0,
            }
        return {
            "native_to_dir_only_logit_kl": self.native_to_dir_only_sum / self.count,
            "native_to_mag_only_logit_kl": self.native_to_mag_only_sum / self.count,
            "native_to_full_mix_logit_kl": self.native_to_full_mix_sum / self.count,
            "full_mix_to_dir_only_logit_kl": self.full_mix_to_dir_only_sum / self.count,
            "full_mix_to_mag_only_logit_kl": self.full_mix_to_mag_only_sum / self.count,
            "count": self.count,
        }


def rescale_block_to_reference_norm(source_block: torch.Tensor, reference_block: torch.Tensor) -> torch.Tensor:
    if source_block.shape != reference_block.shape:
        raise ValueError(
            "Source/reference blocks must have identical shapes, "
            f"got {tuple(source_block.shape)} vs {tuple(reference_block.shape)}"
        )
    source_norm = source_block.float().norm(dim=-1, keepdim=True).clamp_min(1e-8)
    reference_norm = reference_block.float().norm(dim=-1, keepdim=True)
    scaled = source_block.float() * (reference_norm / source_norm)
    return scaled.to(dtype=source_block.dtype)


def build_direction_only_window(
    translated_key_block: torch.Tensor,
    translated_value_block: torch.Tensor,
    native_key_block: torch.Tensor,
    native_value_block: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        rescale_block_to_reference_norm(translated_key_block, native_key_block),
        rescale_block_to_reference_norm(translated_value_block, native_value_block),
    )


def build_magnitude_only_window(
    translated_key_block: torch.Tensor,
    translated_value_block: torch.Tensor,
    native_key_block: torch.Tensor,
    native_value_block: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        rescale_block_to_reference_norm(native_key_block, translated_key_block),
        rescale_block_to_reference_norm(native_value_block, translated_value_block),
    )


def compute_next_token_log_probs(
    model: PreTrainedModel,
    past_key_values: PastKeyValues,
    seed_token: torch.Tensor,
) -> torch.Tensor:
    outputs = model(
        input_ids=seed_token,
        past_key_values=past_key_values,
        use_cache=False,
    )
    return F.log_softmax(outputs.logits[:, -1, :].float(), dim=-1)


def compute_logit_kl(reference_log_probs: torch.Tensor, candidate_log_probs: torch.Tensor) -> float:
    if reference_log_probs.shape != candidate_log_probs.shape:
        raise ValueError(
            "Reference/candidate log-prob shapes must match, "
            f"got {tuple(reference_log_probs.shape)} vs {tuple(candidate_log_probs.shape)}"
        )
    reference_probs = reference_log_probs.exp()
    kl = torch.sum(reference_probs * (reference_log_probs - candidate_log_probs), dim=-1)
    return float(kl.mean().item())


def sanitize_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "default"



def resolve_run_position_label(config: LayerPositionConfig) -> str:
    return f"injection_layer_start_idx_{config.injection_layer_start_idx:03d}"


def resolve_target_num_layers(
    model_ids: str,
    model_directions: str,
) -> int:
    nodes, edges = build_nodes_and_edges(
        model_ids=model_ids,
        model_directions=model_directions,
    )
    node_map = build_node_map(nodes)
    reference_edge = edges[0]
    target_model_id = node_map[reference_edge.tgt_id].model_id
    return infer_model_spec_from_config(AutoConfig.from_pretrained(target_model_id)).num_layers


def build_control_window_variants(
    native_key_block: torch.Tensor,
    native_value_block: torch.Tensor,
    translated_key_block: torch.Tensor,
    translated_value_block: torch.Tensor,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    dir_only_key, dir_only_value = build_direction_only_window(
        translated_key_block=translated_key_block,
        translated_value_block=translated_value_block,
        native_key_block=native_key_block,
        native_value_block=native_value_block,
    )
    mag_only_key, mag_only_value = build_magnitude_only_window(
        translated_key_block=translated_key_block,
        translated_value_block=translated_value_block,
        native_key_block=native_key_block,
        native_value_block=native_value_block,
    )
    return {
        "dir_only": (dir_only_key, dir_only_value),
        "mag_only": (mag_only_key, mag_only_value),
        "full_mix": (translated_key_block, translated_value_block),
    }


def build_study_dir(config: LayerPositionConfig) -> Path:
    study_id = config.study_id or f"run_{sanitize_slug(config.model_directions)}"
    return Path(config.output_path) / study_id


def build_run_output_dir(config: LayerPositionConfig) -> Path:
    return build_study_dir(config) / resolve_run_position_label(config)


def format_layer_range(start_idx: int, end_idx: int) -> str:
    if start_idx == end_idx:
        return f"L{start_idx}"
    return f"L{start_idx}-L{end_idx}"


def format_window_title(injection_window_size: int) -> str:
    if injection_window_size < 1:
        raise ValueError("injection_window_size must be >= 1")
    return f"win={injection_window_size}"


def build_train_log_path(run_dir: Path) -> Path:
    return run_dir / "target_injection_training.log"


def build_eval_log_path(run_dir: Path) -> Path:
    return run_dir / "target_injection_evaluation.log"


def build_config_path(run_dir: Path) -> Path:
    return run_dir / "target_injection_run_config.json"


def build_metrics_path(run_dir: Path) -> Path:
    return run_dir / "target_injection_evaluation_metrics.json"


def run_train(
    ctx: Context,
    run_dir: Path,
) -> LayerWindowTranslatorPool:
    config = ctx.config
    nodes = ctx.nodes
    node_map = build_node_map(nodes)
    logging.info("Starting layer-window position training with target-layer replay")
    logging.info("experiment_config=%s", asdict(config))

    translator_pool = build_translator_pool(
        ctx=ctx,
    )
    translator_pool.train()
    logging.info("[Setup] translator trainable params = %s", f"{count_trainable_parameters(translator_pool):,}")

    dataloaders_by_target = build_training_dataloaders_by_target(ctx)

    optimizer = torch.optim.AdamW(
        translator_pool.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = WarmupCosineScheduler(optimizer, config.warmup_steps, config.max_steps)
    gpu_memory_tracker = GPUMemoryTracker(config.device)
    running_loss = 0.0

    progress_bar = tqdm(range(1, config.max_steps + 1), desc="LayerPositionTrain")
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
                    past_by_node_id = {
                        node.id: extract_past_key_values(ctx.mm.get_model(node.id), prefix_cache_ids)
                        for node in nodes
                    }
                target_batches[target_node_id] = (prefix_cache_ids, lm_input_ids, lm_labels, past_by_node_id)

            total_direction_loss = 0.0
            for edge in ctx.edges:
                prefix_cache_ids, lm_input_ids, lm_labels, past_by_node_id = target_batches[edge.tgt_id]
                translated_key, translated_value = translator_pool.translate_layer_window(
                    past_key_values=past_by_node_id[edge.src_id],
                    src_node_id=edge.src_id,
                    tgt_node_id=edge.tgt_id,
                )
                mixed_target_past = replay_target_prefill_with_injected_window(
                    target_model=ctx.mm.get_model(edge.tgt_id),
                    prefix_input_ids=prefix_cache_ids,
                    target_layer_indices=ctx.cm.get_tgt_layer_indices(edge.id),
                    injected_key_block=translated_key,
                    injected_value_block=translated_value,
                    tgt_spec=ctx.mm.get_model_spec(edge.tgt_id),
                    target_model_id=node_map[edge.tgt_id].model_id,
                )
                total_direction_loss = total_direction_loss + compute_prefix_correction_and_suffix_lm_loss(
                    target_model=ctx.mm.get_model(edge.tgt_id),
                    past_key_values=mixed_target_past,
                    lm_input_ids=lm_input_ids,
                    lm_labels=lm_labels,
                    native_target_past_key_values=past_by_node_id[edge.tgt_id],
                    target_layer_indices=ctx.cm.get_tgt_layer_indices(edge.id),
                )

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
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{scheduler.lr:.2e}")
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

    logging.info("[Done] training complete")
    return translator_pool


@torch.inference_mode()
def evaluate_logit_dataset(
    ctx: Context,
    spec: HFDatasetSpec,
    dataloader: DataLoader,
    translator_pool: LayerWindowTranslatorPool,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    config = ctx.config
    nodes = ctx.nodes
    node_map = build_node_map(nodes)
    edges = ctx.edges
    path_metrics = {edge.id: ControlMetricMeter("accuracy") for edge in edges}
    path_logit_kl = {edge.id: LogitKLMeter() for edge in edges}
    processed_examples = 0

    for batch_idx, batch in enumerate(dataloader, start=1):
        for example in batch:
            prepared_inputs = prepare_logit_task_inputs(
                spec=spec,
                tokenizer=ctx.mm.get_tokenizer(edges[0].tgt_id),
                context=example.get("context"),
                question=example["question"],
                device=config.device,
                choices=example.get("choices"),
                choice_texts=example.get("choice_texts"),
                subject=example.get("subject"),
            )
            candidate_token_ids = build_logit_answer_candidates(tokenizer=ctx.mm.get_tokenizer(edges[0].tgt_id), spec=spec)
            gold_answer = example["answer"]
            context_input_ids = prepared_inputs["prefix_input_ids"]
            suffix_cache_ids = prepared_inputs["suffix_cache_ids"]
            seed_token = prepared_inputs["seed_token"]
            past_by_node_id = {
                node.id: extract_past_key_values(ctx.mm.get_model(node.id), context_input_ids)
                for node in nodes
            }

            for edge in edges:
                translated_key, translated_value = translator_pool.translate_layer_window(
                    past_key_values=past_by_node_id[edge.src_id],
                    src_node_id=edge.src_id,
                    tgt_node_id=edge.tgt_id,
                )
                native_target_past = past_by_node_id[edge.tgt_id]
                native_key_block, native_value_block = extract_layer_window_blocks(
                    past_key_values=native_target_past,
                    start_layer_idx=ctx.cm.get_tgt_layer_start_idx(edge.id),
                    num_layers=config.injection_window_size,
                )
                control_windows = build_control_window_variants(
                    native_key_block=native_key_block,
                    native_value_block=native_value_block,
                    translated_key_block=translated_key,
                    translated_value_block=translated_value,
                )
                dir_only_past = replay_target_prefill_with_injected_window(
                    target_model=ctx.mm.get_model(edge.tgt_id),
                    prefix_input_ids=context_input_ids,
                    target_layer_indices=ctx.cm.get_tgt_layer_indices(edge.id),
                    injected_key_block=control_windows["dir_only"][0],
                    injected_value_block=control_windows["dir_only"][1],
                    tgt_spec=ctx.mm.get_model_spec(edge.tgt_id),
                    target_model_id=node_map[edge.tgt_id].model_id,
                )
                mag_only_past = replay_target_prefill_with_injected_window(
                    target_model=ctx.mm.get_model(edge.tgt_id),
                    prefix_input_ids=context_input_ids,
                    target_layer_indices=ctx.cm.get_tgt_layer_indices(edge.id),
                    injected_key_block=control_windows["mag_only"][0],
                    injected_value_block=control_windows["mag_only"][1],
                    tgt_spec=ctx.mm.get_model_spec(edge.tgt_id),
                    target_model_id=node_map[edge.tgt_id].model_id,
                )
                full_mix_past = replay_target_prefill_with_injected_window(
                    target_model=ctx.mm.get_model(edge.tgt_id),
                    prefix_input_ids=context_input_ids,
                    target_layer_indices=ctx.cm.get_tgt_layer_indices(edge.id),
                    injected_key_block=control_windows["full_mix"][0],
                    injected_value_block=control_windows["full_mix"][1],
                    tgt_spec=ctx.mm.get_model_spec(edge.tgt_id),
                    target_model_id=node_map[edge.tgt_id].model_id,
                )

                native_scoring_past = prepare_answer_scoring_past(
                    model=ctx.mm.get_model(edge.tgt_id),
                    past_key_values=native_target_past,
                    suffix_cache_ids=suffix_cache_ids,
                )
                dir_only_scoring_past = prepare_answer_scoring_past(
                    model=ctx.mm.get_model(edge.tgt_id),
                    past_key_values=dir_only_past,
                    suffix_cache_ids=suffix_cache_ids,
                )
                mag_only_scoring_past = prepare_answer_scoring_past(
                    model=ctx.mm.get_model(edge.tgt_id),
                    past_key_values=mag_only_past,
                    suffix_cache_ids=suffix_cache_ids,
                )
                full_mix_scoring_past = prepare_answer_scoring_past(
                    model=ctx.mm.get_model(edge.tgt_id),
                    past_key_values=full_mix_past,
                    suffix_cache_ids=suffix_cache_ids,
                )

                native_scores = score_answer_choices(
                    model=ctx.mm.get_model(edge.tgt_id),
                    past_key_values=native_scoring_past,
                    seed_token=seed_token,
                    choice_token_ids=candidate_token_ids,
                    normalize_by_length=True,
                )
                dir_only_scores = score_answer_choices(
                    model=ctx.mm.get_model(edge.tgt_id),
                    past_key_values=dir_only_scoring_past,
                    seed_token=seed_token,
                    choice_token_ids=candidate_token_ids,
                    normalize_by_length=True,
                )
                mag_only_scores = score_answer_choices(
                    model=ctx.mm.get_model(edge.tgt_id),
                    past_key_values=mag_only_scoring_past,
                    seed_token=seed_token,
                    choice_token_ids=candidate_token_ids,
                    normalize_by_length=True,
                )
                full_mix_scores = score_answer_choices(
                    model=ctx.mm.get_model(edge.tgt_id),
                    past_key_values=full_mix_scoring_past,
                    seed_token=seed_token,
                    choice_token_ids=candidate_token_ids,
                    normalize_by_length=True,
                )

                native_pred = predict_answer_label(native_scores)
                dir_only_pred = predict_answer_label(dir_only_scores)
                mag_only_pred = predict_answer_label(mag_only_scores)
                full_mix_pred = predict_answer_label(full_mix_scores)
                path_metrics[edge.id].update(
                    native_value=1.0 if is_logit_answer_correct(native_pred, gold_answer) else 0.0,
                    dir_only_value=1.0 if is_logit_answer_correct(dir_only_pred, gold_answer) else 0.0,
                    mag_only_value=1.0 if is_logit_answer_correct(mag_only_pred, gold_answer) else 0.0,
                    full_mix_value=1.0 if is_logit_answer_correct(full_mix_pred, gold_answer) else 0.0,
                    n=1,
                )

                native_log_probs = compute_next_token_log_probs(
                    model=ctx.mm.get_model(edge.tgt_id),
                    past_key_values=native_scoring_past,
                    seed_token=seed_token,
                )
                dir_only_log_probs = compute_next_token_log_probs(
                    model=ctx.mm.get_model(edge.tgt_id),
                    past_key_values=dir_only_scoring_past,
                    seed_token=seed_token,
                )
                mag_only_log_probs = compute_next_token_log_probs(
                    model=ctx.mm.get_model(edge.tgt_id),
                    past_key_values=mag_only_scoring_past,
                    seed_token=seed_token,
                )
                full_mix_log_probs = compute_next_token_log_probs(
                    model=ctx.mm.get_model(edge.tgt_id),
                    past_key_values=full_mix_scoring_past,
                    seed_token=seed_token,
                )
                path_logit_kl[edge.id].update(
                    native_to_dir_only=compute_logit_kl(native_log_probs, dir_only_log_probs),
                    native_to_mag_only=compute_logit_kl(native_log_probs, mag_only_log_probs),
                    native_to_full_mix=compute_logit_kl(native_log_probs, full_mix_log_probs),
                    full_mix_to_dir_only=compute_logit_kl(full_mix_log_probs, dir_only_log_probs),
                    full_mix_to_mag_only=compute_logit_kl(full_mix_log_probs, mag_only_log_probs),
                    n=1,
                )

            processed_examples += 1

        if batch_idx % 50 == 0:
            logging.info(
                "[%s] progress: %d/%d examples",
                spec.name_for_log,
                processed_examples,
                config.eval_max_examples_per_dataset,
            )

    summarized_metrics = {edge_id: meter.summary() for edge_id, meter in path_metrics.items()}
    summarized_logit_kl = {edge_id: meter.summary() for edge_id, meter in path_logit_kl.items()}
    return summarized_metrics, summarized_logit_kl


@torch.inference_mode()
def evaluate_generation_dataset(
    ctx: Context,
    spec: HFDatasetSpec,
    dataloader: DataLoader,
    translator_pool: LayerWindowTranslatorPool,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    config = ctx.config
    nodes = ctx.nodes
    node_map = build_node_map(nodes)
    edges = ctx.edges
    path_metrics = {edge.id: ControlMetricMeter("f1") for edge in edges}
    path_logit_kl = {edge.id: LogitKLMeter() for edge in edges}
    processed_examples = 0

    for batch_idx, batch in enumerate(dataloader, start=1):
        for example in batch:
            question = example["question"]
            context_text = example["context"]
            gold_answers = example["answers"]

            context_budget = compute_benchmark_context_budget(
                ctx=ctx,
                spec=spec,
                question=question,
                eval_config=config,
                tokenizer=ctx.mm.get_tokenizer(edges[0].tgt_id),
                target_node_id=edges[0].tgt_id,
            )
            prepared_inputs = prepare_generation_task_inputs(
                spec=spec,
                tokenizer=ctx.mm.get_tokenizer(edges[0].tgt_id),
                context=context_text,
                question=question,
                device=config.device,
                max_input_tokens=context_budget,
            )
            prefix_input_ids = prepared_inputs["prefix_input_ids"]
            suffix_cache_ids = prepared_inputs["suffix_cache_ids"]
            seed_token = prepared_inputs["seed_token"]

            if prepared_inputs.get("was_truncated") and processed_examples < 3:
                suffix_cache_tokens = 0 if suffix_cache_ids is None else suffix_cache_ids.shape[1]
                logging.info(
                    "[%s] truncated prefix to %d tokens to fit model context window (suffix_cache_tokens=%d, answer_token_budget=%d)",
                    spec.name_for_log,
                    prefix_input_ids.shape[1],
                    suffix_cache_tokens,
                    get_answer_token_budget(config),
                )

            past_by_node_id = {
                node.id: extract_past_key_values(ctx.mm.get_model(node.id), prefix_input_ids)
                for node in nodes
            }

            for edge in edges:
                translated_key, translated_value = translator_pool.translate_layer_window(
                    past_key_values=past_by_node_id[edge.src_id],
                    src_node_id=edge.src_id,
                    tgt_node_id=edge.tgt_id,
                )
                native_target_past = past_by_node_id[edge.tgt_id]
                native_key_block, native_value_block = extract_layer_window_blocks(
                    past_key_values=native_target_past,
                    start_layer_idx=ctx.cm.get_tgt_layer_start_idx(edge.id),
                    num_layers=config.injection_window_size,
                )
                control_windows = build_control_window_variants(
                    native_key_block=native_key_block,
                    native_value_block=native_value_block,
                    translated_key_block=translated_key,
                    translated_value_block=translated_value,
                )
                dir_only_past = replay_target_prefill_with_injected_window(
                    target_model=ctx.mm.get_model(edge.tgt_id),
                    prefix_input_ids=prefix_input_ids,
                    target_layer_indices=ctx.cm.get_tgt_layer_indices(edge.id),
                    injected_key_block=control_windows["dir_only"][0],
                    injected_value_block=control_windows["dir_only"][1],
                    tgt_spec=ctx.mm.get_model_spec(edge.tgt_id),
                    target_model_id=node_map[edge.tgt_id].model_id,
                )
                mag_only_past = replay_target_prefill_with_injected_window(
                    target_model=ctx.mm.get_model(edge.tgt_id),
                    prefix_input_ids=prefix_input_ids,
                    target_layer_indices=ctx.cm.get_tgt_layer_indices(edge.id),
                    injected_key_block=control_windows["mag_only"][0],
                    injected_value_block=control_windows["mag_only"][1],
                    tgt_spec=ctx.mm.get_model_spec(edge.tgt_id),
                    target_model_id=node_map[edge.tgt_id].model_id,
                )
                full_mix_past = replay_target_prefill_with_injected_window(
                    target_model=ctx.mm.get_model(edge.tgt_id),
                    prefix_input_ids=prefix_input_ids,
                    target_layer_indices=ctx.cm.get_tgt_layer_indices(edge.id),
                    injected_key_block=control_windows["full_mix"][0],
                    injected_value_block=control_windows["full_mix"][1],
                    tgt_spec=ctx.mm.get_model_spec(edge.tgt_id),
                    target_model_id=node_map[edge.tgt_id].model_id,
                )

                native_answer = predict_generation_task_answer(
                    model=ctx.mm.get_model(edge.tgt_id),
                    tokenizer=ctx.mm.get_tokenizer(edge.tgt_id),
                    past_key_values=native_target_past,
                    seed_token=seed_token,
                    eval_config=config,
                    suffix_cache_ids=suffix_cache_ids,
                )
                dir_only_answer = predict_generation_task_answer(
                    model=ctx.mm.get_model(edge.tgt_id),
                    tokenizer=ctx.mm.get_tokenizer(edge.tgt_id),
                    past_key_values=dir_only_past,
                    seed_token=seed_token,
                    eval_config=config,
                    suffix_cache_ids=suffix_cache_ids,
                )
                mag_only_answer = predict_generation_task_answer(
                    model=ctx.mm.get_model(edge.tgt_id),
                    tokenizer=ctx.mm.get_tokenizer(edge.tgt_id),
                    past_key_values=mag_only_past,
                    seed_token=seed_token,
                    eval_config=config,
                    suffix_cache_ids=suffix_cache_ids,
                )
                full_mix_answer = predict_generation_task_answer(
                    model=ctx.mm.get_model(edge.tgt_id),
                    tokenizer=ctx.mm.get_tokenizer(edge.tgt_id),
                    past_key_values=full_mix_past,
                    seed_token=seed_token,
                    eval_config=config,
                    suffix_cache_ids=suffix_cache_ids,
                )

                path_metrics[edge.id].update(
                    native_value=compute_generation_f1(native_answer, gold_answers),
                    dir_only_value=compute_generation_f1(dir_only_answer, gold_answers),
                    mag_only_value=compute_generation_f1(mag_only_answer, gold_answers),
                    full_mix_value=compute_generation_f1(full_mix_answer, gold_answers),
                    n=1,
                )

                native_scoring_past = prepare_answer_scoring_past(
                    model=ctx.mm.get_model(edge.tgt_id),
                    past_key_values=native_target_past,
                    suffix_cache_ids=suffix_cache_ids,
                )
                dir_only_scoring_past = prepare_answer_scoring_past(
                    model=ctx.mm.get_model(edge.tgt_id),
                    past_key_values=dir_only_past,
                    suffix_cache_ids=suffix_cache_ids,
                )
                mag_only_scoring_past = prepare_answer_scoring_past(
                    model=ctx.mm.get_model(edge.tgt_id),
                    past_key_values=mag_only_past,
                    suffix_cache_ids=suffix_cache_ids,
                )
                full_mix_scoring_past = prepare_answer_scoring_past(
                    model=ctx.mm.get_model(edge.tgt_id),
                    past_key_values=full_mix_past,
                    suffix_cache_ids=suffix_cache_ids,
                )

                native_log_probs = compute_next_token_log_probs(
                    model=ctx.mm.get_model(edge.tgt_id),
                    past_key_values=native_scoring_past,
                    seed_token=seed_token,
                )
                dir_only_log_probs = compute_next_token_log_probs(
                    model=ctx.mm.get_model(edge.tgt_id),
                    past_key_values=dir_only_scoring_past,
                    seed_token=seed_token,
                )
                mag_only_log_probs = compute_next_token_log_probs(
                    model=ctx.mm.get_model(edge.tgt_id),
                    past_key_values=mag_only_scoring_past,
                    seed_token=seed_token,
                )
                full_mix_log_probs = compute_next_token_log_probs(
                    model=ctx.mm.get_model(edge.tgt_id),
                    past_key_values=full_mix_scoring_past,
                    seed_token=seed_token,
                )
                path_logit_kl[edge.id].update(
                    native_to_dir_only=compute_logit_kl(native_log_probs, dir_only_log_probs),
                    native_to_mag_only=compute_logit_kl(native_log_probs, mag_only_log_probs),
                    native_to_full_mix=compute_logit_kl(native_log_probs, full_mix_log_probs),
                    full_mix_to_dir_only=compute_logit_kl(full_mix_log_probs, dir_only_log_probs),
                    full_mix_to_mag_only=compute_logit_kl(full_mix_log_probs, mag_only_log_probs),
                    n=1,
                )

            processed_examples += 1

        if batch_idx % 25 == 0:
            logging.info(
                "[%s] generation progress: %d/%d examples",
                spec.name_for_log,
                processed_examples,
                config.eval_max_examples_per_dataset,
            )

    summarized_metrics = {edge_id: meter.summary() for edge_id, meter in path_metrics.items()}
    summarized_logit_kl = {edge_id: meter.summary() for edge_id, meter in path_logit_kl.items()}
    return summarized_metrics, summarized_logit_kl


@torch.inference_mode()
def compute_openwebtext_native_and_full_mix_losses(
    *,
    ctx: Context,
    edge: Edge,
    prefix_cache_ids: torch.Tensor,
    lm_input_ids: torch.Tensor,
    lm_labels: torch.Tensor,
    past_by_node_id,
    translator_pool: LayerWindowTranslatorPool,
) -> Dict[str, float]:
    translated_key, translated_value = translator_pool.translate_layer_window(
        past_key_values=past_by_node_id[edge.src_id],
        src_node_id=edge.src_id,
        tgt_node_id=edge.tgt_id,
    )
    native_target_past = past_by_node_id[edge.tgt_id]
    full_mix_past = replay_target_prefill_with_injected_window(
        target_model=ctx.mm.get_model(edge.tgt_id),
        prefix_input_ids=prefix_cache_ids,
        target_layer_indices=ctx.cm.get_tgt_layer_indices(edge.id),
        injected_key_block=translated_key,
        injected_value_block=translated_value,
        tgt_spec=ctx.mm.get_model_spec(edge.tgt_id),
        target_model_id=build_node_map(ctx.nodes)[edge.tgt_id].model_id,
    )

    target_model = ctx.mm.get_model(edge.tgt_id)
    target_layer_indices = ctx.cm.get_tgt_layer_indices(edge.id)

    native_loss = float(
        compute_prefix_correction_and_suffix_lm_loss(
            target_model=target_model,
            past_key_values=native_target_past,
            lm_input_ids=lm_input_ids,
            lm_labels=lm_labels,
            native_target_past_key_values=native_target_past,
            target_layer_indices=target_layer_indices,
        ).item()
    )
    full_mix_loss = float(
        compute_prefix_correction_and_suffix_lm_loss(
            target_model=target_model,
            past_key_values=full_mix_past,
            lm_input_ids=lm_input_ids,
            lm_labels=lm_labels,
            native_target_past_key_values=native_target_past,
            target_layer_indices=target_layer_indices,
        ).item()
    )
    return {
        "native": native_loss,
        "full_mix": full_mix_loss,
    }


def compute_average_metric(
    logit_results: Dict[str, Dict[str, Dict[str, float]]],
    metric_key: str,
) -> float:
    values = []
    for dataset_results in logit_results.values():
        for direction_results in dataset_results.values():
            value = direction_results.get(metric_key)
            if value is not None and value == value:
                values.append(float(value))
    if not values:
        return float("nan")
    return sum(values) / len(values)


class KVSimilarityAccumulator:
    def __init__(self, num_layers: int) -> None:
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.num_layers = num_layers
        self.sum_matrix = torch.zeros((num_layers, 0), dtype=torch.float64)
        self.count_matrix = torch.zeros((num_layers, 0), dtype=torch.float64)
        self.group_labels: List[str] = []
        self.segment_group_counts = {"prefix": 0, "suffix": 0, "generated": 0}

    def _ensure_width(self, width: int) -> None:
        if width <= self.sum_matrix.shape[1]:
            return
        pad_width = width - self.sum_matrix.shape[1]
        self.sum_matrix = torch.cat(
            [self.sum_matrix, torch.zeros((self.num_layers, pad_width), dtype=self.sum_matrix.dtype)],
            dim=1,
        )
        self.count_matrix = torch.cat(
            [self.count_matrix, torch.zeros((self.num_layers, pad_width), dtype=self.count_matrix.dtype)],
            dim=1,
        )

    def update(
        self,
        similarity_matrix: torch.Tensor,
        group_labels: List[str],
        segment_group_counts: Dict[str, int],
    ) -> None:
        matrix = similarity_matrix.detach().cpu().to(dtype=torch.float64)
        if matrix.ndim != 2 or matrix.shape[0] != self.num_layers:
            raise ValueError(
                f"similarity_matrix must have shape ({self.num_layers}, G), got {tuple(matrix.shape)}"
            )
        self._ensure_width(matrix.shape[1])
        self.sum_matrix[:, : matrix.shape[1]] += matrix
        self.count_matrix[:, : matrix.shape[1]] += 1.0
        if len(group_labels) > len(self.group_labels):
            self.group_labels = list(group_labels)
        for segment_name in self.segment_group_counts:
            self.segment_group_counts[segment_name] = max(
                self.segment_group_counts[segment_name],
                int(segment_group_counts.get(segment_name, 0)),
            )

    def summary(self) -> Dict[str, Any]:
        if self.sum_matrix.shape[1] == 0:
            return {
                "matrix": torch.empty((self.num_layers, 0), dtype=torch.float32),
                "count_matrix": torch.empty((self.num_layers, 0), dtype=torch.float32),
                "group_labels": [],
                "segment_group_counts": dict(self.segment_group_counts),
            }
        average_matrix = torch.full_like(self.sum_matrix, float("nan"))
        valid_mask = self.count_matrix > 0
        average_matrix[valid_mask] = self.sum_matrix[valid_mask] / self.count_matrix[valid_mask]
        return {
            "matrix": average_matrix.to(dtype=torch.float32),
            "count_matrix": self.count_matrix.to(dtype=torch.float32),
            "group_labels": list(self.group_labels),
            "segment_group_counts": dict(self.segment_group_counts),
        }


def slice_past_key_values_batch(
    past_key_values: PastKeyValues,
    batch_idx: int,
) -> PastKeyValues:
    return tuple(
        (key[batch_idx : batch_idx + 1].contiguous(), value[batch_idx : batch_idx + 1].contiguous())
        for key, value in past_key_values
    )


def build_token_group_layout(
    *,
    prefix_tokens: int,
    suffix_tokens: int,
    generated_tokens: int,
    token_group_size: int,
) -> Tuple[List[Tuple[int, int]], List[str], Dict[str, int]]:
    if token_group_size < 1:
        raise ValueError("token_group_size must be >= 1")

    spans: List[Tuple[int, int]] = []
    labels: List[str] = []
    segment_group_counts: Dict[str, int] = {}
    cursor = 0
    segment_specs = [
        ("prefix", "P", prefix_tokens),
        ("suffix", "S", suffix_tokens),
        ("generated", "G", generated_tokens),
    ]
    for segment_name, label_prefix, token_count in segment_specs:
        group_idx = 0
        for local_start in range(0, max(0, token_count), token_group_size):
            start = cursor + local_start
            end = cursor + min(local_start + token_group_size, token_count)
            spans.append((start, end))
            labels.append(f"{label_prefix}{group_idx:02d}")
            group_idx += 1
        segment_group_counts[segment_name] = group_idx
        cursor += max(0, token_count)
    return spans, labels, segment_group_counts


def compute_flat_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.float().reshape(-1)
    b_flat = b.float().reshape(-1)
    a_norm = torch.linalg.norm(a_flat)
    b_norm = torch.linalg.norm(b_flat)
    denom = (a_norm * b_norm).clamp_min(1e-8)
    return float(torch.dot(a_flat, b_flat).div(denom).item())


def compute_full_mix_vs_native_kv_similarity_matrix(
    *,
    native_past_key_values: PastKeyValues,
    full_mix_past_key_values: PastKeyValues,
    prefix_tokens: int,
    suffix_tokens: int,
    generated_tokens: int,
    token_group_size: int,
) -> Tuple[torch.Tensor, List[str], Dict[str, int]]:
    if len(native_past_key_values) != len(full_mix_past_key_values):
        raise ValueError(
            "Native/full-mix pasts must have the same number of layers, "
            f"got {len(native_past_key_values)} vs {len(full_mix_past_key_values)}"
        )

    total_tokens = prefix_tokens + suffix_tokens + generated_tokens
    spans, group_labels, segment_group_counts = build_token_group_layout(
        prefix_tokens=prefix_tokens,
        suffix_tokens=suffix_tokens,
        generated_tokens=generated_tokens,
        token_group_size=token_group_size,
    )
    matrix = torch.empty((len(native_past_key_values), len(spans)), dtype=torch.float32)

    for layer_idx, ((native_key, native_value), (full_mix_key, full_mix_value)) in enumerate(
        zip(native_past_key_values, full_mix_past_key_values)
    ):
        if native_key.shape[2] < total_tokens or native_value.shape[2] < total_tokens:
            raise ValueError(
                f"Native cache at layer {layer_idx} is shorter than required total_tokens={total_tokens}"
            )
        if full_mix_key.shape[2] < total_tokens or full_mix_value.shape[2] < total_tokens:
            raise ValueError(
                f"Full-mix cache at layer {layer_idx} is shorter than required total_tokens={total_tokens}"
            )

        for group_idx, (start, end) in enumerate(spans):
            key_cosine = compute_flat_cosine_similarity(
                native_key[:, :, start:end, :],
                full_mix_key[:, :, start:end, :],
            )
            value_cosine = compute_flat_cosine_similarity(
                native_value[:, :, start:end, :],
                full_mix_value[:, :, start:end, :],
            )
            matrix[layer_idx, group_idx] = 0.5 * (key_cosine + value_cosine)

    return matrix, group_labels, segment_group_counts


@torch.inference_mode()
def generate_greedy_with_final_past(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    past_key_values: PastKeyValues,
    seed_token: torch.Tensor,
    max_new_tokens: int,
) -> Dict[str, Any]:
    if seed_token.ndim != 2 or seed_token.shape[0] != 1:
        raise ValueError(f"seed_token must have shape [1, 1], got {tuple(seed_token.shape)}")

    current_past = past_key_values
    current_input_ids = seed_token
    current_input_in_past = False
    generated_token_ids: List[int] = []
    past_after_seed: Optional[PastKeyValues] = None
    eos_token_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=current_input_ids,
            past_key_values=current_past,
            use_cache=True,
        )
        current_past = outputs.past_key_values
        current_input_in_past = True
        if past_after_seed is None:
            past_after_seed = current_past

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        next_token_id = int(next_token.item())
        if eos_token_id is not None and next_token_id == eos_token_id:
            break

        generated_token_ids.append(next_token_id)
        current_input_ids = next_token
        current_input_in_past = False

    if past_after_seed is None:
        past_after_seed = append_input_ids_to_past(
            model=model,
            past_key_values=past_key_values,
            input_ids=seed_token,
        )
        current_past = past_after_seed
        current_input_in_past = True

    if not current_input_in_past:
        current_past = append_input_ids_to_past(
            model=model,
            past_key_values=current_past,
            input_ids=current_input_ids,
        )

    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    return {
        "past_after_seed": past_after_seed,
        "final_past": current_past,
        "generated_token_ids": generated_token_ids,
        "generated_text": postprocess_generated_answer(generated_text),
    }


@dataclass
class SummaryRow:
    study_id: str
    benchmark_mode: str
    metric_name: str
    injection_layer_start_idx: int
    translated_num_layers: int
    source_layer_start_idx: int
    source_layer_end_idx: int
    target_layer_start_idx: int
    target_layer_end_idx: int
    average_metric: float
    average_native_metric: float
    average_dir_only_metric: float
    average_mag_only_metric: float
    average_full_mix_metric: float
    average_delta_dir_only: float
    average_delta_mag_only: float
    average_delta_full_mix: float
    average_native_loss: float
    average_full_mix_loss: float
    average_native_to_dir_only_logit_kl: float
    average_native_to_mag_only_logit_kl: float
    average_native_to_full_mix_logit_kl: float
    average_full_mix_to_dir_only_logit_kl: float
    average_full_mix_to_mag_only_logit_kl: float
    run_dir: str

def build_summary_csv_path(study_dir: Path) -> Path:
    return study_dir / "summary.csv"


def extract_eval_metrics(combined_metrics: Dict[str, Any]) -> Dict[str, Any]:
    metric_name = str(combined_metrics["metric_name"])
    dataset_results_key = "dataset_accuracies" if metric_name == "accuracy" else "dataset_f1"
    return {
        "benchmark_mode": combined_metrics["benchmark_mode"],
        "metric_name": metric_name,
        dataset_results_key: combined_metrics[dataset_results_key],
        "average_metric": combined_metrics["average_metric"],
        "average_native_metric": combined_metrics["average_native_metric"],
        "average_dir_only_metric": combined_metrics["average_dir_only_metric"],
        "average_mag_only_metric": combined_metrics["average_mag_only_metric"],
        "average_full_mix_metric": combined_metrics["average_full_mix_metric"],
        "average_delta_dir_only": combined_metrics["average_delta_dir_only"],
        "average_delta_mag_only": combined_metrics["average_delta_mag_only"],
        "average_delta_full_mix": combined_metrics["average_delta_full_mix"],
        "openwebtext_validation_loss": combined_metrics["openwebtext_validation_loss"],
        "average_native_loss": combined_metrics["average_native_loss"],
        "average_full_mix_loss": combined_metrics["average_full_mix_loss"],
        f"average_{metric_name}": combined_metrics[f"average_{metric_name}"],
        f"average_native_{metric_name}": combined_metrics[f"average_native_{metric_name}"],
    }


def extract_analysis_metrics(combined_metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "benchmark_mode": combined_metrics["benchmark_mode"],
        "metric_name": combined_metrics["metric_name"],
        "dataset_logit_kl": combined_metrics["dataset_logit_kl"],
        "average_native_to_dir_only_logit_kl": combined_metrics["average_native_to_dir_only_logit_kl"],
        "average_native_to_mag_only_logit_kl": combined_metrics["average_native_to_mag_only_logit_kl"],
        "average_native_to_full_mix_logit_kl": combined_metrics["average_native_to_full_mix_logit_kl"],
        "average_full_mix_to_dir_only_logit_kl": combined_metrics["average_full_mix_to_dir_only_logit_kl"],
        "average_full_mix_to_mag_only_logit_kl": combined_metrics["average_full_mix_to_mag_only_logit_kl"],
        "average_delta_dir_only": combined_metrics["average_delta_dir_only"],
        "average_delta_mag_only": combined_metrics["average_delta_mag_only"],
        "average_delta_full_mix": combined_metrics["average_delta_full_mix"],
        "openwebtext_validation_loss": combined_metrics["openwebtext_validation_loss"],
        "average_native_loss": combined_metrics["average_native_loss"],
        "average_full_mix_loss": combined_metrics["average_full_mix_loss"],
        "openwebtext_kv_similarity_heatmaps": combined_metrics.get("openwebtext_kv_similarity_heatmaps", {}),
        "openwebtext_kv_similarity_metadata": combined_metrics.get("openwebtext_kv_similarity_metadata", {}),
    }


def build_summary_row(
    ctx: Context,
    run_dir: Path,
    metrics: Dict[str, Any],
) -> SummaryRow:
    config = ctx.config
    reference_edge = ctx.edges[0]
    reference_edge_id = reference_edge.id
    return SummaryRow(
        study_id=config.study_id or "",
        benchmark_mode=metrics["benchmark_mode"],
        metric_name=metrics["metric_name"],
        injection_layer_start_idx=config.injection_layer_start_idx,
        translated_num_layers=config.injection_window_size,
        source_layer_start_idx=ctx.cm.get_src_layer_start_idx(reference_edge_id),
        source_layer_end_idx=ctx.cm.get_src_layer_end_idx(reference_edge_id),
        target_layer_start_idx=ctx.cm.get_tgt_layer_start_idx(reference_edge_id),
        target_layer_end_idx=ctx.cm.get_tgt_layer_end_idx(reference_edge_id),
        average_metric=float(metrics["average_metric"]),
        average_native_metric=float(metrics["average_native_metric"]),
        average_dir_only_metric=float(metrics["average_dir_only_metric"]),
        average_mag_only_metric=float(metrics["average_mag_only_metric"]),
        average_full_mix_metric=float(metrics["average_full_mix_metric"]),
        average_delta_dir_only=float(metrics["average_delta_dir_only"]),
        average_delta_mag_only=float(metrics["average_delta_mag_only"]),
        average_delta_full_mix=float(metrics["average_delta_full_mix"]),
        average_native_loss=float(metrics["average_native_loss"]),
        average_full_mix_loss=float(metrics["average_full_mix_loss"]),
        average_native_to_dir_only_logit_kl=float(metrics["average_native_to_dir_only_logit_kl"]),
        average_native_to_mag_only_logit_kl=float(metrics["average_native_to_mag_only_logit_kl"]),
        average_native_to_full_mix_logit_kl=float(metrics["average_native_to_full_mix_logit_kl"]),
        average_full_mix_to_dir_only_logit_kl=float(metrics["average_full_mix_to_dir_only_logit_kl"]),
        average_full_mix_to_mag_only_logit_kl=float(metrics["average_full_mix_to_mag_only_logit_kl"]),
        run_dir=str(run_dir),
    )

def read_summary_rows(summary_path: Path) -> List[SummaryRow]:
    rows: List[SummaryRow] = []
    if not summary_path.exists():
        return rows
    field_names = SummaryRow.__dataclass_fields__
    with summary_path.open("r", encoding="utf-8", newline="") as fp:
        for existing in csv.DictReader(fp):
            try:
                payload = {}
                for field_name, field_def in field_names.items():
                    raw_value = existing[field_name]
                    if field_def.type is int:
                        payload[field_name] = int(raw_value)
                    elif field_def.type is float:
                        payload[field_name] = float(raw_value)
                    else:
                        payload[field_name] = raw_value
                rows.append(SummaryRow(**payload))
            except (KeyError, ValueError):
                continue
    return rows

def write_summary(study_dir: Path, rows: List[SummaryRow]) -> Path:
    summary_path = build_summary_csv_path(study_dir)
    rows = sorted(rows, key=lambda row: row.injection_layer_start_idx)
    fieldnames = list(SummaryRow.__dataclass_fields__.keys())
    with summary_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return summary_path


def update_summary(
    ctx: Context,
    run_dir: Path,
    metrics: Dict[str, Any],
) -> Path:
    study_dir = run_dir.parent
    study_dir.mkdir(parents=True, exist_ok=True)
    summary_path = build_summary_csv_path(study_dir)
    row = build_summary_row(ctx, run_dir, metrics)

    rows = [existing for existing in read_summary_rows(summary_path) if existing.injection_layer_start_idx != row.injection_layer_start_idx]
    rows.append(row)
    rows.sort(key=lambda item: item.injection_layer_start_idx)
    return write_summary(study_dir, rows)


def annotate_injected_layer_ranges(ax, rows: List[Any], y_getter) -> None:
    for row in rows:
        x_value = row.injection_layer_start_idx
        ax.annotate(
            format_layer_range(row.target_layer_start_idx, row.target_layer_end_idx),
            (x_value, float(y_getter(row))),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
        )



def _style_paper_axes(ax, *, x_values: Optional[List[int]] = None) -> None:
    style_axes_common(ax)
    ax.minorticks_on()
    ax.margins(x=0.03, y=0.08)
    if x_values is not None:
        ax.set_xticks(x_values)


def _save_paper_figure(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)


def build_analysis_metrics_path(run_dir: Path) -> Path:
    return run_dir / "control_analysis_metrics.json"


def build_metric_controls_chart_path(study_dir: Path, metric_name: str) -> Path:
    return study_dir / f"layer_idx_vs_{sanitize_slug(metric_name)}_controls.png"


def build_logit_kl_chart_path(study_dir: Path) -> Path:
    return study_dir / "layer_idx_vs_logit_kl.png"


def build_openwebtext_loss_chart_path(study_dir: Path) -> Path:
    return study_dir / "layer_idx_vs_openwebtext_validation_loss.png"


def build_kv_similarity_heatmap_path(run_dir: Path, edge_id: str) -> Path:
    return run_dir / f"{sanitize_slug(edge_id)}_full_mix_vs_native_kv_similarity_heatmap.png"


def build_kv_similarity_metadata_path(run_dir: Path, edge_id: str) -> Path:
    return run_dir / f"{sanitize_slug(edge_id)}_full_mix_vs_native_kv_similarity_metadata.json"


def plot_full_mix_vs_native_kv_similarity_heatmap(
    *,
    similarity_matrix: torch.Tensor,
    group_labels: List[str],
    segment_group_counts: Dict[str, int],
    token_group_size: int,
    title: str,
    output_path: Path,
) -> Path:
    if similarity_matrix.ndim != 2:
        raise ValueError(f"similarity_matrix must be 2D, got {tuple(similarity_matrix.shape)}")

    import matplotlib.pyplot as plt
    import numpy as np

    matrix_np = similarity_matrix.detach().cpu().numpy()
    masked_matrix = np.ma.masked_invalid(matrix_np)
    num_layers, num_groups = masked_matrix.shape
    fig_width = max(10.0, 0.42 * max(1, num_groups))
    fig_height = max(5.0, 0.35 * max(1, num_layers))

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(111)
    image = ax.imshow(
        masked_matrix,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        vmin=-1.0,
        vmax=1.0,
    )

    prefix_group_count = int(segment_group_counts.get("prefix", 0))
    suffix_group_count = int(segment_group_counts.get("suffix", 0))
    generated_group_count = int(segment_group_counts.get("generated", 0))

    for boundary in [prefix_group_count, prefix_group_count + suffix_group_count]:
        if 0 < boundary < num_groups:
            ax.axvline(boundary - 0.5, linestyle="--", linewidth=1.0, alpha=0.8)

    x_tick_step = max(1, num_groups // 24) if num_groups > 0 else 1
    x_tick_positions = list(range(0, num_groups, x_tick_step))
    if num_groups > 0 and (num_groups - 1) not in x_tick_positions:
        x_tick_positions.append(num_groups - 1)
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels([group_labels[idx] for idx in x_tick_positions], rotation=90)

    y_tick_step = max(1, num_layers // 16) if num_layers > 0 else 1
    y_tick_positions = list(range(0, num_layers, y_tick_step))
    if num_layers > 0 and (num_layers - 1) not in y_tick_positions:
        y_tick_positions.append(num_layers - 1)
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels([str(idx) for idx in y_tick_positions])

    ax.set_xlabel(f"Token groups (group_size={token_group_size})")
    ax.set_ylabel("Layer index")
    ax.set_title(title)

    segment_specs = [
        ("Prefix", 0, prefix_group_count),
        ("Observed suffix", prefix_group_count, suffix_group_count),
        ("Generated suffix", prefix_group_count + suffix_group_count, generated_group_count),
    ]
    text_y = num_layers - 0.35 if num_layers > 0 else 0.0
    for segment_name, start_idx, width in segment_specs:
        if width < 1:
            continue
        center = start_idx + (width - 1) / 2.0
        ax.text(center, text_y, segment_name, ha="center", va="bottom", fontsize=9)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Cosine similarity (Native vs Full-Mix)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def summarize_full_mix_vs_native_kv_similarity_artifacts(
    *,
    run_dir: Path,
    edge_id: str,
    accumulator: KVSimilarityAccumulator,
    token_group_size: int,
    title: str,
) -> Tuple[Path, Path]:
    summary = accumulator.summary()
    heatmap_path = build_kv_similarity_heatmap_path(run_dir, edge_id)
    metadata_path = build_kv_similarity_metadata_path(run_dir, edge_id)

    plot_full_mix_vs_native_kv_similarity_heatmap(
        similarity_matrix=summary["matrix"],
        group_labels=summary["group_labels"],
        segment_group_counts=summary["segment_group_counts"],
        token_group_size=token_group_size,
        title=title,
        output_path=heatmap_path,
    )

    coverage_by_group = []
    count_matrix = summary["count_matrix"]
    if count_matrix.numel() > 0:
        coverage_by_group = [float(value) for value in count_matrix[0].tolist()]

    write_json(
        str(metadata_path),
        {
            "edge_id": edge_id,
            "token_group_size": token_group_size,
            "group_labels": summary["group_labels"],
            "segment_group_counts": summary["segment_group_counts"],
            "group_coverage": coverage_by_group,
            "mean_similarity_matrix": summary["matrix"].tolist(),
        },
    )
    return heatmap_path, metadata_path


@torch.inference_mode()
def evaluate_openwebtext_losses_and_kv_similarity(
    *,
    ctx: Context,
    run_dir: Path,
    translator_pool: LayerWindowTranslatorPool,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, str], Dict[str, str]]:
    config = ctx.config
    max_examples = max(1, config.eval_max_examples_per_dataset)
    node_map = build_node_map(ctx.nodes)
    target_node_ids = sorted({edge.tgt_id for edge in ctx.edges})
    edges_by_target = {
        target_node_id: [edge for edge in ctx.edges if edge.tgt_id == target_node_id]
        for target_node_id in target_node_ids
    }

    loss_sums = {edge.id: {"native": 0.0, "full_mix": 0.0} for edge in ctx.edges}
    counts = {edge.id: 0 for edge in ctx.edges}
    similarity_accumulators: Dict[str, KVSimilarityAccumulator] = {
        edge.id: KVSimilarityAccumulator(num_layers=ctx.mm.get_model_spec(edge.tgt_id).num_layers)
        for edge in ctx.edges
    }

    for target_node_id in target_node_ids:
        dataloader = build_openwebtext_eval_dataloader(
            tokenizer=ctx.mm.get_tokenizer(target_node_id),
            config=config,
            batch_size=config.eval_batch_size,
            num_workers=config.eval_num_workers,
            shuffle=config.eval_shuffle_stream,
            seed=config.seed,
            shuffle_buffer=config.shuffle_buffer,
        )
        processed_examples = 0

        for batch_idx, input_ids in enumerate(dataloader, start=1):
            if processed_examples >= max_examples:
                break

            remaining_examples = max_examples - processed_examples
            if input_ids.shape[0] > remaining_examples:
                input_ids = input_ids[:remaining_examples]
            input_ids = input_ids.to(config.device)

            prefix_cache_ids, lm_input_ids, lm_labels = split_prefix_and_suffix_for_exact_next_token_loss(
                input_ids=input_ids,
                prefix_tokens=config.prefix_tokens,
            )
            past_by_node_id = {
                node.id: extract_past_key_values(ctx.mm.get_model(node.id), prefix_cache_ids)
                for node in ctx.nodes
            }

            batch_examples = input_ids.shape[0]
            for edge in edges_by_target[target_node_id]:
                edge_losses = compute_openwebtext_native_and_full_mix_losses(
                    ctx=ctx,
                    edge=edge,
                    prefix_cache_ids=prefix_cache_ids,
                    lm_input_ids=lm_input_ids,
                    lm_labels=lm_labels,
                    past_by_node_id=past_by_node_id,
                    translator_pool=translator_pool,
                )
                for metric_name in ["native", "full_mix"]:
                    loss_sums[edge.id][metric_name] += float(edge_losses[metric_name]) * batch_examples
                counts[edge.id] += batch_examples

                translated_key, translated_value = translator_pool.translate_layer_window(
                    past_key_values=past_by_node_id[edge.src_id],
                    src_node_id=edge.src_id,
                    tgt_node_id=edge.tgt_id,
                )
                full_mix_prefix_past = replay_target_prefill_with_injected_window(
                    target_model=ctx.mm.get_model(edge.tgt_id),
                    prefix_input_ids=prefix_cache_ids,
                    target_layer_indices=ctx.cm.get_tgt_layer_indices(edge.id),
                    injected_key_block=translated_key,
                    injected_value_block=translated_value,
                    tgt_spec=ctx.mm.get_model_spec(edge.tgt_id),
                    target_model_id=node_map[edge.tgt_id].model_id,
                    cache_injected_window=True,
                )
                native_prefix_past = past_by_node_id[edge.tgt_id]
                target_model = ctx.mm.get_model(edge.tgt_id)
                target_tokenizer = ctx.mm.get_tokenizer(edge.tgt_id)

                for example_idx in range(batch_examples):
                    native_prefix_example = slice_past_key_values_batch(native_prefix_past, example_idx)
                    full_mix_prefix_example = slice_past_key_values_batch(full_mix_prefix_past, example_idx)
                    example_lm_input_ids = lm_input_ids[example_idx : example_idx + 1]
                    example_seed_token = lm_labels[example_idx : example_idx + 1, -1:]

                    native_past_before_seed = append_input_ids_to_past(
                        model=target_model,
                        past_key_values=native_prefix_example,
                        input_ids=example_lm_input_ids,
                    )
                    full_mix_past_before_seed = append_input_ids_to_past(
                        model=target_model,
                        past_key_values=full_mix_prefix_example,
                        input_ids=example_lm_input_ids,
                    )

                    native_generation = generate_greedy_with_final_past(
                        model=target_model,
                        tokenizer=target_tokenizer,
                        past_key_values=native_past_before_seed,
                        seed_token=example_seed_token,
                        max_new_tokens=config.generation_max_new_tokens,
                    )
                    full_mix_generation = generate_greedy_with_final_past(
                        model=target_model,
                        tokenizer=target_tokenizer,
                        past_key_values=full_mix_past_before_seed,
                        seed_token=example_seed_token,
                        max_new_tokens=config.generation_max_new_tokens,
                    )

                    comparable_generated_tokens = min(
                        len(native_generation["generated_token_ids"]),
                        len(full_mix_generation["generated_token_ids"]),
                    )
                    similarity_matrix, group_labels, segment_group_counts = compute_full_mix_vs_native_kv_similarity_matrix(
                        native_past_key_values=native_generation["final_past"],
                        full_mix_past_key_values=full_mix_generation["final_past"],
                        prefix_tokens=prefix_cache_ids.shape[1],
                        suffix_tokens=example_lm_input_ids.shape[1] + 1,
                        generated_tokens=comparable_generated_tokens,
                        token_group_size=config.kv_similarity_token_group_size,
                    )
                    similarity_accumulators[edge.id].update(
                        similarity_matrix=similarity_matrix,
                        group_labels=group_labels,
                        segment_group_counts=segment_group_counts,
                    )

            processed_examples += batch_examples
            if batch_idx % 10 == 0:
                logging.info(
                    "[OpenWebText/validation][target=%s] loss+kv-sim progress: %d/%d sequences",
                    target_node_id,
                    processed_examples,
                    max_examples,
                )

    loss_summary_by_edge: Dict[str, Dict[str, float]] = {}
    heatmap_paths: Dict[str, str] = {}
    metadata_paths: Dict[str, str] = {}
    for edge in ctx.edges:
        count = counts[edge.id]
        native_loss = float(loss_sums[edge.id]["native"] / count) if count > 0 else float("nan")
        full_mix_loss = float(loss_sums[edge.id]["full_mix"] / count) if count > 0 else float("nan")
        loss_summary_by_edge[edge.id] = {
            "native_loss": native_loss,
            "full_mix_loss": full_mix_loss,
            "delta_full_mix_loss": full_mix_loss - native_loss if math.isfinite(native_loss) and math.isfinite(full_mix_loss) else float("nan"),
            "count": count,
        }
        heatmap_path, metadata_path = summarize_full_mix_vs_native_kv_similarity_artifacts(
            run_dir=run_dir,
            edge_id=edge.id,
            accumulator=similarity_accumulators[edge.id],
            token_group_size=config.kv_similarity_token_group_size,
            title=(
                f"Full-Mix vs Native KV similarity ({edge.id})\n"
                f"OpenWebText | grouped tokens across prefix, observed suffix, and generated suffix"
            ),
        )
        heatmap_paths[edge.id] = str(heatmap_path)
        metadata_paths[edge.id] = str(metadata_path)
        logging.info(
            "[OpenWebText/validation] %s | native_loss=%.6f | full_mix_loss=%.6f | kv_similarity_heatmap=%s",
            edge.id,
            native_loss,
            full_mix_loss,
            heatmap_path,
        )

    return loss_summary_by_edge, heatmap_paths, metadata_paths

def plot_metric_controls_summary(summary_path: Path) -> Path:
    rows = read_summary_rows(summary_path)
    if not rows:
        raise ValueError(f"No plottable rows found in {summary_path}")

    rows.sort(key=lambda row: row.injection_layer_start_idx)
    x_values = [row.injection_layer_start_idx for row in rows]
    metric_name = rows[0].metric_name or "metric"
    metric_label = metric_name.upper() if metric_name == "f1" else metric_name.capitalize()
    window_title = format_window_title(rows[0].translated_num_layers)
    study_dir = summary_path.parent

    import matplotlib.pyplot as plt

    apply_ai_paper_style()

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(
        x_values,
        [row.average_full_mix_metric for row in rows],
        color=ACCENT_RED,
        marker=AI_PAPER_MARKERS[3],
        markerfacecolor="white",
        markeredgecolor=ACCENT_RED,
        markeredgewidth=1.4,
        label=f"Full-mix {metric_label}",
    )
    ax.plot(
        x_values,
        [row.average_native_metric for row in rows],
        color=ACCENT_BLACK,
        marker=AI_PAPER_MARKERS[0],
        markerfacecolor="white",
        markeredgecolor=ACCENT_BLACK,
        markeredgewidth=1.4,
        label=f"Native {metric_label}",
    )
    ax.plot(
        x_values,
        [row.average_dir_only_metric for row in rows],
        color=ACCENT_AQUA,
        marker=AI_PAPER_MARKERS[1],
        markerfacecolor="white",
        markeredgecolor=ACCENT_AQUA,
        markeredgewidth=1.4,
        label=f"Dir-only {metric_label}",
    )
    ax.plot(
        x_values,
        [row.average_mag_only_metric for row in rows],
        color=ACCENT_PURPLE,
        marker=AI_PAPER_MARKERS[2],
        markerfacecolor="white",
        markeredgecolor=ACCENT_PURPLE,
        markeredgewidth=1.4,
        label=f"Mag-only {metric_label}",
    )
    annotate_injected_layer_ranges(ax, rows, lambda row: row.average_full_mix_metric)
    ax.set_xlabel("Injection target layer start index")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} decomposition vs injection target layer start index ({window_title})", pad=8)
    _style_paper_axes(ax, x_values=x_values)
    ax.legend(handlelength=2.6)

    chart_path = build_metric_controls_chart_path(study_dir, metric_name)
    _save_paper_figure(fig, chart_path)
    plt.close(fig)
    return chart_path


def plot_logit_kl_summary(summary_path: Path) -> Path:
    rows = read_summary_rows(summary_path)
    if not rows:
        raise ValueError(f"No plottable rows found in {summary_path}")

    rows.sort(key=lambda row: row.injection_layer_start_idx)
    x_values = [row.injection_layer_start_idx for row in rows]
    window_title = format_window_title(rows[0].translated_num_layers)
    study_dir = summary_path.parent

    import matplotlib.pyplot as plt

    apply_ai_paper_style()

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(
        x_values,
        [row.average_native_to_full_mix_logit_kl for row in rows],
        color=ACCENT_RED,
        marker=AI_PAPER_MARKERS[2],
        markerfacecolor="white",
        markeredgecolor=ACCENT_RED,
        markeredgewidth=1.4,
        label="KL(native || full-mix)",
    )
    ax.plot(
        x_values,
        [row.average_native_to_dir_only_logit_kl for row in rows],
        color=ACCENT_AQUA,
        marker=AI_PAPER_MARKERS[0],
        markerfacecolor="white",
        markeredgecolor=ACCENT_AQUA,
        markeredgewidth=1.4,
        label="KL(native || dir-only)",
    )
    ax.plot(
        x_values,
        [row.average_native_to_mag_only_logit_kl for row in rows],
        color=ACCENT_PURPLE,
        marker=AI_PAPER_MARKERS[1],
        markerfacecolor="white",
        markeredgecolor=ACCENT_PURPLE,
        markeredgewidth=1.4,
        label="KL(native || mag-only)",
    )
    ax.plot(
        x_values,
        [row.average_full_mix_to_dir_only_logit_kl for row in rows],
        color=ACCENT_GREEN,
        marker=AI_PAPER_MARKERS[6],
        markerfacecolor="white",
        markeredgecolor=ACCENT_GREEN,
        markeredgewidth=1.4,
        label="KL(full-mix || dir-only)",
    )
    ax.plot(
        x_values,
        [row.average_full_mix_to_mag_only_logit_kl for row in rows],
        color=ACCENT_ORANGE,
        marker=AI_PAPER_MARKERS[3],
        markerfacecolor="white",
        markeredgecolor=ACCENT_ORANGE,
        markeredgewidth=1.4,
        label="KL(full-mix || mag-only)",
    )
    ax.set_xlabel("Injection target layer start index")
    ax.set_ylabel("KL divergence")
    ax.set_title(f"Logit KL comparison vs layer index ({window_title})", pad=8)
    _style_paper_axes(ax, x_values=x_values)
    ax.legend(handlelength=2.6)

    chart_path = build_logit_kl_chart_path(study_dir)
    _save_paper_figure(fig, chart_path)
    plt.close(fig)
    return chart_path


def plot_openwebtext_loss_summary(summary_path: Path) -> Path:
    rows = read_summary_rows(summary_path)
    if not rows:
        raise ValueError(f"No plottable rows found in {summary_path}")

    rows.sort(key=lambda row: row.injection_layer_start_idx)
    x_values = [row.injection_layer_start_idx for row in rows]
    window_title = format_window_title(rows[0].translated_num_layers)
    study_dir = summary_path.parent

    import matplotlib.pyplot as plt

    apply_ai_paper_style()

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(
        x_values,
        [row.average_full_mix_loss for row in rows],
        color=ACCENT_RED,
        marker=AI_PAPER_MARKERS[3],
        markerfacecolor="white",
        markeredgecolor=ACCENT_RED,
        markeredgewidth=1.4,
        label="Full-mix Loss",
    )
    ax.plot(
        x_values,
        [row.average_native_loss for row in rows],
        color=ACCENT_BLACK,
        marker=AI_PAPER_MARKERS[0],
        markerfacecolor="white",
        markeredgecolor=ACCENT_BLACK,
        markeredgewidth=1.4,
        label="Native Loss",
    )
    annotate_injected_layer_ranges(ax, rows, lambda row: row.average_full_mix_loss)
    ax.set_xlabel("Injection target layer start index")
    ax.set_ylabel("OpenWebText validation Loss")
    ax.set_title(f"OpenWebText validation Loss vs layer index ({window_title})", pad=8)
    _style_paper_axes(ax, x_values=x_values)
    ax.legend(handlelength=2.6)

    chart_path = build_openwebtext_loss_chart_path(study_dir)
    _save_paper_figure(fig, chart_path)
    plt.close(fig)
    return chart_path


def remove_stale_summary_artifacts(study_dir: Path, run_dir: Path) -> None:
    stale_paths = [
        run_dir / "eval_summary.md",
        run_dir / "drift_metrics.json",
        run_dir / "control_analysis_metrics.json",
        study_dir / "drift_summary.md",
        study_dir / "study_summary.csv",
        study_dir / "drift_summary.csv",
        study_dir / "drift_cosine.png",
        study_dir / "drift_l2.png",
        study_dir / "layer_idx_vs_openwebtext_validation_loss.png",
    ]
    for stale_path in stale_paths:
        if stale_path.exists():
            stale_path.unlink()


def save_analysis_artifacts(run_dir: Path, metrics: Dict[str, Any]) -> Path:
    write_json(str(build_analysis_metrics_path(run_dir)), metrics)
    return build_analysis_metrics_path(run_dir)


def save_run_artifacts(
    ctx: Context,
    run_dir: Path,
    eval_metrics: Dict[str, Any],
    combined_metrics: Dict[str, Any],
) -> Tuple[Path, Path, Path, Path, Path]:
    study_dir = run_dir.parent
    study_dir.mkdir(parents=True, exist_ok=True)
    remove_stale_summary_artifacts(study_dir, run_dir)
    write_json(str(build_config_path(run_dir)), asdict(ctx.config))
    write_json(str(build_metrics_path(run_dir)), eval_metrics)
    summary_path = update_summary(ctx, run_dir, combined_metrics)
    metric_controls_chart_path = plot_metric_controls_summary(summary_path)
    logit_kl_chart_path = plot_logit_kl_summary(summary_path)
    openwebtext_loss_chart_path = plot_openwebtext_loss_summary(summary_path)
    return summary_path, build_metrics_path(run_dir), metric_controls_chart_path, logit_kl_chart_path, openwebtext_loss_chart_path

def run_eval(
    ctx: Context,
    run_dir: Path,
    translator_pool: LayerWindowTranslatorPool,
) -> Dict[str, Any]:
    config = ctx.config
    edges = ctx.edges
    logging.info("Starting layer-window position evaluation with target-layer replay")
    logging.info("experiment_config=%s", asdict(config))

    translator_pool.eval()
    for node in ctx.nodes:
        ctx.mm.get_model(node.id).eval()

    eval_config = SimpleNamespace(
        batch_size=config.eval_batch_size,
        num_workers=config.eval_num_workers,
        max_examples_per_dataset=config.eval_max_examples_per_dataset,
        output_path=str(run_dir),
        seed=config.seed,
        shuffle_eval_stream=config.eval_shuffle_stream,
        shuffle_buffer=config.shuffle_buffer,
    )

    dataset_results_by_name: Dict[str, Dict[str, Dict[str, float]]] = {}
    dataset_logit_kl_by_name: Dict[str, Dict[str, Dict[str, float]]] = {}

    logging.info("Preparing validation dataloader for OpenWebText/validation")
    raw_openwebtext_loss_by_edge, openwebtext_kv_similarity_heatmaps, openwebtext_kv_similarity_metadata = evaluate_openwebtext_losses_and_kv_similarity(
        ctx=ctx,
        run_dir=run_dir,
        translator_pool=translator_pool,
    )

    openwebtext_loss_by_edge: Dict[str, Dict[str, float]] = {}
    for edge in ctx.edges:
        row = dict(raw_openwebtext_loss_by_edge[edge.id])
        full_mix_loss = float(row.get("loss", float("nan")))
        native_loss = float(row.get("native_loss", float("nan")))
        row["full_mix_loss"] = full_mix_loss
        row["delta_full_mix_loss"] = (
            full_mix_loss - native_loss
            if math.isfinite(full_mix_loss) and math.isfinite(native_loss)
            else float("nan")
        )
        openwebtext_loss_by_edge[edge.id] = row

    for edge in ctx.edges:
        row = openwebtext_loss_by_edge[edge.id]
        logging.info(
            "[OpenWebText/validation] %s | native_loss=%.6f | full_mix_loss=%.6f | count=%d",
            edge.id,
            row["native_loss"],
            row["full_mix_loss"],
            row["count"],
        )

    if config.benchmark_mode == "logit_qa":
        metric_name = "accuracy"
        dataset_results_key = "dataset_accuracies"
        dataset_specs = get_eval_spec_group("logit_qa")
        dataset_evaluator = evaluate_logit_dataset
        dataloader_builder = build_eval_dataloader
        progress_log_template = (
            "[%s] %s | native_%s=%.6f | dir_only_%s=%.6f | mag_only_%s=%.6f | "
            "full_mix_%s=%.6f | delta_dir_only=%.6f | delta_mag_only=%.6f | "
            "delta_full_mix=%.6f | kl(native||dir)=%.6f | "
            "kl(native||mag)=%.6f | kl(native||full)=%.6f | kl(full||dir)=%.6f | "
            "kl(full||mag)=%.6f | count=%d"
        )
    elif config.benchmark_mode == "gen_qa":
        metric_name = "f1"
        dataset_results_key = "dataset_f1"
        dataset_specs = get_eval_spec_group("gen_qa")
        dataset_evaluator = evaluate_generation_dataset
        dataloader_builder = build_generation_eval_dataloader
        progress_log_template = (
            "[%s] %s | native_%s=%.6f | dir_only_%s=%.6f | mag_only_%s=%.6f | "
            "full_mix_%s=%.6f | delta_dir_only=%.6f | delta_mag_only=%.6f | "
            "delta_full_mix=%.6f | kl(native||dir)=%.6f | "
            "kl(native||mag)=%.6f | kl(native||full)=%.6f | kl(full||dir)=%.6f | "
            "kl(full||mag)=%.6f | count=%d"
        )
    else:
        raise ValueError(f"Unsupported benchmark_mode: {config.benchmark_mode}")

    for spec in dataset_specs:
        dataloader = dataloader_builder(spec=spec, eval_config=eval_config)
        dataset_results, dataset_logit_kl = dataset_evaluator(
            ctx=ctx,
            spec=spec,
            dataloader=dataloader,
            translator_pool=translator_pool,
        )
        dataset_results_by_name[spec.name_for_log] = dataset_results
        dataset_logit_kl_by_name[spec.name_for_log] = dataset_logit_kl
        for edge in edges:
            metric_row = dataset_results[edge.id]
            logit_row = dataset_logit_kl[edge.id]
            logging.info(
                progress_log_template,
                spec.name_for_log,
                edge.id,
                metric_name, metric_row[f"native_{metric_name}"],
                metric_name, metric_row[f"dir_only_{metric_name}"],
                metric_name, metric_row[f"mag_only_{metric_name}"],
                metric_name, metric_row[f"full_mix_{metric_name}"],
                metric_row["delta_dir_only"],
                metric_row["delta_mag_only"],
                metric_row["delta_full_mix"],
                logit_row["native_to_dir_only_logit_kl"],
                logit_row["native_to_mag_only_logit_kl"],
                logit_row["native_to_full_mix_logit_kl"],
                logit_row["full_mix_to_dir_only_logit_kl"],
                logit_row["full_mix_to_mag_only_logit_kl"],
                metric_row["count"],
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    average_native_metric = compute_average_metric(dataset_results_by_name, f"native_{metric_name}")
    average_dir_only_metric = compute_average_metric(dataset_results_by_name, f"dir_only_{metric_name}")
    average_mag_only_metric = compute_average_metric(dataset_results_by_name, f"mag_only_{metric_name}")
    average_full_mix_metric = compute_average_metric(dataset_results_by_name, f"full_mix_{metric_name}")
    average_delta_dir_only = compute_average_metric(dataset_results_by_name, "delta_dir_only")
    average_delta_mag_only = compute_average_metric(dataset_results_by_name, "delta_mag_only")
    average_delta_full_mix = compute_average_metric(dataset_results_by_name, "delta_full_mix")
    average_native_to_dir_only_logit_kl = compute_average_metric(dataset_logit_kl_by_name, "native_to_dir_only_logit_kl")
    average_native_to_mag_only_logit_kl = compute_average_metric(dataset_logit_kl_by_name, "native_to_mag_only_logit_kl")
    average_native_to_full_mix_logit_kl = compute_average_metric(dataset_logit_kl_by_name, "native_to_full_mix_logit_kl")
    average_full_mix_to_dir_only_logit_kl = compute_average_metric(dataset_logit_kl_by_name, "full_mix_to_dir_only_logit_kl")
    average_full_mix_to_mag_only_logit_kl = compute_average_metric(dataset_logit_kl_by_name, "full_mix_to_mag_only_logit_kl")
    openwebtext_loss_results = {"OpenWebText/validation": openwebtext_loss_by_edge}
    average_native_loss = compute_average_metric(openwebtext_loss_results, "native_loss")
    average_full_mix_loss = compute_average_metric(openwebtext_loss_results, "full_mix_loss")

    logging.info(
        "[Summary] metric=%s | native=%.6f | dir_only=%.6f | mag_only=%.6f | full_mix=%.6f",
        metric_name,
        average_native_metric,
        average_dir_only_metric,
        average_mag_only_metric,
        average_full_mix_metric,
    )
    logging.info(
        "[Summary] delta_dir_only=%.6f | delta_mag_only=%.6f | delta_full_mix=%.6f",
        average_delta_dir_only,
        average_delta_mag_only,
        average_delta_full_mix,
    )
    logging.info(
        "[Summary] avg_kl(native||dir)=%.6f | avg_kl(native||mag)=%.6f | avg_kl(native||full)=%.6f | avg_kl(full||dir)=%.6f | avg_kl(full||mag)=%.6f",
        average_native_to_dir_only_logit_kl,
        average_native_to_mag_only_logit_kl,
        average_native_to_full_mix_logit_kl,
        average_full_mix_to_dir_only_logit_kl,
        average_full_mix_to_mag_only_logit_kl,
    )
    logging.info(
        "[Summary] OpenWebText/validation loss | native=%.6f | full_mix=%.6f",
        average_native_loss,
        average_full_mix_loss,
    )
    return {
        "benchmark_mode": config.benchmark_mode,
        "metric_name": metric_name,
        dataset_results_key: dataset_results_by_name,
        "average_metric": average_full_mix_metric,
        "average_native_metric": average_native_metric,
        "average_dir_only_metric": average_dir_only_metric,
        "average_mag_only_metric": average_mag_only_metric,
        "average_full_mix_metric": average_full_mix_metric,
        "average_delta_dir_only": average_delta_dir_only,
        "average_delta_mag_only": average_delta_mag_only,
        "average_delta_full_mix": average_delta_full_mix,
        "openwebtext_validation_loss": openwebtext_loss_by_edge,
        "average_native_loss": average_native_loss,
        "average_full_mix_loss": average_full_mix_loss,
        f"average_{metric_name}": average_full_mix_metric,
        f"average_native_{metric_name}": average_native_metric,
        "dataset_logit_kl": dataset_logit_kl_by_name,
        "average_native_to_dir_only_logit_kl": average_native_to_dir_only_logit_kl,
        "average_native_to_mag_only_logit_kl": average_native_to_mag_only_logit_kl,
        "average_native_to_full_mix_logit_kl": average_native_to_full_mix_logit_kl,
        "average_full_mix_to_dir_only_logit_kl": average_full_mix_to_dir_only_logit_kl,
        "average_full_mix_to_mag_only_logit_kl": average_full_mix_to_mag_only_logit_kl,
        "openwebtext_kv_similarity_heatmaps": openwebtext_kv_similarity_heatmaps,
        "openwebtext_kv_similarity_metadata": openwebtext_kv_similarity_metadata,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep the injection target-layer start index for the layer-window translator and evaluate \
cache injection quality on benchmark tasks."
        )
    )
    parser.add_argument("--alg", default="layer_position")
    parser.add_argument(
        "--default-config-path",
        dest="default_config_path",
        default="configs/layer_position.json",
    )
    parser.add_argument("--print-target-num-layers", action="store_true")
    add_dataclass_arguments(
        parser,
        LayerPositionConfig,
        exclude_fields={"alg"},
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    args = parse_args()
    config_kwargs = build_dataclass_kwargs_from_json_and_namespace(
        config_cls=LayerPositionConfig,
        default_config_path=args.default_config_path,
        args=args,
        exclude_fields={"alg"},
    )

    if args.print_target_num_layers:
        print(resolve_target_num_layers(config_kwargs["model_ids"], config_kwargs["model_directions"]))
        return

    config = LayerPositionConfig(
        alg=args.alg,
        **config_kwargs,
    )

    set_seed(config.seed)
    nodes, edges = build_nodes_and_edges(config.model_ids, config.model_directions)
    models, tokenizers = build_models_and_tokenizers(config, nodes)
    ctx = Context(
        config,
        nodes,
        edges,
        ModelManager(models, tokenizers),
        ChannelManager(edges),
    )
    run_dir = build_run_output_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(build_train_log_path(run_dir))
    translator_pool = run_train(
        ctx=ctx,
        run_dir=run_dir,
    )
    setup_logging(build_eval_log_path(run_dir))
    combined_metrics = run_eval(
        ctx=ctx,
        run_dir=run_dir,
        translator_pool=translator_pool,
    )
    eval_metrics = extract_eval_metrics(combined_metrics)
    analysis_metrics = extract_analysis_metrics(combined_metrics)

    summary_path, metrics_path, metric_controls_chart_path, logit_kl_chart_path, openwebtext_loss_chart_path = save_run_artifacts(
        ctx=ctx,
        run_dir=run_dir,
        eval_metrics=eval_metrics,
        combined_metrics=combined_metrics,
    )
    analysis_metrics_path = save_analysis_artifacts(run_dir=run_dir, metrics=analysis_metrics)

    print(f"Run directory: {run_dir}")
    print(f"Metrics: {metrics_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"Metric controls chart: {metric_controls_chart_path}")
    print(f"Control analysis metrics: {analysis_metrics_path}")
    print(f"Logit KL chart: {logit_kl_chart_path}")
    print(f"OpenWebText validation Loss chart: {openwebtext_loss_chart_path}")
    kv_similarity_heatmaps = combined_metrics.get("openwebtext_kv_similarity_heatmaps", {})
    if kv_similarity_heatmaps:
        print("OpenWebText Full-Mix vs Native KV similarity heatmaps:")
        for edge_id, heatmap_path in kv_similarity_heatmaps.items():
            print(f"  {edge_id}: {heatmap_path}")


if __name__ == "__main__":
    main()
