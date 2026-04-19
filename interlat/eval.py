from __future__ import annotations

from dataclasses import asdict
import logging
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import DataLoader

from core.common import (
    cosine_similarity_between_past,
    extract_past_key_values,
    set_seed,
    write_json,
)
from core.context import Context
from core.eval_util import (
    EvalConfig,
    GEN_QA_SPEC_GROUP_FACTORIES,
    LOGIT_QA_SPEC_GROUP_FACTORIES,
    HFDatasetSpec,
    HFQAPairStream,
    build_eval_dataloader,
    build_logit_answer_candidates,
    compute_benchmark_context_budget,
    compute_generation_f1,
    get_answer_token_budget,
    get_eval_config_path,
    get_eval_log_path,
    predict_answer_label,
    predict_generation_task_answer,
    prepare_answer_scoring_past,
    prepare_generation_task_inputs,
    prepare_logit_task_inputs,
    resolve_progress_total_examples,
    score_answer_choices,
)
from interlat.train import (
    NodeTokenizerPool,
    OpenWebTextRawStream,
    build_latent_conditioned_past,
    compute_alignment_losses,
    extract_last_hidden_states,
    tokenize_valid_texts,
)


def _concat_prefix_ids(*parts: torch.Tensor | None) -> torch.Tensor:
    valid_parts = [part for part in parts if part is not None and part.shape[1] > 0]
    if not valid_parts:
        raise ValueError("At least one prefix tensor is required.")
    return torch.cat(valid_parts, dim=1)


@torch.inference_mode()
def _evaluate_openwebtext_validation(
    *,
    ctx: Context,
    eval_config: EvalConfig,
    translator_pool,
    node_tokenizers: NodeTokenizerPool,
) -> Dict[str, Dict[str, float]]:
    train_config = ctx.config
    dataset = OpenWebTextRawStream(
        split="validation",
        shuffle=eval_config.shuffle_eval_stream,
        seed=eval_config.seed,
        shuffle_buffer=eval_config.shuffle_buffer,
    )
    dataloader = DataLoader(dataset, batch_size=eval_config.batch_size, num_workers=0, collate_fn=lambda batch: batch)
    results = {
        edge.id: {
            "translated_loss_sum": 0.0,
            "native_loss_sum": 0.0,
            "cosine_sum": 0.0,
            "count": 0,
        }
        for edge in ctx.edges
    }
    processed = 0

    for texts in dataloader:
        tokenized_by_node = {}
        valid_mask = None
        for node in ctx.nodes:
            tokenized, node_valid = tokenize_valid_texts(
                tokenizer=node_tokenizers[node.id],
                texts=texts,
                total_tokens=train_config.total_tokens,
                device=train_config.device,
            )
            tokenized_by_node[node.id] = tokenized
            valid_mask = node_valid if valid_mask is None else (valid_mask & node_valid)
        if valid_mask is None or not bool(valid_mask.any()):
            continue
        for node_id, tokenized in tokenized_by_node.items():
            tokenized.input_ids = tokenized.input_ids[valid_mask.to(tokenized.input_ids.device)]

        for row_idx in range(tokenized_by_node[ctx.nodes[0].id].input_ids.shape[0]):
            for edge in ctx.edges:
                source_input_ids = tokenized_by_node[edge.src_id].input_ids[row_idx : row_idx + 1]
                target_input_ids = tokenized_by_node[edge.tgt_id].input_ids[row_idx : row_idx + 1]
                src_prefix_ids = source_input_ids[:, : train_config.prefix_tokens - 1]
                tgt_prefix_ids = target_input_ids[:, : train_config.prefix_tokens - 1]
                lm_input_ids = target_input_ids[:, train_config.prefix_tokens - 1 : -1]
                lm_labels = target_input_ids[:, train_config.prefix_tokens:]

                source_hidden = extract_last_hidden_states(ctx.mm.get_model(edge.src_id), src_prefix_ids)
                target_hidden = extract_last_hidden_states(ctx.mm.get_model(edge.tgt_id), tgt_prefix_ids)
                translated_latents = translator_pool.translate_hidden_states(
                    edge_id=edge.id,
                    source_hidden_states=source_hidden,
                )
                translated_past = build_latent_conditioned_past(
                    ctx.mm.get_model(edge.tgt_id),
                    prefix_input_ids=tgt_prefix_ids,
                    latent_prefix=translated_latents,
                )
                native_past = extract_past_key_values(ctx.mm.get_model(edge.tgt_id), tgt_prefix_ids)
                translated_loss = float(
                    ctx.mm.get_model(edge.tgt_id)(
                        input_ids=lm_input_ids,
                        past_key_values=translated_past,
                        labels=lm_labels,
                        use_cache=False,
                    ).loss.item()
                )
                native_loss = float(
                    ctx.mm.get_model(edge.tgt_id)(
                        input_ids=lm_input_ids,
                        past_key_values=native_past,
                        labels=lm_labels,
                        use_cache=False,
                    ).loss.item()
                )
                _, _, positive_cosine = compute_alignment_losses(
                    translated_latents=translated_latents,
                    target_hidden_states=target_hidden,
                    contrastive_margin=train_config.contrastive_margin,
                )
                meter = results[edge.id]
                meter["translated_loss_sum"] += translated_loss
                meter["native_loss_sum"] += native_loss
                meter["cosine_sum"] += float(positive_cosine.item())
                meter["count"] += 1
            processed += 1
            if processed >= eval_config.max_examples_per_dataset:
                break
        if processed >= eval_config.max_examples_per_dataset:
            break

    summarized = {}
    for edge_id, row in results.items():
        count = max(1, int(row["count"]))
        summarized[edge_id] = {
            "translated_loss": row["translated_loss_sum"] / count,
            "native_loss": row["native_loss_sum"] / count,
            "positive_cosine": row["cosine_sum"] / count,
            "count": int(row["count"]),
        }
    return summarized


@torch.inference_mode()
def _evaluate_logit_dataset(
    *,
    ctx: Context,
    spec: HFDatasetSpec,
    dataloader: DataLoader,
    eval_config: EvalConfig,
    translator_pool,
    node_tokenizers: NodeTokenizerPool,
) -> Dict[str, Dict[str, float]]:
    results = {
        edge.id: {
            "accuracy_sum": 0.0,
            "native_accuracy_sum": 0.0,
            "cosine_sum": 0.0,
            "count": 0,
        }
        for edge in ctx.edges
    }
    progress_total = resolve_progress_total_examples(spec, dataloader.dataset, eval_config.max_examples_per_dataset)
    processed = 0

    for batch in dataloader:
        for example in batch:
            question = example["question"]
            gold_answer = example["answer"]
            context_text = example.get("context")
            for edge in ctx.edges:
                src_tokenizer = node_tokenizers[edge.src_id]
                tgt_tokenizer = node_tokenizers[edge.tgt_id]
                src_prepared = prepare_logit_task_inputs(
                    spec=spec,
                    tokenizer=src_tokenizer,
                    context=context_text,
                    question=question,
                    device=ctx.config.device,
                    choices=example.get("choices"),
                    subject=example.get("subject"),
                )
                tgt_prepared = prepare_logit_task_inputs(
                    spec=spec,
                    tokenizer=tgt_tokenizer,
                    context=context_text,
                    question=question,
                    device=ctx.config.device,
                    choices=example.get("choices"),
                    subject=example.get("subject"),
                )

                source_hidden = extract_last_hidden_states(
                    ctx.mm.get_model(edge.src_id),
                    src_prepared["cache_input_ids"],
                )
                translated_latents = translator_pool.translate_hidden_states(
                    edge_id=edge.id,
                    source_hidden_states=source_hidden,
                )
                translated_past = build_latent_conditioned_past(
                    ctx.mm.get_model(edge.tgt_id),
                    prefix_input_ids=tgt_prepared["cache_input_ids"],
                    latent_prefix=translated_latents,
                )
                native_past = extract_past_key_values(
                    ctx.mm.get_model(edge.tgt_id),
                    tgt_prepared["cache_input_ids"],
                )
                translated_scoring_past = prepare_answer_scoring_past(
                    model=ctx.mm.get_model(edge.tgt_id),
                    past_key_values=translated_past,
                    question_cache_ids=tgt_prepared["question_cache_ids"],
                )
                native_scoring_past = prepare_answer_scoring_past(
                    model=ctx.mm.get_model(edge.tgt_id),
                    past_key_values=native_past,
                    question_cache_ids=tgt_prepared["question_cache_ids"],
                )
                candidate_token_ids = build_logit_answer_candidates(tgt_tokenizer, spec)
                translated_scores = score_answer_choices(
                    model=ctx.mm.get_model(edge.tgt_id),
                    past_key_values=translated_scoring_past,
                    seed_token=tgt_prepared["seed_token"],
                    choice_token_ids=candidate_token_ids,
                    normalize_by_length=True,
                )
                native_scores = score_answer_choices(
                    model=ctx.mm.get_model(edge.tgt_id),
                    past_key_values=native_scoring_past,
                    seed_token=tgt_prepared["seed_token"],
                    choice_token_ids=candidate_token_ids,
                    normalize_by_length=True,
                )
                translated_pred = predict_answer_label(translated_scores)
                native_pred = predict_answer_label(native_scores)
                target_hidden = extract_last_hidden_states(
                    ctx.mm.get_model(edge.tgt_id),
                    tgt_prepared["cache_input_ids"],
                )
                _, _, positive_cosine = compute_alignment_losses(
                    translated_latents=translated_latents,
                    target_hidden_states=target_hidden,
                    contrastive_margin=ctx.config.contrastive_margin,
                )
                meter = results[edge.id]
                meter["accuracy_sum"] += 1.0 if translated_pred == gold_answer else 0.0
                meter["native_accuracy_sum"] += 1.0 if native_pred == gold_answer else 0.0
                meter["cosine_sum"] += float(positive_cosine.item())
                meter["count"] += 1
            processed += 1
            if processed % 50 == 0:
                logging.info("[%s] progress: %d/%d examples", spec.name_for_log, processed, progress_total)

    summarized = {}
    for edge_id, row in results.items():
        count = max(1, int(row["count"]))
        summarized[edge_id] = {
            "accuracy": row["accuracy_sum"] / count,
            "native_accuracy": row["native_accuracy_sum"] / count,
            "cosine": row["cosine_sum"] / count,
            "count": int(row["count"]),
        }
    return summarized


@torch.inference_mode()
def _evaluate_generation_dataset(
    *,
    ctx: Context,
    spec: HFDatasetSpec,
    dataloader: DataLoader,
    eval_config: EvalConfig,
    translator_pool,
    node_tokenizers: NodeTokenizerPool,
) -> Dict[str, Dict[str, float]]:
    results = {
        edge.id: {
            "f1_sum": 0.0,
            "native_f1_sum": 0.0,
            "cosine_sum": 0.0,
            "count": 0,
        }
        for edge in ctx.edges
    }
    processed = 0

    for batch in dataloader:
        for example in batch:
            question = example["question"]
            context_text = example["context"]
            gold_answers = example["answers"]
            for edge in ctx.edges:
                src_tokenizer = node_tokenizers[edge.src_id]
                tgt_tokenizer = node_tokenizers[edge.tgt_id]
                context_budget = None
                if spec.answer_mode in {"squad", "newsqa", "multinews"}:
                    context_budget = compute_benchmark_context_budget(
                        ctx=ctx,
                        spec=spec,
                        question=question,
                        eval_config=eval_config,
                    )
                src_prepared = prepare_generation_task_inputs(
                    spec=spec,
                    tokenizer=src_tokenizer,
                    context=context_text,
                    question=question,
                    device=ctx.config.device,
                    max_input_tokens=context_budget,
                )
                tgt_prepared = prepare_generation_task_inputs(
                    spec=spec,
                    tokenizer=tgt_tokenizer,
                    context=context_text,
                    question=question,
                    device=ctx.config.device,
                    max_input_tokens=context_budget,
                )

                source_hidden = extract_last_hidden_states(
                    ctx.mm.get_model(edge.src_id),
                    src_prepared["cache_input_ids"],
                )
                translated_latents = translator_pool.translate_hidden_states(
                    edge_id=edge.id,
                    source_hidden_states=source_hidden,
                )
                translated_past = build_latent_conditioned_past(
                    ctx.mm.get_model(edge.tgt_id),
                    prefix_input_ids=tgt_prepared["cache_input_ids"],
                    latent_prefix=translated_latents,
                )
                native_past = extract_past_key_values(
                    ctx.mm.get_model(edge.tgt_id),
                    tgt_prepared["cache_input_ids"],
                )
                translated_answer = predict_generation_task_answer(
                    model=ctx.mm.get_model(edge.tgt_id),
                    tokenizer=tgt_tokenizer,
                    past_key_values=translated_past,
                    seed_token=tgt_prepared["seed_token"],
                    eval_config=eval_config,
                    question_cache_ids=tgt_prepared["question_cache_ids"],
                )
                native_answer = predict_generation_task_answer(
                    model=ctx.mm.get_model(edge.tgt_id),
                    tokenizer=tgt_tokenizer,
                    past_key_values=native_past,
                    seed_token=tgt_prepared["seed_token"],
                    eval_config=eval_config,
                    question_cache_ids=tgt_prepared["question_cache_ids"],
                )
                target_hidden = extract_last_hidden_states(
                    ctx.mm.get_model(edge.tgt_id),
                    tgt_prepared["cache_input_ids"],
                )
                _, _, positive_cosine = compute_alignment_losses(
                    translated_latents=translated_latents,
                    target_hidden_states=target_hidden,
                    contrastive_margin=ctx.config.contrastive_margin,
                )
                meter = results[edge.id]
                meter["f1_sum"] += compute_generation_f1(translated_answer, gold_answers)
                meter["native_f1_sum"] += compute_generation_f1(native_answer, gold_answers)
                meter["cosine_sum"] += float(positive_cosine.item())
                meter["count"] += 1
            processed += 1
            if processed % 25 == 0:
                logging.info("[%s] generation progress: %d/%d examples", spec.name_for_log, processed, eval_config.max_examples_per_dataset)

    summarized = {}
    for edge_id, row in results.items():
        count = max(1, int(row["count"]))
        summarized[edge_id] = {
            "f1": row["f1_sum"] / count,
            "native_f1": row["native_f1_sum"] / count,
            "cosine": row["cosine_sum"] / count,
            "count": int(row["count"]),
        }
    return summarized


def _log_metric_table(title: str, metrics: Dict[str, Dict[str, float]]) -> None:
    logging.info("===== %s =====", title)
    for edge_id, row in metrics.items():
        pretty_fields = " | ".join(f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}" for key, value in row.items())
        logging.info("%s | %s", edge_id, pretty_fields)



def run_eval(
    ctx: Context,
    eval_config: EvalConfig,
    translator_pool,
    node_tokenizers: NodeTokenizerPool,
) -> Path:
    set_seed(eval_config.seed)
    config_path = get_eval_config_path(eval_config.output_path)
    write_json(str(config_path), asdict(eval_config))
    log_path = get_eval_log_path(eval_config.output_path)

    logging.info("Starting Interlat evaluation")
    logging.info("checkpoint_dir_path=%s", eval_config.checkpoint_dir_path)
    logging.info("eval_config=%s", asdict(eval_config))

    translator_pool.eval()
    for node in ctx.nodes:
        ctx.mm.get_model(node.id).eval()

    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    openwebtext_metrics = _evaluate_openwebtext_validation(
        ctx=ctx,
        eval_config=eval_config,
        translator_pool=translator_pool,
        node_tokenizers=node_tokenizers,
    )
    _log_metric_table("OpenWebText/validation", openwebtext_metrics)
    all_results["openwebtext_validation"] = openwebtext_metrics

    for factory in LOGIT_QA_SPEC_GROUP_FACTORIES:
        spec = factory()
        dataloader = build_eval_dataloader(spec, eval_config)
        metrics = _evaluate_logit_dataset(
            ctx=ctx,
            spec=spec,
            dataloader=dataloader,
            eval_config=eval_config,
            translator_pool=translator_pool,
            node_tokenizers=node_tokenizers,
        )
        _log_metric_table(spec.name_for_log, metrics)
        all_results[spec.name_for_log] = metrics

    for factory in GEN_QA_SPEC_GROUP_FACTORIES:
        spec = factory()
        dataloader = build_eval_dataloader(spec, eval_config)
        metrics = _evaluate_generation_dataset(
            ctx=ctx,
            spec=spec,
            dataloader=dataloader,
            eval_config=eval_config,
            translator_pool=translator_pool,
            node_tokenizers=node_tokenizers,
        )
        _log_metric_table(spec.name_for_log, metrics)
        all_results[spec.name_for_log] = metrics

    result_path = Path(eval_config.output_path) / "interlat_eval_results.json"
    write_json(str(result_path), all_results)
    logging.info("Saved Interlat evaluation results to %s", result_path)
    logging.info("Saved eval log to %s", log_path)
    return log_path
