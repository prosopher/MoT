from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

try:
    from transformers.cache_utils import DynamicCache
except Exception:  # pragma: no cover
    DynamicCache = None

from core.common import (
    compute_suffix_lm_loss,
    cosine_similarity_between_past,
    extract_past_key_values,
    set_seed,
        write_json,
)
from core.context import Context
from core.eval_util import *
from kvcomm.train import KVCommSelectionPool



def _build_calibration_eval_name(config) -> str:
    return f"{str(config.calibration_dataset).strip()}/validation"


def _build_kvcomm_selected_layer_views(
    *,
    translator_pool: KVCommSelectionPool,
    edge_id: str,
    source_past_key_values,
    target_past_key_values,
    replayed_target_past_key_values,
):
    selected_source_layers = translator_pool.get_selected_source_layers(edge_id)
    selected_target_layers = translator_pool.get_selected_target_layers(edge_id)
    return (
        select_past_layers_by_indices(source_past_key_values, selected_source_layers),
        select_past_layers_by_indices(replayed_target_past_key_values, selected_target_layers),
        select_past_layers_by_indices(target_past_key_values, selected_target_layers),
    )


def _compute_selected_layer_cosine(
    *,
    translator_pool: KVCommSelectionPool,
    edge_id: str,
    source_past_key_values,
    target_past_key_values,
    replayed_target_past_key_values,
) -> float:
    _, translated_selected_past, target_selected_past = _build_kvcomm_selected_layer_views(
        translator_pool=translator_pool,
        edge_id=edge_id,
        source_past_key_values=source_past_key_values,
        target_past_key_values=target_past_key_values,
        replayed_target_past_key_values=replayed_target_past_key_values,
    )
    return cosine_similarity_between_past(translated_selected_past, target_selected_past)


def _ensure_model_cache(past_key_values):
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "get_seq_length"):
        return past_key_values
    if isinstance(past_key_values, tuple) and DynamicCache is not None:
        return DynamicCache.from_legacy_cache(past_key_values)
    return past_key_values


def _openwebtext_total_tokens(config) -> int:
    return max(8, int(getattr(config, "total_tokens", 128)))

def _openwebtext_prefix_tokens(config) -> int:
    total_tokens = _openwebtext_total_tokens(config)
    return max(2, min(total_tokens - 1, int(getattr(config, "prefix_tokens", total_tokens // 2))))


@torch.inference_mode()
def _predict_direct_context_logit(model, spec, tokenizer, context: str, question: str, device: str):
    prepared = prepare_logit_task_inputs(
        spec=spec,
        tokenizer=tokenizer,
        context=context,
        question=question,
        device=device,
    )
    choice_token_ids = build_logit_answer_candidates(tokenizer=tokenizer, spec=spec)
    context_past = _ensure_model_cache(extract_past_key_values(model, prepared["cache_input_ids"]))
    scoring_past = prepare_answer_scoring_past(
        model=model,
        past_key_values=context_past,
        question_cache_ids=prepared["question_cache_ids"],
    )
    scoring_past = _ensure_model_cache(scoring_past)
    scores = score_answer_choices(
        model=model,
        past_key_values=scoring_past,
        seed_token=prepared["seed_token"],
        choice_token_ids=choice_token_ids,
        normalize_by_length=True,
    )
    return predict_answer_label(scores)


@torch.inference_mode()
def _predict_direct_context_generation(
    model,
    spec,
    tokenizer,
    context: str,
    question: str,
    eval_config: EvalConfig,
    device: str,
    context_budget: Optional[int],
) -> str:
    prepared = prepare_generation_task_inputs(
        spec=spec,
        tokenizer=tokenizer,
        context=context,
        question=question,
        device=device,
        max_input_tokens=context_budget,
    )
    context_past = _ensure_model_cache(extract_past_key_values(model, prepared["cache_input_ids"]))
    context_past = _ensure_model_cache(context_past)
    return predict_generation_task_answer(
        model=model,
        tokenizer=tokenizer,
        past_key_values=context_past,
        seed_token=prepared["seed_token"],
        eval_config=eval_config,
        question_cache_ids=prepared["question_cache_ids"],
    )


@torch.inference_mode()
def _predict_kvcomm_logit(
    *,
    pool: KVCommSelectionPool,
    edge_id: str,
    source_model,
    target_model,
    spec,
    tokenizer,
    context: str,
    question: str,
    device: str,
):
    prepared = prepare_logit_task_inputs(
        spec=spec,
        tokenizer=tokenizer,
        context=context,
        question=question,
        device=device,
    )
    choice_token_ids = build_logit_answer_candidates(tokenizer=tokenizer, spec=spec)
    source_past = extract_past_key_values(source_model, prepared["cache_input_ids"])
    kvcomm_past = _ensure_model_cache(
        pool.build_replayed_target_past(
            edge_id=edge_id,
            source_past_key_values=source_past,
        )
    )
    scoring_past = prepare_answer_scoring_past(
        model=target_model,
        past_key_values=kvcomm_past,
        question_cache_ids=prepared["question_cache_ids"],
    )
    scoring_past = _ensure_model_cache(scoring_past)
    scores = score_answer_choices(
        model=target_model,
        past_key_values=scoring_past,
        seed_token=prepared["seed_token"],
        choice_token_ids=choice_token_ids,
        normalize_by_length=True,
    )
    return predict_answer_label(scores)


@torch.inference_mode()
def _predict_kvcomm_generation(
    *,
    pool: KVCommSelectionPool,
    edge_id: str,
    source_model,
    target_model,
    spec,
    tokenizer,
    context: str,
    question: str,
    eval_config: EvalConfig,
    device: str,
    context_budget: Optional[int],
) -> str:
    prepared = prepare_generation_task_inputs(
        spec=spec,
        tokenizer=tokenizer,
        context=context,
        question=question,
        device=device,
        max_input_tokens=context_budget,
    )
    source_past = extract_past_key_values(source_model, prepared["cache_input_ids"])
    kvcomm_past = _ensure_model_cache(
        pool.build_replayed_target_past(
            edge_id=edge_id,
            source_past_key_values=source_past,
        )
    )
    kvcomm_past = _ensure_model_cache(kvcomm_past)
    return predict_generation_task_answer(
        model=target_model,
        tokenizer=tokenizer,
        past_key_values=kvcomm_past,
        seed_token=prepared["seed_token"],
        eval_config=eval_config,
        question_cache_ids=prepared["question_cache_ids"],
    )


def evaluate_dataset(
    *,
    ctx: Context,
    spec,
    dataloader: DataLoader,
    eval_config: EvalConfig,
    translator_pool: KVCommSelectionPool,
) -> Dict[str, Dict[str, float]]:
    device = ctx.config.device
    tokenizer = ctx.tokenizer
    edge_map = {edge.id: edge for edge in ctx.edges}

    path_metrics = {
        edge.id: RunningAverage()
        for edge in ctx.edges
    }

    processed_examples = 0

    for batch_idx, batch in enumerate(dataloader, start=1):
        for example in batch:
            question = example["question"]
            gold_answer = example["answer"]
            context_text = example.get("context")

            prepared_inputs = prepare_logit_task_inputs(
                spec=spec,
                tokenizer=tokenizer,
                context=context_text,
                question=question,
                device=device,
            )
            cache_input_ids = prepared_inputs["cache_input_ids"]
            question_cache_ids = prepared_inputs["question_cache_ids"]
            seed_token = prepared_inputs["seed_token"]

            candidate_token_ids = build_logit_answer_candidates(
                tokenizer=tokenizer,
                spec=spec,
            )

            past_by_node_id = {
                node.id: _ensure_model_cache(
                    extract_past_key_values(ctx.mm.get_model(node.id), cache_input_ids)
                )
                for node in ctx.nodes
            }

            for edge_id, edge in edge_map.items():
                target_model = ctx.mm.get_model(edge.tgt_id)

                kvcomm_past = _ensure_model_cache(
                    translator_pool.build_replayed_target_past(
                        edge_id=edge_id,
                        source_past_key_values=past_by_node_id[edge.src_id],
                    )
                )

                cosine_value = _compute_selected_layer_cosine(
                    translator_pool=translator_pool,
                    edge_id=edge_id,
                    source_past_key_values=past_by_node_id[edge.src_id],
                    target_past_key_values=past_by_node_id[edge.tgt_id],
                    replayed_target_past_key_values=kvcomm_past,
                )

                kvcomm_scoring_past = _ensure_model_cache(
                    prepare_answer_scoring_past(
                        model=target_model,
                        past_key_values=kvcomm_past,
                        question_cache_ids=question_cache_ids,
                    )
                )
                native_scoring_past = _ensure_model_cache(
                    prepare_answer_scoring_past(
                        model=target_model,
                        past_key_values=past_by_node_id[edge.tgt_id],
                        question_cache_ids=question_cache_ids,
                    )
                )

                kvcomm_scores = score_answer_choices(
                    model=target_model,
                    past_key_values=kvcomm_scoring_past,
                    seed_token=seed_token,
                    choice_token_ids=candidate_token_ids,
                    normalize_by_length=True,
                )
                native_scores = score_answer_choices(
                    model=target_model,
                    past_key_values=native_scoring_past,
                    seed_token=seed_token,
                    choice_token_ids=candidate_token_ids,
                    normalize_by_length=True,
                )

                pred_kvcomm = predict_answer_label(kvcomm_scores)
                pred_native = predict_answer_label(native_scores)

                acc = 1.0 if pred_kvcomm == gold_answer else 0.0
                native_acc = 1.0 if pred_native == gold_answer else 0.0
                path_metrics[edge_id].update(cosine_value, acc, native_acc, 1)

            processed_examples += 1

        if batch_idx % 50 == 0:
            logging.info(
                "[%s] progress: %d/%d examples",
                spec.name_for_log,
                processed_examples,
                eval_config.max_examples_per_dataset,
            )

    summarized = summarize_path_metrics(path_metrics)
    for edge in ctx.edges:
        row = summarized[edge.id]
        row["direct_context_accuracy"] = row["native_accuracy"]
        row["kvcomm_accuracy"] = row["accuracy"]
    return summarized



def evaluate_generation_dataset(
    *,
    ctx: Context,
    spec,
    dataloader: DataLoader,
    eval_config: EvalConfig,
    translator_pool: KVCommSelectionPool,
) -> Dict[str, Dict[str, float]]:
    device = ctx.config.device
    tokenizer = ctx.tokenizer
    path_metrics = {
        edge.id: GenerationRunningAverage()
        for edge in ctx.edges
    }

    processed_examples = 0

    for batch_idx, batch in enumerate(dataloader, start=1):
        for example in batch:
            question = example["question"]
            context = example["context"]
            gold_answers = example["answers"]

            context_budget = None
            if spec.answer_mode in {"squad", "newsqa"}:
                context_budget = compute_benchmark_context_budget(
                    ctx=ctx,
                    spec=spec,
                    question=question,
                    eval_config=eval_config,
                )

            prepared_generation_inputs = prepare_generation_task_inputs(
                spec=spec,
                tokenizer=tokenizer,
                context=context,
                question=question,
                device=device,
                max_input_tokens=context_budget,
            )
            cache_input_ids = prepared_generation_inputs["cache_input_ids"]

            for edge in ctx.edges:
                source_model = ctx.mm.get_model(edge.src_id)
                target_model = ctx.mm.get_model(edge.tgt_id)

                pred_direct = _predict_direct_context_generation(
                    model=target_model,
                    spec=spec,
                    tokenizer=tokenizer,
                    context=context,
                    question=question,
                    eval_config=eval_config,
                    device=device,
                    context_budget=context_budget,
                )
                pred_kvcomm = _predict_kvcomm_generation(
                    pool=translator_pool,
                    edge_id=edge.id,
                    source_model=source_model,
                    target_model=target_model,
                    spec=spec,
                    tokenizer=tokenizer,
                    context=context,
                    question=question,
                    eval_config=eval_config,
                    device=device,
                    context_budget=context_budget,
                )

                kvcomm_source_past = _ensure_model_cache(extract_past_key_values(source_model, cache_input_ids))
                kvcomm_replayed_past = _ensure_model_cache(
                    translator_pool.build_replayed_target_past(
                        edge_id=edge.id,
                        source_past_key_values=kvcomm_source_past,
                    )
                )
                native_target_past = _ensure_model_cache(extract_past_key_values(target_model, cache_input_ids))
                cosine_value = _compute_selected_layer_cosine(
                    translator_pool=translator_pool,
                    edge_id=edge.id,
                    source_past_key_values=kvcomm_source_past,
                    target_past_key_values=native_target_past,
                    replayed_target_past_key_values=kvcomm_replayed_past,
                )

                f1_value = compute_generation_f1(pred_kvcomm, gold_answers)
                native_f1_value = compute_generation_f1(pred_direct, gold_answers)
                path_metrics[edge.id].update(cosine_value, f1_value, native_f1_value, 1)

            processed_examples += 1

        if batch_idx % 25 == 0:
            logging.info(
                "[%s] generation progress: %d/%d examples",
                spec.name_for_log,
                processed_examples,
                eval_config.max_examples_per_dataset,
            )

    summarized = summarize_generation_path_metrics(path_metrics)
    for edge in ctx.edges:
        row = summarized[edge.id]
        row["direct_context_f1"] = row["native_f1"]
        row["kvcomm_f1"] = row["f1"]
    return summarized



def run_eval(
    ctx: Context,
    eval_config: EvalConfig,
    translator_pool: KVCommSelectionPool,
) -> Path:
    train_config = ctx.config
    nodes = ctx.nodes
    edges = ctx.edges
    set_seed(eval_config.seed)

    checkpoint_dir_path = eval_config.checkpoint_dir_path

    config_path = get_eval_config_path(eval_config.output_path)
    write_json(str(config_path), asdict(eval_config))

    log_path = get_eval_log_path(eval_config.output_path)
    logging.info("Starting evaluation")
    logging.info("checkpoint_dir_path=%s", checkpoint_dir_path)
    logging.info("eval_config=%s", asdict(eval_config))

    translator_pool.eval()
    for node in nodes:
        ctx.mm.get_model(node.id).eval()

    logging.info("restored_train_config=%s", asdict(train_config))
    logging.info("nodes=%s", [asdict(node) for node in nodes])
    logging.info("edges=%s", [edge.id for edge in edges])
    logging.info(
        "layer_selection_source=%s/train | selection_total_tokens=%d | selection_prefix_tokens=%d",
        train_config.calibration_dataset,
        _openwebtext_total_tokens(train_config),
        _openwebtext_prefix_tokens(train_config),
    )
    for edge in edges:
        logging.info(
            "%s | selected_target_layers=%s | selected_source_layers=%s",
            edge.id,
            translator_pool.get_selected_target_layers(edge.id),
            translator_pool.get_selected_source_layers(edge.id),
        )

    calibration_eval_name = _build_calibration_eval_name(train_config)
    logging.info("Preparing validation dataloader for %s", calibration_eval_name)

    def build_translated_target_past_fn(*, edge: Edge, past_by_node_id) -> PastKeyValues:
        return _ensure_model_cache(
            translator_pool.build_replayed_target_past(
                edge_id=edge.id,
                source_past_key_values=past_by_node_id[edge.src_id],
            )
        )

    def build_visualization_pasts_fn(*, edge: Edge, past_by_node_id, **_) -> Dict[str, PastKeyValues]:
        native_past = _ensure_model_cache(past_by_node_id[edge.tgt_id])
        translated_past = _ensure_model_cache(
            translator_pool.build_replayed_target_past(
                edge_id=edge.id,
                source_past_key_values=past_by_node_id[edge.src_id],
            )
        )
        source_selected_past, translated_selected_past, target_selected_past = _build_kvcomm_selected_layer_views(
            translator_pool=translator_pool,
            edge_id=edge.id,
            source_past_key_values=past_by_node_id[edge.src_id],
            target_past_key_values=native_past,
            replayed_target_past_key_values=translated_past,
        )
        return build_openwebtext_tsne_named_pasts(
            source_top_past_key_values=source_selected_past,
            translated_past_key_values=translated_selected_past,
            target_top_past_key_values=target_selected_past,
        )

    openwebtext_loss_results = evaluate_openwebtext_validation_loss(
        ctx=ctx,
        eval_config=eval_config,
        translator_pool=translator_pool,
        build_translated_target_past_fn=build_translated_target_past_fn,
        build_visualization_pasts_fn=build_visualization_pasts_fn,
    )
    for edge in edges:
        row = openwebtext_loss_results[edge.id]
        logging.info(
            "[%s] %s | native_loss=%.6f | native_profile=%s | kvcomm_loss=%.6f | kvcomm_profile=%s | count=%d",
            calibration_eval_name,
            edge.id,
            row["native_loss"],
            build_openwebtext_profile_cell(row, prefix="native"),
            row["loss"],
            build_openwebtext_profile_cell(row),
            row["count"],
        )
        if row.get("tsne_plot_path"):
            logging.info("[%s] %s | tsne_plot=%s", calibration_eval_name, edge.id, row["tsne_plot_path"])
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    all_logit_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for spec in get_default_logit_qa_dataset_specs():
        logging.info("Preparing dataloader for %s", spec.name_for_log)
        dataloader = build_eval_dataloader(
            spec=spec,
            eval_config=eval_config,
        )
        results = evaluate_dataset(
            ctx=ctx,
            spec=spec,
            dataloader=dataloader,
            eval_config=eval_config,
            translator_pool=translator_pool,
        )
        all_logit_results[spec.name_for_log] = results
        log_dataset_result(
            dataset_name=spec.name_for_log,
            results=results,
            nodes=nodes,
            edges=edges,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_generation_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for spec in get_default_gen_qa_dataset_specs():
        logging.info("Preparing generation dataloader for %s", spec.name_for_log)
        dataloader = build_generation_eval_dataloader(
            spec=spec,
            eval_config=eval_config,
        )
        results = evaluate_generation_dataset(
            ctx=ctx,
            spec=spec,
            dataloader=dataloader,
            eval_config=eval_config,
            translator_pool=translator_pool,
        )
        all_generation_results[spec.name_for_log] = results
        log_generation_dataset_result(
            dataset_name=spec.name_for_log,
            results=results,
            nodes=nodes,
            edges=edges,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    aggregate_metrics = {
        "train_config": asdict(train_config),
        "selection_source": f"{train_config.calibration_dataset}/train",
        "selection_total_tokens": _openwebtext_total_tokens(train_config),
        "selection_prefix_tokens": _openwebtext_prefix_tokens(train_config),
        "selected_target_layers_by_edge": {
            edge.id: translator_pool.get_selected_target_layers(edge.id)
            for edge in edges
        },
        "selected_source_layers_by_edge": {
            edge.id: translator_pool.get_selected_source_layers(edge.id)
            for edge in edges
        },
        "openwebtext_loss_results": openwebtext_loss_results,
        "logit_results": all_logit_results,
        "generation_results": all_generation_results,
    }

    metrics_path = Path(eval_config.output_path) / "metrics.json"
    write_json(str(metrics_path), aggregate_metrics)

    selected_layer_lines: List[str] = ["## KVComm selected replay layers", ""]
    for edge in edges:
        pretty_name = build_edge_pretty_name(edge.id, nodes, edges)
        selected_layer_lines.append(
            f"- {pretty_name}: target={translator_pool.get_selected_target_layers(edge.id)}, "
            f"source={translator_pool.get_selected_source_layers(edge.id)}"
        )
    selected_layer_lines.append("")

    final_summary_markdown = build_final_summary_markdown(
        alg=eval_config.alg,
        nodes=nodes,
        edges=edges,
        all_logit_results=all_logit_results,
        all_generation_results=all_generation_results,
        openwebtext_loss_results=openwebtext_loss_results,
    )
    summary_markdown = "\n".join(selected_layer_lines) + "\n" + final_summary_markdown
    summary_path = Path(eval_config.output_path) / "summary.md"
    summary_path.write_text(summary_markdown, encoding="utf-8")

    logging.info("===== FINAL MARKDOWN SUMMARY =====\n%s", summary_markdown)
    logging.info("Saved metrics to %s", metrics_path)
    logging.info("Saved summary to %s", summary_path)
    logging.info("Done. Saved log to %s", log_path)
    return log_path
