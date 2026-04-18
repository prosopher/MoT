from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from core.context import Context
from core.eval_util import *
from core.train_util import blocks_to_partial_past_key_values
from .train import (
    extract_model_prefill_artifacts,
    extract_selected_layer_canonical_attn_input_block,
)



def extract_selected_layer_blocks(
    past_key_values: PastKeyValues,
    layer_indices: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    selected_past = tuple(past_key_values[layer_idx] for layer_idx in layer_indices)
    return past_key_values_to_blocks(selected_past)



def build_partial_past_from_layer_indices(
    past_key_values: PastKeyValues,
    layer_indices: List[int],
    *,
    num_heads: int,
    head_dim: int,
) -> PastKeyValues:
    key_block, value_block = extract_selected_layer_blocks(
        past_key_values=past_key_values,
        layer_indices=layer_indices,
    )
    return blocks_to_partial_past_key_values(
        key_block=key_block,
        value_block=value_block,
        num_heads=num_heads,
        head_dim=head_dim,
    )


@torch.inference_mode()
def evaluate_dataset(
    ctx: Context,
    spec: HFDatasetSpec,
    dataloader: DataLoader,
    eval_config: EvalConfig,
    translator_pool,
    logger: logging.Logger,
) -> Dict[str, Dict[str, float]]:
    train_config = ctx.config
    nodes = ctx.nodes
    edges = ctx.edges
    device = train_config.device
    tokenizer = ctx.tokenizer
    path_metrics = {edge.id: RunningAverage() for edge in edges}

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

            prefill_by_node_id = {
                node.id: extract_model_prefill_artifacts(ctx.mm.get_model(node.id), cache_input_ids)
                for node in nodes
            }
            past_by_node_id = {
                node.id: prefill_by_node_id[node.id][0]
                for node in nodes
            }
            hidden_states_by_node_id = {
                node.id: prefill_by_node_id[node.id][1]
                for node in nodes
            }

            for edge in edges:
                mixed_target_past, translated_window_past = translator_pool.build_replayed_target_past(
                    source_past_key_values=past_by_node_id[edge.src_id],
                    prefix_input_ids=cache_input_ids,
                    target_model=ctx.mm.get_model(edge.tgt_id),
                    src_node_id=edge.src_id,
                    tgt_node_id=edge.tgt_id,
                    tgt_spec=ctx.mm.get_model_spec(edge.tgt_id),
                    source_canonical_attn_input_block=extract_selected_layer_canonical_attn_input_block(
                        ctx.mm.get_model(edge.src_id),
                        hidden_states_by_node_id[edge.src_id],
                        ctx.cm.get_src_layer_indices(edge.id),
                    ),
                )

                native_target_window = build_partial_past_from_layer_indices(
                    past_key_values=past_by_node_id[edge.tgt_id],
                    layer_indices=ctx.cm.get_tgt_layer_indices(edge.id),
                    num_heads=ctx.mm.get_model_spec(edge.tgt_id).num_heads,
                    head_dim=ctx.mm.get_model_spec(edge.tgt_id).head_dim,
                )
                cosine_value = cosine_similarity_between_past(translated_window_past, native_target_window)

                translated_scoring_past = prepare_answer_scoring_past(
                    model=ctx.mm.get_model(edge.tgt_id),
                    past_key_values=mixed_target_past,
                    question_cache_ids=question_cache_ids,
                )
                native_scoring_past = prepare_answer_scoring_past(
                    model=ctx.mm.get_model(edge.tgt_id),
                    past_key_values=past_by_node_id[edge.tgt_id],
                    question_cache_ids=question_cache_ids,
                )

                translated_scores = score_answer_choices(
                    model=ctx.mm.get_model(edge.tgt_id),
                    past_key_values=translated_scoring_past,
                    seed_token=seed_token,
                    choice_token_ids=candidate_token_ids,
                    normalize_by_length=True,
                )
                native_scores = score_answer_choices(
                    model=ctx.mm.get_model(edge.tgt_id),
                    past_key_values=native_scoring_past,
                    seed_token=seed_token,
                    choice_token_ids=candidate_token_ids,
                    normalize_by_length=True,
                )

                translated_pred = predict_answer_label(translated_scores)
                native_pred = predict_answer_label(native_scores)

                acc = 1.0 if translated_pred == gold_answer else 0.0
                native_acc = 1.0 if native_pred == gold_answer else 0.0

                path_metrics[edge.id].update(cosine_value, acc, native_acc, 1)

            processed_examples += 1

        if batch_idx % 50 == 0:
            logger.info(
                "[%s] progress: %d/%d examples",
                spec.name_for_log,
                processed_examples,
                eval_config.max_examples_per_dataset,
            )

    return summarize_path_metrics(path_metrics)


@torch.inference_mode()
def evaluate_generation_dataset(
    ctx: Context,
    spec: HFDatasetSpec,
    dataloader: DataLoader,
    eval_config: EvalConfig,
    translator_pool,
    logger: logging.Logger,
) -> Dict[str, Dict[str, float]]:
    train_config = ctx.config
    nodes = ctx.nodes
    edges = ctx.edges
    device = train_config.device
    tokenizer = ctx.tokenizer
    path_metrics = {edge.id: GenerationRunningAverage() for edge in edges}

    processed_examples = 0

    for batch_idx, batch in enumerate(dataloader, start=1):
        for example in batch:
            question = example["question"]
            context_text = example["context"]
            gold_answers = example["answers"]

            context_budget = None
            if spec.answer_mode in {"squad", "newsqa"}:
                context_budget = compute_benchmark_context_budget(
                    ctx=ctx,
                    spec=spec,
                    question=question,
                    eval_config=eval_config,
                )

            prepared_inputs = prepare_generation_task_inputs(
                spec=spec,
                tokenizer=tokenizer,
                context=context_text,
                question=question,
                device=device,
                max_input_tokens=context_budget,
            )
            cache_input_ids = prepared_inputs["cache_input_ids"]
            question_cache_ids = prepared_inputs["question_cache_ids"]
            seed_token = prepared_inputs["seed_token"]

            if prepared_inputs.get("was_truncated") and processed_examples < 3:
                question_cache_tokens = 0 if question_cache_ids is None else question_cache_ids.shape[1]
                logger.info(
                    "[%s] truncated context to %d tokens to fit model context window (question_cache_tokens=%d, answer_token_budget=%d)",
                    spec.name_for_log,
                    cache_input_ids.shape[1],
                    question_cache_tokens,
                    get_answer_token_budget(eval_config),
                )

            prefill_by_node_id = {
                node.id: extract_model_prefill_artifacts(ctx.mm.get_model(node.id), cache_input_ids)
                for node in nodes
            }
            past_by_node_id = {
                node.id: prefill_by_node_id[node.id][0]
                for node in nodes
            }
            hidden_states_by_node_id = {
                node.id: prefill_by_node_id[node.id][1]
                for node in nodes
            }

            for edge in edges:
                mixed_target_past, translated_window_past = translator_pool.build_replayed_target_past(
                    source_past_key_values=past_by_node_id[edge.src_id],
                    prefix_input_ids=cache_input_ids,
                    target_model=ctx.mm.get_model(edge.tgt_id),
                    src_node_id=edge.src_id,
                    tgt_node_id=edge.tgt_id,
                    tgt_spec=ctx.mm.get_model_spec(edge.tgt_id),
                    source_canonical_attn_input_block=extract_selected_layer_canonical_attn_input_block(
                        ctx.mm.get_model(edge.src_id),
                        hidden_states_by_node_id[edge.src_id],
                        ctx.cm.get_src_layer_indices(edge.id),
                    ),
                )

                native_target_window = build_partial_past_from_layer_indices(
                    past_key_values=past_by_node_id[edge.tgt_id],
                    layer_indices=ctx.cm.get_tgt_layer_indices(edge.id),
                    num_heads=ctx.mm.get_model_spec(edge.tgt_id).num_heads,
                    head_dim=ctx.mm.get_model_spec(edge.tgt_id).head_dim,
                )
                cosine_value = cosine_similarity_between_past(translated_window_past, native_target_window)

                translated_answer = predict_generation_task_answer(
                    model=ctx.mm.get_model(edge.tgt_id),
                    tokenizer=tokenizer,
                    past_key_values=mixed_target_past,
                    seed_token=seed_token,
                    eval_config=eval_config,
                    question_cache_ids=question_cache_ids,
                )
                native_answer = predict_generation_task_answer(
                    model=ctx.mm.get_model(edge.tgt_id),
                    tokenizer=tokenizer,
                    past_key_values=past_by_node_id[edge.tgt_id],
                    seed_token=seed_token,
                    eval_config=eval_config,
                    question_cache_ids=question_cache_ids,
                )

                f1 = compute_generation_f1(translated_answer, gold_answers)
                native_f1 = compute_generation_f1(native_answer, gold_answers)

                path_metrics[edge.id].update(
                    cosine_value=cosine_value,
                    f1_value=f1,
                    native_f1_value=native_f1,
                    n=1,
                )

            processed_examples += 1

        if batch_idx % 25 == 0:
            logger.info(
                "[%s] generation progress: %d/%d examples",
                spec.name_for_log,
                processed_examples,
                eval_config.max_examples_per_dataset,
            )

    return summarize_generation_path_metrics(path_metrics)



def run_eval(
    ctx: Context,
    eval_config: EvalConfig,
    translator_pool,
) -> Path:
    train_config = ctx.config
    nodes = ctx.nodes
    edges = ctx.edges
    set_seed(eval_config.seed)

    checkpoint_dir_path = eval_config.checkpoint_dir_path

    config_path = get_eval_config_path(eval_config.output_path)
    write_json(str(config_path), asdict(eval_config))

    log_path = get_eval_log_path(eval_config.output_path)
    logger = setup_logger(f"{eval_config.alg}_eval", log_path)
    logger.info("Starting evaluation")
    logger.info("checkpoint_dir_path=%s", checkpoint_dir_path)
    logger.info("eval_config=%s", asdict(eval_config))

    translator_pool.eval()
    for node in nodes:
        ctx.mm.get_model(node.id).eval()

    logger.info("restored_train_config=%s", asdict(train_config))
    logger.info("nodes=%s", [asdict(node) for node in nodes])
    logger.info("edges=%s", [edge.id for edge in edges])
    logger.info(
        "resolved_channels=%s",
        {
            edge.id: {
                "src": [ctx.cm.get_src_layer_start_idx(edge.id), ctx.cm.get_src_layer_end_idx(edge.id)],
                "tgt": [ctx.cm.get_tgt_layer_start_idx(edge.id), ctx.cm.get_tgt_layer_end_idx(edge.id)],
                "src_indices": ctx.cm.get_src_layer_indices(edge.id),
                "tgt_indices": ctx.cm.get_tgt_layer_indices(edge.id),
                "num_layers": len(ctx.cm.get_channels(edge.id)),
            }
            for edge in edges
        },
    )
    logger.info("translation_mode=translate_canonical_attn_input_window_and_restore_target_kv")
    logger.info("qa_eval_log_path=%s", log_path)

    all_logit_results = {}
    all_generation_results = {}

    logger.info("Preparing validation dataloader for OpenWebText/validation")

    def build_source_window_past(edge: Edge, past_by_node_id) -> PastKeyValues:
        return build_partial_past_from_layer_indices(
            past_key_values=past_by_node_id[edge.src_id],
            layer_indices=ctx.cm.get_src_layer_indices(edge.id),
            num_heads=ctx.mm.get_model_spec(edge.src_id).num_heads,
            head_dim=ctx.mm.get_model_spec(edge.src_id).head_dim,
        )

    def build_target_window_past(edge: Edge, past_by_node_id) -> PastKeyValues:
        return build_partial_past_from_layer_indices(
            past_key_values=past_by_node_id[edge.tgt_id],
            layer_indices=ctx.cm.get_tgt_layer_indices(edge.id),
            num_heads=ctx.mm.get_model_spec(edge.tgt_id).num_heads,
            head_dim=ctx.mm.get_model_spec(edge.tgt_id).head_dim,
        )

    def build_translated_target_past_fn(
        *,
        edge: Edge,
        prefix_cache_ids: torch.Tensor,
        past_by_node_id,
    ) -> PastKeyValues:
        _, source_hidden_states = extract_model_prefill_artifacts(
            ctx.mm.get_model(edge.src_id),
            prefix_cache_ids,
        )
        mixed_target_past, _ = translator_pool.build_replayed_target_past(
            source_past_key_values=past_by_node_id[edge.src_id],
            prefix_input_ids=prefix_cache_ids,
            target_model=ctx.mm.get_model(edge.tgt_id),
            src_node_id=edge.src_id,
            tgt_node_id=edge.tgt_id,
            tgt_spec=ctx.mm.get_model_spec(edge.tgt_id),
            source_canonical_attn_input_block=extract_selected_layer_canonical_attn_input_block(
                ctx.mm.get_model(edge.src_id),
                source_hidden_states,
                ctx.cm.get_src_layer_indices(edge.id),
            ),
        )
        return mixed_target_past

    def build_visualization_pasts_fn(
        *,
        edge: Edge,
        prefix_cache_ids: torch.Tensor,
        past_by_node_id,
        **_,
    ) -> Dict[str, PastKeyValues]:
        _, source_hidden_states = extract_model_prefill_artifacts(
            ctx.mm.get_model(edge.src_id),
            prefix_cache_ids,
        )
        _, translated_window_past = translator_pool.build_replayed_target_past(
            source_past_key_values=past_by_node_id[edge.src_id],
            prefix_input_ids=prefix_cache_ids,
            target_model=ctx.mm.get_model(edge.tgt_id),
            src_node_id=edge.src_id,
            tgt_node_id=edge.tgt_id,
            tgt_spec=ctx.mm.get_model_spec(edge.tgt_id),
            source_canonical_attn_input_block=extract_selected_layer_canonical_attn_input_block(
                ctx.mm.get_model(edge.src_id),
                source_hidden_states,
                ctx.cm.get_src_layer_indices(edge.id),
            ),
        )
        return build_openwebtext_tsne_named_pasts(
            source_top_past_key_values=build_source_window_past(edge, past_by_node_id),
            translated_past_key_values=translated_window_past,
            target_top_past_key_values=build_target_window_past(edge, past_by_node_id),
        )

    openwebtext_loss_results = evaluate_openwebtext_validation_loss(
        ctx=ctx,
        eval_config=eval_config,
        translator_pool=translator_pool,
        logger=logger,
        build_translated_target_past_fn=build_translated_target_past_fn,
        build_visualization_pasts_fn=build_visualization_pasts_fn,
    )
    for edge in edges:
        row = openwebtext_loss_results[edge.id]
        logger.info(
            "[OpenWebText/validation] %s | native_loss=%.6f | native_profile=%s | translated_loss=%.6f | translated_profile=%s | count=%d",
            edge.id,
            row["native_loss"],
            build_openwebtext_profile_cell(row, prefix="native"),
            row["loss"],
            build_openwebtext_profile_cell(row),
            row["count"],
        )
        tsne_plot_path = row.get("tsne_plot_path")
        if isinstance(tsne_plot_path, str) and tsne_plot_path:
            logger.info("[OpenWebText/validation] %s | tsne_plot=%s", edge.id, tsne_plot_path)

    logit_dataset_specs = get_default_logit_qa_dataset_specs()
    for spec in logit_dataset_specs:
        logger.info("Preparing dataloader for %s", spec.name_for_log)
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
            logger=logger,
        )
        all_logit_results[spec.name_for_log] = results

        log_dataset_result(
            logger=logger,
            dataset_name=spec.name_for_log,
            results=results,
            nodes=nodes,
            edges=edges,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    generation_dataset_specs = get_default_gen_qa_dataset_specs()
    for spec in generation_dataset_specs:
        logger.info("Preparing generation dataloader for %s", spec.name_for_log)
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
            logger=logger,
        )
        all_generation_results[spec.name_for_log] = results

        log_generation_dataset_result(
            logger=logger,
            dataset_name=spec.name_for_log,
            results=results,
            nodes=nodes,
            edges=edges,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_summary_markdown = build_final_summary_markdown(
        alg=eval_config.alg,
        nodes=nodes,
        edges=edges,
        all_logit_results=all_logit_results,
        all_generation_results=all_generation_results,
        openwebtext_loss_results=openwebtext_loss_results,
    )
    logger.info("===== FINAL MARKDOWN SUMMARY =====\n%s", final_summary_markdown)

    logger.info("Done. Saved log to %s", log_path)
    return log_path
