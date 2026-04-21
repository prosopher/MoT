from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from core.context import Context
from core.eval_util import *
from core.train_util import blocks_to_partial_past_key_values

import logging

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


def _build_logit_example_state(
    *,
    ctx: Context,
    prefix_input_ids: torch.Tensor,
    **_,
):
    return {
        "past_by_node_id": {
            node.id: extract_past_key_values(ctx.mm.get_model(node.id), prefix_input_ids)
            for node in ctx.nodes
        }
    }


def _build_logit_edge_artifacts(
    *,
    ctx: Context,
    edge: Edge,
    prefix_input_ids: torch.Tensor,
    example_state,
    translator_pool,
    **_,
) -> LogitEvalEdgeArtifacts:
    past_by_node_id = example_state["past_by_node_id"]
    mixed_target_past, _ = translator_pool.build_replayed_target_past(
        source_past_key_values=past_by_node_id[edge.src_id],
        prefix_input_ids=prefix_input_ids,
        target_model=ctx.mm.get_model(edge.tgt_id),
        src_node_id=edge.src_id,
        tgt_node_id=edge.tgt_id,
        tgt_spec=ctx.mm.get_model_spec(edge.tgt_id),
    )
    native_past = past_by_node_id[edge.tgt_id]
    return LogitEvalEdgeArtifacts(
        translated_past_key_values=mixed_target_past,
        native_past_key_values=native_past,
        cosine_value=cosine_similarity_between_past(mixed_target_past, native_past),
    )


@torch.inference_mode()
def evaluate_generation_dataset(
    ctx: Context,
    spec: HFDatasetSpec,
    dataloader: DataLoader,
    eval_config: EvalConfig,
    translator_pool,
    logger: logging.Logger,
    requested_context_budget: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    train_config = ctx.config
    nodes = ctx.nodes
    edges = ctx.edges
    device = train_config.device
    tokenizer = ctx.tokenizer
    path_metrics = {edge.id: GenerationRunningAverage() for edge in edges}

    processed_examples = 0
    truncation_logs_emitted = 0

    for batch_idx, batch in enumerate(dataloader, start=1):
        for example in batch:
            question = example["question"]
            context_text = example["context"]
            gold_answers = example["answers"]

            budget_resolution = None
            context_budget = None
            if spec.answer_mode in {"squad", "newsqa", "hotpotqa"}:
                budget_resolution = resolve_generation_context_budget(
                    ctx=ctx,
                    spec=spec,
                    question=question,
                    eval_config=eval_config,
                    requested_budget=requested_context_budget,
                )
                context_budget = budget_resolution.effective_budget

            prepared_inputs = prepare_generation_task_inputs(
                spec=spec,
                tokenizer=tokenizer,
                context=context_text,
                question=question,
                device=device,
                max_input_tokens=context_budget,
            )
            prefix_input_ids = prepared_inputs["prefix_input_ids"]
            suffix_cache_ids = prepared_inputs["suffix_cache_ids"]
            seed_token = prepared_inputs["seed_token"]

            if (
                prepared_inputs.get("was_truncated")
                and truncation_logs_emitted < eval_config.generation_truncation_log_limit
            ):
                question_cache_tokens = (
                    0 if budget_resolution is None else budget_resolution.question_cache_tokens
                )
                answer_budget = (
                    get_answer_token_budget(eval_config)
                    if budget_resolution is None
                    else budget_resolution.answer_token_budget
                )
                max_budget = None if budget_resolution is None else budget_resolution.max_budget
                requested_label = format_generation_context_budget_label(requested_context_budget)
                effective_budget = (
                    prefix_input_ids.shape[1]
                    if budget_resolution is None
                    else budget_resolution.effective_budget
                )

                logger.info(
                    "[%s][ctx=%s] truncated context to %d tokens "
                    "(requested_budget=%s, effective_budget=%d, max_budget=%s, "
                    "question_cache_tokens=%d, answer_token_budget=%d)",
                    spec.name_for_log,
                    requested_label,
                    effective_budget,
                    requested_label,
                    effective_budget,
                    "N/A" if max_budget is None else max_budget,
                    question_cache_tokens,
                    answer_budget,
                )
                truncation_logs_emitted += 1

            past_by_node_id = {
                node.id: extract_past_key_values(ctx.mm.get_model(node.id), prefix_input_ids)
                for node in nodes
            }

            for edge in edges:
                mixed_target_past, _ = translator_pool.build_replayed_target_past(
                    source_past_key_values=past_by_node_id[edge.src_id],
                    prefix_input_ids=prefix_input_ids,
                    target_model=ctx.mm.get_model(edge.tgt_id),
                    src_node_id=edge.src_id,
                    tgt_node_id=edge.tgt_id,
                    tgt_spec=ctx.mm.get_model_spec(edge.tgt_id),
                )

                native_past = past_by_node_id[edge.tgt_id]
                cosine_value = cosine_similarity_between_past(mixed_target_past, native_past)

                translated_answer = predict_generation_task_answer(
                    model=ctx.mm.get_model(edge.tgt_id),
                    tokenizer=tokenizer,
                    past_key_values=mixed_target_past,
                    seed_token=seed_token,
                    eval_config=eval_config,
                    suffix_cache_ids=suffix_cache_ids,
                )
                native_answer = predict_generation_task_answer(
                    model=ctx.mm.get_model(edge.tgt_id),
                    tokenizer=tokenizer,
                    past_key_values=past_by_node_id[edge.tgt_id],
                    seed_token=seed_token,
                    eval_config=eval_config,
                    suffix_cache_ids=suffix_cache_ids,
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
            requested_label = format_generation_context_budget_label(requested_context_budget)
            logger.info(
                "[%s][ctx=%s] generation progress: %d/%d examples",
                spec.name_for_log,
                requested_label,
                processed_examples,
                eval_config.max_examples_per_dataset,
            )

    return summarize_generation_path_metrics(path_metrics)



def run_eval(
    ctx: Context,
    eval_config: EvalConfig,
    translator_pool,
) -> Path:
    logger = logging.getLogger(__name__)

    train_config = ctx.config
    nodes = ctx.nodes
    edges = ctx.edges
    set_seed(eval_config.seed)

    checkpoint_dir_path = eval_config.checkpoint_dir_path

    config_path = get_eval_config_path(eval_config.output_path)
    write_json(str(config_path), asdict(eval_config))

    log_path = get_eval_log_path(eval_config.output_path)
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
    logger.info("translation_mode=translate_window_and_replay_target_prefill")
    logger.info("qa_eval_log_path=%s", log_path)

    all_logit_results = {}
    all_generation_results = {}

    openwebtext_loss_results = None
    if ENABLE_OPENWEBTEXT_VALIDATION:
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

        def build_visualization_pasts_fn(
            *,
            edge: Edge,
            prefix_cache_ids: torch.Tensor,
            past_by_node_id,
            **_,
        ) -> Dict[str, PastKeyValues]:
            _, translated_window_past = translator_pool.build_replayed_target_past(
                source_past_key_values=past_by_node_id[edge.src_id],
                prefix_input_ids=prefix_cache_ids,
                target_model=ctx.mm.get_model(edge.tgt_id),
                src_node_id=edge.src_id,
                tgt_node_id=edge.tgt_id,
                tgt_spec=ctx.mm.get_model_spec(edge.tgt_id),
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
    else:
        logger.info("Skipping OpenWebText/validation for long-context-only evaluation.")

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
            build_example_state_fn=_build_logit_example_state,
            build_edge_artifacts_fn=_build_logit_edge_artifacts,
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

    generation_dataset_specs = get_default_gen_qa_dataset_specs()
    generation_budget_requests = parse_generation_context_budgets(eval_config.generation_context_budgets)
    logger.info(
        "generation_context_budgets=%s",
        [format_generation_context_budget_label(value) for value in generation_budget_requests],
    )

    for spec in generation_dataset_specs:
        for requested_context_budget in generation_budget_requests:
            dataset_result_key = build_generation_eval_result_key(
                spec.name_for_log,
                requested_context_budget=requested_context_budget,
            )
            logger.info("Preparing generation dataloader for %s", dataset_result_key)

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
                requested_context_budget=requested_context_budget,
            )
            all_generation_results[dataset_result_key] = results

            log_generation_dataset_result(
                logger=logger,
                dataset_name=dataset_result_key,
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
