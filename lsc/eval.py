from dataclasses import asdict
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from core.context import Context
from core.eval_util import *


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
    example_state,
    translator_pool,
    **_,
) -> LogitEvalEdgeArtifacts:
    past_by_node_id = example_state["past_by_node_id"]

    translated_past = translator_pool.translate_layers(
        past_key_values=past_by_node_id[edge.src_id],
        src_node_id=edge.src_id,
        tgt_node_id=edge.tgt_id,
        tgt_spec=ctx.mm.get_model_spec(edge.tgt_id),
    )
    native_past = past_by_node_id[edge.tgt_id]
    return LogitEvalEdgeArtifacts(
        translated_past_key_values=translated_past,
        native_past_key_values=native_past,
        cosine_value=cosine_similarity_between_past(translated_past, native_past),
    )


@torch.inference_mode()
def evaluate_generation_dataset(
    ctx: Context,
    spec: HFDatasetSpec,
    dataloader: DataLoader,
    eval_config: EvalConfig,
    translator_pool,
) -> Dict[str, Dict[str, float]]:
    nodes = ctx.nodes
    edges = ctx.edges
    train_config = ctx.config
    device = train_config.device
    path_metrics = {edge.id: GenerationRunningAverage() for edge in edges}

    processed_examples = 0

    for batch_idx, batch in enumerate(dataloader, start=1):
        for example in batch:
            question = example["question"]
            context_text = example["context"]
            gold_answers = example["answers"]

            for edge in edges:
                tokenizer = ctx.mm.get_tokenizer(edge.tgt_id)
                context_budget = None
                if spec.answer_mode in {"squad", "newsqa"}:
                    context_budget = compute_benchmark_context_budget(
                        ctx=ctx,
                        spec=spec,
                        question=question,
                        eval_config=eval_config,
                        tokenizer=tokenizer,
                        target_node_id=edge.tgt_id,
                    )

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
                native_prefix_size_bytes = compute_generation_prefix_text_bytes(prepared_inputs)

                if prepared_inputs.get("was_truncated") and processed_examples < 3:
                    suffix_cache_tokens = 0 if suffix_cache_ids is None else suffix_cache_ids.shape[1]
                    logging.info(
                        "[%s][%s] truncated prefix to %d tokens to fit model context window (suffix_cache_tokens=%d, answer_token_budget=%d)",
                        spec.name_for_log,
                        edge.id,
                        prefix_input_ids.shape[1],
                        suffix_cache_tokens,
                        get_answer_token_budget(eval_config),
                    )

                register_generation_source_prefill_start(device=prefix_input_ids.device)
                source_past = extract_past_key_values(ctx.mm.get_model(edge.src_id), prefix_input_ids)

                translated_past = translator_pool.translate_layers(
                    past_key_values=source_past,
                    src_node_id=edge.src_id,
                    tgt_node_id=edge.tgt_id,
                    tgt_spec=ctx.mm.get_model_spec(edge.tgt_id),
                )

                translated_answer = predict_generation_task_answer(
                    model=ctx.mm.get_model(edge.tgt_id),
                    tokenizer=tokenizer,
                    past_key_values=translated_past,
                    seed_token=seed_token,
                    eval_config=eval_config,
                    suffix_cache_ids=suffix_cache_ids,
                )

                native_profile_state = start_native_generation_profile(
                    prefix_input_ids=prefix_input_ids,
                    prefix_size_bytes=native_prefix_size_bytes,
                )
                native_past = extract_past_key_values(ctx.mm.get_model(edge.tgt_id), prefix_input_ids)
                native_profile = finish_native_generation_profile(
                    native_profile_state,
                    device=prefix_input_ids.device,
                )
                cosine_value = cosine_similarity_between_past(translated_past, native_past)
                native_answer = predict_generation_task_answer(
                    model=ctx.mm.get_model(edge.tgt_id),
                    tokenizer=tokenizer,
                    past_key_values=native_past,
                    seed_token=seed_token,
                    eval_config=eval_config,
                    suffix_cache_ids=suffix_cache_ids,
                    **native_profile,
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
            logging.info(
                "[%s] generation progress: %d/%d examples | %s",
                spec.name_for_log,
                processed_examples,
                eval_config.max_examples_per_dataset,
                format_generation_progress_ttft(
                    path_metrics,
                    method_name=(
                        getattr(eval_config, "alg", None)
                        or getattr(ctx.config, "alg", None)
                        or "Method"
                    ),
                ),
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
    logging.info("Starting evaluation")
    logging.info("checkpoint_dir_path=%s", checkpoint_dir_path)
    logging.info("eval_config=%s", asdict(eval_config))

    translator_pool.eval()
    for node in nodes:
        ctx.mm.get_model(node.id).eval()

    logging.info("restored_train_config=%s", asdict(train_config))
    logging.info("nodes=%s", [asdict(node) for node in nodes])
    logging.info("edges=%s", [edge.id for edge in edges])
    for node in nodes:
        logging.info(
            "translation_spec: %s layers=%d hidden=%d heads=%d (%s)",
            node.id,
            ctx.mm.get_model_spec(node.id).num_layers,
            ctx.mm.get_model_spec(node.id).hidden_size,
            ctx.mm.get_model_spec(node.id).num_heads,
            node.model_id,
        )
    logging.info("translation_mode=translate_all_layers")
    logging.info("qa_eval_log_path=%s", log_path)

    all_logit_results = {}
    all_generation_results = {}

    logging.info("Preparing validation dataloader for OpenWebText/validation")

    def build_translated_target_past_fn(*, edge: Edge, past_by_node_id) -> PastKeyValues:
        return translator_pool.translate_layers(
            past_key_values=past_by_node_id[edge.src_id],
            src_node_id=edge.src_id,
            tgt_node_id=edge.tgt_id,
            tgt_spec=ctx.mm.get_model_spec(edge.tgt_id),
        )

    def build_visualization_pasts_fn(*, edge: Edge, past_by_node_id, **_) -> Dict[str, PastKeyValues]:
        source_past = past_by_node_id[edge.src_id]
        target_past = past_by_node_id[edge.tgt_id]
        return build_openwebtext_tsne_named_pasts(
            source_top_past_key_values=source_past,
            translated_past_key_values=translator_pool.translate_layers(
                past_key_values=source_past,
                src_node_id=edge.src_id,
                tgt_node_id=edge.tgt_id,
                tgt_spec=ctx.mm.get_model_spec(edge.tgt_id),
            ),
            target_top_past_key_values=target_past,
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
            logging.info("[OpenWebText/validation] %s | tsne_plot=%s", edge.id, tsne_plot_path)

    logit_dataset_specs = get_default_logit_qa_dataset_specs()
    for spec in logit_dataset_specs:
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
    for spec in generation_dataset_specs:
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

    final_summary_markdown = build_final_summary_markdown(
        alg=eval_config.alg,
        nodes=nodes,
        edges=edges,
        all_logit_results=all_logit_results,
        all_generation_results=all_generation_results,
        openwebtext_loss_results=openwebtext_loss_results,
    )
    logging.info("===== FINAL MARKDOWN SUMMARY =====\n%s", final_summary_markdown)

    logging.info("Done. Saved log to %s", log_path)
    return log_path
