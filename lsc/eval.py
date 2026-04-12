from dataclasses import asdict
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from core.context import Context
from core.eval_util import *


@torch.inference_mode()
def evaluate_dataset(
    ctx: Context,
    spec: HFDatasetSpec,
    dataloader: DataLoader,
    eval_config: EvalConfig,
    translator_pool,
    logger: logging.Logger,
) -> Dict[str, Dict[str, float]]:
    nodes = ctx.nodes
    edges = ctx.edges
    train_config = ctx.config
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

            past_by_node_id = {
                node.id: extract_past_key_values(ctx.mm.get_model(node.id), cache_input_ids)
                for node in nodes
            }

            for edge in edges:
                translated_top_past = translator_pool.translate_top_layers(
                    past_key_values=past_by_node_id[edge.src_id],
                    src_name=edge.src_id,
                    tgt_name=edge.tgt_id,
                    tgt_spec=ctx.mm.get_model_spec(edge.tgt_id),
                )

                target_top = slice_top_layers(
                    past_key_values=past_by_node_id[edge.tgt_id],
                    top_layers_to_translate=ctx.mm.get_model_spec(edge.tgt_id).num_layers,
                )
                cosine_value = cosine_similarity_between_past(translated_top_past, target_top)

                mixed_target_past = replace_top_layers(
                    base_past_key_values=past_by_node_id[edge.tgt_id],
                    translated_top_past_key_values=translated_top_past,
                )

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
    nodes = ctx.nodes
    edges = ctx.edges
    train_config = ctx.config
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

            past_by_node_id = {
                node.id: extract_past_key_values(ctx.mm.get_model(node.id), cache_input_ids)
                for node in nodes
            }

            for edge in edges:
                translated_top_past = translator_pool.translate_top_layers(
                    past_key_values=past_by_node_id[edge.src_id],
                    src_name=edge.src_id,
                    tgt_name=edge.tgt_id,
                    tgt_spec=ctx.mm.get_model_spec(edge.tgt_id),
                )

                target_top = slice_top_layers(
                    past_key_values=past_by_node_id[edge.tgt_id],
                    top_layers_to_translate=ctx.mm.get_model_spec(edge.tgt_id).num_layers,
                )
                cosine_value = cosine_similarity_between_past(translated_top_past, target_top)

                mixed_target_past = replace_top_layers(
                    base_past_key_values=past_by_node_id[edge.tgt_id],
                    translated_top_past_key_values=translated_top_past,
                )

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
    full_model_specs = {node.id: get_model_spec(ctx.mm.get_model(node.id)) for node in nodes}
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
    logger.info("top_layers_ratio=%.6f", train_config.top_layers_ratio)
    for node in nodes:
        logger.info(
            "translated_layers: %s_top=%d/%d (%s)",
            node.id,
            ctx.mm.get_model_spec(node.id).num_layers,
            full_model_specs[node.id].num_layers,
            node.model_id,
        )
    logger.info("edges=%s", [edge.id for edge in edges])
    logger.info("translation_mode=replace_top_layers_after_target_forward")
    logger.info("qa_eval_log_path=%s", log_path)

    all_logit_results = {}
    all_generation_results = {}

    logger.info("Preparing validation dataloader for OpenWebText/validation")
    openwebtext_loss_results = evaluate_openwebtext_validation_loss(
        ctx=ctx,
        eval_config=eval_config,
        translator_pool=translator_pool,
        logger=logger,
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
