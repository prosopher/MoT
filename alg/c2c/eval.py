from dataclasses import asdict
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from core.context import Context
from core.eval_util import *
from alg.c2c.train import (
    get_top_layers_to_translate_for_edge,
    get_top_layers_to_translate,
    get_translation_mode_name,
    is_projection_only_variant,
    extract_aligned_sharer_past_for_c2c_project,
    translate_top_layers,
)


def _evaluate_openwebtext_validation_loss_c2c(
    ctx: Context,
    eval_config: EvalConfig,
    translator_pool,
    *,
    build_translated_target_past_fn,
    build_visualization_pasts_fn=None,
) -> Dict[str, Dict[str, float]]:
    """C2C-local copy of the top-layer evaluator with token IDs forwarded.

    The shared top-layer evaluator intentionally exposes only past_key_values to
    algorithm callbacks. C2C-Project cross-tokenizer alignment additionally
    needs the receiver token IDs, which are already available in the generic
    OpenWebText metrics loop. Keep that C2C-specific requirement here instead
    of widening the shared eval_util callback contract.
    """
    train_config = ctx.config
    profiler = InferenceProfiler(
        train_config.device,
        models=ctx.tp.models.values(),
        translator_pool=translator_pool,
    )

    def evaluate_edge_losses_fn(
        *,
        edge_id: str,
        edge: Edge,
        source_context_token_ids: TokenIDs,
        target_context_token_ids: TokenIDs,
        prompt_token_ids: TokenIDs,
        label_token_ids: TokenIDs,
        past_by_node_id,
    ):
        del edge_id
        profile_tokens = int(label_token_ids.numel())
        seed_token = prompt_token_ids[:, :1]
        generation_steps = int(label_token_ids.shape[1])

        translated_target_past = build_translated_target_past_fn(
            edge=edge,
            source_context_token_ids=source_context_token_ids,
            target_context_token_ids=target_context_token_ids,
            past_by_node_id=past_by_node_id,
        )
        translated_loss = float(
            compute_suffix_lm_loss(
                target_model=ctx.tp.get_model(edge.tgt_id),
                past_key_values=translated_target_past,
                prompt_token_ids=prompt_token_ids,
                label_token_ids=label_token_ids,
            ).item()
        )
        native_loss = float(
            compute_suffix_lm_loss(
                target_model=ctx.tp.get_model(edge.tgt_id),
                past_key_values=past_by_node_id[edge.tgt_id],
                prompt_token_ids=prompt_token_ids,
                label_token_ids=label_token_ids,
            ).item()
        )

        def run_translated_inference():
            return run_openwebtext_greedy_inference(
                model=ctx.tp.get_model(edge.tgt_id),
                past_key_values=translated_target_past,
                seed_token=seed_token,
                max_new_tokens=generation_steps,
            )

        def run_native_inference():
            return run_openwebtext_greedy_inference(
                model=ctx.tp.get_model(edge.tgt_id),
                past_key_values=past_by_node_id[edge.tgt_id],
                seed_token=seed_token,
                max_new_tokens=generation_steps,
            )

        translated_result, translated_profile = profiler.measure(
            run_translated_inference,
            tokens=profile_tokens,
        )
        del translated_result
        with temporarily_offload_module(translator_pool, train_config.device):
            native_result, native_profile = profiler.measure(
                run_native_inference,
                tokens=profile_tokens,
            )
            del native_result
        return (
            {
                "translated": translated_loss,
                "native": native_loss,
            },
            {
                "translated": translated_profile,
                "native": native_profile,
            },
        )

    return evaluate_openwebtext_validation_loss_metrics(
        ctx=ctx,
        output_path=eval_config.output_path,
        batch_size=eval_config.batch_size,
        num_workers=eval_config.num_workers,
        shuffle=eval_config.shuffle_eval_stream,
        seed=eval_config.seed,
        shuffle_buffer=eval_config.shuffle_buffer,
        max_examples=eval_config.max_examples_per_dataset,
        evaluate_edge_losses_fn=evaluate_edge_losses_fn,
        build_visualization_pasts_fn=build_visualization_pasts_fn,
    )


def _build_logit_edge_artifacts(
    ctx: Context,
    edge: Edge,
    source_context_token_ids: TokenIDs,
    target_context_token_ids: TokenIDs,
    past_by_node_id,
    translator_pool,
) -> LogitEvalEdgeArtifacts:
    train_config = ctx.config

    sharer_past_key_values = past_by_node_id[edge.src_id]
    if is_projection_only_variant(train_config):
        sharer_past_key_values = extract_aligned_sharer_past_for_c2c_project(
            receiver_context_token_ids=target_context_token_ids,
            receiver_model=ctx.tp.get_model(edge.tgt_id),
            sharer_model=ctx.tp.get_model(edge.src_id),
        )

    translated_top_past = translate_top_layers(
        translator_pool=translator_pool,
        train_config=train_config,
        sharer_past_key_values=sharer_past_key_values,
        receiver_past_key_values=past_by_node_id[edge.tgt_id],
        src_node_id=edge.src_id,
        tgt_node_id=edge.tgt_id,
        tgt_spec=ctx.tp.get_model_spec(edge.tgt_id),
    )

    native_past = past_by_node_id[edge.tgt_id]
    translated_target_past = replace_top_layers(
        base_past_key_values=native_past,
        translated_top_past_key_values=translated_top_past,
    )
    return LogitEvalEdgeArtifacts(
        translated_past_key_values=translated_target_past,
        native_past_key_values=native_past,
        cosine_value=cosine_similarity_between_past(translated_target_past, native_past),
    )


@torch.inference_mode()
def evaluate_generation_dataset(
    ctx: Context,
    spec: HFDatasetSpec,
    dataloader: DataLoader,
    eval_config: EvalConfig,
    translator_pool,
) -> Dict[str, Dict[str, float]]:
    train_config = ctx.config
    nodes = ctx.nodes
    edges = ctx.edges
    device = train_config.device
    path_metrics = {edge.id: GenerationRunningAverage() for edge in edges}

    processed_examples = 0

    for batch_idx, batch in enumerate(dataloader, start=1):
        for example in batch:
            question = example["question"]
            context_text = example["context"]
            gold_answers = example["answers"]

            prepared_inputs_by_node_id = {}
            for node in nodes:
                model = ctx.tp.get_model(node.id)
                context_budget = None
                if spec.answer_mode in {"squad", "newsqa"}:
                    context_budget = compute_benchmark_context_budget(
                        ctx=ctx,
                        spec=spec,
                        question=question,
                        eval_config=eval_config,
                        model=model,
                    )
                prepared_inputs = prepare_generation_task_inputs(
                    spec=spec,
                    model=model,
                    context=context_text,
                    question=question,
                    device=device,
                    max_input_tokens=context_budget,
                )
                prepared_inputs_by_node_id[node.id] = prepared_inputs
                if prepared_inputs.get("was_truncated") and processed_examples < 3:
                    prompt_token_ids = prepared_inputs["prompt_token_ids"]
                    prompt_tokens = 0 if prompt_token_ids is None else prompt_token_ids.shape[1]
                    logging.info(
                        "[%s][%s] truncated context to %d tokens to fit model context window (prompt_tokens=%d, answer_token_budget=%d)",
                        spec.name_for_log,
                        node.id,
                        prepared_inputs["context_token_ids"].shape[1],
                        prompt_tokens,
                        get_answer_token_budget(eval_config),
                    )

            past_by_node_id = {
                node.id: extract_past_key_values(
                    ctx.tp.get_model(node.id),
                    prepared_inputs_by_node_id[node.id]["context_token_ids"],
                )
                for node in nodes
            }

            for edge in edges:
                target_model = ctx.tp.get_model(edge.tgt_id)
                target_inputs = prepared_inputs_by_node_id[edge.tgt_id]
                prompt_token_ids = target_inputs["prompt_token_ids"]
                seed_token = target_inputs["seed_token"]
                sharer_past_key_values = past_by_node_id[edge.src_id]
                if is_projection_only_variant(train_config):
                    sharer_past_key_values = extract_aligned_sharer_past_for_c2c_project(
                        receiver_context_token_ids=target_inputs["context_token_ids"],
                        receiver_model=target_model,
                        sharer_model=ctx.tp.get_model(edge.src_id),
                    )
                translated_top_past = translate_top_layers(
                    translator_pool=translator_pool,
                    train_config=train_config,
                    sharer_past_key_values=sharer_past_key_values,
                    receiver_past_key_values=past_by_node_id[edge.tgt_id],
                    src_node_id=edge.src_id,
                    tgt_node_id=edge.tgt_id,
                    tgt_spec=ctx.tp.get_model_spec(edge.tgt_id),
                )

                native_past = past_by_node_id[edge.tgt_id]
                translated_target_past = replace_top_layers(
                    base_past_key_values=native_past,
                    translated_top_past_key_values=translated_top_past,
                )
                cosine_value = cosine_similarity_between_past(translated_target_past, native_past)

                translated_answer = predict_generation_task_answer(
                    model=target_model,
                    past_key_values=translated_target_past,
                    seed_token=seed_token,
                    eval_config=eval_config,
                    prompt_token_ids=prompt_token_ids,
                )
                native_answer = predict_generation_task_answer(
                    model=target_model,
                    past_key_values=past_by_node_id[edge.tgt_id],
                    seed_token=seed_token,
                    eval_config=eval_config,
                    prompt_token_ids=prompt_token_ids,
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
    logging.info("Starting evaluation")
    logging.info("checkpoint_dir_path=%s", checkpoint_dir_path)
    logging.info("eval_config=%s", asdict(eval_config))

    translator_pool.eval()
    for node in nodes:
        ctx.tp.get_model(node.id).eval()

    logging.info("restored_train_config=%s", asdict(train_config))
    logging.info("nodes=%s", [asdict(node) for node in nodes])
    logging.info("edges=%s", [edge.id for edge in edges])
    logging.info("top_layers_to_translate=%s", get_top_layers_to_translate(train_config))
    for edge in edges:
        logging.info(
            "top_layers_to_translate[%s]=%d",
            edge.id,
            get_top_layers_to_translate_for_edge(train_config, ctx, edge),
        )
    logging.info("translation_mode=%s", get_translation_mode_name(train_config))
    logging.info("qa_eval_log_path=%s", log_path)

    all_logit_results = {}
    all_generation_results = {}

    logging.info("Preparing validation dataloader for OpenWebText/validation")

    def build_translated_target_past_fn(
        *,
        edge: Edge,
        past_by_node_id,
        source_context_token_ids: TokenIDs = None,
        target_context_token_ids: TokenIDs = None,
    ) -> PastKeyValues:
        sharer_past_key_values = past_by_node_id[edge.src_id]
        if is_projection_only_variant(train_config):
            if target_context_token_ids is None:
                raise ValueError("C2C-Project cross-tokenizer evaluation requires target_context_token_ids.")
            sharer_past_key_values = extract_aligned_sharer_past_for_c2c_project(
                receiver_context_token_ids=target_context_token_ids,
                receiver_model=ctx.tp.get_model(edge.tgt_id),
                sharer_model=ctx.tp.get_model(edge.src_id),
            )
        translated_top_past = translate_top_layers(
            translator_pool=translator_pool,
            train_config=train_config,
            sharer_past_key_values=sharer_past_key_values,
            receiver_past_key_values=past_by_node_id[edge.tgt_id],
            src_node_id=edge.src_id,
            tgt_node_id=edge.tgt_id,
            tgt_spec=ctx.tp.get_model_spec(edge.tgt_id),
        )
        return replace_top_layers(
            base_past_key_values=past_by_node_id[edge.tgt_id],
            translated_top_past_key_values=translated_top_past,
        )

    def build_visualization_pasts_fn(
        *,
        edge: Edge,
        past_by_node_id,
        target_context_token_ids: TokenIDs = None,
        **_,
    ) -> Dict[str, PastKeyValues]:
        sharer_past_key_values = past_by_node_id[edge.src_id]
        if is_projection_only_variant(train_config):
            if target_context_token_ids is None:
                raise ValueError("C2C-Project cross-tokenizer visualization requires target_context_token_ids.")
            sharer_past_key_values = extract_aligned_sharer_past_for_c2c_project(
                receiver_context_token_ids=target_context_token_ids,
                receiver_model=ctx.tp.get_model(edge.tgt_id),
                sharer_model=ctx.tp.get_model(edge.src_id),
            )
        return build_openwebtext_tsne_named_pasts(
            source_top_past_key_values=slice_top_layers(
                past_key_values=sharer_past_key_values,
                top_layers_to_translate=get_top_layers_to_translate_for_edge(train_config, ctx, edge),
            ),
            translated_past_key_values=translate_top_layers(
                translator_pool=translator_pool,
                train_config=train_config,
                sharer_past_key_values=sharer_past_key_values,
                receiver_past_key_values=past_by_node_id[edge.tgt_id],
                src_node_id=edge.src_id,
                tgt_node_id=edge.tgt_id,
                tgt_spec=ctx.tp.get_model_spec(edge.tgt_id),
            ),
            target_top_past_key_values=slice_top_layers(
                past_key_values=past_by_node_id[edge.tgt_id],
                top_layers_to_translate=get_top_layers_to_translate_for_edge(train_config, ctx, edge),
            ),
        )

    openwebtext_loss_results = _evaluate_openwebtext_validation_loss_c2c(
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
        if row.get("tsne_plot_path"):
            logging.info(
                "[OpenWebText/validation] %s | tsne_plot=%s",
                edge.id,
                row["tsne_plot_path"],
            )

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
