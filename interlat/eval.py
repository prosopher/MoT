from dataclasses import asdict
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from core.context import Context
from core.eval_util import *
from core import eval_util as eval_util_module
from interlat.train import (
    extract_model_prefill_artifacts,
    trim_communication_prefix_from_past,
)



def _build_logit_example_state(
    *,
    ctx: Context,
    cache_input_ids: torch.Tensor,
    **_,
):
    prefill_by_node_id = {
        node.id: extract_model_prefill_artifacts(ctx.mm.get_model(node.id), cache_input_ids)
        for node in ctx.nodes
    }
    return {
        "past_by_node_id": {
            node.id: prefill_by_node_id[node.id][0]
            for node in ctx.nodes
        },
        "last_hidden_by_node_id": {
            node.id: prefill_by_node_id[node.id][1]
            for node in ctx.nodes
        },
    }



def _build_logit_edge_artifacts(
    *,
    ctx: Context,
    edge: Edge,
    cache_input_ids: torch.Tensor,
    example_state,
    translator_pool,
    **_,
) -> LogitEvalEdgeArtifacts:
    past_by_node_id = example_state["past_by_node_id"]
    last_hidden_by_node_id = example_state["last_hidden_by_node_id"]
    translated_past = translator_pool.build_target_past_from_source_latents(
        source_last_hidden_state=last_hidden_by_node_id[edge.src_id],
        prefix_input_ids=cache_input_ids,
        target_model=ctx.mm.get_model(edge.tgt_id),
        src_node_id=edge.src_id,
        tgt_node_id=edge.tgt_id,
    )
    communication_length = translated_past[0][0].shape[2] - cache_input_ids.shape[1]
    native_past = past_by_node_id[edge.tgt_id]
    trimmed_translated_past = trim_communication_prefix_from_past(translated_past, communication_length)
    return LogitEvalEdgeArtifacts(
        translated_past_key_values=translated_past,
        native_past_key_values=native_past,
        cosine_value=cosine_similarity_between_past(trimmed_translated_past, native_past),
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
    tokenizer = ctx.tokenizer
    path_metrics = {edge.id: GenerationRunningAverage() for edge in edges}

    processed_examples = 0

    for batch_idx, batch in enumerate(dataloader, start=1):
        for example in batch:
            question = example["question"]
            context_text = example["context"]
            gold_answers = example["answers"]

            context_budget = None
            if spec.answer_mode in {"squad", "newsqa", "multinews"}:
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
                logging.info(
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
            last_hidden_by_node_id = {
                node.id: prefill_by_node_id[node.id][1]
                for node in nodes
            }

            for edge in edges:
                translated_past = translator_pool.build_target_past_from_source_latents(
                    source_last_hidden_state=last_hidden_by_node_id[edge.src_id],
                    prefix_input_ids=cache_input_ids,
                    target_model=ctx.mm.get_model(edge.tgt_id),
                    src_node_id=edge.src_id,
                    tgt_node_id=edge.tgt_id,
                )
                native_past = past_by_node_id[edge.tgt_id]
                communication_length = translated_past[0][0].shape[2] - cache_input_ids.shape[1]
                trimmed_translated_past = trim_communication_prefix_from_past(translated_past, communication_length)
                cosine_value = cosine_similarity_between_past(trimmed_translated_past, native_past)

                translated_answer = predict_generation_task_answer(
                    model=ctx.mm.get_model(edge.tgt_id),
                    tokenizer=tokenizer,
                    past_key_values=translated_past,
                    seed_token=seed_token,
                    eval_config=eval_config,
                    question_cache_ids=question_cache_ids,
                )
                native_answer = predict_generation_task_answer(
                    model=ctx.mm.get_model(edge.tgt_id),
                    tokenizer=tokenizer,
                    past_key_values=native_past,
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
            logging.info(
                "[%s] generation progress: %d/%d examples",
                spec.name_for_log,
                processed_examples,
                eval_config.max_examples_per_dataset,
            )

    return summarize_generation_path_metrics(path_metrics)


@torch.inference_mode()
def evaluate_openwebtext_validation_loss_interlat(
    ctx: Context,
    eval_config: EvalConfig,
    translator_pool,
) -> Dict[str, Dict[str, float]]:
    train_config = ctx.config
    profiler = InferenceProfiler(train_config.device)
    dataloader = build_openwebtext_eval_dataloader(
        tokenizer=ctx.tokenizer,
        config=train_config,
        batch_size=eval_config.batch_size,
        num_workers=eval_config.num_workers,
        shuffle=eval_config.shuffle_eval_stream,
        seed=eval_config.seed,
        shuffle_buffer=eval_config.shuffle_buffer,
    )

    max_examples = max(1, eval_config.max_examples_per_dataset)
    processed_examples = 0
    loss_sums = {edge.id: {"translated": 0.0, "native": 0.0} for edge in ctx.edges}
    counts = {edge.id: 0 for edge in ctx.edges}
    profile_accumulators = {
        edge.id: {
            "translated": InferenceProfileAccumulator(),
            "native": InferenceProfileAccumulator(),
        }
        for edge in ctx.edges
    }
    tsne_features = {
        edge.id: {label: [] for label in OPENWEBTEXT_TSNE_LABEL_ORDER}
        for edge in ctx.edges
    }

    for batch_idx, input_ids in enumerate(dataloader, start=1):
        if processed_examples >= max_examples:
            break
        remaining_examples = max_examples - processed_examples
        if input_ids.shape[0] > remaining_examples:
            input_ids = input_ids[:remaining_examples]
        input_ids = input_ids.to(train_config.device)

        prefix_cache_ids, lm_input_ids, lm_labels = split_prefix_and_suffix_for_exact_next_token_loss(
            input_ids=input_ids,
            prefix_tokens=train_config.prefix_tokens,
        )
        prefill_by_node_id = {
            node.id: extract_model_prefill_artifacts(ctx.mm.get_model(node.id), prefix_cache_ids)
            for node in ctx.nodes
        }
        past_by_node_id = {
            node.id: prefill_by_node_id[node.id][0]
            for node in ctx.nodes
        }
        last_hidden_by_node_id = {
            node.id: prefill_by_node_id[node.id][1]
            for node in ctx.nodes
        }

        batch_examples = input_ids.shape[0]
        profile_tokens = int(lm_labels.numel())
        seed_token = lm_input_ids[:, :1]
        generation_steps = int(lm_labels.shape[1])

        for edge in ctx.edges:
            translated_past = translator_pool.build_target_past_from_source_latents(
                source_last_hidden_state=last_hidden_by_node_id[edge.src_id],
                prefix_input_ids=prefix_cache_ids,
                target_model=ctx.mm.get_model(edge.tgt_id),
                src_node_id=edge.src_id,
                tgt_node_id=edge.tgt_id,
            )
            native_past = past_by_node_id[edge.tgt_id]

            translated_loss = float(
                compute_suffix_lm_loss(
                    target_model=ctx.mm.get_model(edge.tgt_id),
                    past_key_values=translated_past,
                    lm_input_ids=lm_input_ids,
                    lm_labels=lm_labels,
                ).item()
            )
            native_loss = float(
                compute_suffix_lm_loss(
                    target_model=ctx.mm.get_model(edge.tgt_id),
                    past_key_values=native_past,
                    lm_input_ids=lm_input_ids,
                    lm_labels=lm_labels,
                ).item()
            )

            def run_translated_inference() -> int:
                return run_openwebtext_greedy_inference(
                    model=ctx.mm.get_model(edge.tgt_id),
                    past_key_values=translated_past,
                    seed_token=seed_token,
                    max_new_tokens=generation_steps,
                )

            def run_native_inference() -> int:
                return run_openwebtext_greedy_inference(
                    model=ctx.mm.get_model(edge.tgt_id),
                    past_key_values=native_past,
                    seed_token=seed_token,
                    max_new_tokens=generation_steps,
                )

            _, translated_profile = profiler.measure(run_translated_inference, tokens=profile_tokens)
            with temporarily_offload_module(translator_pool, train_config.device):
                _, native_profile = profiler.measure(run_native_inference, tokens=profile_tokens)

            loss_sums[edge.id]["translated"] += translated_loss * batch_examples
            loss_sums[edge.id]["native"] += native_loss * batch_examples
            counts[edge.id] += batch_examples
            profile_accumulators[edge.id]["translated"].update(
                latency_sec=float(translated_profile.get("latency_sec", 0.0)),
                tokens=translated_profile.get("tokens", 0),
                peak_memory_bytes=translated_profile.get("peak_memory_bytes"),
            )
            profile_accumulators[edge.id]["native"].update(
                latency_sec=float(native_profile.get("latency_sec", 0.0)),
                tokens=native_profile.get("tokens", 0),
                peak_memory_bytes=native_profile.get("peak_memory_bytes"),
            )

            communication_length = translated_past[0][0].shape[2] - prefix_cache_ids.shape[1]
            trimmed_translated_past = trim_communication_prefix_from_past(translated_past, communication_length)
            named_pasts = build_openwebtext_tsne_named_pasts(
                source_top_past_key_values=past_by_node_id[edge.src_id],
                translated_past_key_values=trimmed_translated_past,
                target_top_past_key_values=native_past,
            )
            eval_util_module._accumulate_openwebtext_tsne_samples(
                tsne_features,
                edge_id=edge.id,
                named_pasts=named_pasts,
            )

        processed_examples += batch_examples
        if batch_idx % 25 == 0:
            logging.info(
                "[OpenWebText/validation] progress: %d/%d sequences",
                processed_examples,
                max_examples,
            )

    summaries = {}
    for edge in ctx.edges:
        count = counts[edge.id]
        average_losses = {
            metric_name: float(total_loss / count)
            for metric_name, total_loss in loss_sums[edge.id].items()
        } if count > 0 else {}
        profile_summaries = {
            metric_name: accumulator.summary()
            for metric_name, accumulator in profile_accumulators[edge.id].items()
        }
        summaries[edge.id] = summarize_openwebtext_named_losses(
            average_losses,
            count,
            primary_name="translated",
            loss_field_by_name={"native": "native_loss"},
            profile_summary_by_name=profile_summaries,
            profile_field_prefix_by_name={"native": "native"},
        )

    tsne_paths = eval_util_module._finalize_openwebtext_tsne_plots(
        output_path=eval_config.output_path,
        seed=eval_config.seed,
        features_by_edge_and_group=tsne_features,
        perplexity=50.0,
        max_iter=1000,
    )
    for edge in ctx.edges:
        if edge.id in tsne_paths:
            summaries[edge.id]["tsne_plot_path"] = tsne_paths[edge.id]
    return summaries



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
    logging.info("translation_mode=prepend_interlat_hidden_communication")
    logging.info("qa_eval_log_path=%s", log_path)

    all_logit_results = {}
    all_generation_results = {}

    openwebtext_loss_results = evaluate_openwebtext_validation_loss_interlat(
        ctx=ctx,
        eval_config=eval_config,
        translator_pool=translator_pool,
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
