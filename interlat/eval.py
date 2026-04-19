from __future__ import annotations

from dataclasses import asdict
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from core.eval_util import *
from core.eval_util import (
    _accumulate_openwebtext_tsne_samples,
    _finalize_openwebtext_tsne_plots,
)
from interlat.train import (
    NodeTokenizerPool,
    build_latent_conditioned_past,
    extract_last_hidden_states,
    get_model_context_limit,
    trim_input_ids_from_left,
)


def _fit_cache_input_ids_to_model_limit(
    *,
    model,
    cache_input_ids: torch.Tensor,
    reserved_tail_tokens: int,
) -> torch.Tensor:
    model_context_limit = get_model_context_limit(model)
    max_cache_tokens = model_context_limit - reserved_tail_tokens
    if max_cache_tokens < 1:
        raise ValueError(
            "Insufficient context window for InterLat evaluation: "
            f"model_context_limit={model_context_limit}, reserved_tail_tokens={reserved_tail_tokens}"
        )
    return trim_input_ids_from_left(cache_input_ids, max_length=max_cache_tokens)


def _max_candidate_token_length(choice_token_ids: Dict[str, torch.Tensor]) -> int:
    if not choice_token_ids:
        return 1
    return max(int(token_ids.shape[0]) for token_ids in choice_token_ids.values())


def _slice_past_to_last_tokens(past_key_values, *, target_seq_len: int):
    if target_seq_len < 1:
        raise ValueError(f"target_seq_len must be >= 1, got {target_seq_len}")
    sliced_layers = []
    for key, value in past_key_values:
        if key.shape[2] < target_seq_len or value.shape[2] < target_seq_len:
            raise ValueError(
                "Cannot align InterLat past lengths for cosine: "
                f"source_seq_len={key.shape[2]}, target_seq_len={target_seq_len}"
            )
        sliced_layers.append(
            (
                key[:, :, -target_seq_len:, :].contiguous(),
                value[:, :, -target_seq_len:, :].contiguous(),
            )
        )
    return tuple(sliced_layers)


def _cosine_similarity_for_interlat_past(
    translated_past,
    native_past,
) -> float:
    if len(translated_past) != len(native_past):
        raise ValueError(
            "InterLat past cosine requires the same number of layers: "
            f"translated={len(translated_past)}, native={len(native_past)}"
        )
    aligned_seq_len = min(
        min(int(key.shape[2]) for key, _ in translated_past),
        min(int(key.shape[2]) for key, _ in native_past),
    )
    aligned_translated_past = _slice_past_to_last_tokens(
        translated_past,
        target_seq_len=aligned_seq_len,
    )
    aligned_native_past = _slice_past_to_last_tokens(
        native_past,
        target_seq_len=aligned_seq_len,
    )
    return cosine_similarity_between_past(aligned_translated_past, aligned_native_past)


@torch.inference_mode()
def _evaluate_openwebtext_validation(
    *,
    ctx: Context,
    eval_config: EvalConfig,
    translator_pool,
    node_tokenizers: NodeTokenizerPool,
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

    loss_sums = {edge.id: {"translated": 0.0, "native": 0.0} for edge in ctx.edges}
    cosine_sums = {edge.id: 0.0 for edge in ctx.edges}
    counts = {edge.id: 0 for edge in ctx.edges}
    profile_accumulators = {
        edge.id: {
            "translated": InferenceProfileAccumulator(),
            "native": InferenceProfileAccumulator(),
        }
        for edge in ctx.edges
    }
    tsne_features = {
        edge.id: {
            "source_top": [],
            "translated": [],
            "target_top": [],
        }
        for edge in ctx.edges
    }

    processed_examples = 0
    for batch_idx, input_ids in enumerate(dataloader, start=1):
        if processed_examples >= eval_config.max_examples_per_dataset:
            break

        remaining_examples = eval_config.max_examples_per_dataset - processed_examples
        if input_ids.shape[0] > remaining_examples:
            input_ids = input_ids[:remaining_examples]
        input_ids = input_ids.to(train_config.device)

        prefix_cache_ids, lm_input_ids, lm_labels = split_prefix_and_suffix_for_exact_next_token_loss(
            input_ids=input_ids,
            prefix_tokens=train_config.prefix_tokens,
        )

        batch_examples = int(input_ids.shape[0])
        processed_examples += batch_examples

        decoded_texts = ctx.tokenizer.batch_decode(
            input_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        for edge in ctx.edges:
            source_tokenizer = node_tokenizers[edge.src_id]
            source_encoded = source_tokenizer(
                list(decoded_texts),
                add_special_tokens=False,
                truncation=True,
                max_length=train_config.total_tokens,
                padding="max_length",
                return_tensors="pt",
            )
            source_input_ids = source_encoded["input_ids"].to(train_config.device)
            source_prefix_ids = source_input_ids[:, : train_config.prefix_tokens - 1]

            source_model = ctx.mm.get_model(edge.src_id)
            target_model = ctx.mm.get_model(edge.tgt_id)

            source_hidden = extract_last_hidden_states(source_model, source_prefix_ids)
            translated_latents = translator_pool.translate_hidden_states(
                edge_id=edge.id,
                source_hidden_states=source_hidden,
            )
            translated_prefix_ids = _fit_cache_input_ids_to_model_limit(
                model=target_model,
                cache_input_ids=prefix_cache_ids,
                reserved_tail_tokens=train_config.latent_tokens + int(lm_input_ids.shape[1]),
            )
            translated_past = build_latent_conditioned_past(
                target_model,
                prefix_input_ids=translated_prefix_ids,
                latent_prefix=translated_latents,
            )
            native_past = extract_past_key_values(target_model, prefix_cache_ids)

            translated_loss = float(
                compute_suffix_lm_loss(
                    target_model=target_model,
                    past_key_values=translated_past,
                    lm_input_ids=lm_input_ids,
                    lm_labels=lm_labels,
                ).item()
            )
            native_loss = float(
                compute_suffix_lm_loss(
                    target_model=target_model,
                    past_key_values=native_past,
                    lm_input_ids=lm_input_ids,
                    lm_labels=lm_labels,
                ).item()
            )
            cosine_value = _cosine_similarity_for_interlat_past(translated_past, native_past)

            profile_tokens = int(lm_labels.numel())
            seed_token = lm_input_ids[:, :1]
            generation_steps = int(lm_labels.shape[1])

            def run_translated_inference() -> int:
                return run_openwebtext_greedy_inference(
                    model=target_model,
                    past_key_values=translated_past,
                    seed_token=seed_token,
                    max_new_tokens=generation_steps,
                )

            def run_native_inference() -> int:
                return run_openwebtext_greedy_inference(
                    model=target_model,
                    past_key_values=native_past,
                    seed_token=seed_token,
                    max_new_tokens=generation_steps,
                )

            _, translated_profile = profiler.measure(run_translated_inference, tokens=profile_tokens)
            with temporarily_offload_module(translator_pool, train_config.device):
                _, native_profile = profiler.measure(run_native_inference, tokens=profile_tokens)

            loss_sums[edge.id]["translated"] += translated_loss * batch_examples
            loss_sums[edge.id]["native"] += native_loss * batch_examples
            cosine_sums[edge.id] += cosine_value * batch_examples
            counts[edge.id] += batch_examples
            profile_accumulators[edge.id]["translated"].update(
                latency_sec=float(translated_profile.get("latency_sec", 0.0)),
                tokens=int(translated_profile.get("tokens", 0)),
                peak_memory_bytes=translated_profile.get("peak_memory_bytes"),
            )
            profile_accumulators[edge.id]["native"].update(
                latency_sec=float(native_profile.get("latency_sec", 0.0)),
                tokens=int(native_profile.get("tokens", 0)),
                peak_memory_bytes=native_profile.get("peak_memory_bytes"),
            )

            named_pasts = build_openwebtext_tsne_named_pasts(
                source_top_past_key_values=extract_past_key_values(source_model, source_prefix_ids),
                translated_past_key_values=translated_past,
                target_top_past_key_values=native_past,
            )
            _accumulate_openwebtext_tsne_samples(
                tsne_features,
                edge_id=edge.id,
                named_pasts=named_pasts,
            )

        if batch_idx % 25 == 0:
            logging.info(
                "[OpenWebText/validation] progress: %d/%d sequences",
                processed_examples,
                eval_config.max_examples_per_dataset,
            )

    summaries: Dict[str, Dict[str, float]] = {}
    for edge in ctx.edges:
        count = counts[edge.id]
        average_losses = (
            {
                "translated": float(loss_sums[edge.id]["translated"] / count),
                "native": float(loss_sums[edge.id]["native"] / count),
            }
            if count > 0
            else {}
        )
        profile_summaries = {
            metric_name: accumulator.summary()
            for metric_name, accumulator in profile_accumulators[edge.id].items()
        }
        row = summarize_openwebtext_named_losses(
            average_losses,
            count,
            primary_name="translated",
            loss_field_by_name={"native": "native_loss"},
            profile_summary_by_name=profile_summaries,
            profile_field_prefix_by_name={"native": "native"},
        )
        row["cosine"] = float(cosine_sums[edge.id] / count) if count > 0 else float("nan")
        summaries[edge.id] = row

    tsne_paths = _finalize_openwebtext_tsne_plots(
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
    max_candidate_len = _max_candidate_token_length(
        build_logit_answer_candidates(ctx.tokenizer, spec)
    )

    def build_example_state_fn(
        *,
        ctx: Context,
        spec: HFDatasetSpec,
        example: Dict[str, Any],
        **_,
    ):
        source_hidden_by_edge: Dict[str, torch.Tensor] = {}
        for edge in ctx.edges:
            source_prepared = prepare_logit_task_inputs(
                spec=spec,
                tokenizer=node_tokenizers[edge.src_id],
                context=example.get("context"),
                question=example["question"],
                device=ctx.config.device,
                choices=example.get("choices"),
                subject=example.get("subject"),
            )
            source_hidden_by_edge[edge.id] = extract_last_hidden_states(
                ctx.mm.get_model(edge.src_id),
                source_prepared["cache_input_ids"],
            )
        return {"source_hidden_by_edge": source_hidden_by_edge}

    def build_edge_artifacts_fn(
        *,
        ctx: Context,
        edge: Edge,
        prepared_inputs,
        example_state,
        **_,
    ) -> LogitEvalEdgeArtifacts:
        target_model = ctx.mm.get_model(edge.tgt_id)
        question_cache_ids = prepared_inputs["question_cache_ids"]
        reserved_tail_tokens = (
            ctx.config.latent_tokens
            + (0 if question_cache_ids is None else int(question_cache_ids.shape[1]))
            + max_candidate_len
        )
        translated_cache_input_ids = _fit_cache_input_ids_to_model_limit(
            model=target_model,
            cache_input_ids=prepared_inputs["cache_input_ids"],
            reserved_tail_tokens=reserved_tail_tokens,
        )
        translated_latents = translator_pool.translate_hidden_states(
            edge_id=edge.id,
            source_hidden_states=example_state["source_hidden_by_edge"][edge.id],
        )
        translated_past = build_latent_conditioned_past(
            target_model,
            prefix_input_ids=translated_cache_input_ids,
            latent_prefix=translated_latents,
        )
        native_past = extract_past_key_values(target_model, prepared_inputs["cache_input_ids"])
        return LogitEvalEdgeArtifacts(
            translated_past_key_values=translated_past,
            native_past_key_values=native_past,
            cosine_value=_cosine_similarity_for_interlat_past(translated_past, native_past),
        )

    return evaluate_dataset(
        ctx=ctx,
        spec=spec,
        dataloader=dataloader,
        eval_config=eval_config,
        translator_pool=translator_pool,
        build_example_state_fn=build_example_state_fn,
        build_edge_artifacts_fn=build_edge_artifacts_fn,
    )


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
    path_metrics = {edge.id: GenerationRunningAverage() for edge in ctx.edges}
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
                tokenizer=ctx.tokenizer,
                context=context_text,
                question=question,
                device=ctx.config.device,
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

            for edge in ctx.edges:
                source_prepared = prepare_generation_task_inputs(
                    spec=spec,
                    tokenizer=node_tokenizers[edge.src_id],
                    context=context_text,
                    question=question,
                    device=ctx.config.device,
                    max_input_tokens=context_budget,
                )
                source_hidden = extract_last_hidden_states(
                    ctx.mm.get_model(edge.src_id),
                    source_prepared["cache_input_ids"],
                )

                target_model = ctx.mm.get_model(edge.tgt_id)
                reserved_tail_tokens = (
                    ctx.config.latent_tokens
                    + (0 if question_cache_ids is None else int(question_cache_ids.shape[1]))
                    + get_answer_token_budget(eval_config)
                )
                translated_cache_input_ids = _fit_cache_input_ids_to_model_limit(
                    model=target_model,
                    cache_input_ids=cache_input_ids,
                    reserved_tail_tokens=reserved_tail_tokens,
                )
                translated_latents = translator_pool.translate_hidden_states(
                    edge_id=edge.id,
                    source_hidden_states=source_hidden,
                )
                translated_past = build_latent_conditioned_past(
                    target_model,
                    prefix_input_ids=translated_cache_input_ids,
                    latent_prefix=translated_latents,
                )
                native_past = extract_past_key_values(target_model, cache_input_ids)

                translated_answer = predict_generation_task_answer(
                    model=target_model,
                    tokenizer=ctx.tokenizer,
                    past_key_values=translated_past,
                    seed_token=seed_token,
                    eval_config=eval_config,
                    question_cache_ids=question_cache_ids,
                )
                native_answer = predict_generation_task_answer(
                    model=target_model,
                    tokenizer=ctx.tokenizer,
                    past_key_values=native_past,
                    seed_token=seed_token,
                    eval_config=eval_config,
                    question_cache_ids=question_cache_ids,
                )

                path_metrics[edge.id].update(
                    cosine_value=_cosine_similarity_for_interlat_past(translated_past, native_past),
                    f1_value=compute_generation_f1(translated_answer, gold_answers),
                    native_f1_value=compute_generation_f1(native_answer, gold_answers),
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
    logging.info("===== OpenWebText/validation =====")
    for edge in ctx.edges:
        row = openwebtext_metrics[edge.id]
        pretty_name = build_edge_pretty_name(edge.id, ctx.nodes, ctx.edges)
        logging.info(
            "[OpenWebText/validation] %s | native_loss=%.6f | native_profile=%s | translated_loss=%.6f | translated_profile=%s | count=%d",
            pretty_name,
            row["native_loss"],
            build_openwebtext_profile_cell(row, prefix="native"),
            row["loss"],
            build_openwebtext_profile_cell(row),
            row["count"],
        )
        tsne_plot_path = row.get("tsne_plot_path")
        if isinstance(tsne_plot_path, str) and tsne_plot_path:
            logging.info("[OpenWebText/validation] %s | tsne_plot=%s", edge.id, tsne_plot_path)

    all_logit_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    all_generation_results: Dict[str, Dict[str, Dict[str, float]]] = {}
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
        log_dataset_result(spec.name_for_log, metrics, ctx.nodes, ctx.edges)
        all_logit_results[spec.name_for_log] = metrics
        all_results[spec.name_for_log] = metrics
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for factory in GEN_QA_SPEC_GROUP_FACTORIES:
        spec = factory()
        dataloader = build_generation_eval_dataloader(spec, eval_config)
        metrics = _evaluate_generation_dataset(
            ctx=ctx,
            spec=spec,
            dataloader=dataloader,
            eval_config=eval_config,
            translator_pool=translator_pool,
            node_tokenizers=node_tokenizers,
        )
        log_generation_dataset_result(spec.name_for_log, metrics, ctx.nodes, ctx.edges)
        all_generation_results[spec.name_for_log] = metrics
        all_results[spec.name_for_log] = metrics
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_summary_markdown = build_final_summary_markdown(
        alg=eval_config.alg,
        nodes=ctx.nodes,
        edges=ctx.edges,
        all_logit_results=all_logit_results,
        all_generation_results=all_generation_results,
        openwebtext_loss_results=openwebtext_metrics,
    )
    result_path = Path(eval_config.output_path) / "interlat_eval_results.json"
    write_json(str(result_path), all_results)
    logging.info("===== FINAL MARKDOWN SUMMARY =====\n%s", final_summary_markdown)
    logging.info("Saved Interlat evaluation results to %s", result_path)
    logging.info("Saved eval log to %s", log_path)
    return log_path
