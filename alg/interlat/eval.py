from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from core.context import Context
from core.eval_util import *
from alg.interlat.train import (
    build_latent_conditioned_past,
    extract_interlat_source_hidden_states,
    translate_hidden_states,
)


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


def _cosine_similarity_for_interlat_past(translated_past, native_past) -> float:
    if len(translated_past) != len(native_past):
        raise ValueError(
            "InterLat past cosine requires the same number of layers: "
            f"translated={len(translated_past)}, native={len(native_past)}"
        )
    aligned_seq_len = min(
        min(int(key.shape[2]) for key, _ in translated_past),
        min(int(key.shape[2]) for key, _ in native_past),
    )
    aligned_translated_past = _slice_past_to_last_tokens(translated_past, target_seq_len=aligned_seq_len)
    aligned_native_past = _slice_past_to_last_tokens(native_past, target_seq_len=aligned_seq_len)
    return cosine_similarity_between_past(aligned_translated_past, aligned_native_past)


@torch.inference_mode()
def _build_interlat_target_past(
    *,
    ctx: Context,
    edge: Edge,
    context_token_ids: TokenIDs,
    translator_pool,
) -> PastKeyValues:
    target_model = ctx.tp.get_model(edge.tgt_id)
    source_hidden_states = extract_interlat_source_hidden_states(
        ctx.tp.get_model(edge.src_id),
        context_token_ids,
    )
    translated_latents = translate_hidden_states(
        translator_pool=translator_pool,
        src_node_id=edge.src_id,
        tgt_node_id=edge.tgt_id,
        source_hidden_states=source_hidden_states,
    )
    return build_latent_conditioned_past(
        target_model,
        latent_prefix=translated_latents,
    )


def _build_logit_example_state(
    *,
    ctx: Context,
    context_token_ids: TokenIDs,
    **_,
):
    return {
        "past_by_node_id": {
            node.id: extract_past_key_values(ctx.tp.get_model(node.id), context_token_ids)
            for node in ctx.nodes
        }
    }


def _build_logit_edge_artifacts(
    *,
    ctx: Context,
    edge: Edge,
    context_token_ids: TokenIDs,
    prepared_inputs,
    example_state,
    translator_pool,
    **_,
) -> LogitEvalEdgeArtifacts:
    del prepared_inputs
    translated_past = _build_interlat_target_past(
        ctx=ctx,
        edge=edge,
        context_token_ids=context_token_ids,
        translator_pool=translator_pool,
    )
    native_past = example_state["past_by_node_id"][edge.tgt_id]
    return LogitEvalEdgeArtifacts(
        translated_past_key_values=translated_past,
        native_past_key_values=native_past,
        cosine_value=_cosine_similarity_for_interlat_past(translated_past, native_past),
    )


@torch.inference_mode()
def evaluate_openwebtext_validation_loss_interlat(
    ctx: Context,
    eval_config: EvalConfig,
    translator_pool,
    *,
    build_visualization_pasts_fn=None,
) -> Dict[str, Dict[str, float]]:
    """OpenWebText evaluation using the common metric loop and true source hidden states.

    InterLat communicates last-layer hidden states, so the translated past must be
    built from the same prefix ids used by the common evaluation split rather than
    from a KV value cache.
    """

    train_config = ctx.config
    profiler = InferenceProfiler(train_config.device)

    def evaluate_edge_losses_fn(
        *,
        edge_id: str,
        edge: Edge,
        context_token_ids: TokenIDs,
        prompt_token_ids: TokenIDs,
        label_token_ids: TokenIDs,
        past_by_node_id,
    ):
        del edge_id
        profile_tokens = int(label_token_ids.numel())
        seed_token = prompt_token_ids[:, :1]
        generation_steps = int(label_token_ids.shape[1])

        translated_target_past = _build_interlat_target_past(
            ctx=ctx,
            edge=edge,
            context_token_ids=context_token_ids,
            translator_pool=translator_pool,
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

        def run_translated_inference() -> int:
            return run_openwebtext_greedy_inference(
                model=ctx.tp.get_model(edge.tgt_id),
                past_key_values=translated_target_past,
                seed_token=seed_token,
                max_new_tokens=generation_steps,
            )

        def run_native_inference() -> int:
            return run_openwebtext_greedy_inference(
                model=ctx.tp.get_model(edge.tgt_id),
                past_key_values=past_by_node_id[edge.tgt_id],
                seed_token=seed_token,
                max_new_tokens=generation_steps,
            )

        _, translated_profile = profiler.measure(run_translated_inference, tokens=profile_tokens)
        with temporarily_offload_module(translator_pool, train_config.device):
            _, native_profile = profiler.measure(run_native_inference, tokens=profile_tokens)
        return (
            {"translated": translated_loss, "native": native_loss},
            {"translated": translated_profile, "native": native_profile},
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


@torch.inference_mode()
def evaluate_generation_dataset(
    ctx: Context,
    spec: HFDatasetSpec,
    dataloader: DataLoader,
    eval_config: EvalConfig,
    translator_pool,
) -> Dict[str, Dict[str, float]]:
    train_config = ctx.config
    edges = ctx.edges
    device = train_config.device
    path_metrics = {edge.id: GenerationRunningAverage() for edge in edges}

    processed_examples = 0

    for batch_idx, batch in enumerate(dataloader, start=1):
        for example in batch:
            question = example["question"]
            context_text = example["context"]
            gold_answers = example["answers"]

            for edge in edges:
                tokenizer = ctx.tp.get_tokenizer(edge.tgt_id)
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
                context_token_ids = prepared_inputs["context_token_ids"]
                prompt_token_ids = prepared_inputs["prompt_token_ids"]
                seed_token = prepared_inputs["seed_token"]

                if prepared_inputs.get("was_truncated") and processed_examples < 3:
                    prompt_tokens = 0 if prompt_token_ids is None else prompt_token_ids.shape[1]
                    logging.info(
                        "[%s][%s] truncated context to %d tokens to fit model context window (prompt_tokens=%d, answer_token_budget=%d)",
                        spec.name_for_log,
                        edge.id,
                        context_token_ids.shape[1],
                        prompt_tokens,
                        get_answer_token_budget(eval_config),
                    )

                past_by_node_id = {
                    node.id: extract_past_key_values(ctx.tp.get_model(node.id), context_token_ids)
                    for node in ctx.nodes
                }
                translated_past = _build_interlat_target_past(
                    ctx=ctx,
                    edge=edge,
                    context_token_ids=context_token_ids,
                    translator_pool=translator_pool,
                )
                native_past = past_by_node_id[edge.tgt_id]
                cosine_value = _cosine_similarity_for_interlat_past(translated_past, native_past)

                translated_answer = predict_generation_task_answer(
                    model=ctx.tp.get_model(edge.tgt_id),
                    tokenizer=tokenizer,
                    past_key_values=translated_past,
                    seed_token=seed_token,
                    eval_config=eval_config,
                    prompt_token_ids=prompt_token_ids,
                )
                native_answer = predict_generation_task_answer(
                    model=ctx.tp.get_model(edge.tgt_id),
                    tokenizer=tokenizer,
                    past_key_values=native_past,
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
    logging.info("Starting InterLat evaluation")
    logging.info("checkpoint_dir_path=%s", checkpoint_dir_path)
    logging.info("eval_config=%s", asdict(eval_config))

    translator_pool.eval()
    for node in nodes:
        ctx.tp.get_model(node.id).eval()

    logging.info("restored_train_config=%s", asdict(train_config))
    logging.info("nodes=%s", [asdict(node) for node in nodes])
    logging.info("edges=%s", [edge.id for edge in edges])
    logging.info(
        "translation_mode=interlat_latent_prefix_from_source_hidden_states"
    )
    logging.info("qa_eval_log_path=%s", log_path)

    all_logit_results = {}
    all_generation_results = {}

    logging.info("Preparing validation dataloader for OpenWebText/validation")

    def build_visualization_pasts_fn(
        *,
        edge: Edge,
        context_token_ids: TokenIDs,
        past_by_node_id,
        **_,
    ) -> Dict[str, PastKeyValues]:
        translated_past = _build_interlat_target_past(
            ctx=ctx,
            edge=edge,
            context_token_ids=context_token_ids,
            translator_pool=translator_pool,
        )
        return build_openwebtext_tsne_named_pasts(
            source_top_past_key_values=past_by_node_id[edge.src_id],
            translated_past_key_values=translated_past,
            target_top_past_key_values=past_by_node_id[edge.tgt_id],
        )

    openwebtext_loss_results = evaluate_openwebtext_validation_loss_interlat(
        ctx=ctx,
        eval_config=eval_config,
        translator_pool=translator_pool,
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
    logging.info("Saved eval log to %s", log_path)
    return log_path
