from dataclasses import asdict
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset

from eval_util import *
from c2c.train import (
    get_top_layers_to_translate,
    get_translation_loss_name,
    get_translation_mode_name,
    load_translator_pool_from_checkpoint,
    translate_top_layers,
)


MMLU_CHOICE_LABELS = ["A", "B", "C", "D"]


def get_c2c_logit_qa_dataset_specs() -> List[HFDatasetSpec]:
    specs = list(get_default_logit_qa_dataset_specs())
    specs.append(
        HFDatasetSpec(
            name_for_log="MMLU/validation",
            dataset_path="cais/mmlu",
            dataset_name="all",
            split="validation",
            answer_mode="mmlu",
            question_field="question",
            subject_field="subject",
            streaming=False,
        )
    )
    return specs


def extract_mmlu_question_and_answer(spec: HFDatasetSpec, example: Dict) -> Optional[Dict[str, Any]]:
    question = example.get(spec.question_field, "")
    if not isinstance(question, str) or not question.strip():
        return None

    choices = example.get("choices", None)
    if not isinstance(choices, list) or len(choices) < 2:
        return None
    normalized_choices = []
    for choice in choices:
        if not isinstance(choice, str) or not choice.strip():
            return None
        normalized_choices.append(choice.strip())

    answer_value = example.get("answer", None)
    if not isinstance(answer_value, int) or not (0 <= answer_value < len(normalized_choices)):
        return None

    subject = None
    if spec.subject_field:
        raw_subject = example.get(spec.subject_field, None)
        if isinstance(raw_subject, str) and raw_subject.strip():
            subject = raw_subject.strip()

    return {
        "question": question.strip(),
        "choices": normalized_choices,
        "subject": subject,
        "answer": MMLU_CHOICE_LABELS[answer_value],
    }


class C2CLogitExampleStream(IterableDataset):
    def __init__(
        self,
        spec: HFDatasetSpec,
        max_examples: int,
        shuffle: bool,
        seed: int,
        shuffle_buffer: int,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.max_examples = max_examples
        self.shuffle = shuffle
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer

    def _load_dataset(self):
        if self.spec.dataset_name is None:
            return load_dataset(
                self.spec.dataset_path,
                split=self.spec.split,
                streaming=self.spec.streaming,
            )
        return load_dataset(
            self.spec.dataset_path,
            self.spec.dataset_name,
            split=self.spec.split,
            streaming=self.spec.streaming,
        )

    def __iter__(self):
        dataset = self._load_dataset()
        if self.shuffle:
            if self.spec.streaming:
                dataset = dataset.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)
            else:
                dataset = dataset.shuffle(seed=self.seed)

        emitted = 0
        for example in dataset:
            if self.spec.answer_mode == "mmlu":
                extracted = extract_mmlu_question_and_answer(self.spec, example)
            else:
                extracted = extract_question_and_answer(self.spec, example)

            if extracted is None:
                continue

            yield extracted
            emitted += 1
            if emitted >= self.max_examples:
                return


def build_c2c_eval_dataloader(
    spec: HFDatasetSpec,
    eval_config: EvalConfig,
) -> DataLoader:
    dataset = C2CLogitExampleStream(
        spec=spec,
        max_examples=eval_config.max_examples_per_dataset,
        shuffle=eval_config.shuffle_eval_stream,
        seed=eval_config.seed,
        shuffle_buffer=eval_config.shuffle_buffer,
    )
    return DataLoader(
        dataset,
        batch_size=eval_config.batch_size,
        num_workers=eval_config.num_workers,
        collate_fn=lambda batch: batch,
    )


def prepare_c2c_logit_task_inputs(
    spec: HFDatasetSpec,
    tokenizer,
    context: Optional[str],
    question: str,
    device: str,
    choices: Optional[List[str]] = None,
    subject: Optional[str] = None,
) -> Dict[str, Any]:
    if spec.answer_mode != "mmlu":
        return prepare_logit_task_inputs(
            spec=spec,
            tokenizer=tokenizer,
            context=context,
            question=question,
            device=device,
        )

    prefix = prepare_question_prefix(
        tokenizer=tokenizer,
        question=question,
        choices=choices,
        subject=subject,
        device=device,
        answer_mode=spec.answer_mode,
    )
    return {
        "cache_input_ids": prefix["cache_ids"],
        "question_cache_ids": None,
        "seed_token": prefix["seed_token"],
        "was_truncated": False,
    }


def build_c2c_logit_answer_candidates(
    tokenizer,
    spec: HFDatasetSpec,
) -> Dict[str, torch.Tensor]:
    if spec.answer_mode != "mmlu":
        return build_logit_answer_candidates(
            tokenizer=tokenizer,
            spec=spec,
        )

    return build_text_candidate_token_ids(
        tokenizer,
        {label: label for label in MMLU_CHOICE_LABELS},
    )


@torch.inference_mode()
def evaluate_dataset(
    spec: HFDatasetSpec,
    dataloader: DataLoader,
    tokenizer,
    train_config,
    eval_config: EvalConfig,
    translator_pool,
    model_specs,
    models,
    nodes,
    edges,
    logger: logging.Logger,
) -> Dict[str, Dict[str, float]]:
    device = train_config.device
    path_metrics = {edge.id: RunningAverage() for edge in edges}

    processed_examples = 0

    for batch_idx, batch in enumerate(dataloader, start=1):
        for example in batch:
            question = example["question"]
            gold_answer = example["answer"]
            context_text = example.get("context")
            choices = example.get("choices")
            subject = example.get("subject")

            prepared_inputs = prepare_c2c_logit_task_inputs(
                spec=spec,
                tokenizer=tokenizer,
                context=context_text,
                question=question,
                device=device,
                choices=choices,
                subject=subject,
            )
            cache_input_ids = prepared_inputs["cache_input_ids"]
            question_cache_ids = prepared_inputs["question_cache_ids"]
            seed_token = prepared_inputs["seed_token"]

            candidate_token_ids = build_c2c_logit_answer_candidates(
                tokenizer=tokenizer,
                spec=spec,
            )

            past_by_node_id = {
                node.id: extract_past_key_values(models[node.id], cache_input_ids)
                for node in nodes
            }

            for edge in edges:

                translated_top_past = translate_top_layers(
                    translator_pool=translator_pool,
                    train_config=train_config,
                    sharer_past_key_values=past_by_node_id[edge.src_id],
                    receiver_past_key_values=past_by_node_id[edge.dst_id],
                    src_name=edge.src_id,
                    dst_name=edge.dst_id,
                    dst_spec=model_specs[edge.dst_id],
                )

                target_top = slice_top_layers(
                    past_key_values=past_by_node_id[edge.dst_id],
                    top_layers_to_translate=get_top_layers_to_translate(train_config),
                )
                cosine_value = cosine_similarity_between_past(translated_top_past, target_top)

                translated_target_past = replace_top_layers(
                    base_past_key_values=past_by_node_id[edge.dst_id],
                    translated_top_past_key_values=translated_top_past,
                )

                translated_scoring_past = prepare_answer_scoring_past(
                    model=models[edge.dst_id],
                    past_key_values=translated_target_past,
                    question_cache_ids=question_cache_ids,
                )
                native_scoring_past = prepare_answer_scoring_past(
                    model=models[edge.dst_id],
                    past_key_values=past_by_node_id[edge.dst_id],
                    question_cache_ids=question_cache_ids,
                )

                translated_scores = score_answer_choices(
                    model=models[edge.dst_id],
                    past_key_values=translated_scoring_past,
                    seed_token=seed_token,
                    choice_token_ids=candidate_token_ids,
                    normalize_by_length=True,
                )
                native_scores = score_answer_choices(
                    model=models[edge.dst_id],
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
    spec: HFDatasetSpec,
    dataloader: DataLoader,
    tokenizer,
    train_config,
    eval_config: EvalConfig,
    translator_pool,
    model_specs,
    models,
    nodes,
    edges,
    logger: logging.Logger,
) -> Dict[str, Dict[str, float]]:
    device = train_config.device
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
                    tokenizer=tokenizer,
                    spec=spec,
                    question=question,
                    eval_config=eval_config,
                    models=models,
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
                question_cache_tokens = 0 if question_cache_ids is None else int(question_cache_ids.shape[1])
                logger.info(
                    "[%s] truncated context to %d tokens to fit model context window (question_cache_tokens=%d, answer_token_budget=%d)",
                    spec.name_for_log,
                    int(cache_input_ids.shape[1]),
                    question_cache_tokens,
                    get_answer_token_budget(eval_config),
                )

            past_by_node_id = {
                node.id: extract_past_key_values(models[node.id], cache_input_ids)
                for node in nodes
            }

            for edge in edges:

                translated_top_past = translate_top_layers(
                    translator_pool=translator_pool,
                    train_config=train_config,
                    sharer_past_key_values=past_by_node_id[edge.src_id],
                    receiver_past_key_values=past_by_node_id[edge.dst_id],
                    src_name=edge.src_id,
                    dst_name=edge.dst_id,
                    dst_spec=model_specs[edge.dst_id],
                )

                target_top = slice_top_layers(
                    past_key_values=past_by_node_id[edge.dst_id],
                    top_layers_to_translate=get_top_layers_to_translate(train_config),
                )
                cosine_value = cosine_similarity_between_past(translated_top_past, target_top)

                translated_target_past = replace_top_layers(
                    base_past_key_values=past_by_node_id[edge.dst_id],
                    translated_top_past_key_values=translated_top_past,
                )

                translated_answer = predict_generation_task_answer(
                    model=models[edge.dst_id],
                    tokenizer=tokenizer,
                    past_key_values=translated_target_past,
                    seed_token=seed_token,
                    eval_config=eval_config,
                    question_cache_ids=question_cache_ids,
                )
                native_answer = predict_generation_task_answer(
                    model=models[edge.dst_id],
                    tokenizer=tokenizer,
                    past_key_values=past_by_node_id[edge.dst_id],
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


@torch.inference_mode()
def evaluate_openwebtext_validation_loss(
    tokenizer,
    train_config,
    eval_config: EvalConfig,
    translator_pool,
    model_specs: Dict[str, ModelSpec],
    models,
    nodes,
    edges,
    logger: logging.Logger,
) -> Dict[str, Dict[str, float]]:
    profiler = InferenceProfiler(train_config.device)

    def evaluate_edge_losses_fn(
        *,
        edge_id: str,
        edge: Edge,
        prefix_cache_ids: torch.Tensor,
        lm_input_ids: torch.Tensor,
        lm_labels: torch.Tensor,
        past_by_node_id,
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, Optional[float]]]]:
        profile_tokens = int(lm_labels.numel())
        translation_loss_name = get_translation_loss_name(train_config)

        def compute_translated_loss_value() -> float:
            translated_top_past = translate_top_layers(
                translator_pool=translator_pool,
                train_config=train_config,
                sharer_past_key_values=past_by_node_id[edge.src_id],
                receiver_past_key_values=past_by_node_id[edge.dst_id],
                src_name=edge.src_id,
                dst_name=edge.dst_id,
                dst_spec=model_specs[edge.dst_id],
            )
            translated_target_past = replace_top_layers(
                base_past_key_values=past_by_node_id[edge.dst_id],
                translated_top_past_key_values=translated_top_past,
            )
            return float(
                compute_suffix_lm_loss(
                    target_model=models[edge.dst_id],
                    past_key_values=translated_target_past,
                    lm_input_ids=lm_input_ids,
                    lm_labels=lm_labels,
                ).item()
            )

        def compute_native_loss_value() -> float:
            return float(
                compute_suffix_lm_loss(
                    target_model=models[edge.dst_id],
                    past_key_values=past_by_node_id[edge.dst_id],
                    lm_input_ids=lm_input_ids,
                    lm_labels=lm_labels,
                ).item()
            )

        translated_loss, translated_profile = profiler.measure(
            compute_translated_loss_value,
            tokens=profile_tokens,
        )
        native_loss, native_profile = profiler.measure(
            compute_native_loss_value,
            tokens=profile_tokens,
        )
        return (
            {
                translation_loss_name: translated_loss,
                "native": native_loss,
            },
            {
                translation_loss_name: translated_profile,
                "native": native_profile,
            },
        )

    return evaluate_openwebtext_validation_loss_metrics(
        tokenizer=tokenizer,
        config=train_config,
        batch_size=eval_config.batch_size,
        num_workers=eval_config.num_workers,
        shuffle=eval_config.shuffle_eval_stream,
        seed=eval_config.seed,
        shuffle_buffer=eval_config.shuffle_buffer,
        max_examples=eval_config.max_examples_per_dataset,
        models=models,
        nodes=nodes,
        edges=edges,
        logger=logger,
        evaluate_direction_losses_fn=evaluate_edge_losses_fn,
        summarize_direction_fn=lambda average_losses, count, profile_summaries: summarize_openwebtext_named_losses(
            average_losses,
            count,
            primary_name=get_translation_loss_name(train_config),
            loss_field_by_name={
                "native": "native_loss",
            },
            profile_summary_by_name=profile_summaries,
            profile_field_prefix_by_name={
                "native": "native",
            },
        ),
    )



def run_eval(eval_config: EvalConfig) -> Path:
    if eval_config.checkpoint_path is None:
        raise ValueError("EvalConfig.checkpoint_path must be set before run_eval.")
    if eval_config.output_path is None:
        raise ValueError("EvalConfig.output_path must be initialized before run_eval.")

    set_seed(eval_config.seed)

    checkpoint_path = Path(eval_config.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config_path = get_eval_config_path(eval_config.output_path)
    write_json(str(config_path), asdict(eval_config))

    log_path = get_eval_log_path(eval_config.output_path)
    logger = setup_logger(f"{eval_config.alg}_eval", log_path)
    logger.info("Starting evaluation")
    logger.info("checkpoint_path=%s", checkpoint_path)
    logger.info("eval_config=%s", asdict(eval_config))

    (
        train_config,
        translator_pool,
        model_specs,
        models,
        tokenizer,
        nodes,
        edges,
    ) = load_translator_pool_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        device_override=eval_config.device,
    )


    translator_pool.eval()
    for model in models.values():
        model.eval()

    logger.info("restored_train_config=%s", asdict(train_config))
    logger.info("nodes=%s", [asdict(node) for node in nodes])
    logger.info("top_layers_to_translate=%d", get_top_layers_to_translate(train_config))
    logger.info("edges=%s", [edge.id for edge in edges])
    logger.info("translation_mode=%s", get_translation_mode_name(train_config))
    logger.info("qa_eval_log_path=%s", log_path)

    all_logit_results = {}
    all_generation_results = {}

    logger.info("Preparing validation dataloader for OpenWebText/validation")
    openwebtext_loss_results = evaluate_openwebtext_validation_loss(
        tokenizer=tokenizer,
        train_config=train_config,
        eval_config=eval_config,
        translator_pool=translator_pool,
        model_specs=model_specs,
        models=models,
        nodes=nodes,
        edges=edges,
        logger=logger,
    )
    for edge in edges:
        row = openwebtext_loss_results[edge.id]
        translation_loss_name = get_translation_loss_name(train_config)
        logger.info(
            "[OpenWebText/validation] %s | native_loss=%.6f | native_profile=%s | %s_loss=%.6f | %s_profile=%s | count=%d",
            edge.id,
            row["native_loss"],
            build_openwebtext_profile_cell(row, prefix="native"),
            translation_loss_name,
            row["loss"],
            translation_loss_name,
            build_openwebtext_profile_cell(row),
            int(row["count"]),
        )

    logit_dataset_specs = get_c2c_logit_qa_dataset_specs()
    for spec in logit_dataset_specs:
        logger.info("Preparing dataloader for %s", spec.name_for_log)
        dataloader = build_c2c_eval_dataloader(
            spec=spec,
            eval_config=eval_config,
        )

        results = evaluate_dataset(
            spec=spec,
            dataloader=dataloader,
            tokenizer=tokenizer,
            train_config=train_config,
            eval_config=eval_config,
            translator_pool=translator_pool,
            model_specs=model_specs,
            models=models,
            nodes=nodes,
            edges=edges,
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
            spec=spec,
            dataloader=dataloader,
            tokenizer=tokenizer,
            train_config=train_config,
            eval_config=eval_config,
            translator_pool=translator_pool,
            model_specs=model_specs,
            models=models,
            nodes=nodes,
            edges=edges,
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
