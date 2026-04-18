import importlib
import time
from typing import Callable, Tuple

import numpy as np

from core.common import *
from core.config import Config
from core.context import Context
from core.train_util import get_train_config_path


@dataclass
class EvalConfig(Config):
    outputs_path: str
    checkpoint_dir_path: Optional[str]

    # evaluation sampling
    batch_size: int
    num_workers: int
    max_examples_per_dataset: int
    seed: int

    # streaming / shuffling
    shuffle_eval_stream: bool
    shuffle_buffer: int

    # generation QA
    generation_max_new_tokens: int


    def __post_init__(self) -> None:
        super().__post_init__()
        initialize_eval_output_paths(self)


OPENWEBTEXT_TSNE_LABEL_ORDER = (
    "source_top",
    "translated",
    "target_top",
)
OPENWEBTEXT_TSNE_DISPLAY_NAMES = {
    "source_top": "Source Top KV",
    "translated": "Translated KV",
    "target_top": "Target Top KV",
}
OPENWEBTEXT_TSNE_FILE_BASENAME = "openwebtext_validation_tsne"


@dataclass
class InferenceProfileAccumulator:
    total_latency_sec: float = 0.0
    total_tokens: int = 0
    num_calls: int = 0
    peak_memory_bytes: Optional[int] = None

    def update(
        self,
        *,
        latency_sec: float,
        tokens: int,
        peak_memory_bytes: Optional[int],
    ) -> None:
        self.total_latency_sec += float(latency_sec)
        self.total_tokens += tokens
        self.num_calls += 1
        if peak_memory_bytes is not None:
            peak_value = peak_memory_bytes
            if self.peak_memory_bytes is None or peak_value > self.peak_memory_bytes:
                self.peak_memory_bytes = peak_value

    def summary(self) -> Dict[str, float]:
        if self.num_calls <= 0:
            avg_latency_ms = float("nan")
        else:
            avg_latency_ms = (self.total_latency_sec / self.num_calls) * 1000.0

        if self.total_latency_sec > 0.0 and self.total_tokens > 0:
            throughput_tokens_per_sec = self.total_tokens / self.total_latency_sec
        else:
            throughput_tokens_per_sec = float("nan")

        if self.peak_memory_bytes is None:
            peak_memory_gib = float("nan")
        else:
            peak_memory_gib = float(self.peak_memory_bytes) / (1024 ** 3)

        return {
            "avg_latency_ms": avg_latency_ms,
            "throughput_tokens_per_sec": throughput_tokens_per_sec,
            "peak_memory_gib": peak_memory_gib,
        }


class InferenceProfiler:
    def __init__(self, device: str, *, sample_interval_sec: float = 0.005) -> None:
        self.device = device
        self.reader = CurrentProcessGPUMemoryReader(device)
        self.enabled = self.reader.enabled
        self.sample_interval_sec = max(float(sample_interval_sec), 0.001)
        if self.enabled:
            device_index = torch.device(self.device).index
            self.device_index = torch.cuda.current_device() if device_index is None else device_index
        else:
            self.device_index = None

    def _measure_peak_allocated_bytes(self, fn: Callable[[], T]) -> Tuple[T, Optional[int]]:
        if not self.enabled:
            return fn(), None

        peak_memory_bytes = 0
        stop_event = threading.Event()
        peak_lock = threading.Lock()

        def sample_memory() -> None:
            nonlocal peak_memory_bytes
            while not stop_event.is_set():
                allocated = self.reader.read_allocated_bytes()
                if allocated is not None:
                    with peak_lock:
                        peak_memory_bytes = max(peak_memory_bytes, int(allocated))
                stop_event.wait(self.sample_interval_sec)

        sampler = threading.Thread(
            target=sample_memory,
            name="inference-memory-profiler",
            daemon=True,
        )

        initial_allocated = self.reader.read_allocated_bytes()
        if initial_allocated is not None:
            peak_memory_bytes = max(peak_memory_bytes, int(initial_allocated))

        sampler.start()
        try:
            result = fn()
            torch.cuda.synchronize(self.device_index)
        finally:
            stop_event.set()
            sampler.join(timeout=max(1.0, self.sample_interval_sec * 4.0))

        final_allocated = self.reader.read_allocated_bytes()
        if final_allocated is not None:
            peak_memory_bytes = max(peak_memory_bytes, int(final_allocated))

        return result, peak_memory_bytes

    def measure(self, fn: Callable[[], T], *, tokens: int) -> Tuple[T, Dict[str, Optional[float]]]:
        if self.enabled:
            torch.cuda.synchronize(self.device_index)

        started_at = time.perf_counter()
        result, peak_memory_bytes = self._measure_peak_allocated_bytes(fn)
        latency_sec = time.perf_counter() - started_at

        return result, {
            "latency_sec": float(latency_sec),
            "tokens": tokens,
            "peak_memory_bytes": peak_memory_bytes,
        }


@dataclass
class HFDatasetSpec:
    name_for_log: str
    dataset_path: str
    dataset_name: Optional[str]
    split: str
    answer_mode: str
    question_field: str = "question"
    context_field: Optional[str] = None
    answers_field: Optional[str] = None
    subject_field: Optional[str] = None
    streaming: bool = False


class HFQAPairStream(IterableDataset):
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
            qa_pair = extract_question_and_answer(self.spec, example)
            if qa_pair is None:
                continue

            yield qa_pair
            emitted += 1
            if emitted >= self.max_examples:
                return


DEFAULT_MULTINEWS_SUMMARY_TASK = "Summarize the news articles above."


def get_boolq_dataset_spec() -> HFDatasetSpec:
    return HFDatasetSpec(
        name_for_log="BoolQ/validation",
        dataset_path="google/boolq",
        dataset_name=None,
        split="validation",
        answer_mode="boolq",
        question_field="question",
        context_field="passage",
        streaming=False,
    )


def get_pubmedqa_dataset_spec() -> HFDatasetSpec:
    return HFDatasetSpec(
        name_for_log="PubMedQA/pqa_labeled/train",
        dataset_path="qiaojin/PubMedQA",
        dataset_name="pqa_labeled",
        split="train",
        answer_mode="pubmed_qa",
        question_field="question",
        context_field="context",
        streaming=False,
    )


def get_squad_v11_dataset_spec() -> HFDatasetSpec:
    return HFDatasetSpec(
        name_for_log="SQuAD-v1.1/validation",
        dataset_path="rajpurkar/squad",
        dataset_name=None,
        split="validation",
        answer_mode="squad",
        question_field="question",
        context_field="context",
        answers_field="answers",
        streaming=False,
    )


def get_newsqa_generation_dataset_spec() -> HFDatasetSpec:
    return HFDatasetSpec(
        name_for_log="NewsQA/validation",
        dataset_path="gabrieltorresgamez/newsqa",
        dataset_name=None,
        split="validation",
        answer_mode="newsqa",
        question_field="questions",
        context_field="paragraph",
        answers_field="answers",
        streaming=False,
    )


def get_multinews_generation_dataset_spec() -> HFDatasetSpec:
    return HFDatasetSpec(
        name_for_log="MultiNews/validation",
        dataset_path="Awesome075/multi_news_parquet",
        dataset_name=None,
        split="validation",
        answer_mode="multinews",
        context_field="document",
        answers_field="summary",
        streaming=False,
    )


LOGIT_QA_SPEC_GROUP_FACTORIES = [
    get_boolq_dataset_spec,
    get_pubmedqa_dataset_spec,
]

GEN_QA_SPEC_GROUP_FACTORIES = [
    get_squad_v11_dataset_spec,
    get_newsqa_generation_dataset_spec,
]

EVAL_SPEC_GROUP_FACTORIES = {
    "logit_qa": LOGIT_QA_SPEC_GROUP_FACTORIES,
    "gen_qa": GEN_QA_SPEC_GROUP_FACTORIES,
}


def build_openwebtext_eval_dataloader(
    tokenizer: PreTrainedTokenizerBase,
    config,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: Optional[int] = None,
    shuffle_buffer: Optional[int] = None,
    seed_offset: int = 10_000,
) -> DataLoader:
    dataset = OpenWebTextSequenceStream(
        tokenizer=tokenizer,
        sequence_length=config.total_tokens,
        split="train",
        shuffle=shuffle,
        shuffle_buffer=config.shuffle_buffer if shuffle_buffer is None else shuffle_buffer,
        seed=(config.seed if seed is None else seed) + seed_offset,
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


def select_past_layers_by_indices(
    past_key_values: PastKeyValues,
    layer_indices: List[int],
) -> PastKeyValues:
    if not layer_indices:
        raise ValueError("layer_indices must contain at least one layer index.")

    selected_layers = []
    num_layers = len(past_key_values)
    for layer_idx in layer_indices:
        normalized_idx = int(layer_idx)
        if not (0 <= normalized_idx < num_layers):
            raise ValueError(
                f"layer_idx={normalized_idx} must be in [0, {num_layers - 1}]"
            )
        selected_layers.append(past_key_values[normalized_idx])
    return tuple(selected_layers)


def select_last_layer_past_key_values(
    past_key_values: PastKeyValues,
) -> PastKeyValues:
    if len(past_key_values) < 1:
        raise ValueError("past_key_values must contain at least one layer.")
    return (past_key_values[-1],)


def summarize_past_key_values_for_tsne(
    past_key_values: PastKeyValues,
) -> np.ndarray:
    if len(past_key_values) < 1:
        raise ValueError("past_key_values must contain at least one layer.")

    last_layer_past_key_values = select_last_layer_past_key_values(past_key_values)
    flat_features = flatten_past_key_values(last_layer_past_key_values)
    return flat_features.detach().to(torch.float32).cpu().numpy().astype(np.float32, copy=False)


def build_openwebtext_tsne_named_pasts(
    *,
    source_top_past_key_values: PastKeyValues,
    translated_past_key_values: PastKeyValues,
    target_top_past_key_values: PastKeyValues,
) -> Dict[str, PastKeyValues]:
    return {
        "source_top": source_top_past_key_values,
        "translated": translated_past_key_values,
        "target_top": target_top_past_key_values,
    }

def _load_openwebtext_tsne_plotting_deps():
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ModuleNotFoundError as exc:
        missing_name = exc.name or "optional plotting dependency"
        raise ModuleNotFoundError(
            "OpenWebText t-SNE plotting requires optional dependency "
            f"'{missing_name}'. Install matplotlib and scikit-learn to enable plotting."
        ) from exc
    return plt, TSNE


def _build_openwebtext_tsne_output_dir(output_path: Union[str, Path]) -> Path:
    return Path(output_path) / "tsne" / "openwebtext_validation"


def _sanitize_edge_id_for_filename(edge_id: str) -> str:
    allowed = []
    for char in edge_id:
        if char.isalnum() or char in {"-", "_"}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed)


def _build_openwebtext_tsne_plot_path(output_path: Union[str, Path], edge_id: str) -> Path:
    safe_edge_id = _sanitize_edge_id_for_filename(edge_id)
    return _build_openwebtext_tsne_output_dir(output_path) / f"{OPENWEBTEXT_TSNE_FILE_BASENAME}_{safe_edge_id}.png"


def _accumulate_openwebtext_tsne_samples(
    features_by_edge_and_group: Dict[str, Dict[str, List[np.ndarray]]],
    *,
    edge_id: str,
    named_pasts: Dict[str, PastKeyValues],
) -> None:
    group_store = features_by_edge_and_group[edge_id]
    reference_batch_size = None

    for label in OPENWEBTEXT_TSNE_LABEL_ORDER:
        past_key_values = named_pasts.get(label)
        if past_key_values is None:
            continue

        features = summarize_past_key_values_for_tsne(past_key_values)
        if reference_batch_size is None:
            reference_batch_size = features.shape[0]
        elif reference_batch_size != features.shape[0]:
            raise ValueError(
                "All OpenWebText t-SNE groups must share the same batch size, "
                f"got {reference_batch_size} vs {features.shape[0]} for edge {edge_id}."
            )
        group_store[label].append(features)


def _finalize_openwebtext_tsne_plots(
    *,
    output_path: Union[str, Path],
    seed: int,
    logger: logging.Logger,
    features_by_edge_and_group: Dict[str, Dict[str, List[np.ndarray]]],
    perplexity: float = 50.0,
    max_iter: int = 1000,
) -> Dict[str, str]:
    try:
        plt, TSNE = _load_openwebtext_tsne_plotting_deps()
    except ModuleNotFoundError as exc:
        logger.warning("Skipping OpenWebText t-SNE plots: %s", exc)
        return {}

    output_dir = _build_openwebtext_tsne_output_dir(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    visible_labels = OPENWEBTEXT_TSNE_LABEL_ORDER

    saved_paths: Dict[str, str] = {}
    for edge_id, group_store in features_by_edge_and_group.items():
        ordered_features = []
        ordered_labels = []
        group_counts = {}

        for label in visible_labels:
            feature_batches = group_store.get(label, [])
            if not feature_batches:
                continue
            group_features = np.concatenate(feature_batches, axis=0)
            ordered_features.append(group_features)
            ordered_labels.extend([label] * group_features.shape[0])
            group_counts[label] = int(group_features.shape[0])

        if len(ordered_features) < 2:
            logger.warning(
                "Skipping OpenWebText t-SNE for %s because fewer than two visible groups were collected.",
                edge_id,
            )
            continue

        max_feature_dim = max(features.shape[1] for features in ordered_features)
        padded_features = []
        for features in ordered_features:
            if features.shape[1] < max_feature_dim:
                features = np.pad(
                    features,
                    pad_width=((0, 0), (0, max_feature_dim - features.shape[1])),
                    mode="constant",
                    constant_values=0.0,
                )
            padded_features.append(features)

        feature_matrix = np.concatenate(padded_features, axis=0)
        if feature_matrix.shape[0] < 3:
            logger.warning(
                "Skipping OpenWebText t-SNE for %s because only %d total samples were collected.",
                edge_id,
                feature_matrix.shape[0],
            )
            continue

        feature_mean = feature_matrix.mean(axis=0, keepdims=True)
        feature_std = feature_matrix.std(axis=0, keepdims=True)
        feature_matrix = (feature_matrix - feature_mean) / np.clip(feature_std, 1e-6, None)

        effective_perplexity = float(perplexity)
        if feature_matrix.shape[0] <= effective_perplexity:
            effective_perplexity = float(max(1, feature_matrix.shape[0] - 1))
            logger.warning(
                "Adjusted OpenWebText t-SNE perplexity for %s from %.1f to %.1f because only %d total samples were collected.",
                edge_id,
                float(perplexity),
                effective_perplexity,
                feature_matrix.shape[0],
            )

        embedding = TSNE(
            n_components=2,
            perplexity=effective_perplexity,
            max_iter=max_iter,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        ).fit_transform(feature_matrix)

        fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
        scatter_style = {
            "source_top": {"s": 44, "alpha": 0.62, "zorder": 2},
            "translated": {"s": 68, "alpha": 0.90, "zorder": 4},
            "target_top": {"s": 44, "alpha": 0.62, "zorder": 2},
        }
        for label in visible_labels:
            display_name = OPENWEBTEXT_TSNE_DISPLAY_NAMES[label]
            mask = np.asarray([row_label == label for row_label in ordered_labels], dtype=bool)
            if not np.any(mask):
                continue
            style = scatter_style.get(label, {"s": 36, "alpha": 0.60, "zorder": 1})
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=style["s"],
                alpha=style["alpha"],
                zorder=style["zorder"],
                linewidths=0.0,
                label=display_name,
            )

        ax.set_title(
            f"OpenWebText validation t-SNE ({edge_id})\n"
            f"last-layer raw flattened KV | perplexity={int(round(effective_perplexity))}, max_iter={max_iter}",
            fontsize=14,
        )
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)

        plot_path = _build_openwebtext_tsne_plot_path(output_path, edge_id)
        fig.tight_layout()
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)

        saved_paths[edge_id] = str(plot_path)
        count_summary = ", ".join(
            f"{label}={group_counts.get(label, 0)}"
            for label in visible_labels
            if label in group_counts
        )
        logger.info(
            "Saved OpenWebText t-SNE plot for %s to %s (%s)",
            edge_id,
            plot_path,
            count_summary,
        )

    return saved_paths


def summarize_openwebtext_named_losses(
    average_losses: Dict[str, float],
    count: int,
    *,
    primary_name: str,
    loss_field_by_name: Optional[Dict[str, str]] = None,
    loss_delta_reference_name: Optional[str] = None,
    loss_delta_field_by_name: Optional[Dict[str, str]] = None,
    profile_summary_by_name: Optional[Dict[str, Dict[str, float]]] = None,
    profile_field_prefix_by_name: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    loss_field_by_name = dict(loss_field_by_name or {})
    loss_delta_field_by_name = dict(loss_delta_field_by_name or {})
    profile_summary_by_name = dict(profile_summary_by_name or {})
    profile_field_prefix_by_name = dict(profile_field_prefix_by_name or {})

    metric_names = set(loss_field_by_name)
    metric_names.add(primary_name)
    if loss_delta_reference_name is not None:
        metric_names.add(loss_delta_reference_name)
    metric_names.update(loss_delta_field_by_name)

    summary: Dict[str, float] = {"count": count}

    for name in sorted(metric_names):
        average_loss = float(average_losses.get(name, float("nan"))) if count > 0 else float("nan")
        if name == primary_name:
            summary["loss"] = average_loss

        loss_field = loss_field_by_name.get(name)
        if loss_field is not None:
            summary[loss_field] = average_loss

    for name, profile_summary in profile_summary_by_name.items():
        if name == primary_name:
            prefix = ""
        else:
            prefix = profile_field_prefix_by_name.get(name)
            if prefix is None:
                continue

        field_prefix = f"{prefix}_" if prefix else ""
        summary[f"{field_prefix}latency_ms"] = float(profile_summary.get("avg_latency_ms", float("nan")))
        summary[f"{field_prefix}throughput_tokens_per_sec"] = float(profile_summary.get("throughput_tokens_per_sec", float("nan")))
        summary[f"{field_prefix}peak_memory_gib"] = float(profile_summary.get("peak_memory_gib", float("nan")))

    if loss_delta_reference_name is not None:
        if loss_delta_reference_name == primary_name:
            reference_loss = float(summary.get("loss", float("nan")))
        else:
            reference_loss = float(summary.get(loss_field_by_name.get(loss_delta_reference_name, ""), float("nan")))

        for name, delta_field in loss_delta_field_by_name.items():
            if name == primary_name:
                candidate_loss = float(summary.get("loss", float("nan")))
            else:
                candidate_loss = float(summary.get(loss_field_by_name.get(name, ""), float("nan")))

            if math.isfinite(reference_loss) and math.isfinite(candidate_loss):
                summary[delta_field] = candidate_loss - reference_loss
            else:
                summary[delta_field] = float("nan")

    return summary


@torch.inference_mode()
def run_openwebtext_greedy_inference(
    *,
    model,
    past_key_values: PastKeyValues,
    seed_token: torch.Tensor,
    max_new_tokens: int,
) -> int:
    if max_new_tokens <= 0:
        return 0

    current_input_ids = seed_token
    current_past = past_key_values
    total_generated_tokens = 0

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=current_input_ids,
            past_key_values=current_past,
            use_cache=True,
        )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        total_generated_tokens += int(next_token.numel())
        current_input_ids = next_token
        current_past = outputs.past_key_values

    return total_generated_tokens



@torch.inference_mode()
def evaluate_openwebtext_validation_loss_metrics(
    *,
    ctx: Context,
    output_path: Union[str, Path],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    shuffle_buffer: int,
    max_examples: int,
    logger: logging.Logger,
    evaluate_edge_losses_fn: Callable[..., Tuple[Dict[str, float], Dict[str, Dict[str, Optional[float]]]]],
    build_visualization_pasts_fn: Optional[Callable[..., Dict[str, PastKeyValues]]] = None,
) -> Dict[str, Dict[str, float]]:
    dataloader = build_openwebtext_eval_dataloader(
        tokenizer=ctx.tokenizer,
        config=ctx.config,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        seed=seed,
        shuffle_buffer=shuffle_buffer,
    )
    device = ctx.config.device
    max_examples = max(1, max_examples)

    loss_sums = {edge.id: {} for edge in ctx.edges}
    counts = {edge.id: 0 for edge in ctx.edges}
    profile_accumulators = {edge.id: {} for edge in ctx.edges}
    tsne_features = None
    if build_visualization_pasts_fn is not None:
        tsne_features = {
            edge.id: {label: [] for label in OPENWEBTEXT_TSNE_LABEL_ORDER}
            for edge in ctx.edges
        }

    processed_examples = 0
    for batch_idx, input_ids in enumerate(dataloader, start=1):
        if processed_examples >= max_examples:
            break

        remaining_examples = max_examples - processed_examples
        if input_ids.shape[0] > remaining_examples:
            input_ids = input_ids[:remaining_examples]
        input_ids = input_ids.to(device)

        prefix_cache_ids, lm_input_ids, lm_labels = split_prefix_and_suffix_for_exact_next_token_loss(
            input_ids=input_ids,
            prefix_tokens=ctx.config.prefix_tokens,
        )
        past_by_node_id = {
            node.id: extract_past_key_values(ctx.mm.get_model(node.id), prefix_cache_ids)
            for node in ctx.nodes
        }

        batch_examples = input_ids.shape[0]
        for edge in ctx.edges:
            edge_losses, edge_profiles = evaluate_edge_losses_fn(
                edge_id=edge.id,
                edge=edge,
                prefix_cache_ids=prefix_cache_ids,
                lm_input_ids=lm_input_ids,
                lm_labels=lm_labels,
                past_by_node_id=past_by_node_id,
            )
            if not edge_losses:
                continue
            for metric_name, loss_value in edge_losses.items():
                loss_sums[edge.id][metric_name] = (
                    float(loss_sums[edge.id].get(metric_name, 0.0))
                    + float(loss_value) * batch_examples
                )
            for metric_name, profile_values in edge_profiles.items():
                accumulator = profile_accumulators[edge.id].setdefault(
                    metric_name,
                    InferenceProfileAccumulator(),
                )
                accumulator.update(
                    latency_sec=float(profile_values.get("latency_sec", 0.0)),
                    tokens=profile_values.get("tokens", 0),
                    peak_memory_bytes=profile_values.get("peak_memory_bytes"),
                )
            counts[edge.id] += batch_examples

            if tsne_features is not None:
                named_pasts = build_visualization_pasts_fn(
                    edge_id=edge.id,
                    edge=edge,
                    prefix_cache_ids=prefix_cache_ids,
                    lm_input_ids=lm_input_ids,
                    lm_labels=lm_labels,
                    past_by_node_id=past_by_node_id,
                )
                if named_pasts:
                    _accumulate_openwebtext_tsne_samples(
                        tsne_features,
                        edge_id=edge.id,
                        named_pasts=named_pasts,
                    )

        processed_examples += batch_examples
        if batch_idx % 25 == 0:
            logger.info(
                "[OpenWebText/validation] progress: %d/%d sequences",
                processed_examples,
                max_examples,
            )

    summaries = {}
    for edge in ctx.edges:
        count = counts[edge.id]
        if count > 0:
            average_losses = {
                metric_name: float(total_loss / count)
                for metric_name, total_loss in loss_sums[edge.id].items()
            }
        else:
            average_losses = {}
        profile_summaries = {
            metric_name: accumulator.summary()
            for metric_name, accumulator in profile_accumulators[edge.id].items()
        }
        summaries[edge.id] = summarize_openwebtext_named_losses(
            average_losses,
            count,
            primary_name="translated",
            loss_field_by_name={
                "native": "native_loss",
            },
            profile_summary_by_name=profile_summaries,
            profile_field_prefix_by_name={
                "native": "native",
            },
        )

    if tsne_features is not None:
        tsne_paths = _finalize_openwebtext_tsne_plots(
            output_path=output_path,
            seed=seed,
            logger=logger,
            features_by_edge_and_group=tsne_features,
            perplexity=50.0,
            max_iter=1000,
        )
        for edge in ctx.edges:
            if edge.id in tsne_paths:
                summaries[edge.id]["tsne_plot_path"] = tsne_paths[edge.id]

    return summaries


@torch.inference_mode()
def evaluate_openwebtext_validation_loss_top_layers(
    ctx: Context,
    eval_config: EvalConfig,
    translator_pool,
    logger: logging.Logger,
    *,
    build_translated_target_past_fn: Callable[..., PastKeyValues],
    build_visualization_pasts_fn: Optional[Callable[..., Dict[str, PastKeyValues]]] = None,
) -> Dict[str, Dict[str, float]]:
    train_config = ctx.config
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
        seed_token = lm_input_ids[:, :1]
        generation_steps = int(lm_labels.shape[1])

        translated_target_past_for_loss = build_translated_target_past_fn(
            edge=edge,
            past_by_node_id=past_by_node_id,
        )
        translated_loss = float(
            compute_suffix_lm_loss(
                target_model=ctx.mm.get_model(edge.tgt_id),
                past_key_values=translated_target_past_for_loss,
                lm_input_ids=lm_input_ids,
                lm_labels=lm_labels,
            ).item()
        )
        native_loss = float(
            compute_suffix_lm_loss(
                target_model=ctx.mm.get_model(edge.tgt_id),
                past_key_values=past_by_node_id[edge.tgt_id],
                lm_input_ids=lm_input_ids,
                lm_labels=lm_labels,
            ).item()
        )

        def run_translated_inference() -> int:
            translated_target_past = build_translated_target_past_fn(
                edge=edge,
                past_by_node_id=past_by_node_id,
            )
            return run_openwebtext_greedy_inference(
                model=ctx.mm.get_model(edge.tgt_id),
                past_key_values=translated_target_past,
                seed_token=seed_token,
                max_new_tokens=generation_steps,
            )

        def run_native_inference() -> int:
            return run_openwebtext_greedy_inference(
                model=ctx.mm.get_model(edge.tgt_id),
                past_key_values=past_by_node_id[edge.tgt_id],
                seed_token=seed_token,
                max_new_tokens=generation_steps,
            )

        _, translated_profile = profiler.measure(
            run_translated_inference,
            tokens=profile_tokens,
        )
        _, native_profile = profiler.measure(
            run_native_inference,
            tokens=profile_tokens,
        )
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
        logger=logger,
        evaluate_edge_losses_fn=evaluate_edge_losses_fn,
        build_visualization_pasts_fn=build_visualization_pasts_fn,
    )


@torch.inference_mode()
def evaluate_openwebtext_validation_loss_replay(
    ctx: Context,
    eval_config: EvalConfig,
    translator_pool,
    logger: logging.Logger,
    *,
    build_translated_target_past_fn: Callable[..., PastKeyValues],
    build_visualization_pasts_fn: Optional[Callable[..., Dict[str, PastKeyValues]]] = None,
) -> Dict[str, Dict[str, float]]:
    train_config = ctx.config
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
        seed_token = lm_input_ids[:, :1]
        generation_steps = int(lm_labels.shape[1])

        mixed_target_past_for_loss = build_translated_target_past_fn(
            edge=edge,
            prefix_cache_ids=prefix_cache_ids,
            past_by_node_id=past_by_node_id,
        )
        translated_loss = float(
            compute_prefix_correction_and_suffix_lm_loss(
                target_model=ctx.mm.get_model(edge.tgt_id),
                past_key_values=mixed_target_past_for_loss,
                lm_input_ids=lm_input_ids,
                lm_labels=lm_labels,
                native_target_past_key_values=past_by_node_id[edge.tgt_id],
                target_layer_indices=ctx.cm.get_tgt_layer_indices(edge.id),
            ).item()
        )
        native_loss = float(
            compute_prefix_correction_and_suffix_lm_loss(
                target_model=ctx.mm.get_model(edge.tgt_id),
                past_key_values=past_by_node_id[edge.tgt_id],
                lm_input_ids=lm_input_ids,
                lm_labels=lm_labels,
                native_target_past_key_values=past_by_node_id[edge.tgt_id],
                target_layer_indices=ctx.cm.get_tgt_layer_indices(edge.id),
            ).item()
        )

        def run_translated_inference() -> int:
            mixed_target_past = build_translated_target_past_fn(
                edge=edge,
                prefix_cache_ids=prefix_cache_ids,
                past_by_node_id=past_by_node_id,
            )
            return run_openwebtext_greedy_inference(
                model=ctx.mm.get_model(edge.tgt_id),
                past_key_values=mixed_target_past,
                seed_token=seed_token,
                max_new_tokens=generation_steps,
            )

        def run_native_inference() -> int:
            return run_openwebtext_greedy_inference(
                model=ctx.mm.get_model(edge.tgt_id),
                past_key_values=past_by_node_id[edge.tgt_id],
                seed_token=seed_token,
                max_new_tokens=generation_steps,
            )

        _, translated_profile = profiler.measure(
            run_translated_inference,
            tokens=profile_tokens,
        )
        _, native_profile = profiler.measure(
            run_native_inference,
            tokens=profile_tokens,
        )
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
        logger=logger,
        evaluate_edge_losses_fn=evaluate_edge_losses_fn,
        build_visualization_pasts_fn=build_visualization_pasts_fn,
    )


@torch.inference_mode()
def evaluate_openwebtext_validation_loss(
    ctx: Context,
    eval_config: EvalConfig,
    translator_pool,
    logger: logging.Logger,
    *,
    build_translated_target_past_fn: Optional[Callable[..., PastKeyValues]] = None,
    build_visualization_pasts_fn: Optional[Callable[..., Dict[str, PastKeyValues]]] = None,
) -> Dict[str, Dict[str, float]]:
    if eval_config.alg in {"mot", "mot-h"}:
        return evaluate_openwebtext_validation_loss_replay(
            ctx=ctx,
            eval_config=eval_config,
            translator_pool=translator_pool,
            logger=logger,
            build_translated_target_past_fn=build_translated_target_past_fn,
            build_visualization_pasts_fn=build_visualization_pasts_fn,
        )
    return evaluate_openwebtext_validation_loss_top_layers(
        ctx=ctx,
        eval_config=eval_config,
        translator_pool=translator_pool,
        logger=logger,
        build_translated_target_past_fn=build_translated_target_past_fn,
        build_visualization_pasts_fn=build_visualization_pasts_fn,
    )


def get_eval_spec_group(group_name: str) -> List[HFDatasetSpec]:
    try:
        factories = EVAL_SPEC_GROUP_FACTORIES[group_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported eval spec group: {group_name}") from exc
    return [factory() for factory in factories]


def get_default_logit_qa_dataset_specs() -> List[HFDatasetSpec]:
    return get_eval_spec_group("logit_qa")


def get_default_gen_qa_dataset_specs() -> List[HFDatasetSpec]:
    return get_eval_spec_group("gen_qa")


def _normalize_answer_texts(raw_value: Any) -> List[str]:
    if isinstance(raw_value, str):
        text = raw_value.strip()
        return [text] if text else []

    if isinstance(raw_value, list):
        return [
            item.strip()
            for item in raw_value
            if isinstance(item, str) and item.strip()
        ]

    if isinstance(raw_value, dict):
        raw_texts = raw_value.get("text", [])
        if isinstance(raw_texts, list):
            return [
                item.strip()
                for item in raw_texts
                if isinstance(item, str) and item.strip()
            ]

    return []


def normalize_multinews_context_text(raw_value: Any) -> Optional[str]:
    context = normalize_context_text(raw_value)
    if context is None:
        return None
    normalized = context.replace(" ||||| ", "\n\n").replace("|||||", "\n\n").strip()
    return normalized or None


def extract_generation_examples(spec: HFDatasetSpec, example: Dict[str, Any]) -> List[Dict[str, Any]]:
    if spec.answer_mode == "squad":
        question = example.get(spec.question_field, "")
        if not isinstance(question, str) or not question.strip():
            return []

        context_field = spec.context_field or "context"
        answers_field = spec.answers_field or "answers"

        context = example.get(context_field, "")
        if not isinstance(context, str) or not context.strip():
            return []

        answer_texts = _normalize_answer_texts(example.get(answers_field, None))
        if not answer_texts:
            return []

        return [{
            "question": question.strip(),
            "context": context.strip(),
            "answers": answer_texts,
        }]

    if spec.answer_mode == "newsqa":
        context_field = spec.context_field or "paragraph"
        question_field = spec.question_field or "questions"
        answers_field = spec.answers_field or "answers"

        context = normalize_context_text(example.get(context_field, None))
        if context is None:
            return []

        raw_questions = example.get(question_field, None)
        raw_answers = example.get(answers_field, None)
        if not isinstance(raw_questions, list) or not isinstance(raw_answers, list):
            return []

        generation_examples: List[Dict[str, Any]] = []
        for raw_question, raw_answer in zip(raw_questions, raw_answers):
            if not isinstance(raw_question, str) or not raw_question.strip():
                continue

            answer_texts = _normalize_answer_texts(raw_answer)
            if not answer_texts and isinstance(raw_answer, str) and raw_answer.strip():
                answer_texts = [raw_answer.strip()]
            if not answer_texts:
                continue

            generation_examples.append({
                "question": raw_question.strip(),
                "context": context,
                "answers": answer_texts,
            })
        return generation_examples


    # if spec.answer_mode == "multinews":
    #     question_value = example.get(spec.question_field, None)
    #     if isinstance(question_value, str) and question_value.strip():
    #         question = question_value.strip()
    #     else:
    #         question = DEFAULT_MULTINEWS_SUMMARY_TASK
    #
    #     context_field = spec.context_field or "document"
    #     answers_field = spec.answers_field or "summary"
    #
    #     context = normalize_multinews_context_text(example.get(context_field, None))
    #     if context is None:
    #         return []
    #
    #     answer_texts = _normalize_answer_texts(example.get(answers_field, None))
    #     if not answer_texts:
    #         return []
    #
    #     return [{
    #         "question": question,
    #         "context": context,
    #         "answers": answer_texts,
    #     }]

    raise ValueError(f"Unsupported generation answer_mode: {spec.answer_mode}")


class HFGenerationExampleStream(IterableDataset):
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
            generation_examples = extract_generation_examples(self.spec, example)
            if not generation_examples:
                continue

            for generation_example in generation_examples:
                yield generation_example
                emitted += 1
                if emitted >= self.max_examples:
                    return


def build_generation_eval_dataloader(
    spec: HFDatasetSpec,
    eval_config: EvalConfig,
) -> DataLoader:
    dataset = HFGenerationExampleStream(
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



class RunningAverage:
    def __init__(self) -> None:
        self.cosine_sum = 0.0
        self.accuracy_sum = 0.0
        self.native_accuracy_sum = 0.0
        self.count = 0

    def update(self, cosine_value: float, accuracy_value: float, native_accuracy_value: float, n: int) -> None:
        self.cosine_sum += float(cosine_value) * n
        self.accuracy_sum += float(accuracy_value) * n
        self.native_accuracy_sum += float(native_accuracy_value) * n
        self.count += n

    def summary(self) -> Dict[str, float]:
        if self.count == 0:
            return {
                "cosine": float("nan"),
                "accuracy": float("nan"),
                "native_accuracy": float("nan"),
                "count": 0,
            }
        return {
            "cosine": self.cosine_sum / self.count,
            "accuracy": self.accuracy_sum / self.count,
            "native_accuracy": self.native_accuracy_sum / self.count,
            "count": self.count,
        }


class GenerationRunningAverage:
    def __init__(self) -> None:
        self.cosine_sum = 0.0
        self.f1_sum = 0.0
        self.native_f1_sum = 0.0
        self.count = 0

    def update(
        self,
        cosine_value: float,
        f1_value: float,
        native_f1_value: float,
        n: int,
    ) -> None:
        self.cosine_sum += float(cosine_value) * n
        self.f1_sum += float(f1_value) * n
        self.native_f1_sum += float(native_f1_value) * n
        self.count += n

    def summary(self) -> Dict[str, float]:
        if self.count == 0:
            return {
                "cosine": float("nan"),
                "f1": float("nan"),
                "native_f1": float("nan"),
                "count": 0,
            }
        return {
            "cosine": self.cosine_sum / self.count,
            "f1": self.f1_sum / self.count,
            "native_f1": self.native_f1_sum / self.count,
            "count": self.count,
        }


def get_eval_config_path(output_path: Union[str, Path]) -> Path:
    return Path(output_path) / "eval_config.json"


def get_eval_log_path(output_path: Union[str, Path]) -> Path:
    return Path(output_path) / "eval.log"


def initialize_eval_output_paths(config: EvalConfig) -> None:
    output_path = config.output_path
    checkpoint_dir_path = config.checkpoint_dir_path

    if output_path is None:
        if checkpoint_dir_path is not None:
            output_path_obj = Path(checkpoint_dir_path)
        else:
            if not config.alg:
                return
            timestamp = config.timestamp
            if timestamp is None:
                timestamp = build_timestamp_string()
                config.timestamp = timestamp
            output_path_obj = build_timestamped_output_path(
                alg=config.alg,
                outputs_path=config.outputs_path,
                timestamp=timestamp,
            )
    else:
        output_path_obj = Path(output_path)

    config.output_path = str(output_path_obj)



def load_train_config_from_checkpoint(
    alg: str,
    checkpoint_dir_path: str,
    device_override: Optional[str] = None,
):
    checkpoint_dir_path_obj = Path(checkpoint_dir_path)
    if not checkpoint_dir_path_obj.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir_path_obj}")
    train_config_path = get_train_config_path(checkpoint_dir_path_obj)
    if not train_config_path.exists():
        raise FileNotFoundError(f"Train config not found under checkpoint directory: {checkpoint_dir_path}")

    module_name = f"{alg}.train"
    try:
        train_module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise SystemExit(f"Unsupported alg: {alg}") from exc
        raise

    config = train_module.TrainConfig(**read_json(train_config_path))
    if device_override is not None:
        config.device = device_override
    return config


def build_eval_context(
    alg: str,
    eval_config: EvalConfig,
    nodes: List[Node],
    edges: List[Edge],
):
    if eval_config.checkpoint_dir_path is None:
        raise ValueError("EvalConfig.checkpoint_dir_path must be set before build_eval_context.")
    checkpoint_dir_path = eval_config.checkpoint_dir_path
    checkpoint_dir_path_obj = Path(checkpoint_dir_path)
    if not checkpoint_dir_path_obj.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir_path_obj}")

    module_name = f"{alg}.train"
    try:
        train_module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise SystemExit(f"Unsupported alg: {alg}") from exc
        raise

    try:
        load_from_checkpoint = train_module.load_translator_pool_from_checkpoint
    except AttributeError as exc:
        raise AttributeError(f"{module_name} does not define load_translator_pool_from_checkpoint") from exc

    return load_from_checkpoint(
        checkpoint_dir_path=checkpoint_dir_path,
        nodes=nodes,
        edges=edges,
        device_override=eval_config.device,
    )

def resolve_latest_checkpoint_dir_for_alg(
    alg: str,
    outputs_path: str = "outputs",
    checkpoint_name: str = "checkpoint.pt",
) -> Path:
    outputs_path_obj = Path(outputs_path)
    candidates = sorted(
        path
        for path in outputs_path_obj.glob(f"{alg}_*")
        if path.is_dir() and (path / checkpoint_name).exists()
    )
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint directories found for alg={alg!r} under {outputs_path_obj}"
        )
    return candidates[-1]


def build_eval_dataloader(
    spec: HFDatasetSpec,
    eval_config: EvalConfig,
) -> DataLoader:
    dataset = HFQAPairStream(
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


def normalize_context_text(raw_value: Any) -> Optional[str]:
    def _collect_strings(value: Any) -> List[str]:
        if value is None:
            return []

        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []

        if isinstance(value, list):
            parts: List[str] = []
            for item in value:
                parts.extend(_collect_strings(item))
            return parts

        if isinstance(value, dict):
            for key in ("contexts", "context", "text", "abstract", "passage", "sentences"):
                if key in value:
                    parts = _collect_strings(value[key])
                    if parts:
                        return parts
            return []

        return []

    parts = _collect_strings(raw_value)
    if not parts:
        return None
    return "\n".join(parts)


def extract_question_and_answer(spec: HFDatasetSpec, example: Dict) -> Optional[Dict[str, Any]]:
    question = example.get(spec.question_field, "")
    if not isinstance(question, str) or not question.strip():
        return None

    if spec.answer_mode == "boolq":
        context_field = spec.context_field or "passage"
        context = normalize_context_text(example.get(context_field, None))
        if context is None:
            return None

        answer_value = example.get("answer", None)
        if not isinstance(answer_value, bool):
            return None

        return {
            "question": question.strip(),
            "context": context,
            "answer": "yes" if answer_value else "no",
        }

    if spec.answer_mode == "pubmed_qa":
        context_field = spec.context_field or "context"
        context = normalize_context_text(example.get(context_field, None))
        if context is None:
            return None

        answer_value = example.get("final_decision", None)
        if not isinstance(answer_value, str):
            return None

        normalized_answer = answer_value.strip().lower()
        if normalized_answer not in {"yes", "no", "maybe"}:
            return None

        return {
            "question": question.strip(),
            "context": context,
            "answer": normalized_answer,
        }

    if spec.answer_mode == "squad":
        context_field = spec.context_field or "context"
        answers_field = spec.answers_field or "answers"

        context = example.get(context_field, "")
        answers = example.get(answers_field, None)

        if not isinstance(context, str) or not context.strip():
            return None

        answer_texts: List[str] = []
        if isinstance(answers, dict):
            raw_texts = answers.get("text", [])
            if isinstance(raw_texts, list):
                answer_texts = [
                    item.strip()
                    for item in raw_texts
                    if isinstance(item, str) and item.strip()
                ]
        elif isinstance(answers, list):
            answer_texts = [
                item.strip()
                for item in answers
                if isinstance(item, str) and item.strip()
            ]

        if not answer_texts:
            return None

        return {
            "question": question.strip(),
            "context": context.strip(),
            "answers": answer_texts,
        }

    raise ValueError(f"Unsupported answer_mode: {spec.answer_mode}")


def format_boolq_context_prefix(context: str) -> str:
    return (
        "Read the passage and answer the question with yes or no only.\n\n"
        f"Passage: {context.strip()}\n"
    )


def format_boolq_question_prefix(question: str) -> str:
    return (
        f"Question: {question.strip()}\n"
        "Answer:"
    )


def prepare_boolq_context_inputs(tokenizer, context: str, device: str) -> Dict[str, Any]:
    prefix_text = format_boolq_context_prefix(context=context)
    return prepare_full_text_inputs(tokenizer=tokenizer, text=prefix_text, device=device)


def prepare_boolq_question_prefix(tokenizer, question: str, device: str) -> Dict[str, torch.Tensor]:
    prefix_text = format_boolq_question_prefix(question=question)
    return prepare_text_prefix(tokenizer=tokenizer, prefix_text=prefix_text, device=device)


def format_pubmed_qa_context_prefix(context: str) -> str:
    return (
        "Read the abstract and answer the biomedical research question with yes, no, or maybe only.\n\n"
        f"Abstract: {context.strip()}\n"
    )


def format_pubmed_qa_question_prefix(question: str) -> str:
    return (
        f"Question: {question.strip()}\n"
        "Answer:"
    )


def prepare_pubmed_qa_context_inputs(tokenizer, context: str, device: str) -> Dict[str, Any]:
    prefix_text = format_pubmed_qa_context_prefix(context=context)
    return prepare_full_text_inputs(tokenizer=tokenizer, text=prefix_text, device=device)


def prepare_pubmed_qa_question_prefix(tokenizer, question: str, device: str) -> Dict[str, torch.Tensor]:
    prefix_text = format_pubmed_qa_question_prefix(question=question)
    return prepare_text_prefix(tokenizer=tokenizer, prefix_text=prefix_text, device=device)


def format_question_prefix(
    question: str,
    choices: Optional[List[str]] = None,
    subject: Optional[str] = None,
    context: Optional[str] = None,
    answer_mode: Optional[str] = None,
) -> str:
    question = question.strip()

    if answer_mode == "boolq":
        if not isinstance(context, str) or not context.strip():
            raise ValueError("BoolQ requires passage context.")
        return format_boolq_context_prefix(context=context) + format_boolq_question_prefix(question=question)

    if answer_mode == "pubmed_qa":
        if not isinstance(context, str) or not context.strip():
            raise ValueError("PubMedQA requires abstract context.")
        return format_pubmed_qa_context_prefix(context=context) + format_pubmed_qa_question_prefix(question=question)

    if not choices:
        return f"Question: {question}\nAnswer:"

    prompt_lines = [f"Question: {question}", "Choices:"]
    for idx, choice in enumerate(choices):
        label = chr(ord("A") + idx)
        prompt_lines.append(f"{label}. {choice.strip()}")
    prompt_lines.append("Answer:")
    return "\n".join(prompt_lines)


def format_squad_v11_context_prefix(context: str) -> str:
    return (
        "Read the passage and answer the question briefly. Use a short phrase from the passage when possible.\n\n"
        f"Passage: {context.strip()}\n"
    )


def format_squad_v11_question_prefix(question: str) -> str:
    return (
        f"Question: {question.strip()}\n"
        "Answer:"
    )


def prepare_squad_v11_context_inputs(
    tokenizer,
    context: str,
    device: str,
    max_input_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    prefix_text = format_squad_v11_context_prefix(context=context)
    return prepare_full_text_inputs(
        tokenizer=tokenizer,
        text=prefix_text,
        device=device,
        max_input_tokens=max_input_tokens,
    )


def prepare_squad_v11_question_prefix(tokenizer, question: str, device: str) -> Dict[str, torch.Tensor]:
    prefix_text = format_squad_v11_question_prefix(question=question)
    return prepare_text_prefix(tokenizer=tokenizer, prefix_text=prefix_text, device=device)


def format_multinews_context_prefix(context: str) -> str:
    return (
        "Read the following news articles and write a concise summary.\n\n"
        f"Articles:\n{context.strip()}\n"
    )


def format_multinews_question_prefix(question: str) -> str:
    return (
        f"Task: {question.strip()}\n"
        "Summary:"
    )


def prepare_multinews_context_inputs(
    tokenizer,
    context: str,
    device: str,
    max_input_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    prefix_text = format_multinews_context_prefix(context=context)
    return prepare_full_text_inputs(
        tokenizer=tokenizer,
        text=prefix_text,
        device=device,
        max_input_tokens=max_input_tokens,
    )


def prepare_multinews_question_prefix(tokenizer, question: str, device: str) -> Dict[str, torch.Tensor]:
    prefix_text = format_multinews_question_prefix(question=question)
    return prepare_text_prefix(tokenizer=tokenizer, prefix_text=prefix_text, device=device)


def format_generation_prompt(context: str, question: str) -> str:
    return (
        "Read the passage and answer the question briefly.\n\n"
        f"Context: {context.strip()}\n"
        f"Question: {question.strip()}\n"
        "Answer:"
    )


def prepare_text_prefix(tokenizer, prefix_text: str, device: str) -> Dict[str, torch.Tensor]:
    tokenized = tokenizer(prefix_text, return_tensors="pt")
    input_ids = tokenized.input_ids.to(device)
    if input_ids.shape[1] < 2:
        raise ValueError("Prefix must tokenize to at least 2 tokens.")
    cache_ids = input_ids[:, :-1]
    seed_token = input_ids[:, -1:]
    return {
        "prefix_text": prefix_text,
        "full_prefix_ids": input_ids,
        "cache_ids": cache_ids,
        "seed_token": seed_token,
    }


def prepare_question_prefix(
    tokenizer,
    question: str,
    device: str,
    choices: Optional[List[str]] = None,
    subject: Optional[str] = None,
    context: Optional[str] = None,
    answer_mode: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    prefix_text = format_question_prefix(
        question,
        choices=choices,
        subject=subject,
        context=context,
        answer_mode=answer_mode,
    )
    return prepare_text_prefix(tokenizer=tokenizer, prefix_text=prefix_text, device=device)


def prepare_generation_prefix(tokenizer, context: str, question: str, device: str) -> Dict[str, torch.Tensor]:
    prefix_text = format_generation_prompt(context=context, question=question)
    return prepare_text_prefix(tokenizer=tokenizer, prefix_text=prefix_text, device=device)


def format_generation_question_prefix(question: str) -> str:
    return (
        f"Question: {question.strip()}\n"
        "Answer:"
    )


def prepare_full_text_inputs(
    tokenizer,
    text: str,
    device: str,
    max_input_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    tokenizer_kwargs = {"return_tensors": "pt"}
    if max_input_tokens is not None:
        if max_input_tokens < 1:
            raise ValueError("max_input_tokens must be >= 1")
        tokenizer_kwargs["truncation"] = True
        tokenizer_kwargs["max_length"] = max_input_tokens
    tokenized = tokenizer(text, **tokenizer_kwargs)
    input_ids = tokenized.input_ids.to(device)
    if input_ids.shape[1] < 1:
        raise ValueError("Text must tokenize to at least 1 token.")
    return {
        "text": text,
        "input_ids": input_ids,
        "was_truncated": max_input_tokens is not None and input_ids.shape[1] >= max_input_tokens,
    }


def prepare_generation_question_prefix(tokenizer, question: str, device: str) -> Dict[str, torch.Tensor]:
    prefix_text = format_generation_question_prefix(question=question)
    return prepare_text_prefix(tokenizer=tokenizer, prefix_text=prefix_text, device=device)


def get_model_context_limit(model: PreTrainedModel, tokenizer: Optional[PreTrainedTokenizerBase] = None) -> int:
    config = getattr(model, "config", None)
    candidates = [
        getattr(config, "n_positions", None),
        getattr(config, "max_position_embeddings", None),
        getattr(config, "n_ctx", None),
    ]
    if tokenizer is not None:
        tokenizer_limit = getattr(tokenizer, "model_max_length", None)
        if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000:
            candidates.append(tokenizer_limit)

    limits = [value for value in candidates if isinstance(value, int) and value > 0]
    if not limits:
        return 1024
    return min(limits)


def get_answer_token_budget(eval_config) -> int:
    return eval_config.generation_max_new_tokens


def compute_benchmark_context_budget(
    ctx: Context,
    spec: HFDatasetSpec,
    question: str,
    eval_config,
) -> int:
    shared_limit = min(
        get_model_context_limit(ctx.mm.get_model(node.id), ctx.tokenizer)
        for node in ctx.nodes
    )
    question_prefix = prepare_generation_task_question_prefix(
        spec=spec,
        tokenizer=ctx.tokenizer,
        question=question,
        device="cpu",
    )
    reserved_tokens = (
        question_prefix["cache_ids"].shape[1]
        + question_prefix["seed_token"].shape[1]
        + get_answer_token_budget(eval_config)
    )
    budget = shared_limit - reserved_tokens
    if budget < 16:
        raise ValueError(
            f"Insufficient context budget for {spec.name_for_log}: "
            f"shared_limit={shared_limit}, reserved_tokens={reserved_tokens}"
        )
    return budget


def prepare_generation_task_question_prefix(
    spec: HFDatasetSpec,
    tokenizer,
    question: str,
    device: str,
) -> Dict[str, torch.Tensor]:
    if spec.answer_mode in {"squad", "newsqa"}:
        return prepare_squad_v11_question_prefix(
            tokenizer=tokenizer,
            question=question,
            device=device,
        )
    # if spec.answer_mode == "multinews":
    #     return prepare_multinews_question_prefix(
    #         tokenizer=tokenizer,
    #         question=question,
    #         device=device,
    #     )
    return prepare_generation_question_prefix(
        tokenizer=tokenizer,
        question=question,
        device=device,
    )


def prepare_logit_task_inputs(
    spec: HFDatasetSpec,
    tokenizer,
    context: Optional[str],
    question: str,
    device: str,
) -> Dict[str, Any]:
    if spec.answer_mode == "boolq":
        if not isinstance(context, str) or not context.strip():
            raise ValueError("BoolQ requires passage context.")
        context_prefix = prepare_boolq_context_inputs(
            tokenizer=tokenizer,
            context=context,
            device=device,
        )
        question_prefix = prepare_boolq_question_prefix(
            tokenizer=tokenizer,
            question=question,
            device=device,
        )
        return {
            "context_prefix": context_prefix,
            "question_prefix": question_prefix,
            "cache_input_ids": context_prefix["input_ids"],
            "question_cache_ids": question_prefix["cache_ids"],
            "seed_token": question_prefix["seed_token"],
            "was_truncated": False,
        }

    if spec.answer_mode == "pubmed_qa":
        if not isinstance(context, str) or not context.strip():
            raise ValueError("PubMedQA requires abstract context.")
        context_prefix = prepare_pubmed_qa_context_inputs(
            tokenizer=tokenizer,
            context=context,
            device=device,
        )
        question_prefix = prepare_pubmed_qa_question_prefix(
            tokenizer=tokenizer,
            question=question,
            device=device,
        )
        return {
            "context_prefix": context_prefix,
            "question_prefix": question_prefix,
            "cache_input_ids": context_prefix["input_ids"],
            "question_cache_ids": question_prefix["cache_ids"],
            "seed_token": question_prefix["seed_token"],
            "was_truncated": False,
        }

    prefix = prepare_question_prefix(
        tokenizer=tokenizer,
        question=question,
        device=device,
        context=context,
        answer_mode=spec.answer_mode,
    )
    return {
        "cache_input_ids": prefix["cache_ids"],
        "question_cache_ids": None,
        "seed_token": prefix["seed_token"],
        "was_truncated": False,
    }


def prepare_generation_task_inputs(
    spec: HFDatasetSpec,
    tokenizer,
    context: str,
    question: str,
    device: str,
    max_input_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    if spec.answer_mode in {"squad", "newsqa"}:
        context_prefix = prepare_squad_v11_context_inputs(
            tokenizer=tokenizer,
            context=context,
            device=device,
            max_input_tokens=max_input_tokens,
        )
        question_prefix = prepare_squad_v11_question_prefix(
            tokenizer=tokenizer,
            question=question,
            device=device,
        )
        return {
            "context_prefix": context_prefix,
            "question_prefix": question_prefix,
            "cache_input_ids": context_prefix["input_ids"],
            "question_cache_ids": question_prefix["cache_ids"],
            "seed_token": question_prefix["seed_token"],
            "was_truncated": context_prefix.get("was_truncated", False),
        }

    # if spec.answer_mode == "multinews":
    #     context_prefix = prepare_multinews_context_inputs(
    #         tokenizer=tokenizer,
    #         context=context,
    #         device=device,
    #         max_input_tokens=max_input_tokens,
    #     )
    #     question_prefix = prepare_multinews_question_prefix(
    #         tokenizer=tokenizer,
    #         question=question,
    #         device=device,
    #     )
    #     return {
    #         "context_prefix": context_prefix,
    #         "question_prefix": question_prefix,
    #         "cache_input_ids": context_prefix["input_ids"],
    #         "question_cache_ids": question_prefix["cache_ids"],
    #         "seed_token": question_prefix["seed_token"],
    #         "was_truncated": bool(context_prefix.get("was_truncated", False)),
    #     }

    prefix = prepare_generation_prefix(
        tokenizer=tokenizer,
        context=context,
        question=question,
        device=device,
    )
    return {
        "cache_input_ids": prefix["cache_ids"],
        "question_cache_ids": None,
        "seed_token": prefix["seed_token"],
        "was_truncated": False,
    }


def predict_generation_task_answer(
    model,
    tokenizer,
    past_key_values: PastKeyValues,
    seed_token: torch.Tensor,
    eval_config,
    question_cache_ids: Optional[torch.Tensor] = None,
) -> str:
    generation_past = past_key_values
    if question_cache_ids is not None:
        generation_past = append_input_ids_to_past(
            model=model,
            past_key_values=past_key_values,
            input_ids=question_cache_ids,
        )

    return generate_greedy_answer(
        model=model,
        tokenizer=tokenizer,
        past_key_values=generation_past,
        seed_token=seed_token,
        max_new_tokens=eval_config.generation_max_new_tokens,
    )



@torch.inference_mode()
def append_input_ids_to_past(
    model,
    past_key_values: PastKeyValues,
    input_ids: torch.Tensor,
) -> PastKeyValues:
    if input_ids.shape[1] == 0:
        return past_key_values

    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    return outputs.past_key_values


def build_text_candidate_token_ids(
    tokenizer,
    candidates: Dict[str, str],
) -> Dict[str, torch.Tensor]:
    token_ids_by_label: Dict[str, torch.Tensor] = {}

    for label, text in candidates.items():
        normalized_text = text.strip()
        token_ids = tokenizer(
            f" {normalized_text}",
            add_special_tokens=False,
        ).input_ids
        if len(token_ids) < 1:
            raise ValueError(f"Failed to tokenize candidate text for label={label}: {text!r}")
        token_ids_by_label[label] = torch.tensor(token_ids, dtype=torch.long)

    return token_ids_by_label


def build_logit_answer_candidates(
    tokenizer,
    spec: HFDatasetSpec,
) -> Dict[str, torch.Tensor]:
    if spec.answer_mode == "boolq":
        return build_text_candidate_token_ids(
            tokenizer,
            {"yes": "yes", "no": "no"},
        )

    if spec.answer_mode == "pubmed_qa":
        return build_text_candidate_token_ids(
            tokenizer,
            {"yes": "yes", "no": "no", "maybe": "maybe"},
        )

    raise ValueError(f"Unsupported answer_mode for logit scoring: {spec.answer_mode}")


def score_candidate_logprob(
    model,
    past_key_values,
    seed_token: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    normalize_by_length: bool = True,
) -> float:
    device = seed_token.device
    candidate_ids = candidate_token_ids.to(device).unsqueeze(0)

    if candidate_ids.shape[1] == 1:
        scoring_input_ids = seed_token
    else:
        scoring_input_ids = torch.cat([seed_token, candidate_ids[:, :-1]], dim=1)

    outputs = model(
        input_ids=scoring_input_ids,
        past_key_values=past_key_values,
        use_cache=False,
    )
    log_probs = F.log_softmax(outputs.logits, dim=-1)
    token_log_probs = log_probs.gather(-1, candidate_ids.unsqueeze(-1)).squeeze(-1)

    score = token_log_probs.sum(dim=1).item()
    if normalize_by_length:
        score /= candidate_ids.shape[1]
    return score


def score_answer_choices(
    model,
    past_key_values,
    seed_token: torch.Tensor,
    choice_token_ids: Dict[str, torch.Tensor],
    normalize_by_length: bool = True,
) -> Dict[str, float]:
    return {
        label: score_candidate_logprob(
            model=model,
            past_key_values=past_key_values,
            seed_token=seed_token,
            candidate_token_ids=choice_token_ids[label],
            normalize_by_length=normalize_by_length,
        )
        for label in choice_token_ids
    }


def prepare_answer_scoring_past(
    model,
    past_key_values: PastKeyValues,
    question_cache_ids: Optional[torch.Tensor] = None,
) -> PastKeyValues:
    if question_cache_ids is None:
        return past_key_values
    return append_input_ids_to_past(
        model=model,
        past_key_values=past_key_values,
        input_ids=question_cache_ids,
    )


def predict_answer_label(choice_scores: Dict[str, float]) -> str:
    return max(choice_scores.items(), key=lambda item: item[1])[0]


@torch.inference_mode()
def generate_greedy_answer(
    model,
    tokenizer,
    past_key_values: PastKeyValues,
    seed_token: torch.Tensor,
    max_new_tokens: int,
) -> str:
    generated_token_ids: List[int] = []
    current_input_ids = seed_token
    current_past = past_key_values
    eos_token_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=current_input_ids,
            past_key_values=current_past,
            use_cache=True,
        )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        next_token_id = next_token.item()

        if eos_token_id is not None and next_token_id == eos_token_id:
            break

        generated_token_ids.append(next_token_id)
        current_input_ids = next_token
        current_past = outputs.past_key_values

    decoded = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    return postprocess_generated_answer(decoded)


def postprocess_generated_answer(text: str) -> str:
    cleaned = text.strip()
    for stopper in ["\n", "\r", "Question:", "Context:", "Answer:"]:
        if stopper in cleaned:
            cleaned = cleaned.split(stopper, 1)[0].strip()
    return cleaned


def normalize_qa_text(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    def lower(value: str) -> str:
        return value.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def _compute_pair_f1(prediction: str, gold_answer: str) -> float:
    pred_tokens = normalize_qa_text(prediction).split()
    gold_tokens = normalize_qa_text(gold_answer).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def compute_generation_f1(prediction: str, gold_answers: List[str]) -> float:
    return max(_compute_pair_f1(prediction, gold_answer) for gold_answer in gold_answers)


def summarize_path_metrics(path_metrics: Dict[str, RunningAverage]) -> Dict[str, Dict[str, float]]:
    results = {}

    for path_name, meter in path_metrics.items():
        results[path_name] = meter.summary()

    return results


def summarize_generation_path_metrics(path_metrics: Dict[str, GenerationRunningAverage]) -> Dict[str, Dict[str, float]]:
    results = {}

    for path_name, meter in path_metrics.items():
        results[path_name] = meter.summary()

    return results


def build_edge_pretty_name(edge_id: str, nodes: List[Node], edges: List[Edge]) -> str:
    node_map = build_node_map(nodes)
    edge_map = build_edge_map(edges)
    edge = edge_map.get(edge_id)
    if edge is None:
        return edge_id
    src_model_id = node_map[edge.src_id].model_id
    tgt_model_id = node_map[edge.tgt_id].model_id
    return f"{edge.id} ({src_model_id} -> {tgt_model_id})"


def log_dataset_result(
    logger: logging.Logger,
    dataset_name: str,
    results: Dict[str, Dict[str, float]],
    nodes: List[Node],
    edges: List[Edge],
) -> None:
    logger.info("===== %s =====", dataset_name)
    for edge in edges:
        row = results[edge.id]
        pretty_name = build_edge_pretty_name(edge.id, nodes, edges)
        logger.info(
            "%s | cosine=%.6f | accuracy=%.6f | native_accuracy=%.6f | count=%d",
            pretty_name,
            row["cosine"],
            row["accuracy"],
            row["native_accuracy"],
            row["count"],
        )


def log_generation_dataset_result(
    logger: logging.Logger,
    dataset_name: str,
    results: Dict[str, Dict[str, float]],
    nodes: List[Node],
    edges: List[Edge],
) -> None:
    logger.info("===== %s =====", dataset_name)
    for edge in edges:
        row = results[edge.id]
        pretty_name = build_edge_pretty_name(edge.id, nodes, edges)
        logger.info(
            "%s | cosine=%.6f | f1=%.6f | native_f1=%.6f | count=%d",
            pretty_name,
            row["cosine"],
            row["f1"],
            row["native_f1"],
            row["count"],
        )


def _is_valid_summary_value(value: Any) -> bool:
    return isinstance(value, (int, float)) and value == value


def _summary_mean(values: List[float]) -> float:
    valid_values = [float(value) for value in values if _is_valid_summary_value(value)]
    if not valid_values:
        return float("nan")
    return sum(valid_values) / len(valid_values)


def _format_summary_float(value: float) -> str:
    if not _is_valid_summary_value(value):
        return "N/A"
    return f"{float(value):.3f}"


def _format_summary_percent(value: float) -> str:
    if not _is_valid_summary_value(value):
        return "N/A"
    return f"{float(value) * 100.0:.1f}%"




def _format_summary_throughput(value: float) -> str:
    if not _is_valid_summary_value(value):
        return "N/A"
    return f"{float(value):.0f} tok/s"


def build_openwebtext_profile_fields(row: Dict[str, float], *, prefix: str = "") -> Tuple[str, str, str]:
    field_prefix = f"{prefix}_" if prefix else ""
    latency_text = _format_summary_float(row.get(f"{field_prefix}latency_ms", float("nan")))
    if latency_text != "N/A":
        latency_text = f"{latency_text} ms"
    throughput_value = row.get(f"{field_prefix}throughput_tokens_per_sec", float("nan"))
    throughput_text = _format_summary_throughput(throughput_value)
    peak_text = _format_summary_float(row.get(f"{field_prefix}peak_memory_gib", float("nan")))
    if peak_text != "N/A":
        peak_text = f"{peak_text} GiB"
    return latency_text, throughput_text, peak_text

def build_openwebtext_profile_cell(row: Dict[str, float], *, prefix: str = "") -> str:
    latency_text, throughput_text, peak_text = build_openwebtext_profile_fields(row, prefix=prefix)
    return f"{latency_text} · {throughput_text} · {peak_text}"

def build_edge_summary_markdown_table(
    alg: str,
    edge_id: str,
    nodes: List[Node],
    edges: List[Edge],
    all_logit_results: Dict[str, Dict[str, Dict[str, float]]],
    all_generation_results: Dict[str, Dict[str, Dict[str, float]]],
    openwebtext_loss_results: Optional[Dict[str, Dict[str, float]]] = None,
) -> str:
    node_map = build_node_map(nodes)
    edge_map = build_edge_map(edges)
    edge = edge_map.get(edge_id)

    logit_dataset_keys = [
        ("BoolQ", "BoolQ/validation"),
        ("PubMedQA", "PubMedQA/pqa_labeled/train"),
    ]
    generation_dataset_keys = [
        ("SQuAD", "SQuAD-v1.1/validation"),
        ("NewsQA", "NewsQA/validation"),
    ]

    logit_rows = {
        display_name: all_logit_results.get(dataset_key, {}).get(edge_id, {})
        for display_name, dataset_key in logit_dataset_keys
    }
    generation_rows = {
        display_name: all_generation_results.get(dataset_key, {}).get(edge_id, {})
        for display_name, dataset_key in generation_dataset_keys
    }
    loss_row = (openwebtext_loss_results or {}).get(edge_id, {})

    translated_cosine_avg = _summary_mean([
        logit_rows["BoolQ"].get("cosine", float("nan")),
        logit_rows["PubMedQA"].get("cosine", float("nan")),
    ])
    translated_accuracy_avg = _summary_mean([
        logit_rows["BoolQ"].get("accuracy", float("nan")),
        logit_rows["PubMedQA"].get("accuracy", float("nan")),
    ])
    native_accuracy_avg = _summary_mean([
        logit_rows["BoolQ"].get("native_accuracy", float("nan")),
        logit_rows["PubMedQA"].get("native_accuracy", float("nan")),
    ])
    translated_generation_f1_avg = _summary_mean([
        generation_rows["SQuAD"].get("f1", float("nan")),
        generation_rows["NewsQA"].get("f1", float("nan")),
    ])
    native_generation_f1_avg = _summary_mean([
        generation_rows["SQuAD"].get("native_f1", float("nan")),
        generation_rows["NewsQA"].get("native_f1", float("nan")),
    ])

    if edge is None:
        target_model_id = "target"
        direction_title = edge_id
    else:
        src_model_id = node_map[edge.src_id].model_id
        target_model_id = node_map[edge.tgt_id].model_id
        direction_title = f"{edge.id} 방향 ({src_model_id} -> {target_model_id})"

    native_latency_text, native_throughput_text, native_peak_text = build_openwebtext_profile_fields(loss_row, prefix="native")
    translated_latency_text, translated_throughput_text, translated_peak_text = build_openwebtext_profile_fields(loss_row)

    lines = [
        f"### {direction_title}",
        "",
        "| Method | Cosine Sim Avg | BoolQ | PubMedQA | Acc Avg | SQuAD | NewsQA | Gen F1 Avg | OWT Val Loss | OWT Val Latency | OWT Val Throughput | OWT Val GPU Peak Memory |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {target_model_id} (baseline) | N/A | "
            f"{_format_summary_percent(logit_rows['BoolQ'].get('native_accuracy', float('nan')))} | "
            f"{_format_summary_percent(logit_rows['PubMedQA'].get('native_accuracy', float('nan')))} | "
            f"{_format_summary_percent(native_accuracy_avg)} | "
            f"{_format_summary_float(generation_rows['SQuAD'].get('native_f1', float('nan')))} | "
            f"{_format_summary_float(generation_rows['NewsQA'].get('native_f1', float('nan')))} | "
            f"{_format_summary_float(native_generation_f1_avg)} | "
            f"{_format_summary_float(loss_row.get('native_loss', float('nan')))} | "
            f"{native_latency_text} | "
            f"{native_throughput_text} | "
            f"{native_peak_text} |"
        ),
        (
            f"| {alg} | "
            f"{_format_summary_float(translated_cosine_avg)} | "
            f"{_format_summary_percent(logit_rows['BoolQ'].get('accuracy', float('nan')))} | "
            f"{_format_summary_percent(logit_rows['PubMedQA'].get('accuracy', float('nan')))} | "
            f"{_format_summary_percent(translated_accuracy_avg)} | "
            f"{_format_summary_float(generation_rows['SQuAD'].get('f1', float('nan')))} | "
            f"{_format_summary_float(generation_rows['NewsQA'].get('f1', float('nan')))} | "
            f"{_format_summary_float(translated_generation_f1_avg)} | "
            f"{_format_summary_float(loss_row.get('loss', float('nan')))} | "
            f"{translated_latency_text} | "
            f"{translated_throughput_text} | "
            f"{translated_peak_text} |"
        ),
    ]
    return "\n".join(lines)


def build_final_summary_markdown(
    alg: str,
    nodes: List[Node],
    edges: List[Edge],
    all_logit_results: Dict[str, Dict[str, Dict[str, float]]],
    all_generation_results: Dict[str, Dict[str, Dict[str, float]]],
    openwebtext_loss_results: Optional[Dict[str, Dict[str, float]]] = None,
) -> str:
    sections = []

    for edge in edges:
        sections.append(
            build_edge_summary_markdown_table(
                alg=alg,
                edge_id=edge.id,
                nodes=nodes,
                edges=edges,
                all_logit_results=all_logit_results,
                all_generation_results=all_generation_results,
                openwebtext_loss_results=openwebtext_loss_results,
            )
        )

    return "\n\n".join(sections)
