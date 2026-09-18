from contextlib import contextmanager
import importlib
import time
from typing import Any, Callable, Iterable, Optional, Tuple

import numpy as np

from core.common import *
from core.config import Config
from core.context import Context
from core.train_util import InfiniteDataLoader, get_train_config_path


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


@dataclass(frozen=True)
class GPUMemoryBreakdownBytes:
    model_bytes: int
    translator_bytes: int
    kv_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.model_bytes + self.translator_bytes + self.kv_bytes


def _iter_module_like_tensors(module: Optional[Any]):
    if module is None:
        return

    modules_fn = getattr(module, "modules", None)
    if callable(modules_fn):
        for submodule in modules_fn():
            parameters_fn = getattr(submodule, "parameters", None)
            if callable(parameters_fn):
                yield from parameters_fn(recurse=False)
            buffers_fn = getattr(submodule, "buffers", None)
            if callable(buffers_fn):
                yield from buffers_fn(recurse=False)
        return

    parameters_fn = getattr(module, "parameters", None)
    if callable(parameters_fn):
        yield from parameters_fn()


def _cuda_storage_bytes(
    modules: Iterable[Any],
    *,
    device_index: int,
    seen_storage_keys: Optional[set] = None,
) -> int:
    seen = seen_storage_keys if seen_storage_keys is not None else set()
    total = 0
    for module in modules:
        for tensor in _iter_module_like_tensors(module):
            if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cuda":
                continue
            tensor_device_index = torch.cuda.current_device() if tensor.device.index is None else tensor.device.index
            if tensor_device_index != device_index:
                continue
            storage = tensor.untyped_storage()
            key = (tensor_device_index, storage.data_ptr())
            if key in seen:
                continue
            seen.add(key)
            total += int(storage.nbytes())
    return total


def _iter_nested_tensors(value: Any, *, seen_objects: Optional[set] = None):
    """Yield tensors from nested KV/cache containers without walking arbitrary objects.

    Supports tuple/list/dict past_key_values as well as Transformers-style cache
    objects exposing ``key_cache`` and ``value_cache``. Object identities are
    deduplicated to avoid cycles; tensor storage deduplication happens separately.
    """
    if value is None:
        return
    if isinstance(value, torch.Tensor):
        yield value
        return

    seen = seen_objects if seen_objects is not None else set()
    object_id = id(value)
    if object_id in seen:
        return
    seen.add(object_id)

    if isinstance(value, dict):
        for nested in value.values():
            yield from _iter_nested_tensors(nested, seen_objects=seen)
        return
    if isinstance(value, (tuple, list, set)):
        for nested in value:
            yield from _iter_nested_tensors(nested, seen_objects=seen)
        return

    found_cache_attr = False
    for attr_name in ("key_cache", "value_cache"):
        if hasattr(value, attr_name):
            found_cache_attr = True
            yield from _iter_nested_tensors(getattr(value, attr_name), seen_objects=seen)
    if found_cache_attr:
        return


def _cuda_nested_storage_bytes(
    objects: Iterable[Any],
    *,
    device_index: int,
    seen_storage_keys: Optional[set] = None,
) -> int:
    seen = seen_storage_keys if seen_storage_keys is not None else set()
    seen_objects: set = set()
    total = 0
    for obj in objects:
        for tensor in _iter_nested_tensors(obj, seen_objects=seen_objects):
            if tensor.device.type != "cuda":
                continue
            tensor_device_index = torch.cuda.current_device() if tensor.device.index is None else tensor.device.index
            if tensor_device_index != device_index:
                continue
            storage = tensor.untyped_storage()
            key = (tensor_device_index, storage.data_ptr())
            if key in seen:
                continue
            seen.add(key)
            total += int(storage.nbytes())
    return total


def measure_gpu_memory_breakdown_bytes(
    device: str,
    *,
    models: Iterable[Any],
    translator_pool: Optional[Any],
    kv_objects: Optional[Iterable[Any]] = None,
) -> Optional[GPUMemoryBreakdownBytes]:
    """Measure persistent CUDA storage for model, translator, and KV/cache state.

    Unlike the old implementation, KV bytes are *not* inferred as
    ``memory_allocated - model - translator``. That residual also contains unrelated
    persistent tensors and can be dominated by allocator/forward artifacts. KV/cache
    bytes are measured directly from the supplied cache objects and deduplicated by
    underlying CUDA storage, matching the AgentRunner memory metric.
    """
    if not (torch.cuda.is_available() and str(device).startswith("cuda")):
        return None

    device_obj = torch.device(device)
    device_index = torch.cuda.current_device() if device_obj.index is None else device_obj.index
    torch.cuda.synchronize(device_index)

    seen_storages = set()
    model_bytes = _cuda_storage_bytes(
        models,
        device_index=device_index,
        seen_storage_keys=seen_storages,
    )
    translator_bytes = _cuda_storage_bytes(
        [translator_pool] if translator_pool is not None else [],
        device_index=device_index,
        seen_storage_keys=seen_storages,
    )
    kv_bytes = _cuda_nested_storage_bytes(
        list(kv_objects or ()),
        device_index=device_index,
        seen_storage_keys=seen_storages,
    )
    return GPUMemoryBreakdownBytes(
        model_bytes=model_bytes,
        translator_bytes=translator_bytes,
        kv_bytes=kv_bytes,
    )


@dataclass
class InferenceProfileAccumulator:
    total_latency_sec: float = 0.0
    total_tokens: int = 0
    num_calls: int = 0
    model_memory_bytes: Optional[int] = None
    translator_memory_bytes: Optional[int] = None
    kv_memory_bytes: Optional[int] = None

    def update(
        self,
        *,
        latency_sec: float,
        tokens: int,
        model_memory_bytes: Optional[int],
        translator_memory_bytes: Optional[int],
        kv_memory_bytes: Optional[int],
    ) -> None:
        self.total_latency_sec += float(latency_sec)
        self.total_tokens += tokens
        self.num_calls += 1
        for field_name, value in (
            ("model_memory_bytes", model_memory_bytes),
            ("translator_memory_bytes", translator_memory_bytes),
            ("kv_memory_bytes", kv_memory_bytes),
        ):
            if value is None:
                continue
            current = getattr(self, field_name)
            if current is None or value > current:
                setattr(self, field_name, int(value))

    def summary(self) -> Dict[str, float]:
        if self.num_calls <= 0:
            avg_latency_ms = float("nan")
        else:
            avg_latency_ms = (self.total_latency_sec / self.num_calls) * 1000.0

        if self.total_latency_sec > 0.0 and self.total_tokens > 0:
            throughput_tokens_per_sec = self.total_tokens / self.total_latency_sec
        else:
            throughput_tokens_per_sec = float("nan")

        def to_gib(value: Optional[int]) -> float:
            return float("nan") if value is None else float(value) / (1024 ** 3)

        return {
            "avg_latency_ms": avg_latency_ms,
            "throughput_tokens_per_sec": throughput_tokens_per_sec,
            "model_memory_gib": to_gib(self.model_memory_bytes),
            "translator_memory_gib": to_gib(self.translator_memory_bytes),
            "kv_memory_gib": to_gib(self.kv_memory_bytes),
        }


@contextmanager
def temporarily_offload_module(module: Optional[Any], device: str):
    """Temporarily move a module-like object to CPU and restore it afterwards."""
    if module is None:
        yield
        return

    move_to = getattr(module, "to", None)
    if not callable(move_to):
        yield
        return
    if not (torch.cuda.is_available() and str(device).startswith("cuda")):
        yield
        return

    device_obj = torch.device(device)
    device_index = torch.cuda.current_device() if device_obj.index is None else device_obj.index

    try:
        move_to("cpu")
        torch.cuda.synchronize(device_index)
        yield
    finally:
        move_to(device)
        torch.cuda.synchronize(device_index)


class InferenceProfiler:
    def __init__(self, device: str, *, models: Iterable[Any], translator_pool: Optional[Any]) -> None:
        self.device = device
        self.models = list(models)
        self.translator_pool = translator_pool
        self.enabled = torch.cuda.is_available() and str(device).startswith("cuda")
        if self.enabled:
            device_index = torch.device(self.device).index
            self.device_index = torch.cuda.current_device() if device_index is None else device_index
        else:
            self.device_index = None

    def measure(
        self,
        fn: Callable[[], T],
        *,
        tokens: int,
        kv_objects_getter: Optional[Callable[[T], Iterable[Any]]] = None,
    ) -> Tuple[T, Dict[str, Optional[float]]]:
        if self.enabled:
            torch.cuda.synchronize(self.device_index)

        started_at = time.perf_counter()
        result = fn()
        if self.enabled:
            torch.cuda.synchronize(self.device_index)
        latency_sec = time.perf_counter() - started_at

        kv_objects = () if kv_objects_getter is None else tuple(kv_objects_getter(result))
        measured = measure_gpu_memory_breakdown_bytes(
            self.device,
            models=self.models,
            translator_pool=self.translator_pool,
            kv_objects=kv_objects,
        )
        return result, {
            "latency_sec": float(latency_sec),
            "tokens": tokens,
            "model_memory_bytes": None if measured is None else measured.model_bytes,
            "translator_memory_bytes": None if measured is None else measured.translator_bytes,
            "kv_memory_bytes": None if measured is None else measured.kv_bytes,
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
    choices_field: Optional[str] = None
    error_type_field: Optional[str] = None
    corrected_answer_field: Optional[str] = None
    dataset_names: Optional[List[str]] = None
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
        self.max_examples = resolve_max_examples_for_spec(spec, max_examples)
        self.shuffle = shuffle
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        self._cached_multi_config_examples: Optional[List[Dict[str, Any]]] = None

    def _load_dataset(self, dataset_name: Optional[str] = None):
        resolved_dataset_name = self.spec.dataset_name if dataset_name is None else dataset_name
        if resolved_dataset_name is None:
            return load_dataset(
                self.spec.dataset_path,
                split=self.spec.split,
                streaming=self.spec.streaming,
            )
        return load_dataset(
            self.spec.dataset_path,
            resolved_dataset_name,
            split=self.spec.split,
            streaming=self.spec.streaming,
        )

    def _collect_multi_config_examples(self) -> List[Dict[str, Any]]:
        if self._cached_multi_config_examples is not None:
            return self._cached_multi_config_examples
        if not self.spec.dataset_names:
            return []
        if self.spec.streaming:
            raise ValueError("dataset_names multi-config loading does not support streaming datasets.")

        extracted_examples: List[Dict[str, Any]] = []
        for dataset_index, dataset_name in enumerate(self.spec.dataset_names):
            dataset = self._load_dataset(dataset_name)
            if self.shuffle:
                dataset = dataset.shuffle(seed=self.seed + dataset_index)

            subject_examples: List[Dict[str, Any]] = []
            for raw_example in dataset:
                example = dict(raw_example)
                subject_field = self.spec.subject_field or "subject"
                example.setdefault(subject_field, dataset_name)
                qa_pair = extract_question_and_answer(self.spec, example)
                if qa_pair is None:
                    continue
                subject_examples.append(qa_pair)
                if len(subject_examples) >= self.max_examples:
                    break

            extracted_examples.extend(subject_examples)

        if self.shuffle:
            random.Random(self.seed).shuffle(extracted_examples)

        self._cached_multi_config_examples = extracted_examples
        return self._cached_multi_config_examples

    def _iter_multi_config_examples(self):
        for qa_pair in self._collect_multi_config_examples():
            yield qa_pair

    def __len__(self) -> int:
        if self.spec.dataset_names:
            return len(self._collect_multi_config_examples())
        return self.max_examples

    def __iter__(self):
        if self.spec.dataset_names:
            yield from self._iter_multi_config_examples()
            return

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
MMLU_REDUX_MAX_SUBJECT_EXAMPLES = 25
MMLU_REDUX_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]
MMLU_REDUX_CHOICE_MARKERS = ("①", "②", "③", "④")
MMLU_REDUX_LABELS = MMLU_REDUX_CHOICE_MARKERS
MMLU_REDUX_SUBJECT_CATEGORIES = (
    "math",
    "physics",
    "computer science",
    "biology",
    "chemistry",
    "engineering",
    "culture",
    "psychology",
    "politics",
    "economics",
    "geography",
    "philosophy",
    "history",
    "law",
    "health",
    "other",
    "business",
)
MMLU_REDUX_SUBJECT_TO_CATEGORY = {
    "abstract_algebra": "math",
    "anatomy": "health",
    "astronomy": "physics",
    "business_ethics": "business",
    "clinical_knowledge": "health",
    "college_biology": "biology",
    "college_chemistry": "chemistry",
    "college_computer_science": "computer science",
    "college_mathematics": "math",
    "college_medicine": "health",
    "college_physics": "physics",
    "computer_security": "computer science",
    "conceptual_physics": "physics",
    "econometrics": "economics",
    "electrical_engineering": "engineering",
    "elementary_mathematics": "math",
    "formal_logic": "philosophy",
    "global_facts": "other",
    "high_school_biology": "biology",
    "high_school_chemistry": "chemistry",
    "high_school_computer_science": "computer science",
    "high_school_european_history": "history",
    "high_school_geography": "geography",
    "high_school_government_and_politics": "politics",
    "high_school_macroeconomics": "economics",
    "high_school_mathematics": "math",
    "high_school_microeconomics": "economics",
    "high_school_physics": "physics",
    "high_school_psychology": "psychology",
    "high_school_statistics": "math",
    "high_school_us_history": "history",
    "high_school_world_history": "history",
    "human_aging": "health",
    "human_sexuality": "culture",
    "international_law": "law",
    "jurisprudence": "law",
    "logical_fallacies": "philosophy",
    "machine_learning": "computer science",
    "management": "business",
    "marketing": "business",
    "medical_genetics": "health",
    "miscellaneous": "other",
    "moral_disputes": "philosophy",
    "moral_scenarios": "philosophy",
    "nutrition": "health",
    "philosophy": "philosophy",
    "prehistory": "history",
    "professional_accounting": "other",
    "professional_law": "law",
    "professional_medicine": "health",
    "professional_psychology": "psychology",
    "public_relations": "politics",
    "security_studies": "politics",
    "sociology": "culture",
    "us_foreign_policy": "politics",
    "virology": "health",
    "world_religions": "philosophy",
}


def resolve_max_examples_for_spec(spec: HFDatasetSpec, requested_max_examples: int) -> int:
    resolved = int(requested_max_examples)
    if resolved <= 0:
        raise ValueError(f"requested_max_examples must be positive, got {requested_max_examples!r}")
    if spec.answer_mode == "mmlu_redux":
        return min(resolved, MMLU_REDUX_MAX_SUBJECT_EXAMPLES)
    return resolved


def resolve_progress_total_examples(spec: HFDatasetSpec, dataset: IterableDataset, requested_max_examples: int) -> int:
    if spec.answer_mode == "mmlu_redux":
        try:
            return len(dataset)
        except TypeError:
            return len(MMLU_REDUX_SUBJECTS) * resolve_max_examples_for_spec(spec, requested_max_examples)
    return resolve_max_examples_for_spec(spec, requested_max_examples)


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


def get_mmlu_redux_dataset_spec() -> HFDatasetSpec:
    return HFDatasetSpec(
        name_for_log="MMLU-Redux/test",
        dataset_path="edinburgh-dawg/mmlu-redux-2.0",
        dataset_name=None,
        dataset_names=MMLU_REDUX_SUBJECTS,
        split="test",
        answer_mode="mmlu_redux",
        question_field="question",
        choices_field="choices",
        subject_field="subject",
        error_type_field="error_type",
        corrected_answer_field="correct_answer",
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
    get_mmlu_redux_dataset_spec,
]

GEN_QA_SPEC_GROUP_FACTORIES = [
    get_squad_v11_dataset_spec,
    get_newsqa_generation_dataset_spec,
    # get_multinews_generation_dataset_spec,
]

EVAL_SPEC_GROUP_FACTORIES = {
    "logit_qa": LOGIT_QA_SPEC_GROUP_FACTORIES,
    "gen_qa": GEN_QA_SPEC_GROUP_FACTORIES,
}


def build_openwebtext_eval_dataloader(
    model: Model,
    config,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: Optional[int] = None,
    shuffle_buffer: Optional[int] = None,
    seed_offset: int = 10_000,
) -> InfiniteDataLoader:
    dataset = OpenWebTextSequenceStream(
        tokenizer=model.tokenizer,
        sequence_length=config.total_tokens,
        split="train",
        shuffle=shuffle,
        shuffle_buffer=config.shuffle_buffer if shuffle_buffer is None else shuffle_buffer,
        seed=(config.seed if seed is None else seed) + seed_offset,
    )

    def collate_token_ids(examples: List[torch.Tensor]) -> TokenIDs:
        return TokenIDs(torch.stack([torch.as_tensor(example) for example in examples], dim=0), model_id=model.id)

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_token_ids)
    return InfiniteDataLoader(dataloader)


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
        matplotlib.rcParams["savefig.format"] = "pdf"
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
    return _build_openwebtext_tsne_output_dir(output_path) / f"{OPENWEBTEXT_TSNE_FILE_BASENAME}_{safe_edge_id}.pdf"


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
    features_by_edge_and_group: Dict[str, Dict[str, List[np.ndarray]]],
    perplexity: float = 50.0,
    max_iter: int = 1000,
) -> Dict[str, str]:
    try:
        plt, TSNE = _load_openwebtext_tsne_plotting_deps()
    except ModuleNotFoundError as exc:
        logging.warning("Skipping OpenWebText t-SNE plots: %s", exc)
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
            logging.warning(
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
            logging.warning(
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
            logging.warning(
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
        logging.info(
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
        summary[f"{field_prefix}model_memory_gib"] = float(profile_summary.get("model_memory_gib", float("nan")))
        summary[f"{field_prefix}translator_memory_gib"] = float(profile_summary.get("translator_memory_gib", float("nan")))
        summary[f"{field_prefix}kv_memory_gib"] = float(profile_summary.get("kv_memory_gib", float("nan")))

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
    seed_token: TokenIDs,
    max_new_tokens: int,
) -> Tuple[int, PastKeyValues]:
    if max_new_tokens <= 0:
        return 0, past_key_values

    current_token_ids = seed_token
    current_past = past_key_values
    total_generated_tokens = 0

    for _ in range(max_new_tokens):
        ensure_token_ids_model(model, current_token_ids)
        outputs = model(
            input_ids=current_token_ids.as_tensor(),
            past_key_values=current_past,
            use_cache=True,
        )
        next_token = TokenIDs(outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True), model_id=current_token_ids.model_id)
        current_past = outputs.past_key_values
        del outputs
        total_generated_tokens += int(next_token.numel())
        current_token_ids = next_token

    return total_generated_tokens, current_past



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
    evaluate_edge_losses_fn: Callable[..., Tuple[Dict[str, float], Dict[str, Dict[str, Optional[float]]]]],
    build_visualization_pasts_fn: Optional[Callable[..., Dict[str, PastKeyValues]]] = None,
) -> Dict[str, Dict[str, float]]:
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

    eval_dataloaders = {
        node.id: build_openwebtext_eval_dataloader(
            model=ctx.tp.get_model(node.id),
            config=ctx.config,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            seed=seed,
            shuffle_buffer=shuffle_buffer,
        )
        for node in ctx.nodes
    }

    processed_examples = 0
    batch_idx = 0
    while processed_examples < max_examples:
        batch_idx += 1
        past_by_node_id, batches_by_node_id = build_step_pasts_and_batches(ctx, eval_dataloaders)

        batch_examples = min(
            batches_by_node_id[edge.tgt_id][0].shape[0]
            for edge in ctx.edges
        )
        for edge in ctx.edges:
            target_context_token_ids, prompt_token_ids, label_token_ids = batches_by_node_id[edge.tgt_id]
            source_context_token_ids = batches_by_node_id[edge.src_id][0]

            edge_losses, edge_profiles = evaluate_edge_losses_fn(
                edge_id=edge.id,
                edge=edge,
                source_context_token_ids=source_context_token_ids,
                target_context_token_ids=target_context_token_ids,
                prompt_token_ids=prompt_token_ids,
                label_token_ids=label_token_ids,
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
                    model_memory_bytes=profile_values.get("model_memory_bytes"),
                    translator_memory_bytes=profile_values.get("translator_memory_bytes"),
                    kv_memory_bytes=profile_values.get("kv_memory_bytes"),
                )
            counts[edge.id] += batch_examples

            if tsne_features is not None:
                named_pasts = build_visualization_pasts_fn(
                    edge_id=edge.id,
                    edge=edge,
                    source_context_token_ids=source_context_token_ids,
                    target_context_token_ids=target_context_token_ids,
                    prompt_token_ids=prompt_token_ids,
                    label_token_ids=label_token_ids,
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
            logging.info(
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
            loss_field_by_name={"native": "native_loss"},
            loss_delta_reference_name="native",
            profile_summary_by_name=profile_summaries,
            profile_field_prefix_by_name={"native": "native"},
        )

    if tsne_features is not None:
        _finalize_openwebtext_tsne_plots(
            output_path=output_path,
            seed=seed,
            features_by_edge_and_group=tsne_features,
        )
    return summaries


def evaluate_openwebtext_validation_loss_with_context_tokens(
    ctx: Context,
    eval_config: EvalConfig,
    translator_pool,
    *,
    build_translated_target_past_fn,
    build_visualization_pasts_fn=None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate OpenWebText while forwarding source/target context token IDs.

    Cross-tokenizer algorithms need the receiver token sequence to construct a
    source-model KV cache with exactly the receiver's positional length. This
    dedicated evaluator keeps that contract isolated from algorithms that only
    consume precomputed past_key_values.
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
            kv_objects_getter=lambda result: (result[1],),
        )
        del translated_result
        with temporarily_offload_module(translator_pool, train_config.device):
            native_result, native_profile = profiler.measure(
                run_native_inference,
                tokens=profile_tokens,
                kv_objects_getter=lambda result: (result[1],),
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



@torch.inference_mode()
def evaluate_openwebtext_validation_loss_top_layers(
    ctx: Context,
    eval_config: EvalConfig,
    translator_pool,
    *,
    build_translated_target_past_fn: Callable[..., PastKeyValues],
    build_visualization_pasts_fn: Optional[Callable[..., Dict[str, PastKeyValues]]] = None,
) -> Dict[str, Dict[str, float]]:
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
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, Optional[float]]]]:
        profile_tokens = int(label_token_ids.numel())
        seed_token = prompt_token_ids[:, :1]
        generation_steps = int(label_token_ids.shape[1])

        translated_target_past = build_translated_target_past_fn(
            edge=edge,
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
            kv_objects_getter=lambda result: (result[1],),
        )
        del translated_result
        with temporarily_offload_module(translator_pool, train_config.device):
            native_result, native_profile = profiler.measure(
                run_native_inference,
                tokens=profile_tokens,
                kv_objects_getter=lambda result: (result[1],),
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



@torch.inference_mode()
def evaluate_openwebtext_validation_loss_replay(
    ctx: Context,
    eval_config: EvalConfig,
    translator_pool,
    *,
    build_translated_target_past_fn: Callable[..., PastKeyValues],
    build_visualization_pasts_fn: Optional[Callable[..., Dict[str, PastKeyValues]]] = None,
) -> Dict[str, Dict[str, float]]:
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
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, Optional[float]]]]:
        profile_tokens = int(label_token_ids.numel())
        seed_token = prompt_token_ids[:, :1]
        generation_steps = int(label_token_ids.shape[1])

        mixed_target_past_for_loss = build_translated_target_past_fn(
            edge=edge,
            source_context_token_ids=source_context_token_ids,
            target_context_token_ids=target_context_token_ids,
            past_by_node_id=past_by_node_id,
        )
        translated_loss = float(
            compute_prefix_correction_and_suffix_lm_loss(
                target_model=ctx.tp.get_model(edge.tgt_id),
                past_key_values=mixed_target_past_for_loss,
                prompt_token_ids=prompt_token_ids,
                label_token_ids=label_token_ids,
                native_target_past_key_values=past_by_node_id[edge.tgt_id],
                target_layer_indices=ctx.cm.get_tgt_layer_indices(edge.id),
            ).item()
        )
        native_loss = float(
            compute_prefix_correction_and_suffix_lm_loss(
                target_model=ctx.tp.get_model(edge.tgt_id),
                past_key_values=past_by_node_id[edge.tgt_id],
                prompt_token_ids=prompt_token_ids,
                label_token_ids=label_token_ids,
                native_target_past_key_values=past_by_node_id[edge.tgt_id],
                target_layer_indices=ctx.cm.get_tgt_layer_indices(edge.id),
            ).item()
        )

        def run_translated_inference():
            mixed_target_past = build_translated_target_past_fn(
                edge=edge,
                source_context_token_ids=source_context_token_ids,
                target_context_token_ids=target_context_token_ids,
                past_by_node_id=past_by_node_id,
            )
            return run_openwebtext_greedy_inference(
                model=ctx.tp.get_model(edge.tgt_id),
                past_key_values=mixed_target_past,
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
            kv_objects_getter=lambda result: (result[1],),
        )
        del translated_result
        with temporarily_offload_module(translator_pool, train_config.device):
            native_result, native_profile = profiler.measure(
                run_native_inference,
                tokens=profile_tokens,
                kv_objects_getter=lambda result: (result[1],),
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


@torch.inference_mode()
def evaluate_openwebtext_validation_loss(
    ctx: Context,
    eval_config: EvalConfig,
    translator_pool,
    *,
    build_translated_target_past_fn: Optional[Callable[..., PastKeyValues]] = None,
    build_visualization_pasts_fn: Optional[Callable[..., Dict[str, PastKeyValues]]] = None,
) -> Dict[str, Dict[str, float]]:
    if eval_config.alg == "mot":
        return evaluate_openwebtext_validation_loss_replay(
            ctx=ctx,
            eval_config=eval_config,
            translator_pool=translator_pool,
            build_translated_target_past_fn=build_translated_target_past_fn,
            build_visualization_pasts_fn=build_visualization_pasts_fn,
        )
    return evaluate_openwebtext_validation_loss_top_layers(
        ctx=ctx,
        eval_config=eval_config,
        translator_pool=translator_pool,
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

    #     context_field = spec.context_field or "document"
    #     answers_field = spec.answers_field or "summary"

    #     context = normalize_multinews_context_text(example.get(context_field, None))
    #     if context is None:
    #         return []

    #     answer_texts = _normalize_answer_texts(example.get(answers_field, None))
    #     if not answer_texts:
    #         return []

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
        self.max_examples = resolve_max_examples_for_spec(spec, max_examples)
        self.shuffle = shuffle
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        self._cached_multi_config_examples: Optional[List[Dict[str, Any]]] = None

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


class SubjectAccuracyAccumulator:
    def __init__(self, category: str) -> None:
        self.category = category
        self.accuracy_sum = 0.0
        self.native_accuracy_sum = 0.0
        self.count = 0

    def update(self, *, accuracy_value: float, native_accuracy_value: float) -> None:
        self.accuracy_sum += float(accuracy_value)
        self.native_accuracy_sum += float(native_accuracy_value)
        self.count += 1

    def summary(self) -> Dict[str, Any]:
        if self.count <= 0:
            raise ValueError("SubjectAccuracyAccumulator.summary() requires count > 0")
        return {
            "category": self.category,
            "accuracy": self.accuracy_sum / self.count,
            "native_accuracy": self.native_accuracy_sum / self.count,
            "count": self.count,
        }


def build_mmlu_redux_subject_breakdown(
    subject_accumulators_by_edge: Dict[str, Dict[str, SubjectAccuracyAccumulator]],
    *,
    algorithm_name: str,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "algorithm": algorithm_name,
        "edges": {},
    }
    for edge_id, subject_accumulators in subject_accumulators_by_edge.items():
        subject_accuracy: Dict[str, Dict[str, Any]] = {}
        for subject in MMLU_REDUX_SUBJECTS:
            accumulator = subject_accumulators.get(subject)
            if accumulator is None or accumulator.count <= 0:
                raise ValueError(
                    f"MMLU-Redux subject breakdown is incomplete for edge={edge_id}: missing subject={subject}"
                )
            subject_accuracy[subject] = accumulator.summary()

        subject_category_accuracy: Dict[str, Dict[str, Any]] = {}
        for category in MMLU_REDUX_SUBJECT_CATEGORIES:
            subject_rows = [
                (subject, subject_accuracy[subject])
                for subject in MMLU_REDUX_SUBJECTS
                if subject_accuracy[subject]["category"] == category
            ]
            if not subject_rows:
                raise ValueError(f"No MMLU-Redux subjects mapped to category={category}")
            subject_category_accuracy[category] = {
                "accuracy": sum(row["accuracy"] for _, row in subject_rows) / len(subject_rows),
                "native_accuracy": sum(row["native_accuracy"] for _, row in subject_rows) / len(subject_rows),
                "num_subjects_evaluated": len(subject_rows),
                "total_count": sum(int(row["count"]) for _, row in subject_rows),
                "subjects": [subject for subject, _ in subject_rows],
            }

        payload["edges"][edge_id] = {
            "subject_accuracy": subject_accuracy,
            "subject_category_accuracy": subject_category_accuracy,
        }

    return payload


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

    module_name = f"alg.{alg}.train"
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
):
    if eval_config.checkpoint_dir_path is None:
        raise ValueError("EvalConfig.checkpoint_dir_path must be set before build_eval_context.")
    checkpoint_dir_path = eval_config.checkpoint_dir_path
    checkpoint_dir_path_obj = Path(checkpoint_dir_path)
    if not checkpoint_dir_path_obj.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir_path_obj}")

    module_name = f"alg.{alg}.train"
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
        device_override=eval_config.device,
    )

def resolve_latest_checkpoint_dir_for_alg(
    alg: str,
    outputs_path: str = "outputs",
    checkpoint_name: str = "translators",
) -> Path:
    outputs_path_obj = Path(outputs_path)
    candidates = sorted(
        path
        for path in outputs_path_obj.glob(f"{alg}_*")
        if path.is_dir() and (path / checkpoint_name).is_dir()
    )
    if not candidates:
        raise FileNotFoundError(
            f"No translator checkpoint directories found for alg={alg!r} under {outputs_path_obj}"
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


def _normalize_logit_qa_choice_text(text: str) -> str:
    return " ".join(text.strip().lower().split())



def _match_logit_qa_choice(choices: List[str], candidate: str) -> Optional[str]:
    normalized_candidate = _normalize_logit_qa_choice_text(candidate)
    if not normalized_candidate:
        return None

    normalized_choices: List[Tuple[str, str]] = []
    for choice in choices:
        if not isinstance(choice, str):
            continue
        normalized_choice = _normalize_logit_qa_choice_text(choice)
        if normalized_choice:
            normalized_choices.append((choice.strip(), normalized_choice))

    for canonical_choice, normalized_choice in normalized_choices:
        if normalized_candidate == normalized_choice:
            return canonical_choice

    for canonical_choice, normalized_choice in normalized_choices:
        if re.fullmatch(r"[a-z0-9]+(?:\s+[a-z0-9]+)*", normalized_choice):
            pattern = rf"(?<!\w){re.escape(normalized_choice)}(?!\w)"
        else:
            pattern = re.escape(normalized_choice)
        if re.search(pattern, normalized_candidate, flags=re.IGNORECASE):
            return canonical_choice

    return None



def _parse_logit_qa_correct_answers(
    choices: List[str],
    raw_correct_answer: Any,
    *,
    include_original_choice: Optional[str] = None,
) -> List[str]:
    answers: List[str] = []
    if isinstance(include_original_choice, str) and include_original_choice.strip():
        matched_original_choice = _match_logit_qa_choice(choices, include_original_choice)
        if matched_original_choice is not None:
            answers.append(matched_original_choice)

    if isinstance(raw_correct_answer, str) and raw_correct_answer.strip():
        raw_text = raw_correct_answer.strip()
        direct_answer = _match_logit_qa_choice(choices, raw_text)
        if direct_answer is not None:
            answers.append(direct_answer)
        else:
            parts = [
                part.strip()
                for part in re.split(r"(?:,|/|;|\bor\b|\band\b)", raw_text, flags=re.IGNORECASE)
                if part.strip()
            ]
            for part in parts:
                matched_answer = _match_logit_qa_choice(choices, part)
                if matched_answer is not None:
                    answers.append(matched_answer)

    deduped: List[str] = []
    for answer in answers:
        if answer not in deduped:
            deduped.append(answer)
    return deduped



def is_logit_answer_correct(predicted_label: str, gold_answer: Any) -> bool:
    if isinstance(gold_answer, str):
        return predicted_label == gold_answer
    if isinstance(gold_answer, (list, tuple, set)):
        return predicted_label in set(gold_answer)
    return False



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
            "choices": ["yes", "no"],
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
            "choices": ["yes", "no", "maybe"],
            "answer": normalized_answer,
        }

    if spec.answer_mode == "mmlu_redux":
        choices_field = spec.choices_field or "choices"
        error_type_field = spec.error_type_field or "error_type"
        corrected_answer_field = spec.corrected_answer_field or "correct_answer"
        subject_field = spec.subject_field or "subject"

        raw_choice_texts = example.get(choices_field, None)
        if not isinstance(raw_choice_texts, list):
            return None
        choice_texts = [choice.strip() for choice in raw_choice_texts if isinstance(choice, str) and choice.strip()]
        if len(choice_texts) != 4:
            return None

        choices = list(MMLU_REDUX_CHOICE_MARKERS)
        raw_answer_idx = example.get("answer", None)
        if not isinstance(raw_answer_idx, int) or not (0 <= raw_answer_idx < len(choices)):
            return None
        original_choice = choices[raw_answer_idx]

        error_type = str(example.get(error_type_field, "ok") or "ok").strip().lower()
        corrected_answer = example.get(corrected_answer_field, None)
        if error_type == "expert":
            return None

        if error_type == "wrong_groundtruth":
            acceptable_answers = _parse_logit_qa_correct_answers(
                choices,
                corrected_answer,
                include_original_choice=None,
            )
        elif error_type == "multiple_correct_answers":
            acceptable_answers = _parse_logit_qa_correct_answers(
                choices,
                corrected_answer,
                include_original_choice=original_choice,
            )
        elif error_type == "no_correct_answer":
            acceptable_answers = _parse_logit_qa_correct_answers(
                choices,
                corrected_answer,
                include_original_choice=None,
            )
        else:
            acceptable_answers = [original_choice]

        if not acceptable_answers:
            return None

        answer_value: Union[str, List[str]]
        if len(acceptable_answers) == 1:
            answer_value = acceptable_answers[0]
        else:
            answer_value = acceptable_answers

        subject_value = example.get(subject_field, None)
        subject = subject_value.strip() if isinstance(subject_value, str) and subject_value.strip() else None

        return {
            "question": question.strip(),
            "choices": choices,
            "choice_texts": choice_texts,
            "subject": subject,
            "answer": answer_value,
            "error_type": error_type,
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


def format_boolq_question_suffix(question: str) -> str:
    return (
        f"Question: {question.strip()}\n"
        "Answer:"
    )


def prepare_boolq_context_inputs(
    model: Model,
    context: str,
    device: str,
    max_input_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    prefix_text = format_boolq_context_prefix(context=context)
    return prepare_full_text_inputs(
        model=model,
        text=prefix_text,
        device=device,
        max_input_tokens=max_input_tokens,
    )


def prepare_boolq_question_suffix(model: Model, question: str, device: str) -> Dict[str, torch.Tensor]:
    suffix_text = format_boolq_question_suffix(question=question)
    return prepare_cache_text_inputs(model=model, text=suffix_text, device=device)


def format_pubmed_qa_context_prefix(context: str) -> str:
    return (
        "Read the abstract and answer the biomedical research question with yes, no, or maybe only.\n\n"
        f"Abstract: {context.strip()}\n"
    )


def format_pubmed_qa_question_suffix(question: str) -> str:
    return (
        f"Question: {question.strip()}\n"
        "Answer:"
    )


def format_mmlu_redux_question_prefix(question: str, subject: Optional[str] = None) -> str:
    prompt_lines: List[str] = []
    if isinstance(subject, str) and subject.strip():
        pretty_subject = subject.strip().replace("_", " ")
        prompt_lines.append(f"Subject: {pretty_subject}")
    prompt_lines.append(f"Question: {question.strip()}")
    return "\n".join(prompt_lines) + "\n"


def format_mmlu_redux_choices_suffix(
    choices: List[str],
    choice_texts: List[str],
) -> str:
    if len(choice_texts) != len(choices):
        raise ValueError("choices and choice_texts must have the same length.")

    prompt_lines: List[str] = ["Choices:"]
    for choice, choice_text in zip(choices, choice_texts):
        prompt_lines.append(f"{choice.strip()} {choice_text.strip()}")
    prompt_lines.append("Answer:")
    return "\n".join(prompt_lines)


def prepare_pubmed_qa_context_inputs(
    model: Model,
    context: str,
    device: str,
    max_input_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    prefix_text = format_pubmed_qa_context_prefix(context=context)
    return prepare_full_text_inputs(
        model=model,
        text=prefix_text,
        device=device,
        max_input_tokens=max_input_tokens,
    )


def prepare_pubmed_qa_question_suffix(model: Model, question: str, device: str) -> Dict[str, torch.Tensor]:
    suffix = format_pubmed_qa_question_suffix(question=question)
    return prepare_cache_text_inputs(model=model, text=suffix, device=device)


def prepare_mmlu_redux_question_inputs(
    model: Model,
    question: str,
    device: str,
    max_input_tokens: Optional[int] = None,
    subject: Optional[str] = None,
) -> Dict[str, Any]:
    prefix_text = format_mmlu_redux_question_prefix(question=question, subject=subject)
    return prepare_full_text_inputs(
        model=model,
        text=prefix_text,
        device=device,
        max_input_tokens=max_input_tokens,
    )


def prepare_mmlu_redux_choices_suffix(
    model: Model,
    choices: List[str],
    choice_texts: List[str],
    device: str,
) -> Dict[str, torch.Tensor]:
    suffix = format_mmlu_redux_choices_suffix(
        choices=choices,
        choice_texts=choice_texts,
    )
    return prepare_cache_text_inputs(model=model, text=suffix, device=device)


def format_logit_task_prompt(
    question: str,
    choices: Optional[List[str]] = None,
    choice_texts: Optional[List[str]] = None,
    subject: Optional[str] = None,
    context: Optional[str] = None,
    answer_mode: Optional[str] = None,
) -> str:
    question = question.strip()

    if answer_mode == "boolq":
        if not isinstance(context, str) or not context.strip():
            raise ValueError("BoolQ requires passage context.")
        return format_boolq_context_prefix(context=context) + format_boolq_question_suffix(question=question)

    if answer_mode == "pubmed_qa":
        if not isinstance(context, str) or not context.strip():
            raise ValueError("PubMedQA requires abstract context.")
        return format_pubmed_qa_context_prefix(context=context) + format_pubmed_qa_question_suffix(question=question)

    if not choices:
        return f"Question: {question}\nAnswer:"

    if answer_mode == "mmlu_redux":
        if choice_texts is None:
            raise ValueError("MMLU-Redux requires choice_texts.")
        return (
            format_mmlu_redux_question_prefix(question=question, subject=subject)
            + format_mmlu_redux_choices_suffix(
                choices=choices,
                choice_texts=choice_texts,
            )
        )

    prompt_lines: List[str] = [f"Question: {question}", "Choices:"]
    if choice_texts is None:
        for choice in choices:
            prompt_lines.append(choice.strip())
    else:
        if len(choice_texts) != len(choices):
            raise ValueError("choices and choice_texts must have the same length.")
        for choice, choice_text in zip(choices, choice_texts):
            prompt_lines.append(f"{choice.strip()} {choice_text.strip()}")
    prompt_lines.append("Answer:")
    return "\n".join(prompt_lines)


def format_squad_v11_context_prefix(context: str) -> str:
    return (
        "Read the passage and answer the question briefly. Use a short phrase from the passage when possible.\n\n"
        f"Passage: {context.strip()}\n"
    )


def format_squad_v11_question_suffix(question: str) -> str:
    return (
        f"Question: {question.strip()}\n"
        "Answer:"
    )


def prepare_squad_v11_context_inputs(
    model: Model,
    context: str,
    device: str,
    max_input_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    prefix_text = format_squad_v11_context_prefix(context=context)
    return prepare_full_text_inputs(
        model=model,
        text=prefix_text,
        device=device,
        max_input_tokens=max_input_tokens,
    )


def prepare_squad_v11_question_suffix(model: Model, question: str, device: str) -> Dict[str, torch.Tensor]:
    suffix = format_squad_v11_question_suffix(question=question)
    return prepare_cache_text_inputs(model=model, text=suffix, device=device)


def format_multinews_context_prefix(context: str) -> str:
    return (
        "Read the following news articles and write a concise summary.\n\n"
        f"Articles:\n{context.strip()}\n"
    )


def format_multinews_question_suffix(question: str) -> str:
    return (
        f"Task: {question.strip()}\n"
        "Summary:"
    )


def prepare_multinews_context_inputs(
    model: Model,
    context: str,
    device: str,
    max_input_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    prefix_text = format_multinews_context_prefix(context=context)
    return prepare_full_text_inputs(
        model=model,
        text=prefix_text,
        device=device,
        max_input_tokens=max_input_tokens,
    )


def prepare_multinews_question_suffix(model: Model, question: str, device: str) -> Dict[str, torch.Tensor]:
    suffix = format_multinews_question_suffix(question=question)
    return prepare_cache_text_inputs(model=model, text=suffix, device=device)


def format_generation_task_prompt(context: str, question: str) -> str:
    return (
        "Read the passage and answer the question briefly.\n\n"
        f"Context: {context.strip()}\n"
        f"Question: {question.strip()}\n"
        "Answer:"
    )


def prepare_cache_text_inputs(
    model: Model,
    text: str,
    device: str,
    max_input_tokens: Optional[int] = None,
    truncation_side: str = "left",
) -> Dict[str, TokenIDs]:
    tokenized = model.tokenizer(text, return_tensors="pt", add_special_tokens=False)
    token_ids = TokenIDs(tokenized.input_ids, model_id=model.id)
    was_truncated = False

    if max_input_tokens is not None:
        if max_input_tokens < 2:
            raise ValueError("max_input_tokens must be >= 2")
        if token_ids.shape[1] > max_input_tokens:
            was_truncated = True
            if truncation_side == "left":
                token_ids = token_ids[:, -max_input_tokens:]
            elif truncation_side == "right":
                token_ids = token_ids[:, :max_input_tokens]
            else:
                raise ValueError(f"Unsupported truncation_side: {truncation_side}")

    token_ids = token_ids.to(device)
    if token_ids.shape[1] < 2:
        raise ValueError("Cache text must tokenize to at least 2 tokens.")
    prompt_token_ids = token_ids[:, :-1]
    seed_token = token_ids[:, -1:]
    return {
        "text": text,
        "token_ids": token_ids,
        "prompt_token_ids": prompt_token_ids,
        "seed_token": seed_token,
        "was_truncated": was_truncated,
    }


def prepare_logit_task_prompt(
    model: Model,
    question: str,
    device: str,
    choices: Optional[List[str]] = None,
    choice_texts: Optional[List[str]] = None,
    subject: Optional[str] = None,
    context: Optional[str] = None,
    answer_mode: Optional[str] = None,
    max_input_tokens: Optional[int] = None,
    truncation_side: str = "left",
) -> Dict[str, torch.Tensor]:
    prompt_text = format_logit_task_prompt(
        question,
        choices=choices,
        choice_texts=choice_texts,
        subject=subject,
        context=context,
        answer_mode=answer_mode,
    )
    return prepare_cache_text_inputs(
        model=model,
        text=prompt_text,
        device=device,
        max_input_tokens=max_input_tokens,
        truncation_side=truncation_side,
    )


def prepare_generation_task_prompt(model: Model, context: str, question: str, device: str) -> Dict[str, torch.Tensor]:
    prompt_text = format_generation_task_prompt(context=context, question=question)
    return prepare_cache_text_inputs(model=model, text=prompt_text, device=device)


def format_generation_question_suffix(question: str) -> str:
    return (
        f"Question: {question.strip()}\n"
        "Answer:"
    )


def prepare_full_text_inputs(
    model: Model,
    text: str,
    device: str,
    max_input_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    tokenizer_kwargs = {"return_tensors": "pt", "add_special_tokens": False}
    if max_input_tokens is not None:
        if max_input_tokens < 1:
            raise ValueError("max_input_tokens must be >= 1")
        tokenizer_kwargs["truncation"] = True
        tokenizer_kwargs["max_length"] = max_input_tokens
    tokenized = model.tokenizer(text, **tokenizer_kwargs)
    token_ids = TokenIDs(tokenized.input_ids.to(device), model_id=model.id)
    if token_ids.shape[1] < 1:
        raise ValueError("Text must tokenize to at least 1 token.")
    return {
        "text": text,
        "token_ids": token_ids,
        "was_truncated": max_input_tokens is not None and token_ids.shape[1] >= max_input_tokens,
    }


def prepare_generation_question_suffix(model: Model, question: str, device: str) -> Dict[str, torch.Tensor]:
    suffix = format_generation_question_suffix(question=question)
    return prepare_cache_text_inputs(model=model, text=suffix, device=device)


def get_model_context_limit(model: Model) -> int:
    config = getattr(model, "config", None)
    candidates = [
        getattr(config, "n_positions", None),
        getattr(config, "max_position_embeddings", None),
        getattr(config, "n_ctx", None),
    ]
    tokenizer_limit = getattr(model.tokenizer, "model_max_length", None)
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
    *,
    model: Model,
) -> int:
    shared_limit = get_model_context_limit(model)
    suffix = prepare_generation_task_suffix(
        spec=spec,
        model=model,
        question=question,
        device="cpu",
    )
    reserved_tokens = (
        suffix["prompt_token_ids"].shape[1]
        + suffix["seed_token"].shape[1]
        + get_answer_token_budget(eval_config)
    )
    budget = shared_limit - reserved_tokens
    if budget < 16:
        raise ValueError(
            f"Insufficient context budget for {spec.name_for_log}: "
            f"shared_limit={shared_limit}, reserved_tokens={reserved_tokens}"
        )
    return budget


def compute_logit_task_token_budgets(
    ctx: Context,
    spec: HFDatasetSpec,
    question: str,
    eval_config,
    *,
    model: Model,
    choices: Optional[List[str]] = None,
    choice_texts: Optional[List[str]] = None,
    subject: Optional[str] = None,
) -> Dict[str, Optional[int]]:
    shared_limit = get_model_context_limit(model)
    answer_budget = get_answer_token_budget(eval_config)

    if spec.answer_mode == "boolq":
        suffix = prepare_boolq_question_suffix(
            model=model,
            question=question,
            device="cpu",
        )
        reserved_tokens = (
            suffix["prompt_token_ids"].shape[1]
            + suffix["seed_token"].shape[1]
            + answer_budget
        )
        budget = shared_limit - reserved_tokens
        if budget < 16:
            raise ValueError(
                f"Insufficient context budget for {spec.name_for_log}: "
                f"shared_limit={shared_limit}, reserved_tokens={reserved_tokens}"
            )
        return {"max_context_tokens": budget, "max_prefix_tokens": None}

    if spec.answer_mode == "pubmed_qa":
        suffix = prepare_pubmed_qa_question_suffix(
            model=model,
            question=question,
            device="cpu",
        )
        reserved_tokens = (
            suffix["prompt_token_ids"].shape[1]
            + suffix["seed_token"].shape[1]
            + answer_budget
        )
        budget = shared_limit - reserved_tokens
        if budget < 16:
            raise ValueError(
                f"Insufficient context budget for {spec.name_for_log}: "
                f"shared_limit={shared_limit}, reserved_tokens={reserved_tokens}"
            )
        return {"max_context_tokens": budget, "max_prefix_tokens": None}

    if spec.answer_mode == "mmlu_redux":
        if not choices:
            raise ValueError("MMLU-Redux requires choices for prompt budgeting.")
        if not choice_texts:
            raise ValueError("MMLU-Redux requires choice_texts for prompt budgeting.")
        suffix = prepare_mmlu_redux_choices_suffix(
            model=model,
            choices=choices,
            choice_texts=choice_texts,
            device="cpu",
        )
        reserved_tokens = (
            suffix["prompt_token_ids"].shape[1]
            + suffix["seed_token"].shape[1]
            + answer_budget
        )
        budget = shared_limit - reserved_tokens
        if budget < 16:
            raise ValueError(
                f"Insufficient context budget for {spec.name_for_log}: "
                f"shared_limit={shared_limit}, reserved_tokens={reserved_tokens}"
            )
        return {"max_context_tokens": budget, "max_prefix_tokens": None}

    prompt_budget = shared_limit - answer_budget
    if prompt_budget < 16:
        raise ValueError(
            f"Insufficient prompt budget for {spec.name_for_log}: "
            f"shared_limit={shared_limit}, answer_budget={answer_budget}"
        )
    return {"max_context_tokens": None, "max_prefix_tokens": prompt_budget}


def prepare_generation_task_suffix(
    spec: HFDatasetSpec,
    model: Model,
    question: str,
    device: str,
) -> Dict[str, torch.Tensor]:
    if spec.answer_mode in {"squad", "newsqa"}:
        return prepare_squad_v11_question_suffix(
            model=model,
            question=question,
            device=device,
        )
    # if spec.answer_mode == "multinews":
    #     return prepare_multinews_question_suffix(
    #         question=question,
    #         device=device,
    #     )
    return prepare_generation_question_suffix(
        model=model,
        question=question,
        device=device,
    )


def prepare_logit_task_inputs(
    spec: HFDatasetSpec,
    model: Model,
    context: Optional[str],
    question: str,
    device: str,
    choices: Optional[List[str]] = None,
    choice_texts: Optional[List[str]] = None,
    subject: Optional[str] = None,
    max_context_tokens: Optional[int] = None,
    max_prefix_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    if spec.answer_mode == "boolq":
        if not isinstance(context, str) or not context.strip():
            raise ValueError("BoolQ requires passage context.")
        context_prefix = prepare_boolq_context_inputs(
            model=model,
            context=context,
            device=device,
            max_input_tokens=max_context_tokens,
        )
        suffix = prepare_boolq_question_suffix(
            model=model,
            question=question,
            device=device,
        )
        return {
            "prefix": context_prefix,
            "suffix": suffix,
            "context_token_ids": context_prefix["token_ids"],
            "prompt_token_ids": suffix["prompt_token_ids"],
            "seed_token": suffix["seed_token"],
            "was_truncated": bool(context_prefix.get("was_truncated", False)),
        }

    if spec.answer_mode == "pubmed_qa":
        if not isinstance(context, str) or not context.strip():
            raise ValueError("PubMedQA requires abstract context.")
        context_prefix = prepare_pubmed_qa_context_inputs(
            model=model,
            context=context,
            device=device,
            max_input_tokens=max_context_tokens,
        )
        suffix = prepare_pubmed_qa_question_suffix(
            model=model,
            question=question,
            device=device,
        )
        return {
            "prefix": context_prefix,
            "suffix": suffix,
            "context_token_ids": context_prefix["token_ids"],
            "prompt_token_ids": suffix["prompt_token_ids"],
            "seed_token": suffix["seed_token"],
            "was_truncated": bool(context_prefix.get("was_truncated", False)),
        }

    if spec.answer_mode == "mmlu_redux":
        if not choices:
            raise ValueError("MMLU-Redux requires choices.")
        if not choice_texts:
            raise ValueError("MMLU-Redux requires choice_texts.")
        question_prefix = prepare_mmlu_redux_question_inputs(
            model=model,
            question=question,
            device=device,
            max_input_tokens=max_context_tokens,
            subject=subject,
        )
        suffix = prepare_mmlu_redux_choices_suffix(
            model=model,
            choices=choices,
            choice_texts=choice_texts,
            device=device,
        )
        return {
            "prefix": question_prefix,
            "suffix": suffix,
            "context_token_ids": question_prefix["token_ids"],
            "prompt_token_ids": suffix["prompt_token_ids"],
            "seed_token": suffix["seed_token"],
            "was_truncated": bool(question_prefix.get("was_truncated", False)),
        }

    prompt = prepare_logit_task_prompt(
        model=model,
        question=question,
        device=device,
        choices=choices,
        choice_texts=choice_texts,
        subject=subject,
        context=context,
        answer_mode=spec.answer_mode,
        max_input_tokens=max_prefix_tokens,
        truncation_side="left",
    )
    return {
        "context_token_ids": prompt["prompt_token_ids"],
        "prompt_token_ids": None,
        "seed_token": prompt["seed_token"],
        "was_truncated": bool(prompt.get("was_truncated", False)),
    }


def prepare_generation_task_inputs(
    spec: HFDatasetSpec,
    model: Model,
    context: str,
    question: str,
    device: str,
    max_input_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    if spec.answer_mode in {"squad", "newsqa"}:
        context_prefix = prepare_squad_v11_context_inputs(
            model=model,
            context=context,
            device=device,
            max_input_tokens=max_input_tokens,
        )
        suffix = prepare_squad_v11_question_suffix(
            model=model,
            question=question,
            device=device,
        )
        return {
            "prefix": context_prefix,
            "suffix": suffix,
            "context_token_ids": context_prefix["token_ids"],
            "prompt_token_ids": suffix["prompt_token_ids"],
            "seed_token": suffix["seed_token"],
            "was_truncated": context_prefix.get("was_truncated", False),
        }

    # if spec.answer_mode == "multinews":
    #     context_prefix = prepare_multinews_context_inputs(
    #         context=context,
    #         device=device,
    #         max_input_tokens=max_input_tokens,
    #     )
    #     suffix = prepare_multinews_question_suffix(
    #         question=question,
    #         device=device,
    #     )
    #     return {
    #         "prefix": context_prefix,
    #         "suffix": suffix,
    #         "context_token_ids": context_prefix["token_ids"],
    #         "prompt_token_ids": suffix["prompt_token_ids"],
    #         "seed_token": suffix["seed_token"],
    #         "was_truncated": bool(context_prefix.get("was_truncated", False)),
    #     }

    prompt = prepare_generation_task_prompt(
        model=model,
        context=context,
        question=question,
        device=device,
    )
    return {
        "context_token_ids": prompt["prompt_token_ids"],
        "prompt_token_ids": None,
        "seed_token": prompt["seed_token"],
        "was_truncated": False,
    }


def predict_generation_task_answer(
    model: Model,
    past_key_values: PastKeyValues,
    seed_token: TokenIDs,
    eval_config,
    prompt_token_ids: Optional[TokenIDs] = None,
) -> str:
    generation_past = past_key_values
    if prompt_token_ids is not None:
        generation_past = append_token_ids_to_past(
            model=model,
            past_key_values=past_key_values,
            token_ids=prompt_token_ids,
        )

    return generate_greedy_answer(
        model=model,
        past_key_values=generation_past,
        seed_token=seed_token,
        max_new_tokens=eval_config.generation_max_new_tokens,
    )



@torch.inference_mode()
def append_token_ids_to_past(
    model,
    past_key_values: PastKeyValues,
    token_ids: TokenIDs,
) -> PastKeyValues:
    ensure_token_ids_model(model, token_ids)
    if token_ids.shape[1] == 0:
        return past_key_values

    outputs = model(
        input_ids=token_ids.as_tensor(),
        past_key_values=past_key_values,
        use_cache=True,
    )
    return outputs.past_key_values


def build_text_candidate_token_ids(
    model: Model,
    candidates: Dict[str, str],
) -> Dict[str, TokenIDs]:
    token_ids_by_label: Dict[str, TokenIDs] = {}

    for label, text in candidates.items():
        normalized_text = text.strip()
        token_ids = model.tokenizer(
            f" {normalized_text}",
            add_special_tokens=False,
        ).input_ids
        if len(token_ids) < 1:
            raise ValueError(f"Failed to tokenize candidate text for label={label}: {text!r}")
        token_ids_by_label[label] = TokenIDs(torch.tensor(token_ids, dtype=torch.long), model_id=model.id)

    return token_ids_by_label


def build_logit_answer_candidates(
    model: Model,
    spec: HFDatasetSpec,
) -> Dict[str, TokenIDs]:
    if spec.answer_mode == "boolq":
        return build_text_candidate_token_ids(
            model,
            {"yes": "yes", "no": "no"},
        )

    if spec.answer_mode == "pubmed_qa":
        return build_text_candidate_token_ids(
            model,
            {"yes": "yes", "no": "no", "maybe": "maybe"},
        )

    if spec.answer_mode == "mmlu_redux":
        return build_text_candidate_token_ids(
            model,
            {label: label for label in MMLU_REDUX_LABELS},
        )

    raise ValueError(f"Unsupported answer_mode for logit scoring: {spec.answer_mode}")


def score_candidate_logprob(
    model,
    past_key_values,
    seed_token: TokenIDs,
    candidate_token_ids: TokenIDs,
    normalize_by_length: bool = True,
) -> float:
    ensure_token_ids_model(model, seed_token)
    ensure_token_ids_model(model, candidate_token_ids)
    device = seed_token.device
    candidate_ids = candidate_token_ids.as_tensor().to(device).unsqueeze(0)

    if candidate_ids.shape[1] == 1:
        scoring_token_ids = seed_token
    else:
        scoring_token_ids = TokenIDs(torch.cat([seed_token, candidate_ids[:, :-1]], dim=1), model_id=seed_token.model_id)

    outputs = model(
        input_ids=scoring_token_ids.as_tensor(),
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
    seed_token: TokenIDs,
    choice_token_ids: Dict[str, TokenIDs],
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
    prompt_token_ids: Optional[TokenIDs] = None,
) -> PastKeyValues:
    if prompt_token_ids is None:
        return past_key_values
    return append_token_ids_to_past(
        model=model,
        past_key_values=past_key_values,
        token_ids=prompt_token_ids,
    )


def predict_answer_label(choice_scores: Dict[str, float]) -> str:
    return max(choice_scores.items(), key=lambda item: item[1])[0]


def parse_generated_logit_answer(
    spec: HFDatasetSpec,
    prediction: str,
    *,
    choices: Optional[List[str]] = None,
) -> Optional[str]:
    if not choices:
        raise ValueError(f"{spec.answer_mode} generative parsing requires choices.")

    cleaned = postprocess_generated_answer(prediction)
    if not cleaned:
        return None

    direct_answer = _match_logit_qa_choice(choices, cleaned)
    if direct_answer is not None:
        return direct_answer

    parts = [part.strip() for part in re.split(r"[\n\r]|(?:\.\s+)|;|:", cleaned) if part.strip()]
    for part in parts:
        matched_answer = _match_logit_qa_choice(choices, part)
        if matched_answer is not None:
            return matched_answer

    return None


@torch.inference_mode()
def generate_greedy_answer(
    model: Model,
    past_key_values: PastKeyValues,
    seed_token: TokenIDs,
    max_new_tokens: int,
) -> str:
    generated_token_ids: List[int] = []
    current_token_ids = seed_token
    current_past = past_key_values
    eos_token_id = model.tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        ensure_token_ids_model(model, current_token_ids)
        outputs = model(
            input_ids=current_token_ids.as_tensor(),
            past_key_values=current_past,
            use_cache=True,
        )
        next_token = TokenIDs(outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True), model_id=current_token_ids.model_id)
        next_token_id = next_token.item()

        if eos_token_id is not None and next_token_id == eos_token_id:
            break

        generated_token_ids.append(next_token_id)
        current_token_ids = next_token
        current_past = outputs.past_key_values

    decoded = model.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
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


@dataclass
class LogitEvalEdgeArtifacts:
    translated_past_key_values: PastKeyValues
    native_past_key_values: PastKeyValues
    cosine_value: float


@torch.inference_mode()
def evaluate_dataset(
    *,
    ctx: Context,
    spec: HFDatasetSpec,
    dataloader: DataLoader,
    eval_config: EvalConfig,
    translator_pool,
    build_edge_artifacts_fn: Callable[..., LogitEvalEdgeArtifacts],
    prepare_scoring_past_fn: Callable[..., PastKeyValues] = prepare_answer_scoring_past,
    finalize_results_fn: Optional[Callable[..., None]] = None,
    progress_interval: int = 50,
) -> Dict[str, Dict[str, float]]:
    edges = ctx.edges
    device = ctx.config.device
    path_metrics = {edge.id: RunningAverage() for edge in edges}
    subject_accumulators_by_edge: Optional[Dict[str, Dict[str, SubjectAccuracyAccumulator]]]
    if spec.answer_mode == "mmlu_redux":
        subject_accumulators_by_edge = {edge.id: {} for edge in edges}
    else:
        subject_accumulators_by_edge = None

    processed_examples = 0
    progress_total_examples = resolve_progress_total_examples(spec, dataloader.dataset, eval_config.max_examples_per_dataset)
    next_progress_examples = progress_interval if progress_interval > 0 else None

    for batch in dataloader:
        for example in batch:
            question = example["question"]
            gold_answer = example["answer"]
            context_text = example.get("context")
            choices = example.get("choices")
            choice_texts = example.get("choice_texts")

            prepared_inputs_by_node_id = {}
            for node in ctx.nodes:
                model = ctx.tp.get_model(node.id)
                token_budgets = compute_logit_task_token_budgets(
                    ctx=ctx,
                    spec=spec,
                    question=question,
                    eval_config=eval_config,
                    model=model,
                    choices=choices,
                    choice_texts=choice_texts,
                    subject=example.get("subject"),
                )
                prepared_inputs = prepare_logit_task_inputs(
                    spec=spec,
                    model=model,
                    context=context_text,
                    question=question,
                    device=device,
                    choices=choices,
                    choice_texts=choice_texts,
                    subject=example.get("subject"),
                    max_context_tokens=token_budgets["max_context_tokens"],
                    max_prefix_tokens=token_budgets["max_prefix_tokens"],
                )
                prepared_inputs_by_node_id[node.id] = prepared_inputs

                if prepared_inputs.get("was_truncated") and processed_examples < 3:
                    prompt_token_ids = prepared_inputs["prompt_token_ids"]
                    prompt_tokens = 0 if prompt_token_ids is None else prompt_token_ids.shape[1]
                    logging.info(
                        "[%s][%s] truncated context to fit model context window (context_tokens=%d, prompt_tokens=%d, answer_token_budget=%d)",
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
                for node in ctx.nodes
            }

            for edge in edges:
                target_model = ctx.tp.get_model(edge.tgt_id)
                prepared_inputs = prepared_inputs_by_node_id[edge.tgt_id]
                target_context_token_ids = prepared_inputs["context_token_ids"]
                source_context_token_ids = prepared_inputs_by_node_id[edge.src_id]["context_token_ids"]
                prompt_token_ids = prepared_inputs["prompt_token_ids"]
                seed_token = prepared_inputs["seed_token"]

                edge_artifacts = build_edge_artifacts_fn(
                    ctx=ctx,
                    edge=edge,
                    source_context_token_ids=source_context_token_ids,
                    target_context_token_ids=target_context_token_ids,
                    past_by_node_id=past_by_node_id,
                    translator_pool=translator_pool,
                )

                target_model = ctx.tp.get_model(edge.tgt_id)
                translated_generation_past = prepare_scoring_past_fn(
                    model=target_model,
                    past_key_values=edge_artifacts.translated_past_key_values,
                    prompt_token_ids=prompt_token_ids,
                )
                native_generation_past = prepare_scoring_past_fn(
                    model=target_model,
                    past_key_values=edge_artifacts.native_past_key_values,
                    prompt_token_ids=prompt_token_ids,
                )

                translated_answer = generate_greedy_answer(
                    model=target_model,
                    past_key_values=translated_generation_past,
                    seed_token=seed_token,
                    max_new_tokens=eval_config.generation_max_new_tokens,
                )
                native_answer = generate_greedy_answer(
                    model=target_model,
                    past_key_values=native_generation_past,
                    seed_token=seed_token,
                    max_new_tokens=eval_config.generation_max_new_tokens,
                )

                translated_pred = parse_generated_logit_answer(
                    spec,
                    translated_answer,
                    choices=choices,
                )
                native_pred = parse_generated_logit_answer(
                    spec,
                    native_answer,
                    choices=choices,
                )

                acc = 1.0 if translated_pred is not None and is_logit_answer_correct(translated_pred, gold_answer) else 0.0
                native_acc = 1.0 if native_pred is not None and is_logit_answer_correct(native_pred, gold_answer) else 0.0
                path_metrics[edge.id].update(edge_artifacts.cosine_value, acc, native_acc, 1)

                if subject_accumulators_by_edge is not None:
                    subject = example.get("subject")
                    if not isinstance(subject, str) or not subject.strip():
                        raise ValueError("MMLU-Redux examples must include a non-empty subject.")
                    subject = subject.strip()
                    if subject not in MMLU_REDUX_SUBJECT_TO_CATEGORY:
                        raise ValueError(f"Unsupported MMLU-Redux subject: {subject}")
                    subject_accumulator = subject_accumulators_by_edge[edge.id].setdefault(
                        subject,
                        SubjectAccuracyAccumulator(MMLU_REDUX_SUBJECT_TO_CATEGORY[subject]),
                    )
                    subject_accumulator.update(
                        accuracy_value=acc,
                        native_accuracy_value=native_acc,
                    )

            processed_examples += 1
            while next_progress_examples is not None and processed_examples >= next_progress_examples:
                logging.info(
                    "[%s] progress: %d/%d examples",
                    spec.name_for_log,
                    next_progress_examples,
                    progress_total_examples,
                )
                next_progress_examples += progress_interval

    summarized = summarize_path_metrics(path_metrics)
    if subject_accumulators_by_edge is not None:
        breakdown_payload = build_mmlu_redux_subject_breakdown(subject_accumulators_by_edge, algorithm_name=ctx.config.alg)
        breakdown_path = Path(eval_config.output_path) / "mmlu_redux_subject_category_accuracy.json"
        write_json(str(breakdown_path), breakdown_payload)
        logging.info("Saved MMLU-Redux subject/category breakdown to %s", breakdown_path)
    if finalize_results_fn is not None:
        finalize_results_fn(ctx=ctx, spec=spec, results=summarized)
    return summarized


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
    dataset_name: str,
    results: Dict[str, Dict[str, float]],
    nodes: List[Node],
    edges: List[Edge],
) -> None:
    logging.info("===== %s =====", dataset_name)
    for edge in edges:
        row = results[edge.id]
        pretty_name = build_edge_pretty_name(edge.id, nodes, edges)
        logging.info(
            "%s | cosine=%.6f | accuracy=%.6f | native_accuracy=%.6f | count=%d",
            pretty_name,
            row["cosine"],
            row["accuracy"],
            row["native_accuracy"],
            row["count"],
        )


def log_generation_dataset_result(
    dataset_name: str,
    results: Dict[str, Dict[str, float]],
    nodes: List[Node],
    edges: List[Edge],
) -> None:
    logging.info("===== %s =====", dataset_name)
    for edge in edges:
        row = results[edge.id]
        pretty_name = build_edge_pretty_name(edge.id, nodes, edges)
        logging.info(
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

    model_text = _format_summary_float(row.get(f"{field_prefix}model_memory_gib", float("nan")))
    translator_text = _format_summary_float(row.get(f"{field_prefix}translator_memory_gib", float("nan")))
    kv_text = _format_summary_float(row.get(f"{field_prefix}kv_memory_gib", float("nan")))
    if "N/A" in {model_text, translator_text, kv_text}:
        memory_text = "N/A"
    else:
        total = sum(float(value) for value in (model_text, translator_text, kv_text))
        memory_text = (
            f"{total:.6f} GiB total "
            f"(M {model_text} / T {translator_text} / KV {kv_text})"
        )
    return latency_text, throughput_text, memory_text

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
        ("MMLU-Redux", "MMLU-Redux/test"),
    ]
    generation_dataset_keys = [
        ("SQuAD", "SQuAD-v1.1/validation"),
        ("NewsQA", "NewsQA/validation"),
        # ("MultiNews", "MultiNews/validation"),
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
        logit_rows["MMLU-Redux"].get("cosine", float("nan")),
    ])
    translated_accuracy_avg = _summary_mean([
        logit_rows["BoolQ"].get("accuracy", float("nan")),
        logit_rows["PubMedQA"].get("accuracy", float("nan")),
        logit_rows["MMLU-Redux"].get("accuracy", float("nan")),
    ])
    native_accuracy_avg = _summary_mean([
        logit_rows["BoolQ"].get("native_accuracy", float("nan")),
        logit_rows["PubMedQA"].get("native_accuracy", float("nan")),
        logit_rows["MMLU-Redux"].get("native_accuracy", float("nan")),
    ])
    translated_generation_f1_avg = _summary_mean([
        generation_rows["SQuAD"].get("f1", float("nan")),
        generation_rows["NewsQA"].get("f1", float("nan")),
        # generation_rows["MultiNews"].get("f1", float("nan")),
    ])
    native_generation_f1_avg = _summary_mean([
        generation_rows["SQuAD"].get("native_f1", float("nan")),
        generation_rows["NewsQA"].get("native_f1", float("nan")),
        # generation_rows["MultiNews"].get("native_f1", float("nan")),
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
        "| Method | Cosine Sim Avg | BoolQ | PubMedQA | MMLU-Redux | Acc | SQuAD | NewsQA | F1 | Loss | Latency | Throughput | GPU Memory |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {target_model_id} (Native) | N/A | "
            f"{_format_summary_percent(logit_rows['BoolQ'].get('native_accuracy', float('nan')))} | "
            f"{_format_summary_percent(logit_rows['PubMedQA'].get('native_accuracy', float('nan')))} | "
            f"{_format_summary_percent(logit_rows['MMLU-Redux'].get('native_accuracy', float('nan')))} | "
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
            f"{_format_summary_percent(logit_rows['MMLU-Redux'].get('accuracy', float('nan')))} | "
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
