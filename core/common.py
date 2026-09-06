from __future__ import annotations

import argparse
import json
import logging
import math
import random
import string
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from tqdm.auto import tqdm
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type, TypeAlias, TypeVar, Union, get_args, get_origin

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

from .model import Model
from .topology import *


if TYPE_CHECKING:
    from .context import Context
    from .train_util import InfiniteDataLoader


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


def setup_logging(log_path: Union[str, Path]) -> logging.Logger:
    """Configure the process-wide root logger."""
    resolved_log_path = Path(log_path)
    resolved_log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(resolved_log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = TqdmLoggingHandler()
    stream_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    return root_logger



PastKeyValues: TypeAlias = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


class TokenIDs(torch.Tensor):
    model_id: str

    def __new__(
        cls,
        data: Any,
        model_id: str,
    ) -> "TokenIDs":
        tensor = data if isinstance(data, torch.Tensor) else torch.tensor(data)
        token_ids = tensor.as_subclass(cls)
        token_ids.model_id = str(model_id)
        return token_ids

    def as_tensor(self) -> torch.Tensor:
        return self.as_subclass(torch.Tensor)

    def _with_model_id(self, tensor: torch.Tensor) -> "TokenIDs":
        return TokenIDs(tensor, model_id=self.model_id)

    def __getitem__(self, key) -> "TokenIDs":
        return self._with_model_id(super().__getitem__(key))

    def clone(self, *args, **kwargs) -> "TokenIDs":
        return self._with_model_id(super().clone(*args, **kwargs))

    def to(self, *args, **kwargs) -> "TokenIDs":
        return self._with_model_id(super().to(*args, **kwargs))

    def view(self, *shape) -> "TokenIDs":
        return self._with_model_id(super().view(*shape))

    def reshape(self, *shape) -> "TokenIDs":
        return self._with_model_id(super().reshape(*shape))

    def unsqueeze(self, dim: int) -> "TokenIDs":
        return self._with_model_id(super().unsqueeze(dim))

    def squeeze(self, *args) -> "TokenIDs":
        return self._with_model_id(super().squeeze(*args))

    def detach(self) -> "TokenIDs":
        return self._with_model_id(super().detach())

    def cpu(self) -> "TokenIDs":
        return self._with_model_id(super().cpu())


def ensure_token_ids_model(model: Model, token_ids: TokenIDs) -> None:
    if token_ids.model_id != model.id:
        raise ValueError(
            f"TokenIDs model mismatch: model.id={model.id!r}, "
            f"token_ids.model_id={token_ids.model_id!r}"
        )


def split_context_and_prompt_token_ids(
    token_ids: TokenIDs,
    context_tokens: int,
) -> Tuple[TokenIDs, TokenIDs, TokenIDs]:
    if context_tokens < 2:
        raise ValueError("context_tokens must be >= 2")
    context_token_ids = token_ids[:, : context_tokens - 1]
    prompt_token_ids = token_ids[:, context_tokens - 1 : -1]
    label_token_ids = token_ids[:, context_tokens:]
    return context_token_ids, prompt_token_ids, label_token_ids


def build_step_pasts_and_batches(
    ctx: Context,
    dataloaders: Dict[str, InfiniteDataLoader],
) -> Tuple[Dict[str, PastKeyValues], Dict[str, Tuple[TokenIDs, TokenIDs, TokenIDs]]]:
    past_by_node_id: Dict[str, PastKeyValues] = {}
    batches_by_node_id: Dict[str, Tuple[TokenIDs, TokenIDs, TokenIDs]] = {}

    for node in ctx.nodes:
        token_ids = next(dataloaders[node.id]).to(ctx.config.device)
        context_token_ids, prompt_token_ids, label_token_ids = split_context_and_prompt_token_ids(
            token_ids=token_ids,
            context_tokens=ctx.config.prefix_tokens,
        )
        with torch.no_grad():
            past_by_node_id[node.id] = extract_past_key_values(
                ctx.tp.get_model(node.id),
                context_token_ids,
            )
        batches_by_node_id[node.id] = (context_token_ids, prompt_token_ids, label_token_ids)

    return past_by_node_id, batches_by_node_id


class OpenWebTextSequenceStream(IterableDataset):
    """
    Streams OpenWebText and yields fixed-length token chunks.

    This behaves like a rolling token buffer, which is usually a better fit for
    language-model experiments than per-document truncation.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        sequence_length: int,
        split: str = "train",
        shuffle: bool = True,
        shuffle_buffer: int = 10_000,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.split = split
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

    def __iter__(self) -> Iterable[torch.Tensor]:
        stream = load_dataset("openwebtext", split=self.split, streaming=True)
        if self.shuffle:
            stream = stream.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)
        token_buffer: List[int] = []
        eos_token_id = self.tokenizer.eos_token_id
        for example in stream:
            text = example.get("text", "")
            if not text or text.isspace():
                continue
            token_ids = self.tokenizer(text, add_special_tokens=False, verbose=False).input_ids
            if len(token_ids) < 8:
                continue
            token_buffer.extend(token_ids)
            token_buffer.append(eos_token_id)
            while len(token_buffer) >= self.sequence_length:
                chunk = token_buffer[: self.sequence_length]
                token_buffer = token_buffer[self.sequence_length :]
                yield torch.tensor(chunk, dtype=torch.long)


def compute_suffix_lm_loss(
    target_model: Model,
    past_key_values: PastKeyValues,
    prompt_token_ids: TokenIDs,
    label_token_ids: TokenIDs,
) -> torch.Tensor:
    ensure_token_ids_model(target_model, prompt_token_ids)
    ensure_token_ids_model(target_model, label_token_ids)
    outputs = target_model(
        input_ids=prompt_token_ids.as_tensor(),
        past_key_values=past_key_values,
        use_cache=False,
    )
    logits = outputs.logits
    vocab_size = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        label_token_ids.as_tensor().reshape(-1),
        reduction="mean",
    )


def compute_prefix_correction_and_suffix_lm_loss(
    target_model: Model,
    past_key_values: PastKeyValues,
    prompt_token_ids: TokenIDs,
    label_token_ids: TokenIDs,
    native_target_past_key_values: PastKeyValues,
    target_layer_indices: Sequence[int],
    prefix_correction_weight: float = 1.0,
) -> torch.Tensor:
    correction_start_layer_idx = target_layer_indices[0]

    suffix_lm_loss = compute_suffix_lm_loss(
        target_model=target_model,
        past_key_values=past_key_values,
        prompt_token_ids=prompt_token_ids,
        label_token_ids=label_token_ids,
    )

    mixed_key_block, mixed_value_block = past_key_values_to_blocks(past_key_values[correction_start_layer_idx:])
    native_key_block, native_value_block = past_key_values_to_blocks(
        native_target_past_key_values[correction_start_layer_idx:]
    )
    if mixed_key_block.shape != native_key_block.shape:
        raise ValueError(
            "Mixed and native correction key blocks must have the same shape, "
            f"got {tuple(mixed_key_block.shape)} vs {tuple(native_key_block.shape)}"
        )
    if mixed_value_block.shape != native_value_block.shape:
        raise ValueError(
            "Mixed and native correction value blocks must have the same shape, "
            f"got {tuple(mixed_value_block.shape)} vs {tuple(native_value_block.shape)}"
        )

    prefix_correction_loss = (
        F.mse_loss(mixed_key_block, native_key_block, reduction="mean")
        + F.mse_loss(mixed_value_block, native_value_block, reduction="mean")
    )
    return suffix_lm_loss + (prefix_correction_weight * prefix_correction_loss)



def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def get_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = dtype_name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[key]



def _is_qwen2_tokenizer_error(error: Exception) -> bool:
    message = str(error)
    return "Qwen2Tokenizer" in message or "Qwen2TokenizerFast" in message


def _ensure_qwen2_compat_for_old_transformers() -> None:
    try:
        import transformers

        if hasattr(transformers, "Qwen2ForCausalLM"):
            return
    except Exception:
        pass
    from .qwen2_compat import register_qwen2_compat

    register_qwen2_compat()


def _looks_like_qwen2_model_id(model_id: str) -> bool:
    normalized = model_id.lower()
    return "qwen2" in normalized or "qwen2.5" in normalized or "qwen/qwen2" in normalized


def _ensure_qwen3_compat_for_old_transformers() -> None:
    try:
        import transformers

        if hasattr(transformers, "Qwen3ForCausalLM"):
            return
    except Exception:
        pass
    from .qwen3_compat import register_qwen3_compat

    register_qwen3_compat()


def _looks_like_qwen3_model_id(model_id: str) -> bool:
    normalized = model_id.lower()
    return "qwen3" in normalized or "qwen/qwen3" in normalized


def _is_legacy_tokenizers_model_error(error: Exception) -> bool:
    message = str(error)
    return "ModelWrapper" in message or "tokenizer.json" in message and "did not match" in message


def _rewrite_qwen3_tokenizer_json_for_legacy_tokenizers(source_path: str) -> str:
    """Create a tokenizers<0.19-compatible copy of a Qwen3 tokenizer.json.

    New tokenizers releases serialize BPE models with fields/merge-pair shapes
    that tokenizers 0.14-0.15 (the range used with transformers==4.35.2) cannot
    deserialize.  The underlying vocabulary, merge order, pre-tokenizer, decoder,
    and added-token definitions are unchanged, so rewriting only the BPE
    serialization preserves Qwen3 tokenization exactly.
    """
    import os
    import tempfile

    with open(source_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    model = data.get("model")
    if not isinstance(model, dict) or model.get("type") != "BPE":
        raise ValueError(f"Expected a BPE tokenizer in {source_path}")

    # Added to BPE serialization after the tokenizers versions accepted by
    # transformers 4.35.2.  For Qwen3 this is false, matching the old default.
    model.pop("ignore_merges", None)

    # New tokenizers writes merge pairs as [token_a, token_b].  Older releases
    # expect the legacy `token_a token_b` string representation.
    merges = model.get("merges")
    if isinstance(merges, list) and merges and isinstance(merges[0], (list, tuple)):
        legacy_merges = []
        for pair in merges:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("Unexpected Qwen3 BPE merge entry in tokenizer.json")
            legacy_merges.append(f"{pair[0]} {pair[1]}")
        model["merges"] = legacy_merges

    fd, compat_path = tempfile.mkstemp(prefix="mot-qwen3-tokenizer-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        try:
            os.unlink(compat_path)
        except OSError:
            pass
        raise
    return compat_path


def _load_qwen3_tokenizer_with_legacy_tokenizers(model_id: str) -> PreTrainedTokenizerBase:
    import os

    # Import lazily so the project's light-weight transformer stubs remain usable.
    from transformers.utils import cached_file

    tokenizer_json = cached_file(model_id, "tokenizer.json")
    if tokenizer_json is None:
        raise FileNotFoundError(f"tokenizer.json was not found for {model_id}")

    compat_path = _rewrite_qwen3_tokenizer_json_for_legacy_tokenizers(tokenizer_json)
    try:
        # from_pretrained still reads tokenizer_config.json from the Qwen3 repo,
        # so chat_template, EOS/PAD tokens, model_max_length and fixed added-token
        # ids are retained.  Only the backend tokenizer_file is replaced.
        return PreTrainedTokenizerFast.from_pretrained(
            model_id, tokenizer_file=compat_path, trust_remote_code=True
        )
    finally:
        try:
            os.unlink(compat_path)
        except OSError:
            pass


def load_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
    is_qwen3 = _looks_like_qwen3_model_id(model_id)
    if _looks_like_qwen2_model_id(model_id):
        _ensure_qwen2_compat_for_old_transformers()
    elif is_qwen3:
        _ensure_qwen3_compat_for_old_transformers()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except ValueError as error:
        if not _is_qwen2_tokenizer_error(error):
            raise
        # transformers==4.35.x does not ship Qwen2Tokenizer.
        try:
            tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id, trust_remote_code=True)
        except Exception as fast_error:
            if not is_qwen3 or not _is_legacy_tokenizers_model_error(fast_error):
                raise
            tokenizer = _load_qwen3_tokenizer_with_legacy_tokenizers(model_id)
    except Exception as error:
        # Some 4.35.x/tokenizers combinations reach the fast tokenizer directly
        # and fail before AutoTokenizer can report the missing Qwen2Tokenizer.
        if not is_qwen3 or not _is_legacy_tokenizers_model_error(error):
            raise
        tokenizer = _load_qwen3_tokenizer_with_legacy_tokenizers(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def freeze_model(model: PreTrainedModel) -> None:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)


def load_frozen_model(model_id: str, device: str, dtype: str) -> PreTrainedModel:
    if _looks_like_qwen2_model_id(model_id):
        _ensure_qwen2_compat_for_old_transformers()
    elif _looks_like_qwen3_model_id(model_id):
        _ensure_qwen3_compat_for_old_transformers()
    torch_dtype = get_torch_dtype(dtype)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, trust_remote_code=True)
    model.to(device)
    freeze_model(model)
    return model



def get_model_parameter_dtype(model: Model) -> Optional[torch.dtype]:
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return None


def cast_past_key_values_dtype(
    past_key_values: PastKeyValues,
    dtype: Optional[torch.dtype],
) -> PastKeyValues:
    if dtype is None:
        return past_key_values
    return tuple(
        (
            key.to(dtype=dtype) if torch.is_floating_point(key) else key,
            value.to(dtype=dtype) if torch.is_floating_point(value) else value,
        )
        for key, value in past_key_values
    )


@torch.no_grad()
def extract_past_key_values(model: Model, token_ids: TokenIDs) -> PastKeyValues:
    ensure_token_ids_model(model, token_ids)
    outputs = model(input_ids=token_ids.as_tensor(), use_cache=True)
    return cast_past_key_values_dtype(
        outputs.past_key_values,
        get_model_parameter_dtype(model),
    )


def past_key_values_to_blocks(past_key_values: PastKeyValues) -> Tuple[torch.Tensor, torch.Tensor]:
    key_layers = []
    value_layers = []
    for key, value in past_key_values:
        batch_size, num_heads, seq_len, head_dim = key.shape
        key_flat = key.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, num_heads * head_dim)
        value_flat = value.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, num_heads * head_dim)
        key_layers.append(key_flat)
        value_layers.append(value_flat)
    key_block = torch.stack(key_layers, dim=2)
    value_block = torch.stack(value_layers, dim=2)
    return key_block, value_block


def slice_top_layers(
    past_key_values: PastKeyValues,
    top_layers_to_translate: int,
) -> PastKeyValues:
    if top_layers_to_translate < 1:
        raise ValueError("top_layers_to_translate must be >= 1")
    if top_layers_to_translate > len(past_key_values):
        raise ValueError(
            f"Cannot slice {top_layers_to_translate} layers from cache with only {len(past_key_values)} layers."
        )
    return tuple(past_key_values[-top_layers_to_translate:])


def replace_top_layers(
    base_past_key_values: PastKeyValues,
    translated_top_past_key_values: PastKeyValues,
) -> PastKeyValues:
    num_replace = len(translated_top_past_key_values)
    if num_replace < 1:
        raise ValueError("translated_top_past_key_values must contain at least one layer.")
    if num_replace > len(base_past_key_values):
        raise ValueError(
            f"Cannot replace {num_replace} layers in cache with only {len(base_past_key_values)} layers."
        )

    base_list = list(base_past_key_values)
    start_idx = len(base_list) - num_replace

    for offset, translated_layer in enumerate(translated_top_past_key_values):
        base_key, base_value = base_list[start_idx + offset]
        translated_key, translated_value = translated_layer

        if base_key.shape != translated_key.shape:
            raise ValueError(
                f"Key shape mismatch at replaced layer {offset}: "
                f"base={tuple(base_key.shape)} vs translated={tuple(translated_key.shape)}"
            )
        if base_value.shape != translated_value.shape:
            raise ValueError(
                f"Value shape mismatch at replaced layer {offset}: "
                f"base={tuple(base_value.shape)} vs translated={tuple(translated_value.shape)}"
            )

        base_list[start_idx + offset] = (translated_key, translated_value)

    return tuple(base_list)


def flatten_past_key_values(past_key_values: PastKeyValues) -> torch.Tensor:
    flat_parts = []
    for key, value in past_key_values:
        flat_parts.append(key.reshape(key.shape[0], -1))
        flat_parts.append(value.reshape(value.shape[0], -1))
    return torch.cat(flat_parts, dim=1)


def cosine_similarity_between_past(a: PastKeyValues, b: PastKeyValues) -> float:
    flat_a = flatten_past_key_values(a)
    flat_b = flatten_past_key_values(b)
    return F.cosine_similarity(flat_a, flat_b, dim=1).mean().item()


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def format_memory_gib(num_bytes: float) -> str:
    gib = num_bytes / (1024 ** 3)
    return f"{gib:.2f} GiB"


class CurrentProcessGPUMemoryReader:
    def __init__(self, device: str) -> None:
        self.device = device
        self.enabled = torch.cuda.is_available() and device.startswith("cuda")
        if self.enabled:
            device_index = torch.device(device).index
            self.device_index = torch.cuda.current_device() if device_index is None else device_index
        else:
            self.device_index = None

    def read_allocated_bytes(self) -> Optional[int]:
        if not self.enabled:
            return None
        return int(torch.cuda.memory_allocated(self.device_index))


class GPUMemoryTracker:
    def __init__(self, device: str, *, sample_interval_sec: float = 0.02) -> None:
        self.device = device
        self.reader = CurrentProcessGPUMemoryReader(device)
        self.enabled = self.reader.enabled
        self.total_allocated_bytes = 0.0
        self.num_samples = 0
        self.peak_allocated_bytes = 0
        self.sample_interval_sec = max(float(sample_interval_sec), 0.001)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        if self.enabled:
            self._thread = threading.Thread(
                target=self._sample_loop,
                name="gpu-memory-tracker",
                daemon=True,
            )
            self._thread.start()

    def _record_sample(self, allocated: Optional[int]) -> None:
        if allocated is None:
            return
        with self._lock:
            self.total_allocated_bytes += float(allocated)
            self.num_samples += 1
            self.peak_allocated_bytes = max(self.peak_allocated_bytes, int(allocated))

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            self._record_sample(self.reader.read_allocated_bytes())
            self._stop_event.wait(self.sample_interval_sec)

    def update(self) -> None:
        if not self.enabled:
            return
        self._record_sample(self.reader.read_allocated_bytes())

    def close(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=max(1.0, self.sample_interval_sec * 4.0))
        self._thread = None

    @property
    def avg_allocated_bytes(self) -> float:
        with self._lock:
            if self.num_samples == 0:
                return 0.0
            return self.total_allocated_bytes / self.num_samples

    def summary(self) -> Dict[str, object]:
        if not self.enabled:
            return {
                "enabled": False,
                "avg_allocated_bytes": None,
                "peak_allocated_bytes": None,
                "avg_allocated_pretty": "N/A",
                "peak_allocated_pretty": "N/A",
                "num_samples": 0,
            }

        with self._lock:
            avg_allocated_bytes = 0.0 if self.num_samples == 0 else self.total_allocated_bytes / self.num_samples
            peak_allocated_bytes = self.peak_allocated_bytes
            num_samples = self.num_samples

        return {
            "enabled": True,
            "avg_allocated_bytes": avg_allocated_bytes,
            "peak_allocated_bytes": peak_allocated_bytes,
            "avg_allocated_pretty": format_memory_gib(avg_allocated_bytes),
            "peak_allocated_pretty": format_memory_gib(peak_allocated_bytes),
            "num_samples": num_samples,
        }


def read_json(path: Union[str, Path]) -> Dict[str, Any]:
    path_obj = Path(path)
    with path_obj.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    if not isinstance(payload, dict):
        raise ValueError(f"JSON config at {path_obj} must contain a top-level object.")
    return payload


def write_json(path: str, payload: Dict) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)


def build_timestamp_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_timestamped_output_path(
    alg: str,
    outputs_path: str = "outputs",
    timestamp: Optional[str] = None,
) -> Path:
    run_timestamp = timestamp or build_timestamp_string()
    return Path(outputs_path) / f"{alg}_{run_timestamp}"


T = TypeVar("T")


def parse_bool_arg(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _unwrap_optional_type(annotation):
    origin = get_origin(annotation)
    if origin is Union:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _resolve_argparse_type(annotation):
    annotation = _unwrap_optional_type(annotation)
    if annotation is bool:
        return parse_bool_arg
    if annotation in {str, int, float}:
        return annotation
    return str


def add_dataclass_arguments(
    parser: argparse.ArgumentParser,
    config_cls: Type[T],
    exclude_fields: Optional[set[str]] = None,
) -> None:
    if not is_dataclass(config_cls):
        raise TypeError(f"{config_cls} must be a dataclass type.")

    excluded = exclude_fields or set()

    for field_info in fields(config_cls):
        if field_info.name in excluded:
            continue

        option_name = f"--{field_info.name.replace('_', '-')}"
        resolved_type = _unwrap_optional_type(field_info.type)
        if resolved_type is bool:
            parser.add_argument(
                option_name,
                dest=field_info.name,
                nargs="?",
                const=True,
                type=parse_bool_arg,
                default=argparse.SUPPRESS,
            )
            continue

        parser.add_argument(
            option_name,
            dest=field_info.name,
            type=_resolve_argparse_type(field_info.type),
            default=argparse.SUPPRESS,
        )


def extract_dataclass_kwargs_from_namespace(
    config_cls: Type[T],
    args: argparse.Namespace,
    exclude_fields: Optional[set[str]] = None,
) -> Dict[str, Any]:
    if not is_dataclass(config_cls):
        raise TypeError(f"{config_cls} must be a dataclass type.")

    excluded = exclude_fields or set()
    kwargs: Dict[str, Any] = {}

    for field_info in fields(config_cls):
        if field_info.name in excluded:
            continue
        if hasattr(args, field_info.name):
            kwargs[field_info.name] = getattr(args, field_info.name)

    return kwargs


def build_dataclass_kwargs_from_json_and_namespace(
    config_cls: Type[T],
    default_config_path: Union[str, Path],
    args: argparse.Namespace,
    exclude_fields: Optional[set[str]] = None,
) -> Dict[str, Any]:
    if not is_dataclass(config_cls):
        raise TypeError(f"{config_cls} must be a dataclass type.")

    excluded = exclude_fields or set()
    default_kwargs = read_json(default_config_path)

    valid_field_names = {field_info.name for field_info in fields(config_cls)}
    unknown_keys = sorted(set(default_kwargs) - valid_field_names)
    if unknown_keys:
        raise ValueError(
            f"Unknown config keys in {default_config_path}: {unknown_keys}"
        )

    merged_kwargs = {
        key: value
        for key, value in default_kwargs.items()
        if key not in excluded
    }
    merged_kwargs.update(
        extract_dataclass_kwargs_from_namespace(
            config_cls=config_cls,
            args=args,
            exclude_fields=exclude_fields,
        )
    )

    missing_keys = [
        field_info.name
        for field_info in fields(config_cls)
        if field_info.name not in excluded and field_info.name not in merged_kwargs
    ]
    if missing_keys:
        raise ValueError(
            f"Missing required config keys in {default_config_path}: {missing_keys}"
        )

    return merged_kwargs
