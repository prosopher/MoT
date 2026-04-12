import argparse
import json
import logging
import math
import random
import string
from collections import Counter
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from tqdm.auto import tqdm
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union, get_args, get_origin

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .topology import *


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


def setup_logger(name: str, log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = TqdmLoggingHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


PastKeyValues = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


def split_prefix_and_suffix_for_exact_next_token_loss(
    input_ids: torch.Tensor,
    prefix_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if prefix_tokens < 2:
        raise ValueError("prefix_tokens must be >= 2")
    prefix_cache_ids = input_ids[:, : prefix_tokens - 1]
    lm_input_ids = input_ids[:, prefix_tokens - 1 : -1]
    lm_labels = input_ids[:, prefix_tokens:]
    return prefix_cache_ids, lm_input_ids, lm_labels


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
    target_model: PreTrainedModel,
    past_key_values: PastKeyValues,
    lm_input_ids: torch.Tensor,
    lm_labels: torch.Tensor,
) -> torch.Tensor:
    outputs = target_model(
        input_ids=lm_input_ids,
        past_key_values=past_key_values,
        use_cache=False,
    )
    logits = outputs.logits
    vocab_size = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        lm_labels.reshape(-1),
        reduction="mean",
    )


def compute_prefix_correction_and_suffix_lm_loss(
    target_model: PreTrainedModel,
    past_key_values: PastKeyValues,
    lm_input_ids: torch.Tensor,
    lm_labels: torch.Tensor,
    native_target_past_key_values: PastKeyValues,
    target_start_layer_idx: int,
    prefix_correction_weight: float = 1.0,
) -> torch.Tensor:
    if not (0 <= target_start_layer_idx < len(native_target_past_key_values)):
        raise ValueError(
            f"target_start_layer_idx={target_start_layer_idx} must be in [0, {len(native_target_past_key_values) - 1}]"
        )
    if len(past_key_values) != len(native_target_past_key_values):
        raise ValueError(
            "past_key_values and native_target_past_key_values must have the same number of layers, "
            f"got {len(past_key_values)} vs {len(native_target_past_key_values)}"
        )

    suffix_lm_loss = compute_suffix_lm_loss(
        target_model=target_model,
        past_key_values=past_key_values,
        lm_input_ids=lm_input_ids,
        lm_labels=lm_labels,
    )

    mixed_key_block, mixed_value_block = past_key_values_to_blocks(past_key_values[target_start_layer_idx:])
    native_key_block, native_value_block = past_key_values_to_blocks(
        native_target_past_key_values[target_start_layer_idx:]
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


def load_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def freeze_model(model: PreTrainedModel) -> None:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)


def load_frozen_model(model_id: str, device: str, dtype: str = "float32") -> PreTrainedModel:
    torch_dtype = get_torch_dtype(dtype)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
    model.to(device)
    freeze_model(model)
    return model



@torch.no_grad()
def extract_past_key_values(model: PreTrainedModel, input_ids: torch.Tensor) -> PastKeyValues:
    outputs = model(input_ids=input_ids, use_cache=True)
    return outputs.past_key_values


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


class GPUMemoryTracker:
    def __init__(self, device: str) -> None:
        self.device = device
        self.enabled = torch.cuda.is_available() and device.startswith("cuda")
        self.total_allocated_bytes = 0.0
        self.num_samples = 0
        self.peak_allocated_bytes = 0

        if self.enabled:
            self.device_index = torch.device(device).index
            if self.device_index is None:
                self.device_index = torch.cuda.current_device()
            torch.cuda.reset_peak_memory_stats(self.device_index)
        else:
            self.device_index = None

    def update(self) -> None:
        if not self.enabled:
            return

        allocated = torch.cuda.memory_allocated(self.device_index)
        peak = torch.cuda.max_memory_allocated(self.device_index)

        self.total_allocated_bytes += float(allocated)
        self.num_samples += 1
        self.peak_allocated_bytes = max(self.peak_allocated_bytes, peak)

    @property
    def avg_allocated_bytes(self) -> float:
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

        return {
            "enabled": True,
            "avg_allocated_bytes": self.avg_allocated_bytes,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "avg_allocated_pretty": format_memory_gib(self.avg_allocated_bytes),
            "peak_allocated_pretty": format_memory_gib(self.peak_allocated_bytes),
            "num_samples": self.num_samples,
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
