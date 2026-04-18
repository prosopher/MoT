#!/usr/bin/env python3
import argparse
import csv
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.common import (  # noqa: E402
    extract_past_key_values,
    load_frozen_model,
    load_tokenizer,
    parse_bool_arg,
    past_key_values_to_blocks,
    setup_logging,
    write_json,
)
from core.config import resolve_device

PoolMode = Literal["mean", "last"]
MetricName = Literal["linear_cka"]

MODEL_ID_ALIASES: Dict[str, str] = {
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "distilgpt2": "distilgpt2",
    "dialogpt": "microsoft/DialoGPT-small",
    "dialogpt-small": "microsoft/DialoGPT-small",
    "microsoft/dialogpt-small": "microsoft/DialoGPT-small",
    "pythia-70m": "EleutherAI/pythia-70m",
    "eleutherai/pythia-70m": "EleutherAI/pythia-70m",
    "pythia-160m": "EleutherAI/pythia-160m",
    "eleutherai/pythia-160m": "EleutherAI/pythia-160m",
    "mathgpt2": "FlameF0X/MathGPT2",
    "flamef0x/mathgpt2": "FlameF0X/MathGPT2",
}

DATASET_SPECS: Dict[str, Dict[str, object]] = {
    "openwebtext": {
        "hf_path": "openwebtext",
        "hf_name": None,
        "default_split": "train",
        "text_field": "text",
        "streaming": True,
        "filters": {
            "lang_field": None,
            "lang_value": None,
            "role_field": None,
            "allowed_roles": None,
            "deleted_field": None,
            "deleted_keep": None,
            "review_field": None,
            "review_keep": None,
        },
    },
    "oasst1": {
        "hf_path": "OpenAssistant/oasst1",
        "hf_name": None,
        "default_split": "train",
        "text_field": "text",
        "streaming": False,
        "filters": {
            "lang_field": "lang",
            "lang_value": "en",
            "role_field": "role",
            "allowed_roles": ["assistant", "prompter"],
            "deleted_field": "deleted",
            "deleted_keep": False,
            "review_field": "review_result",
            "review_keep": True,
        },
    },
}

PAIR_SUITES: Dict[str, List[Tuple[str, str]]] = {
    "single": [],
    # legacy suites kept for compatibility
    "added_pairs": [
        ("gpt2", "microsoft/DialoGPT-small"),
        ("gpt2", "EleutherAI/pythia-70m"),
    ],
    "recommended": [
        ("gpt2", "gpt2-medium"),
        ("gpt2", "microsoft/DialoGPT-small"),
        ("gpt2", "EleutherAI/pythia-70m"),
    ],
    # dataset-specific suites for the current study
    "openwebtext_main": [
        ("gpt2-medium", "gpt2"),
        ("gpt2-medium", "distilgpt2"),
        ("gpt2-medium", "EleutherAI/pythia-160m"),
        ("gpt2-medium", "microsoft/DialoGPT-small"),
        ("gpt2-medium", "FlameF0X/MathGPT2"),
    ],
    "oasst1_main": [
        ("gpt2-medium", "microsoft/DialoGPT-small"),
    ],
}

DATASET_ALLOWED_SUITES: Dict[str, set] = {
    "openwebtext": {"single", "added_pairs", "recommended", "openwebtext_main"},
    "oasst1": {"single", "oasst1_main"},
}


def normalize_model_id(model_id: str) -> str:
    key = model_id.strip()
    if not key:
        raise ValueError("model id must not be empty")
    return MODEL_ID_ALIASES.get(key.lower(), key)


@dataclass(frozen=True)
class SimpleModelSpec:
    model_id: str
    num_layers: int
    hidden_size: int
    num_heads: int
    head_dim: int
    architecture: str


@dataclass
class LayerSimConfig:
    model_a_id: str = "gpt2"
    model_b_id: str = "gpt2-medium"
    tokenizer_a_id: Optional[str] = None
    tokenizer_b_id: Optional[str] = None
    pair_suite: str = "single"
    include_swapped_pairs: bool = False

    dataset_name: str = "openwebtext"
    output_root: str = "outputs/layer_sim"
    study_id: Optional[str] = None

    split: str = "train"
    num_samples: int = 500
    batch_size: int = 4
    num_workers: int = 0
    prefix_tokens: int = 128
    shuffle_stream: bool = True
    shuffle_buffer: int = 10000
    seed: int = 42
    min_text_chars: int = 256
    max_text_chars: int = 4096

    pool_mode: PoolMode = "mean"
    metric: MetricName = "linear_cka"

    device: str = "cuda"
    dtype: str = "float32"

    figure_dpi: int = 180
    annotate_heatmap: bool = False

    def __post_init__(self) -> None:
        self.model_a_id = normalize_model_id(self.model_a_id)
        self.model_b_id = normalize_model_id(self.model_b_id)
        self.tokenizer_a_id = normalize_model_id(self.tokenizer_a_id or self.model_a_id)
        self.tokenizer_b_id = normalize_model_id(self.tokenizer_b_id or self.model_b_id)
        self.device = resolve_device(self.device)
        self.dataset_name = self.dataset_name.strip().lower()

        if self.dataset_name not in DATASET_SPECS:
            raise ValueError(f"dataset_name must be one of {sorted(DATASET_SPECS)}")
        if self.num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        if self.prefix_tokens < 2:
            raise ValueError("prefix_tokens must be >= 2")
        if self.pool_mode not in {"mean", "last"}:
            raise ValueError("pool_mode must be one of {'mean', 'last'}")
        if self.metric != "linear_cka":
            raise ValueError("metric currently supports only 'linear_cka'")
        if self.figure_dpi < 50:
            raise ValueError("figure_dpi must be >= 50")
        if self.min_text_chars < 1:
            raise ValueError("min_text_chars must be >= 1")
        if self.max_text_chars != 0 and self.max_text_chars < self.min_text_chars:
            raise ValueError("max_text_chars must be 0 or >= min_text_chars")
        if self.pair_suite not in PAIR_SUITES:
            raise ValueError(f"pair_suite must be one of {sorted(PAIR_SUITES)}")
        allowed_suites = DATASET_ALLOWED_SUITES.get(self.dataset_name, {"single"})
        if self.pair_suite not in allowed_suites:
            raise ValueError(
                f"pair_suite '{self.pair_suite}' is not valid for dataset '{self.dataset_name}'. "
                f"Allowed: {sorted(allowed_suites)}"
            )
        if self.study_id is None and self.pair_suite == "single":
            self.study_id = (
                f"{self.dataset_name}__"
                f"{self.model_a_id.replace('/', '_')}__vs__{self.model_b_id.replace('/', '_')}"
            )


class HFTextStream(IterableDataset):
    def __init__(
        self,
        dataset_name: str,
        split: str,
        shuffle: bool,
        shuffle_buffer: int,
        seed: int,
        min_text_chars: int,
        max_text_chars: int,
    ) -> None:
        super().__init__()
        if dataset_name not in DATASET_SPECS:
            raise ValueError(f"Unsupported dataset_name: {dataset_name}")

        self.dataset_name = dataset_name
        self.spec = DATASET_SPECS[dataset_name]
        self.split = split or str(self.spec["default_split"])
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.min_text_chars = min_text_chars
        self.max_text_chars = max_text_chars

    def _is_valid(self, example: dict) -> bool:
        filters = self.spec["filters"]

        lang_field = filters["lang_field"]
        if lang_field and example.get(lang_field) != filters["lang_value"]:
            return False

        role_field = filters["role_field"]
        allowed_roles = filters["allowed_roles"]
        if role_field and allowed_roles is not None:
            if example.get(role_field) not in allowed_roles:
                return False

        deleted_field = filters["deleted_field"]
        if deleted_field is not None and filters["deleted_keep"] is not None:
            if example.get(deleted_field) != filters["deleted_keep"]:
                return False

        review_field = filters["review_field"]
        if review_field is not None and filters["review_keep"] is not None:
            if example.get(review_field) != filters["review_keep"]:
                return False

        return True

    def __iter__(self) -> Iterable[str]:
        ds = load_dataset(
            self.spec["hf_path"],
            self.spec["hf_name"],
            split=self.split,
            streaming=self.spec["streaming"],
        )

        if self.shuffle:
            if self.spec["streaming"]:
                ds = ds.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)
            else:
                ds = ds.shuffle(seed=self.seed)

        for example in ds:
            if not self._is_valid(example):
                continue
            text = str(example.get(self.spec["text_field"], "") or "").strip()
            if not text or len(text) < self.min_text_chars:
                continue
            if self.max_text_chars > 0:
                text = text[: self.max_text_chars]
            if text:
                yield text


class LayerFeatureStore:
    def __init__(self, num_layers: int) -> None:
        self.key_features: List[List[torch.Tensor]] = [[] for _ in range(num_layers)]
        self.value_features: List[List[torch.Tensor]] = [[] for _ in range(num_layers)]
        self.mean_key_block_sum: Optional[torch.Tensor] = None
        self.mean_value_block_sum: Optional[torch.Tensor] = None
        self.num_examples = 0

    def update(self, key_block: torch.Tensor, value_block: torch.Tensor, pool_mode: PoolMode) -> None:
        key_cpu = key_block.detach().to(device="cpu", dtype=torch.float32)
        value_cpu = value_block.detach().to(device="cpu", dtype=torch.float32)
        batch_size = key_cpu.shape[0]

        if self.mean_key_block_sum is None:
            self.mean_key_block_sum = torch.zeros_like(key_cpu[0], dtype=torch.float64)
            self.mean_value_block_sum = torch.zeros_like(value_cpu[0], dtype=torch.float64)

        self.mean_key_block_sum += key_cpu.to(dtype=torch.float64).sum(dim=0)
        self.mean_value_block_sum += value_cpu.to(dtype=torch.float64).sum(dim=0)
        self.num_examples += batch_size

        if pool_mode == "mean":
            pooled_key = key_cpu.mean(dim=1)
            pooled_value = value_cpu.mean(dim=1)
        elif pool_mode == "last":
            pooled_key = key_cpu[:, -1, :, :]
            pooled_value = value_cpu[:, -1, :, :]
        else:
            raise ValueError(f"Unsupported pool_mode: {pool_mode}")

        for layer_idx in range(pooled_key.shape[1]):
            self.key_features[layer_idx].append(pooled_key[:, layer_idx, :].clone())
            self.value_features[layer_idx].append(pooled_value[:, layer_idx, :].clone())

    def finalize(self) -> Dict[str, object]:
        if self.num_examples < 1:
            raise ValueError("No examples were accumulated.")
        return {
            "key_features": [torch.cat(chunks, dim=0) for chunks in self.key_features],
            "value_features": [torch.cat(chunks, dim=0) for chunks in self.value_features],
            "mean_key_block": (self.mean_key_block_sum / float(self.num_examples)).to(dtype=torch.float32),
            "mean_value_block": (self.mean_value_block_sum / float(self.num_examples)).to(dtype=torch.float32),
            "num_examples": self.num_examples,
        }


def get_model_spec_flexible(model) -> SimpleModelSpec:
    config = model.config
    num_heads = getattr(config, "n_head", None)
    if num_heads is None:
        num_heads = getattr(config, "num_attention_heads", None)

    hidden_size = getattr(config, "n_embd", None)
    if hidden_size is None:
        hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(config, "d_model", None)

    num_layers = getattr(config, "n_layer", None)
    if num_layers is None:
        num_layers = getattr(config, "num_hidden_layers", None)
    if num_layers is None:
        num_layers = getattr(config, "n_layers", None)

    if num_heads is None or hidden_size is None or num_layers is None:
        raise ValueError(
            "Could not infer model spec. Expected one of GPT-2/GPT-NeoX-style config fields: "
            "{n_head,num_attention_heads}, {n_embd,hidden_size,d_model}, {n_layer,num_hidden_layers,n_layers}."
        )
    if hidden_size % num_heads != 0:
        raise ValueError(f"hidden_size must be divisible by num_heads, got {hidden_size} and {num_heads}")

    arch = "unknown"
    architectures = getattr(config, "architectures", None)
    if isinstance(architectures, list) and architectures:
        arch = str(architectures[0])

    return SimpleModelSpec(
        model_id=getattr(config, "_name_or_path", "unknown"),
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=hidden_size // num_heads,
        architecture=arch,
    )


@torch.no_grad()
def collect_layer_features(config: LayerSimConfig) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, object]]:
    tokenizer_a = load_tokenizer(config.tokenizer_a_id)
    tokenizer_b = load_tokenizer(config.tokenizer_b_id)
    model_a = load_frozen_model(config.model_a_id, device=config.device, dtype=config.dtype)
    model_b = load_frozen_model(config.model_b_id, device=config.device, dtype=config.dtype)

    spec_a = get_model_spec_flexible(model_a)
    spec_b = get_model_spec_flexible(model_b)
    logging.info(
        "Loaded models: A=%s (layers=%d, hidden=%d, arch=%s), B=%s (layers=%d, hidden=%d, arch=%s)",
        config.model_a_id,
        spec_a.num_layers,
        spec_a.hidden_size,
        spec_a.architecture,
        config.model_b_id,
        spec_b.num_layers,
        spec_b.hidden_size,
        spec_b.architecture,
    )
    logging.info(
        "Loaded tokenizers: A=%s, B=%s (same_tokenizer=%s)",
        config.tokenizer_a_id,
        config.tokenizer_b_id,
        config.tokenizer_a_id == config.tokenizer_b_id,
    )

    dataset = HFTextStream(
        dataset_name=config.dataset_name,
        split=config.split,
        shuffle=config.shuffle_stream,
        shuffle_buffer=config.shuffle_buffer,
        seed=config.seed,
        min_text_chars=config.min_text_chars,
        max_text_chars=config.max_text_chars,
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_workers, collate_fn=list)

    store_a = LayerFeatureStore(num_layers=spec_a.num_layers)
    store_b = LayerFeatureStore(num_layers=spec_b.num_layers)

    processed = 0
    skipped_short = 0
    total_texts_seen = 0
    for batch_idx, texts in enumerate(dataloader, start=1):
        if processed >= config.num_samples:
            break
        total_texts_seen += len(texts)

        encoded_a = tokenizer_a(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=config.prefix_tokens,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        encoded_b = tokenizer_b(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=config.prefix_tokens,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        full_len_a = encoded_a["attention_mask"].sum(dim=1) >= config.prefix_tokens
        full_len_b = encoded_b["attention_mask"].sum(dim=1) >= config.prefix_tokens
        valid_mask = full_len_a & full_len_b
        skipped_short += (~valid_mask).sum().item()
        valid_count = valid_mask.sum().item()
        if valid_count == 0:
            continue

        input_ids_a = encoded_a["input_ids"][valid_mask]
        input_ids_b = encoded_b["input_ids"][valid_mask]
        remaining = config.num_samples - processed
        if input_ids_a.shape[0] > remaining:
            input_ids_a = input_ids_a[:remaining]
            input_ids_b = input_ids_b[:remaining]

        input_ids_a = input_ids_a.to(config.device)
        input_ids_b = input_ids_b.to(config.device)

        past_a = extract_past_key_values(model_a, input_ids_a)
        past_b = extract_past_key_values(model_b, input_ids_b)
        key_a, value_a = past_key_values_to_blocks(past_a)
        key_b, value_b = past_key_values_to_blocks(past_b)

        store_a.update(key_a, value_a, pool_mode=config.pool_mode)
        store_b.update(key_b, value_b, pool_mode=config.pool_mode)
        processed += input_ids_a.shape[0]

        if batch_idx == 1:
            logging.info(
                "First valid batch block shapes: A key=%s value=%s | B key=%s value=%s",
                tuple(key_a.shape),
                tuple(value_a.shape),
                tuple(key_b.shape),
                tuple(value_b.shape),
            )
        if batch_idx % 10 == 0 or processed >= config.num_samples:
            logging.info(
                "Collected %d / %d valid samples on dataset=%s (seen_texts=%d, skipped_short=%d)",
                processed,
                config.num_samples,
                config.dataset_name,
                total_texts_seen,
                skipped_short,
            )

    if processed < config.num_samples:
        logging.warning(
            "Dataset stream ended early. Requested %d samples, collected %d valid samples on dataset=%s "
            "(seen_texts=%d, skipped_short=%d).",
            config.num_samples,
            processed,
            config.dataset_name,
            total_texts_seen,
            skipped_short,
        )

    result_a = store_a.finalize()
    result_b = store_b.finalize()
    metadata = {
        "dataset_name": config.dataset_name,
        "dataset_hf_path": DATASET_SPECS[config.dataset_name]["hf_path"],
        "spec_a": asdict(spec_a),
        "spec_b": asdict(spec_b),
        "num_examples": min(result_a["num_examples"], result_b["num_examples"]),
        "tokenizer_a_id": config.tokenizer_a_id,
        "tokenizer_b_id": config.tokenizer_b_id,
        "total_texts_seen": total_texts_seen,
        "skipped_short_texts": skipped_short,
        "filter_rule": f"Both tokenizers must yield at least {config.prefix_tokens} tokens.",
    }
    return result_a, result_b, metadata


def center_features(x: torch.Tensor) -> torch.Tensor:
    return x.to(dtype=torch.float64) - x.to(dtype=torch.float64).mean(dim=0, keepdim=True)


def linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"linear_cka requires the same number of samples, got {x.shape[0]} and {y.shape[0]}")
    if x.shape[0] < 2:
        return float("nan")

    x_centered = center_features(x)
    y_centered = center_features(y)

    cross_cov = x_centered.transpose(0, 1) @ y_centered
    x_cov = x_centered.transpose(0, 1) @ x_centered
    y_cov = y_centered.transpose(0, 1) @ y_centered

    numerator = torch.linalg.matrix_norm(cross_cov, ord="fro").pow(2)
    denominator = torch.linalg.matrix_norm(x_cov, ord="fro") * torch.linalg.matrix_norm(y_cov, ord="fro")
    score = numerator / denominator.clamp_min(eps)
    return float(score.item())



def compute_similarity_matrix(
    a_layers: Sequence[torch.Tensor],
    b_layers: Sequence[torch.Tensor],
    metric: MetricName,
) -> torch.Tensor:
    matrix = torch.zeros((len(a_layers), len(b_layers)), dtype=torch.float64)
    for layer_a_idx, layer_a_repr in enumerate(a_layers):
        for layer_b_idx, layer_b_repr in enumerate(b_layers):
            if metric == "linear_cka":
                matrix[layer_a_idx, layer_b_idx] = linear_cka(layer_a_repr, layer_b_repr)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
    return matrix.to(dtype=torch.float32)



def build_best_alignment_summary(matrix: torch.Tensor) -> Dict[str, object]:
    per_row = []
    for row_idx in range(matrix.shape[0]):
        best_col = torch.argmax(matrix[row_idx]).item()
        per_row.append(
            {
                "src_layer_idx": row_idx,
                "best_tgt_layer_idx": best_col,
                "score": float(matrix[row_idx, best_col].item()),
            }
        )
    global_flat_idx = torch.argmax(matrix).item()
    global_row = global_flat_idx // matrix.shape[1]
    global_col = global_flat_idx % matrix.shape[1]
    return {
        "best_per_src_layer": per_row,
        "global_best_pair": {
            "src_layer_idx": global_row,
            "tgt_layer_idx": global_col,
            "score": float(matrix[global_row, global_col].item()),
        },
    }



def save_matrix_csv(path: Path, matrix: torch.Tensor, row_prefix: str, col_prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow([""] + [f"{col_prefix}_{idx}" for idx in range(matrix.shape[1])])
        for row_idx in range(matrix.shape[0]):
            writer.writerow([f"{row_prefix}_{row_idx}"] + [f"{float(v):.8f}" for v in matrix[row_idx].tolist()])



def plot_heatmap(
    matrix: torch.Tensor,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    annotate: bool,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    height = max(5.5, 0.42 * matrix.shape[0] + 2.0)
    width = max(7.0, 0.34 * matrix.shape[1] + 2.5)
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(matrix.cpu().numpy(), aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_xticklabels([str(idx) for idx in range(matrix.shape[1])], rotation=45, ha="right")
    ax.set_yticklabels([str(idx) for idx in range(matrix.shape[0])])

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Similarity")

    if annotate and matrix.numel() <= 900:
        values = matrix.cpu().numpy()
        for row_idx in range(values.shape[0]):
            for col_idx in range(values.shape[1]):
                value = values[row_idx, col_idx]
                text_color = "white" if value < 0.5 else "black"
                ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=7)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)



def build_output_dir(config: LayerSimConfig) -> Path:
    output_dir = Path(config.output_root) / config.dataset_name / str(config.study_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir



def run_layer_similarity(config: LayerSimConfig) -> Path:
    output_dir = build_output_dir(config)
    logging.info("Layer similarity config: %s", asdict(config))

    result_a, result_b, metadata = collect_layer_features(config=config)

    key_matrix = compute_similarity_matrix(
        a_layers=result_a["key_features"],
        b_layers=result_b["key_features"],
        metric=config.metric,
    )
    value_matrix = compute_similarity_matrix(
        a_layers=result_a["value_features"],
        b_layers=result_b["value_features"],
        metric=config.metric,
    )
    kv_matrix = 0.5 * (key_matrix + value_matrix)

    save_matrix_csv(output_dir / "key_similarity.csv", key_matrix, row_prefix="a_layer", col_prefix="b_layer")
    save_matrix_csv(output_dir / "value_similarity.csv", value_matrix, row_prefix="a_layer", col_prefix="b_layer")
    save_matrix_csv(output_dir / "kv_similarity.csv", kv_matrix, row_prefix="a_layer", col_prefix="b_layer")

    plot_heatmap(
        matrix=key_matrix,
        title=f"Key cache layer similarity ({config.dataset_name}: {config.model_a_id} vs {config.model_b_id})",
        xlabel=f"{config.model_b_id} layer",
        ylabel=f"{config.model_a_id} layer",
        output_path=output_dir / "key_similarity_heatmap.png",
        annotate=config.annotate_heatmap,
        dpi=config.figure_dpi,
    )
    plot_heatmap(
        matrix=value_matrix,
        title=f"Value cache layer similarity ({config.dataset_name}: {config.model_a_id} vs {config.model_b_id})",
        xlabel=f"{config.model_b_id} layer",
        ylabel=f"{config.model_a_id} layer",
        output_path=output_dir / "value_similarity_heatmap.png",
        annotate=config.annotate_heatmap,
        dpi=config.figure_dpi,
    )
    plot_heatmap(
        matrix=kv_matrix,
        title=f"Mean K/V layer similarity ({config.dataset_name}: {config.model_a_id} vs {config.model_b_id})",
        xlabel=f"{config.model_b_id} layer",
        ylabel=f"{config.model_a_id} layer",
        output_path=output_dir / "kv_similarity_heatmap.png",
        annotate=config.annotate_heatmap,
        dpi=config.figure_dpi,
    )

    torch.save(
        {
            "dataset_name": config.dataset_name,
            "model_id": config.model_a_id,
            "tokenizer_id": config.tokenizer_a_id,
            "mean_key_block": result_a["mean_key_block"],
            "mean_value_block": result_a["mean_value_block"],
        },
        output_dir / "model_a_mean_cache.pt",
    )
    torch.save(
        {
            "dataset_name": config.dataset_name,
            "model_id": config.model_b_id,
            "tokenizer_id": config.tokenizer_b_id,
            "mean_key_block": result_b["mean_key_block"],
            "mean_value_block": result_b["mean_value_block"],
        },
        output_dir / "model_b_mean_cache.pt",
    )

    summary = {
        "config": asdict(config),
        "metadata": metadata,
        "key_alignment": build_best_alignment_summary(key_matrix),
        "value_alignment": build_best_alignment_summary(value_matrix),
        "kv_alignment": build_best_alignment_summary(kv_matrix),
        "artifacts": {
            "key_similarity_csv": str(output_dir / "key_similarity.csv"),
            "value_similarity_csv": str(output_dir / "value_similarity.csv"),
            "kv_similarity_csv": str(output_dir / "kv_similarity.csv"),
            "key_similarity_heatmap_png": str(output_dir / "key_similarity_heatmap.png"),
            "value_similarity_heatmap_png": str(output_dir / "value_similarity_heatmap.png"),
            "kv_similarity_heatmap_png": str(output_dir / "kv_similarity_heatmap.png"),
            "model_a_mean_cache_pt": str(output_dir / "model_a_mean_cache.pt"),
            "model_b_mean_cache_pt": str(output_dir / "model_b_mean_cache.pt"),
        },
    }
    write_json(output_dir / "summary.json", summary)

    logging.info(
        "Global best mean K/V pair on dataset=%s: A layer %d <-> B layer %d (score=%.6f)",
        config.dataset_name,
        summary["kv_alignment"]["global_best_pair"]["src_layer_idx"],
        summary["kv_alignment"]["global_best_pair"]["tgt_layer_idx"],
        summary["kv_alignment"]["global_best_pair"]["score"],
    )
    logging.info("Saved artifacts to %s", output_dir)
    return output_dir



def clone_config_for_pair(base: LayerSimConfig, model_a_id: str, model_b_id: str) -> LayerSimConfig:
    pair_study_id = None
    if base.study_id is not None:
        pair_suffix = f"{normalize_model_id(model_a_id).replace('/', '_')}__vs__{normalize_model_id(model_b_id).replace('/', '_')}"
        pair_study_id = f"{base.study_id}__{pair_suffix}" if base.pair_suite != "single" else base.study_id
    return LayerSimConfig(
        model_a_id=model_a_id,
        model_b_id=model_b_id,
        tokenizer_a_id=None,
        tokenizer_b_id=None,
        pair_suite="single",
        include_swapped_pairs=False,
        dataset_name=base.dataset_name,
        output_root=base.output_root,
        study_id=pair_study_id,
        split=base.split,
        num_samples=base.num_samples,
        batch_size=base.batch_size,
        num_workers=base.num_workers,
        prefix_tokens=base.prefix_tokens,
        shuffle_stream=base.shuffle_stream,
        shuffle_buffer=base.shuffle_buffer,
        seed=base.seed,
        min_text_chars=base.min_text_chars,
        max_text_chars=base.max_text_chars,
        pool_mode=base.pool_mode,
        metric=base.metric,
        device=base.device,
        dtype=base.dtype,
        figure_dpi=base.figure_dpi,
        annotate_heatmap=base.annotate_heatmap,
    )



def build_pair_list(config: LayerSimConfig) -> List[Tuple[str, str]]:
    if config.pair_suite == "single":
        pairs = [(config.model_a_id, config.model_b_id)]
    else:
        pairs = [(normalize_model_id(a), normalize_model_id(b)) for a, b in PAIR_SUITES[config.pair_suite]]

    if config.include_swapped_pairs:
        augmented: List[Tuple[str, str]] = []
        seen = set()
        for pair in pairs:
            for candidate in (pair, (pair[1], pair[0])):
                if candidate not in seen:
                    augmented.append(candidate)
                    seen.add(candidate)
        pairs = augmented
    return pairs



def write_suite_summary(output_root: Path, rows: List[Dict[str, object]]) -> Path:
    path = output_root / "suite_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset_name",
        "model_a_id",
        "model_b_id",
        "study_id",
        "output_dir",
        "key_best_src_layer",
        "key_best_tgt_layer",
        "key_best_score",
        "kv_best_src_layer",
        "kv_best_tgt_layer",
        "kv_best_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure cross-model KV-cache layer similarity on dataset-specific prefills.")
    parser.add_argument("--model-a-id", default="gpt2")
    parser.add_argument("--model-b-id", default="gpt2-medium")
    parser.add_argument("--tokenizer-a-id", default=None)
    parser.add_argument("--tokenizer-b-id", default=None)
    parser.add_argument("--pair-suite", default="single", choices=sorted(PAIR_SUITES.keys()))
    parser.add_argument("--include-swapped-pairs", type=parse_bool_arg, default=False)

    parser.add_argument("--dataset-name", default="openwebtext", choices=sorted(DATASET_SPECS.keys()))
    parser.add_argument("--output-root", default="outputs/layer_sim")
    parser.add_argument("--study-id", default=None)

    parser.add_argument("--split", default="train")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefix-tokens", type=int, default=128)
    parser.add_argument("--shuffle-stream", type=parse_bool_arg, default=True)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-text-chars", type=int, default=256)
    parser.add_argument("--max-text-chars", type=int, default=4096, help="0 disables text truncation before tokenization.")

    parser.add_argument("--pool-mode", default="mean", choices=["mean", "last"])
    parser.add_argument("--metric", default="linear_cka", choices=["linear_cka"])

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32")

    parser.add_argument("--figure-dpi", type=int, default=180)
    parser.add_argument("--annotate-heatmap", type=parse_bool_arg, default=False)
    return parser



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    base_config = LayerSimConfig(**vars(args))

    pair_list = build_pair_list(base_config)
    suite_rows: List[Dict[str, object]] = []
    output_dirs: List[str] = []

    for model_a_id, model_b_id in pair_list:
        pair_config = clone_config_for_pair(base_config, model_a_id=model_a_id, model_b_id=model_b_id)
        setup_logging(build_output_dir(pair_config) / "layer_sim.log")
        output_dir = run_layer_similarity(pair_config)
        output_dirs.append(str(output_dir))

        summary_path = output_dir / "summary.json"
        import json
        with summary_path.open("r", encoding="utf-8") as fp:
            summary = json.load(fp)
        suite_rows.append(
            {
                "dataset_name": pair_config.dataset_name,
                "model_a_id": pair_config.model_a_id,
                "model_b_id": pair_config.model_b_id,
                "study_id": pair_config.study_id,
                "output_dir": str(output_dir),
                "key_best_src_layer": summary["key_alignment"]["global_best_pair"]["src_layer_idx"],
                "key_best_tgt_layer": summary["key_alignment"]["global_best_pair"]["tgt_layer_idx"],
                "key_best_score": f"{summary['key_alignment']['global_best_pair']['score']:.6f}",
                "kv_best_src_layer": summary["kv_alignment"]["global_best_pair"]["src_layer_idx"],
                "kv_best_tgt_layer": summary["kv_alignment"]["global_best_pair"]["tgt_layer_idx"],
                "kv_best_score": f"{summary['kv_alignment']['global_best_pair']['score']:.6f}",
            }
        )

    if len(suite_rows) > 1:
        suite_summary_root = Path(base_config.output_root) / base_config.dataset_name
        suite_summary_path = write_suite_summary(suite_summary_root, suite_rows)
        print(f"Layer similarity outputs: {output_dirs}\nSuite summary: {suite_summary_path}")
    else:
        print(f"Layer similarity outputs: {output_dirs}")


if __name__ == "__main__":
    main()
