from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.channel_manager import ChannelManager
from core.common import GPUMemoryTracker, OpenWebTextSequenceStream, TokenIDs, read_json, set_seed, write_json
from core.config import Config, resolve_device
from core.context import Context
from core.translator_pool import TranslatorPool
from core.topology import Edge, Node, build_edge_map, get_translator_id
from core.train_util import (
    build_models_and_tokenizers,
    get_train_checkpoint_path,
    get_train_config_path,
    get_train_log_path,
    initialize_train_output_paths,
    load_translator_checkpoints,
    save_translator_checkpoints,
)



def _normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    min_value = float(values.min())
    max_value = float(values.max())
    denom = max_value - min_value
    if denom < 1e-9:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - min_value) / denom).astype(np.float32)


def _extract_architecture_signature_from_model(model) -> Tuple[str, str, str, str]:
    config = model.config
    model_class = model.__class__.__name__
    config_class = config.__class__.__name__
    model_type = str(getattr(config, "model_type", "unknown"))
    architectures = getattr(config, "architectures", None) or []
    architecture_name = str(architectures[0]) if architectures else model_class
    return model_class, config_class, model_type, architecture_name


def inspect_kvcomm_model_compatibility(ctx: Context) -> Dict[str, object]:
    inspected_models: List[Dict[str, object]] = []

    for node in ctx.nodes:
        model = ctx.tp.get_model(node.id)
        spec = ctx.tp.get_model_spec(node.id)
        model_class, config_class, model_type, architecture_name = _extract_architecture_signature_from_model(model)
        inspected_models.append(
            {
                "node_id": node.id,
                "model_id": node.model_id,
                "model_class": model_class,
                "config_class": config_class,
                "model_type": model_type,
                "architecture_name": architecture_name,
                "num_layers": int(spec.num_layers),
            }
        )

    reference = inspected_models[0]
    mismatch_messages: List[str] = []
    for item in inspected_models[1:]:
        architecture_matches = (
            item["model_class"] == reference["model_class"]
            and item["config_class"] == reference["config_class"]
            and item["model_type"] == reference["model_type"]
            and item["architecture_name"] == reference["architecture_name"]
        )
        if not architecture_matches:
            mismatch_messages.append(
                "architecture mismatch: "
                f"{reference['model_id']} -> ({reference['model_class']}, {reference['config_class']}, {reference['model_type']}, {reference['architecture_name']}) vs "
                f"{item['model_id']} -> ({item['model_class']}, {item['config_class']}, {item['model_type']}, {item['architecture_name']})"
            )
        if int(item["num_layers"]) != int(reference["num_layers"]):
            mismatch_messages.append(
                "layer-count mismatch: "
                f"{reference['model_id']} -> {reference['num_layers']} layers vs "
                f"{item['model_id']} -> {item['num_layers']} layers"
            )

    if mismatch_messages:
        message = (
            "KVComm requires all model_ids in the config to share the same architecture and the same number of layers. "
            f"Configured model_ids='{ctx.config.model_ids}' are incompatible: "
            + "; ".join(mismatch_messages)
        )
    else:
        message = (
            "KVComm model_ids are compatible: "
            + ", ".join(
                f"{item['model_id']} ({item['architecture_name']}, {item['num_layers']} layers)"
                for item in inspected_models
            )
        )

    return {
        "is_compatible": len(mismatch_messages) == 0,
        "message": message,
        "inspected_models": inspected_models,
    }


@dataclass
class TrainConfig(Config):
    model_ids: str
    model_directions: str
    calibration_dataset: str

    layers_list: List[int]
    top_layers: float
    calib_size: int
    alpha: float
    mu: float
    sigma: float
    random_selection: bool
    shift_back: bool

    max_input_length: int
    seed: int
    dtype: str
    log_level: str = "INFO"
    total_tokens: int = 128
    prefix_tokens: int = 64

    def __post_init__(self) -> None:
        super().__post_init__()
        self.calibration_dataset = str(self.calibration_dataset).strip()
        if not self.calibration_dataset:
            raise ValueError("calibration_dataset must be a non-empty string")
        self.layers_list = [int(x) for x in self.layers_list]
        if self.calib_size < 1:
            raise ValueError("calib_size must be >= 1")
        if not (0.0 <= float(self.top_layers) <= 1.0):
            raise ValueError("top_layers must be in [0, 1]")
        if not (0.0 <= float(self.alpha) <= 1.0):
            raise ValueError("alpha must be in [0, 1]")
        if self.sigma <= 0.0:
            raise ValueError("sigma must be > 0")
        if self.max_input_length < 8:
            raise ValueError("max_input_length must be >= 8")
        initialize_train_output_paths(self)


@dataclass(frozen=True)
class EdgeCalibrationResult:
    selected_target_layers: List[int]
    selected_source_layers: List[int]
    layer_ranking: Optional[List[int]]
    calibration_score: Optional[float]
    attention_importance: Optional[List[float]]


class KVCommSelectionTranslator(nn.Module):
    def __init__(
        self,
        selected_target_layers: List[int],
        selected_source_layers: Optional[List[int]] = None,
        layer_ranking: Optional[List[int]] = None,
        calibration_score: Optional[float] = None,
        attention_importance: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "selected_target_layers_tensor",
            torch.tensor([int(layer_idx) for layer_idx in selected_target_layers], dtype=torch.long),
        )
        self.register_buffer(
            "selected_source_layers_tensor",
            torch.tensor([int(layer_idx) for layer_idx in (selected_source_layers or [])], dtype=torch.long),
        )
        self.register_buffer(
            "layer_ranking_tensor",
            torch.tensor([int(layer_idx) for layer_idx in (layer_ranking or [])], dtype=torch.long),
        )
        score = float("nan") if calibration_score is None else float(calibration_score)
        self.register_buffer("calibration_score_tensor", torch.tensor(score, dtype=torch.float32))
        self.register_buffer(
            "attention_importance_tensor",
            torch.tensor([float(value) for value in (attention_importance or [])], dtype=torch.float32),
        )

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        for name in (
            "selected_target_layers_tensor",
            "selected_source_layers_tensor",
            "layer_ranking_tensor",
            "calibration_score_tensor",
            "attention_importance_tensor",
        ):
            if name in state_dict:
                setattr(self, name, state_dict[name].detach().clone())
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    @property
    def selected_target_layers(self) -> List[int]:
        return [int(value) for value in self.selected_target_layers_tensor.detach().cpu().tolist()]

    @property
    def selected_source_layers(self) -> List[int]:
        return [int(value) for value in self.selected_source_layers_tensor.detach().cpu().tolist()]

    @property
    def layer_ranking(self) -> Optional[List[int]]:
        values = [int(value) for value in self.layer_ranking_tensor.detach().cpu().tolist()]
        return values if values else None

    @property
    def calibration_score(self) -> Optional[float]:
        value = float(self.calibration_score_tensor.detach().cpu().item())
        return None if math.isnan(value) else value

    @property
    def attention_importance(self) -> Optional[List[float]]:
        values = [float(value) for value in self.attention_importance_tensor.detach().cpu().tolist()]
        return values if values else None


def source_to_target_layer_map(num_source_layers: int, num_target_layers: int) -> Dict[int, int]:
    return {
        src_idx: int(min(num_target_layers - 1, max(0, round((src_idx + 0.5) * num_target_layers / num_source_layers - 0.5))))
        for src_idx in range(num_source_layers)
    }


def target_to_source_layer_map(num_source_layers: int, num_target_layers: int) -> Dict[int, int]:
    return {
        tgt_idx: int(min(num_source_layers - 1, max(0, round((tgt_idx + 0.5) * num_source_layers / num_target_layers - 0.5))))
        for tgt_idx in range(num_target_layers)
    }


def validate_edge_compatibility(ctx: Context, edge: Edge) -> None:
    src_spec = ctx.tp.get_model_spec(edge.src_id)
    tgt_spec = ctx.tp.get_model_spec(edge.tgt_id)
    mismatches = []
    if src_spec.hidden_size != tgt_spec.hidden_size:
        mismatches.append(f"hidden_size {src_spec.hidden_size} != {tgt_spec.hidden_size}")
    if src_spec.num_heads != tgt_spec.num_heads:
        mismatches.append(f"num_heads {src_spec.num_heads} != {tgt_spec.num_heads}")
    if src_spec.head_dim != tgt_spec.head_dim:
        mismatches.append(f"head_dim {src_spec.head_dim} != {tgt_spec.head_dim}")
    if mismatches:
        raise ValueError(
            "KVComm requires source/target models with matching KV geometry. "
            f"Edge {edge.id} is incompatible: {', '.join(mismatches)}"
        )


def validate_all_edges(ctx: Context) -> None:
    for edge in ctx.edges:
        validate_edge_compatibility(ctx, edge)


def initialize_translators(
    ctx: Context,
    *,
    selected_target_layers_by_translator_id: Dict[str, List[int]],
    selected_source_layers_by_translator_id: Optional[Dict[str, List[int]]] = None,
    layer_ranking_by_translator_id: Optional[Dict[str, Optional[List[int]]]] = None,
    calibration_score_by_translator_id: Optional[Dict[str, Optional[float]]] = None,
    attention_importance_by_translator_id: Optional[Dict[str, Optional[List[float]]]] = None,
) -> TranslatorPool:
    translator_pool = ctx.tp
    translator_pool.kvcomm_node_model_ids = {node.id: node.model_id for node in ctx.nodes}
    translator_pool.kvcomm_edge_map = build_edge_map(ctx.edges)
    validate_all_edges(ctx)
    selected_source_layers_by_translator_id = selected_source_layers_by_translator_id or {}
    layer_ranking_by_translator_id = layer_ranking_by_translator_id or {}
    calibration_score_by_translator_id = calibration_score_by_translator_id or {}
    attention_importance_by_translator_id = attention_importance_by_translator_id or {}
    for translator_id, target_layers in selected_target_layers_by_translator_id.items():
        translator_pool.add_translator(
            translator_id,
            KVCommSelectionTranslator(
                selected_target_layers=target_layers,
                selected_source_layers=selected_source_layers_by_translator_id.get(translator_id, []),
                layer_ranking=layer_ranking_by_translator_id.get(translator_id),
                calibration_score=calibration_score_by_translator_id.get(translator_id),
                attention_importance=attention_importance_by_translator_id.get(translator_id),
            ),
        )
    return translator_pool


def _get_kvcomm_edge(ctx: Context, translator_pool: TranslatorPool, edge_id: Optional[str], src_node_id: Optional[str], tgt_node_id: Optional[str]) -> Edge:
    edge_map = getattr(translator_pool, "kvcomm_edge_map", None) or build_edge_map(ctx.edges)
    if edge_id is None:
        if src_node_id is None or tgt_node_id is None:
            raise ValueError("Either edge_id or both src_node_id/tgt_node_id must be provided.")
        edge_id = f"{src_node_id}_to_{tgt_node_id}"
    return edge_map[edge_id]


def get_kvcomm_translator_id(translator_pool: TranslatorPool, edge: Edge) -> str:
    node_model_ids = getattr(translator_pool, "kvcomm_node_model_ids", None)
    if node_model_ids is None:
        raise ValueError("KVComm translator metadata has not been initialized.")
    return get_translator_id(node_model_ids[edge.src_id], node_model_ids[edge.tgt_id])


def get_selected_target_layers(ctx: Context, translator_pool: TranslatorPool, edge_id: str) -> List[int]:
    edge = _get_kvcomm_edge(ctx, translator_pool, edge_id, None, None)
    translator_id = get_kvcomm_translator_id(translator_pool, edge)
    return list(translator_pool.translators[translator_id].selected_target_layers)


def get_selected_source_layers(ctx: Context, translator_pool: TranslatorPool, edge_id: str) -> List[int]:
    edge = _get_kvcomm_edge(ctx, translator_pool, edge_id, None, None)
    translator_id = get_kvcomm_translator_id(translator_pool, edge)
    translator = translator_pool.translators[translator_id]
    if translator.selected_source_layers:
        return list(translator.selected_source_layers)
    src_spec = ctx.tp.get_model_spec(edge.src_id)
    tgt_spec = ctx.tp.get_model_spec(edge.tgt_id)
    tgt_to_src = target_to_source_layer_map(src_spec.num_layers, tgt_spec.num_layers)
    return sorted({tgt_to_src[idx] for idx in translator.selected_target_layers})


def build_replayed_target_past(
    ctx: Context,
    translator_pool: TranslatorPool,
    *,
    source_past_key_values,
    edge_id: Optional[str] = None,
    src_node_id: Optional[str] = None,
    tgt_node_id: Optional[str] = None,
    **_: object,
):
    edge = _get_kvcomm_edge(ctx, translator_pool, edge_id, src_node_id, tgt_node_id)
    translator_id = get_kvcomm_translator_id(translator_pool, edge)
    src_spec = ctx.tp.get_model_spec(edge.src_id)
    tgt_spec = ctx.tp.get_model_spec(edge.tgt_id)
    tgt_to_src = target_to_source_layer_map(src_spec.num_layers, tgt_spec.num_layers)
    selected_target_layers = set(translator_pool.translators[translator_id].selected_target_layers)
    replayed = []
    for tgt_layer_idx in range(tgt_spec.num_layers):
        src_layer_idx = tgt_to_src[tgt_layer_idx]
        key_layer, value_layer = source_past_key_values[src_layer_idx]
        if tgt_layer_idx in selected_target_layers or tgt_layer_idx == 0:
            replayed.append((key_layer, value_layer))
        else:
            replayed.append((key_layer[:, :, :1, :].contiguous(), value_layer[:, :, :1, :].contiguous()))
    return tuple(replayed)


def load_kvcomm_translator_checkpoints(ctx: Context) -> TranslatorPool:
    node_model_ids = {node.id: node.model_id for node in ctx.nodes}
    selected_target_layers_by_translator_id = {}
    for edge in ctx.edges:
        translator_id = get_translator_id(node_model_ids[edge.src_id], node_model_ids[edge.tgt_id])
        if translator_id in selected_target_layers_by_translator_id:
            continue
        selected_target_layers_by_translator_id[translator_id] = [0]
    translator_pool = initialize_translators(ctx, selected_target_layers_by_translator_id=selected_target_layers_by_translator_id)
    load_translator_checkpoints(ctx.config.output_path, translator_pool)
    return translator_pool


def _is_openwebtext_dataset(dataset_name: str) -> bool:
    return str(dataset_name).strip().lower() == "openwebtext"


def _openwebtext_total_tokens(config) -> int:
    return max(8, int(getattr(config, "total_tokens", 128)))

def _openwebtext_prefix_tokens(config) -> int:
    total_tokens = _openwebtext_total_tokens(config)
    return max(2, min(total_tokens - 1, int(getattr(config, "prefix_tokens", total_tokens // 2))))


def _resolve_candidate_target_layers(
    *,
    config: TrainConfig,
    target_num_layers: int,
) -> Tuple[List[int], Optional[List[int]], int]:
    manual_layers = list(config.layers_list) if config.layers_list and config.layers_list[0] != -1 else None
    if manual_layers is not None:
        selected = sorted({int(x) for x in manual_layers})
        invalid = [layer_idx for layer_idx in selected if not (0 <= layer_idx < target_num_layers)]
        if invalid:
            raise ValueError(
                f"Manual layers {invalid} are outside target layer range [0, {target_num_layers - 1}]"
            )
        return selected, manual_layers, len(selected)

    # KVComm selects from the full target layer range by default.  The original
    # method defines selection over {1, ..., L}; here we resolve that range only
    # after the target model depth is known, avoiding model-specific config such
    # as GPT-2's last layer index (11).
    candidate_layers = list(range(target_num_layers))
    if not candidate_layers:
        raise ValueError("No candidate target layers were resolved for KVComm selection.")

    if config.top_layers <= 0.0:
        num_layers_to_select = len(candidate_layers)
    else:
        num_layers_to_select = max(1, int(round(config.top_layers * len(candidate_layers))))
        num_layers_to_select = min(num_layers_to_select, len(candidate_layers))

    return candidate_layers, None, num_layers_to_select


@torch.inference_mode()
def _build_openwebtext_calibration_batches(
    *,
    ctx: Context,
    config: TrainConfig,
    tokenizer,
) -> List[torch.Tensor]:
    if not _is_openwebtext_dataset(config.calibration_dataset):
        raise ValueError(
            f"Unsupported calibration_dataset for KVComm layer selection: {config.calibration_dataset}"
        )

    dataset = OpenWebTextSequenceStream(
        tokenizer=tokenizer,
        sequence_length=_openwebtext_total_tokens(config),
        split="train",
        shuffle=True,
        shuffle_buffer=max(1_000, config.calib_size * 8),
        seed=config.seed,
    )
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    batches: List[torch.Tensor] = []
    collected = 0
    for batch in dataloader:
        if batch.dim() == 1:
            batch = batch.unsqueeze(0)
        batches.append(batch)
        collected += int(batch.shape[0])
        if collected >= config.calib_size:
            break
    return batches


@torch.inference_mode()
def _compute_attention_importance_for_batch(
    *,
    model,
    token_ids: TokenIDs,
    context_length: int,
) -> Optional[np.ndarray]:
    query_length = int(token_ids.shape[1]) - context_length
    if context_length < 1 or query_length < 1:
        return None

    outputs = model(
        input_ids=token_ids,
        use_cache=False,
        output_attentions=True,
    )
    attentions = getattr(outputs, "attentions", None)
    if attentions is None:
        return None

    layer_scores: List[float] = []
    for layer_attn in attentions:
        if layer_attn is None:
            layer_scores.append(float("nan"))
            continue
        suffix_to_prefix = layer_attn[:, :, context_length:, :context_length]
        if suffix_to_prefix.numel() == 0:
            layer_scores.append(float("nan"))
            continue
        layer_scores.append(float(suffix_to_prefix.sum(dim=-1).mean().item()))
    return np.asarray(layer_scores, dtype=np.float32)


def _rank_layers_from_scores(
    *,
    raw_scores: np.ndarray,
    alpha: float,
    mu: float,
    sigma: float,
) -> Tuple[List[int], List[float]]:
    num_layers = int(raw_scores.shape[0])
    normalized_scores = _normalize(raw_scores)
    center = float(mu) * max(0, num_layers - 1)
    positions = np.arange(num_layers, dtype=np.float32)
    gaussian = np.exp(-0.5 * ((positions - center) / float(sigma)) ** 2)
    gaussian = _normalize(gaussian)
    combined = alpha * normalized_scores + (1.0 - alpha) * gaussian
    ranking = np.argsort(combined)[::-1].astype(int).tolist()
    return ranking, combined.astype(float).tolist()


def _select_layers_for_edge(
    *,
    ctx: Context,
    edge: Edge,
    config: TrainConfig,
    calibration_batches: List[torch.Tensor],
) -> EdgeCalibrationResult:
    target_spec = ctx.tp.get_model_spec(edge.tgt_id)
    source_spec = ctx.tp.get_model_spec(edge.src_id)
    validate_edge_compatibility(ctx, edge)

    candidate_layers, manual_layers, num_layers_to_select = _resolve_candidate_target_layers(
        config=config,
        target_num_layers=target_spec.num_layers,
    )
    tgt_to_src = target_to_source_layer_map(source_spec.num_layers, target_spec.num_layers)

    if manual_layers is not None:
        selected_target_layers = candidate_layers
        selected_source_layers = sorted({tgt_to_src[layer_idx] for layer_idx in selected_target_layers})
        return EdgeCalibrationResult(
            selected_target_layers=selected_target_layers,
            selected_source_layers=selected_source_layers,
            layer_ranking=None,
            calibration_score=None,
            attention_importance=None,
        )

    if num_layers_to_select >= len(candidate_layers):
        selected_target_layers = candidate_layers
        selected_source_layers = sorted({tgt_to_src[layer_idx] for layer_idx in selected_target_layers})
        return EdgeCalibrationResult(
            selected_target_layers=selected_target_layers,
            selected_source_layers=selected_source_layers,
            layer_ranking=None,
            calibration_score=None,
            attention_importance=None,
        )

    if config.random_selection:
        generator = torch.Generator().manual_seed(config.seed)
        candidate_tensor = torch.tensor(candidate_layers, dtype=torch.long)
        permutation = candidate_tensor[torch.randperm(len(candidate_layers), generator=generator)].tolist()
        selected_target_layers = sorted(permutation[:num_layers_to_select])
        selected_source_layers = sorted({tgt_to_src[layer_idx] for layer_idx in selected_target_layers})
        return EdgeCalibrationResult(
            selected_target_layers=selected_target_layers,
            selected_source_layers=selected_source_layers,
            layer_ranking=None,
            calibration_score=None,
            attention_importance=None,
        )

    target_model = ctx.tp.get_model(edge.tgt_id)
    target_model.eval()

    layer_score_samples: List[np.ndarray] = []
    context_length = _openwebtext_prefix_tokens(config)
    processed = 0
    log_interval = 1 if str(config.log_level).upper() == "DEBUG" else max(1, min(25, config.calib_size))

    for batch_idx, batch in enumerate(calibration_batches, start=1):
        token_ids = batch.to(config.device)
        scores = _compute_attention_importance_for_batch(
            model=target_model,
            token_ids=token_ids,
            context_length=context_length,
        )
        if scores is not None:
            layer_score_samples.append(scores)
        processed += int(token_ids.shape[0])
        if batch_idx % log_interval == 0 or processed >= config.calib_size:
            logging.info("%s | %s selection progress: %d/%d sequences", edge.id, config.calibration_dataset, processed, config.calib_size)
        if processed >= config.calib_size:
            break

    if not layer_score_samples:
        logging.warning(
            "No attention tensors were returned during KVComm %s layer selection for %s. Falling back to Gaussian prior only.",
            config.calibration_dataset,
            edge.id,
        )
        mean_scores = np.zeros(target_spec.num_layers, dtype=np.float32)
    else:
        mean_scores = np.nanmean(np.stack(layer_score_samples, axis=0), axis=0)
        mean_scores = np.nan_to_num(mean_scores, nan=0.0, posinf=0.0, neginf=0.0)

    layer_ranking, combined_scores = _rank_layers_from_scores(
        raw_scores=mean_scores,
        alpha=float(config.alpha),
        mu=float(config.mu),
        sigma=float(config.sigma),
    )
    candidate_set = set(candidate_layers)
    filtered_ranking = [layer_idx for layer_idx in layer_ranking if layer_idx in candidate_set]
    selected_target_layers = sorted(filtered_ranking[:num_layers_to_select])
    selected_source_layers = sorted({tgt_to_src[layer_idx] for layer_idx in selected_target_layers})
    calibration_score = float(np.mean([combined_scores[layer_idx] for layer_idx in selected_target_layers]))

    return EdgeCalibrationResult(
        selected_target_layers=selected_target_layers,
        selected_source_layers=selected_source_layers,
        layer_ranking=filtered_ranking,
        calibration_score=calibration_score,
        attention_importance=combined_scores,
    )


def run_train(ctx: Context, gpu_memory_tracker: GPUMemoryTracker) -> Path:
    config = ctx.config
    nodes = ctx.nodes
    edges = ctx.edges
    set_seed(config.seed)
    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    config_path = get_train_config_path(output_path)
    write_json(str(config_path), asdict(config))

    log_path = get_train_log_path(output_path)
    logging.info("Starting KVComm layer selection")
    logging.info("train_config=%s", asdict(config))
    logging.info("nodes=%s", [node.id for node in nodes])
    logging.info("edges=%s", [edge.id for edge in edges])
    logging.info(
        "layer_selection_source=%s/train",
        config.calibration_dataset,
    )
    logging.info(
        "selection_total_tokens=%d | selection_prefix_tokens=%d | calib_size=%d",
        _openwebtext_total_tokens(config),
        _openwebtext_prefix_tokens(config),
        config.calib_size,
    )

    compatibility = inspect_kvcomm_model_compatibility(ctx)
    if not compatibility["is_compatible"]:
        logging.error(compatibility["message"])
        raise SystemExit(compatibility["message"])
    logging.info(compatibility["message"])

    calibration_batches_by_target = {
        target_node_id: _build_openwebtext_calibration_batches(
            ctx=ctx,
            config=config,
            tokenizer=ctx.tp.get_tokenizer(target_node_id),
        )
        for target_node_id in sorted({edge.tgt_id for edge in edges})
    }
    for target_node_id, calibration_batches in calibration_batches_by_target.items():
        if not calibration_batches:
            raise RuntimeError(f"Failed to sample any {config.calibration_dataset} sequences for target {target_node_id}.")
        logging.info("Collected %d %s batch(es) for target=%s layer selection", len(calibration_batches), config.calibration_dataset, target_node_id)

    node_model_ids = {node.id: node.model_id for node in nodes}
    calibration_by_translator_id: Dict[str, EdgeCalibrationResult] = {}
    for edge in edges:
        translator_id = get_translator_id(node_model_ids[edge.src_id], node_model_ids[edge.tgt_id])
        if translator_id not in calibration_by_translator_id:
            calibration_batches = calibration_batches_by_target[edge.tgt_id]
            calibration_by_translator_id[translator_id] = _select_layers_for_edge(
                ctx=ctx,
                edge=edge,
                config=config,
                calibration_batches=calibration_batches,
            )
        result = calibration_by_translator_id[translator_id]
        logging.info(
            "%s | translator_id=%s | selected_target_layers=%s | selected_source_layers=%s | calibration_score=%s",
            edge.id,
            translator_id,
            result.selected_target_layers,
            result.selected_source_layers,
            "N/A" if result.calibration_score is None else f"{result.calibration_score:.6f}",
        )
        if result.layer_ranking is not None:
            logging.info("%s | translator_id=%s | layer_ranking=%s", edge.id, translator_id, result.layer_ranking)

    translator_pool = initialize_translators(
        ctx,
        selected_target_layers_by_translator_id={
            translator_id: result.selected_target_layers
            for translator_id, result in calibration_by_translator_id.items()
        },
        selected_source_layers_by_translator_id={
            translator_id: result.selected_source_layers
            for translator_id, result in calibration_by_translator_id.items()
        },
        layer_ranking_by_translator_id={
            translator_id: result.layer_ranking
            for translator_id, result in calibration_by_translator_id.items()
        },
        calibration_score_by_translator_id={
            translator_id: result.calibration_score
            for translator_id, result in calibration_by_translator_id.items()
        },
        attention_importance_by_translator_id={
            translator_id: result.attention_importance
            for translator_id, result in calibration_by_translator_id.items()
        },
    )
    checkpoint_dir = save_translator_checkpoints(output_path, translator_pool)
    logging.info("Saved KVComm layer-selection translator checkpoints to %s", checkpoint_dir)
    return checkpoint_dir


def load_translator_pool_from_checkpoint(
    checkpoint_dir_path: str,
    nodes: List[Node],
    edges: List[Edge],
    device_override: Optional[str] = None,
) -> Tuple[Context, TranslatorPool]:
    checkpoint_dir_path_obj = Path(checkpoint_dir_path)
    if not checkpoint_dir_path_obj.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir_path_obj}")
    train_config_path = get_train_config_path(checkpoint_dir_path_obj)
    if not train_config_path.exists():
        raise FileNotFoundError(f"Train config not found under checkpoint directory: {checkpoint_dir_path}")
    config = TrainConfig(**read_json(train_config_path))
    if device_override is not None:
        config.device = resolve_device(device_override)
    config.output_path = str(checkpoint_dir_path_obj)
    models, tokenizers = build_models_and_tokenizers(config, nodes)
    ctx = Context(config, nodes, edges, TranslatorPool(models, tokenizers), ChannelManager(edges))
    translator_pool = load_kvcomm_translator_checkpoints(ctx)
    translator_pool.eval()
    return ctx, translator_pool
